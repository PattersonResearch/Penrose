"""Worker-count resolution and live concurrency throttling for claim runs."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


MAX_REQUESTED_WORKERS = 16
DEFAULT_AUTO_HARD_CAP = 8
DEFAULT_RAM_PER_SANDBOX_GB = 1.5
GOVERNOR_SUCCESS_THRESHOLD = 3
GOVERNOR_DECREASE_DEBOUNCE_S = 1.0


@dataclass(frozen=True)
class WorkerCountResolution:
    count: int
    bound: str
    cpu_ceiling: int
    ram_ceiling: int
    hard_cap: int
    requested: object


def _clamp_int(n: int, low: int = 1, high: int = MAX_REQUESTED_WORKERS) -> int:
    return max(low, min(high, n))


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return _clamp_int(value)


def _available_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        available = float(psutil.virtual_memory().available)
        if available > 0:
            return available / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and pages > 0:
            return (page_size * pages) / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass

    return 8.0


def resolve_worker_count_details(requested: int | str | None) -> WorkerCountResolution:
    if requested is None:
        return WorkerCountResolution(
            count=1,
            bound="serial",
            cpu_ceiling=1,
            ram_ceiling=1,
            hard_cap=1,
            requested=requested,
        )
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        cpu_ceiling = (os.cpu_count() or 2) - 1
        ram_per_sandbox_gb = _float_env(
            "PENROSE_WORKER_RAM_PER_SANDBOX_GB", DEFAULT_RAM_PER_SANDBOX_GB)
        try:
            ram_ceiling = max(1, int(_available_ram_gb() // ram_per_sandbox_gb))
        except Exception:  # noqa: BLE001
            ram_ceiling = max(1, int(8.0 // DEFAULT_RAM_PER_SANDBOX_GB))
        hard_cap = _int_env("PENROSE_WORKER_HARD_CAP", DEFAULT_AUTO_HARD_CAP)
        ceilings = {"cpu": cpu_ceiling, "ram": ram_ceiling, "cap": hard_cap}
        bound = min(ceilings, key=ceilings.get)
        return WorkerCountResolution(
            count=max(1, min(cpu_ceiling, ram_ceiling, hard_cap)),
            bound=bound,
            cpu_ceiling=cpu_ceiling,
            ram_ceiling=ram_ceiling,
            hard_cap=hard_cap,
            requested=requested,
        )
    try:
        return WorkerCountResolution(
            count=_clamp_int(int(requested)),
            bound="requested",
            cpu_ceiling=0,
            ram_ceiling=0,
            hard_cap=MAX_REQUESTED_WORKERS,
            requested=requested,
        )
    except (TypeError, ValueError):
        return WorkerCountResolution(
            count=1,
            bound="fallback",
            cpu_ceiling=0,
            ram_ceiling=0,
            hard_cap=MAX_REQUESTED_WORKERS,
            requested=requested,
        )


def resolve_worker_count(requested: int | str | None) -> int:
    return resolve_worker_count_details(requested).count


DEFAULT_UNSPECIFIED_CAP = 4


def default_worker_count() -> int:
    """The default when the operator specifies nothing: min(cap, auto).

    `auto` picks the hardware-safe max (min of cpu/ram/hard_cap); capping it at 4 keeps the default
    moderate on capable hardware while `auto`'s own ceilings auto-REDUCE it on constrained machines
    (e.g. a 2-core box resolves auto to 1-2, so the default follows down and never swamps). Fail-open
    to the cap if resolution errors.
    """
    try:
        cap = _int_env("PENROSE_WORKER_DEFAULT_CAP", DEFAULT_UNSPECIFIED_CAP)
        return max(1, min(cap, resolve_worker_count("auto")))
    except Exception:  # noqa: BLE001
        return DEFAULT_UNSPECIFIED_CAP


class ClaimConcurrencyGovernor:
    """AIMD permit governor layered below the fixed ThreadPoolExecutor size."""

    def __init__(
        self,
        max_permits: int = 1,
        *,
        success_threshold: int = GOVERNOR_SUCCESS_THRESHOLD,
        decrease_debounce_s: float = GOVERNOR_DECREASE_DEBOUNCE_S,
        clock=time.monotonic,
    ):
        self._condition = threading.Condition()
        self._clock = clock
        self._success_threshold = max(1, int(success_threshold))
        self._decrease_debounce_s = max(0.0, float(decrease_debounce_s))
        self._max_permits = max(1, int(max_permits))
        self._permits = self._max_permits
        self._active = 0
        self._successes = 0
        self._last_decrease_at: float | None = None

    @property
    def max_permits(self) -> int:
        with self._condition:
            return self._max_permits

    @property
    def permits(self) -> int:
        with self._condition:
            return self._permits

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    def reset(self, max_permits: int) -> None:
        with self._condition:
            self._max_permits = max(1, int(max_permits))
            self._permits = self._max_permits
            self._active = 0
            self._successes = 0
            self._last_decrease_at = None
            self._condition.notify_all()

    @contextmanager
    def permit(self):
        acquired = False
        try:
            self.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                self.release()

    def acquire(self) -> None:
        if self.max_permits <= 1:
            return
        with self._condition:
            while self._active >= self._permits:
                self._condition.wait()
            self._active += 1

    def release(self) -> None:
        if self.max_permits <= 1:
            return
        with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()

    def report_rate_limit(self) -> None:
        if self.max_permits <= 1:
            return
        with self._condition:
            now = self._clock()
            if (
                self._last_decrease_at is not None
                and now - self._last_decrease_at < self._decrease_debounce_s
            ):
                return
            self._permits = max(1, self._permits // 2)
            self._successes = 0
            self._last_decrease_at = now
            self._condition.notify_all()

    def report_success(self) -> None:
        if self.max_permits <= 1:
            return
        with self._condition:
            if self._permits >= self._max_permits:
                self._successes = 0
                return
            self._successes += 1
            if self._successes >= self._success_threshold:
                self._permits = min(self._max_permits, self._permits + 1)
                self._successes = 0
                self._condition.notify_all()


CLAIM_CONCURRENCY_GOVERNOR = ClaimConcurrencyGovernor()


def configure_claim_governor(worker_count: int) -> ClaimConcurrencyGovernor:
    try:
        CLAIM_CONCURRENCY_GOVERNOR.reset(worker_count)
    except Exception:  # noqa: BLE001
        pass
    return CLAIM_CONCURRENCY_GOVERNOR


def report_rate_limit() -> None:
    try:
        CLAIM_CONCURRENCY_GOVERNOR.report_rate_limit()
    except Exception:  # noqa: BLE001
        pass


def report_success() -> None:
    try:
        CLAIM_CONCURRENCY_GOVERNOR.report_success()
    except Exception:  # noqa: BLE001
        pass
