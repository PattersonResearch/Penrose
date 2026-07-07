import threading
import time
from concurrent.futures import ThreadPoolExecutor

from penrose import worker_control
from penrose.pipeline import run as runmod


def test_resolve_worker_count_auto_respects_hard_cap(monkeypatch):
    monkeypatch.setattr(worker_control.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(worker_control, "_available_ram_gb", lambda: 64.0)
    monkeypatch.setenv("PENROSE_WORKER_HARD_CAP", "4")
    monkeypatch.setenv("PENROSE_WORKER_RAM_PER_SANDBOX_GB", "1.5")

    resolution = worker_control.resolve_worker_count_details("auto")

    assert runmod.resolve_worker_count("AUTO") == 4
    assert 1 <= resolution.count <= 4
    assert resolution.bound == "cap"


def test_resolve_worker_count_auto_binding_ceilings(monkeypatch):
    monkeypatch.setenv("PENROSE_WORKER_RAM_PER_SANDBOX_GB", "2")
    monkeypatch.setenv("PENROSE_WORKER_HARD_CAP", "8")

    monkeypatch.setattr(worker_control.os, "cpu_count", lambda: 3)
    monkeypatch.setattr(worker_control, "_available_ram_gb", lambda: 64.0)
    cpu_bound = worker_control.resolve_worker_count_details("auto")
    assert cpu_bound.count == 2
    assert cpu_bound.bound == "cpu"

    monkeypatch.setattr(worker_control.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(worker_control, "_available_ram_gb", lambda: 5.0)
    ram_bound = worker_control.resolve_worker_count_details("auto")
    assert ram_bound.count == 2
    assert ram_bound.bound == "ram"

    monkeypatch.setattr(worker_control.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(worker_control, "_available_ram_gb", lambda: 64.0)
    cap_bound = worker_control.resolve_worker_count_details("auto")
    assert cap_bound.count == 8
    assert cap_bound.bound == "cap"


def test_resolve_worker_count_int_none_and_bad_env(monkeypatch):
    assert runmod.resolve_worker_count(None) == 1
    assert runmod.resolve_worker_count(0) == 1
    assert runmod.resolve_worker_count(99) == 16
    assert runmod.resolve_worker_count("6") == 6
    assert runmod.resolve_worker_count("bogus") == 1

    monkeypatch.setenv("PENROSE_MAX_CLAIM_WORKERS", "bogus")
    assert runmod._claim_worker_resolution(None).count == 1

    monkeypatch.setenv("PENROSE_MAX_CLAIM_WORKERS", "auto")
    monkeypatch.setenv("PENROSE_WORKER_HARD_CAP", "not-an-int")
    monkeypatch.setenv("PENROSE_WORKER_RAM_PER_SANDBOX_GB", "not-a-float")
    monkeypatch.setattr(worker_control.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(worker_control, "_available_ram_gb", lambda: 64.0)
    assert runmod._claim_worker_resolution(None).count == worker_control.DEFAULT_AUTO_HARD_CAP


def test_governor_aimd_halves_debounced_and_restores():
    now = [100.0]
    governor = worker_control.ClaimConcurrencyGovernor(
        8,
        success_threshold=3,
        decrease_debounce_s=1.0,
        clock=lambda: now[0],
    )

    assert governor.permits == 8
    governor.report_rate_limit()
    assert governor.permits == 4
    governor.report_rate_limit()
    assert governor.permits == 4

    now[0] += 1.1
    governor.report_rate_limit()
    assert governor.permits == 2

    governor.report_success()
    governor.report_success()
    assert governor.permits == 2
    governor.report_success()
    assert governor.permits == 3

    for _ in range(15):
        governor.report_success()
    assert governor.permits == 8


def test_governor_concurrent_acquire_release_respects_permits():
    governor = worker_control.ClaimConcurrencyGovernor(4, decrease_debounce_s=0.0)
    governor.report_rate_limit()
    assert governor.permits == 2

    active = 0
    max_active = 0
    lock = threading.Lock()
    start = threading.Event()

    def task():
        nonlocal active, max_active
        start.wait()
        with governor.permit():
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(task) for _ in range(4)]
        start.set()
        assert [f.result(timeout=2) for f in futures] == [True, True, True, True]

    assert max_active <= 2
    assert governor.active == 0


def test_default_worker_count_is_capped_at_four():
    """The unspecified-operator default is min(4, auto): 4 on capable hardware, auto-reduced below."""
    from penrose import worker_control as wc
    d = wc.default_worker_count()
    assert 1 <= d <= 4
    assert d == min(4, wc.resolve_worker_count("auto"))


def test_default_worker_count_follows_down_on_small_hardware(monkeypatch):
    """On a constrained box, auto resolves low and the default follows it down (never swamps)."""
    from penrose import worker_control as wc
    monkeypatch.setattr(wc.os, "cpu_count", lambda: 2)          # 2 cores -> cpu_ceiling 1
    monkeypatch.setattr(wc, "_available_ram_gb", lambda: 3.0)   # 3GB -> ram_ceiling ~2
    assert wc.default_worker_count() == 1                        # min(4, min(1, 2, 8)) = 1
