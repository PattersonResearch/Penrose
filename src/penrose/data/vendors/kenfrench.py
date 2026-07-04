"""Ken French Data Library adapter — keyless, free, as-displayed.

Kenneth French's Data Library (Dartmouth) is the canonical, free, decades-old
source of U.S. equity research portfolios and factors: the momentum factor
(WML/UMD), the 10 momentum decile portfolios, size/value/profitability/investment
sorts, industry portfolios, and the short/long-term reversal portfolios. Academic
anomaly papers routinely express their claims against these exact series so any
reader can replicate — which makes ONE adapter here general infrastructure that
unlocks a whole class of equity factor/anomaly claims, not a per-paper scraper.

Keyless (like stooq): `available()` is always True; every network/parse error
fails open to None and never raises into bundle construction.

The files are zipped CSVs with a quirky shape — a text preamble, then one or more
labelled data blocks (e.g. "Average Value Weighted Returns -- Daily", then
"Average Equal Weighted Returns -- Daily"), each a header row of column names
followed by `YYYYMMDD, val, val, ...` rows. Values are in PERCENT with -99.99 /
-999 sentinels for missing. We parse into blocks, pick one block + one column, map
sentinels to NaN, and (by default) convert percent returns to decimal fractions so
downstream modules consume them as plain daily returns.

Grade = "as_displayed": these are current/revised research returns, not a
point-in-time vintage.
"""
from __future__ import annotations

from io import BytesIO
import os
import re
import tempfile
import time
import urllib.request
import zipfile

import pandas as pd

NAME = "kenfrench"
PROVENANCE_GRADE = "as_displayed"

# Public download root for the library's "_CSV.zip" files.
_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# A data row begins with a 6- (YYYYMM) or 8-digit (YYYYMMDD) date as the first cell.
_DATE_ROW = re.compile(r"^\s*(\d{6,8})\s*,")

# Sentinels French uses for missing observations (no real daily return is <= -99%).
_MISSING_FLOOR = -99.0

# On-disk cache: these files update at most monthly, so a 1-day TTL spares
# Dartmouth's server across repeated runs. Fail-open — cache problems are ignored.
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "penrose_kenfrench_cache")
_CACHE_TTL_SEC = 24 * 60 * 60

# Hard caps (defense-in-depth against a hostile/compromised endpoint or zip bomb).
# Real French files are well under 5 MB raw and a few MB decompressed.
_MAX_RAW_BYTES = 64 * 1024 * 1024
_MAX_CSV_BYTES = 128 * 1024 * 1024

# Conservative file-stem charset. We REJECT anything outside it rather than
# rewriting it, so a malformed spec can never silently alias onto a real dataset.
_SAFE_DATASET = re.compile(r"[A-Za-z0-9_.\-]+")


def available() -> bool:
    """Keyless; network failure is handled inside fetch()."""
    return True


def _safe_dataset(dataset: str) -> str | None:
    """Return the dataset stem iff it is a safe file-stem; else None (rejected, not rewritten)."""
    ds = (dataset or "").strip()
    if not ds or ds in (".", "..") or not _SAFE_DATASET.fullmatch(ds):
        return None
    return ds


def _download(dataset: str) -> bytes | None:
    """Return the raw zip bytes for `<dataset>_CSV.zip`, disk-cached. None on error."""
    safe = _safe_dataset(dataset)
    if safe is None:
        return None
    cache_path = os.path.join(_CACHE_DIR, f"{safe}_CSV.zip")
    try:
        st = os.stat(cache_path)
        if (time.time() - st.st_mtime) < _CACHE_TTL_SEC and st.st_size > 0:
            with open(cache_path, "rb") as fh:
                cached = fh.read(_MAX_RAW_BYTES + 1)
            if 0 < len(cached) <= _MAX_RAW_BYTES:
                return cached
    except OSError:
        pass  # no fresh/usable cache -> fetch below
    try:
        url = f"{_BASE}{safe}_CSV.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.1"})
        with urllib.request.urlopen(req, timeout=45) as r:
            raw = r.read(_MAX_RAW_BYTES + 1)       # bounded read — refuse an oversized body
    except Exception:  # noqa: BLE001 — a download error never breaks a run
        return None
    if not raw or len(raw) > _MAX_RAW_BYTES:
        return None
    try:  # best-effort ATOMIC cache write; a partial file never becomes visible to readers
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, cache_path)            # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except OSError:
        pass
    return raw


def _csv_from_zip(raw: bytes, dataset: str) -> str | None:
    """Extract the CSV member matching `dataset` (or the sole member) as text. None on error.

    Refuses to guess: a multi-CSV archive with no unique name match returns None
    rather than picking the first member by archive order. Caps the decompressed
    size before reading (zip-bomb defense).
    """
    try:
        zf = zipfile.ZipFile(BytesIO(raw))
        csvs = [zi for zi in zf.infolist() if zi.filename.lower().endswith(".csv")]
        if not csvs:
            return None
        stem = (dataset or "").lower()
        matched = [zi for zi in csvs if stem and stem in zi.filename.lower()]
        if len(matched) == 1:
            chosen = matched[0]
        elif len(csvs) == 1:
            chosen = csvs[0]
        else:
            return None                            # ambiguous archive -> refuse
        if chosen.file_size > _MAX_CSV_BYTES:
            return None
        return zf.read(chosen).decode("latin-1")
    except Exception:  # noqa: BLE001
        return None


def _blocks(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Split a French CSV into (header_tokens, data_rows) blocks.

    A block is a maximal run of date-led data rows; its header is the nearest
    preceding comma-bearing line (the column-name row). Blocks with no header line
    get an empty header (callers may still select by positional index).
    """
    blocks: list[tuple[list[str], list[list[str]]]] = []
    last_header: list[str] = []
    cur: list[list[str]] = []

    def flush():
        nonlocal cur
        if cur:
            blocks.append((last_header, cur))
            cur = []

    for line in text.splitlines():
        if _DATE_ROW.match(line):
            cur.append([c.strip() for c in line.split(",")])
            continue
        # non-data line: end any open block, then maybe remember a header
        flush()
        if "," in line and line.strip():
            # header tokens minus the (empty) leading date column
            toks = [c.strip() for c in line.split(",")]
            last_header = toks[1:]
    flush()
    return blocks


def _pick_column(header: list[str], n_cols: int, column) -> int | None:
    """Resolve `column` (int index or name) to a 0-based data-column position.

    Fails CLOSED (None) rather than guessing: an empty, missing, or AMBIGUOUS name
    returns None so the series is absent (acceptable) instead of silently returning
    the wrong column's data (a correctness violation). Exact label wins; a
    substring match is honoured ONLY when it uniquely identifies one column.
    """
    if column is None:
        return 0 if n_cols == 1 else None  # unambiguous only when there's one column
    if isinstance(column, int):
        return column if 0 <= column < n_cols else None
    name = str(column).strip().lower()
    if not name:
        return None
    lowered = [h.lower() for h in header[:n_cols]]
    for i, h in enumerate(lowered):                       # exact label first
        if h == name:
            return i
    hits = [i for i, h in enumerate(lowered) if name in h]  # then UNIQUE substring only
    return hits[0] if len(hits) == 1 else None


def parse_french_csv(text: str, *, section: int = 0, column=None,
                     as_return: bool = True) -> pd.Series | None:
    """Pure parser (no network) — pick one block + one column as a daily UTC Series.

    section: which data block (0 = first; for return files that is the
             value-weighted block). column: column name (case-insensitive,
             exact-then-contains) or 0-based index; None means "the only column".
    as_return: divide percent values by 100 to yield decimal returns.
    Returns None on any structural miss so the adapter can fail open.
    """
    blocks = _blocks(text or "")
    if not blocks or not (0 <= section < len(blocks)):
        return None
    header, rows = blocks[section]
    n_cols = max((len(r) - 1 for r in rows), default=0)
    if n_cols <= 0:
        return None
    col = _pick_column(header, n_cols, column)
    if col is None:
        return None
    dates, vals = [], []
    for r in rows:
        if len(r) <= col + 1:
            continue
        raw_date = r[0]
        try:
            v = float(r[col + 1])
        except (TypeError, ValueError):
            continue
        if v <= _MISSING_FLOOR:           # -99.99 / -999 sentinels -> drop
            continue
        dates.append(raw_date)
        vals.append(v / 100.0 if as_return else v)
    if not vals:
        return None
    fmt = "%Y%m%d" if len(dates[0]) == 8 else "%Y%m"
    idx = pd.to_datetime(dates, format=fmt, utc=True, errors="coerce").normalize()
    s = pd.Series(vals, index=idx, name=str(column if column is not None else "value"))
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s if not s.empty else None


def fetch(spec: dict):
    """Return (daily tz-aware UTC pd.Series, provenance) for a French series, or None.

    spec keys:
      dataset (required) — library file stem WITHOUT the "_CSV.zip" suffix,
                           e.g. "F-F_Momentum_Factor_daily" or
                           "10_Portfolios_Prior_12_2_Daily".
      column  (optional) — column name (case-insensitive, exact-then-contains,
                           e.g. "Mom", "Lo PRIOR", "Hi PRIOR") or 0-based int index.
                           Omit only when the block has a single column.
      section (optional) — which data block; default 0 (the value-weighted block).
      as_return (optional) — divide percent by 100 (default True).
      start/end (optional) — ISO date bounds, filtered locally.

    Never raises — any error returns None.
    """
    if not isinstance(spec, dict):
        return None
    dataset = str(spec.get("dataset") or "").strip()
    if not dataset:
        return None
    try:
        raw = _download(dataset)
        if raw is None:
            return None
        text = _csv_from_zip(raw, dataset)
        if text is None:
            return None
        s = parse_french_csv(
            text,
            section=int(spec.get("section", 0) or 0),
            column=spec.get("column"),
            as_return=bool(spec.get("as_return", True)),
        )
        if s is None or s.empty:
            return None
        if spec.get("start"):
            s = s[s.index >= pd.Timestamp(spec["start"], tz="UTC")]
        if spec.get("end"):
            s = s[s.index <= pd.Timestamp(spec["end"], tz="UTC")]
        if s.empty:
            return None
        col = spec.get("column")
        tag = f"{dataset}:{col}" if col is not None else dataset
        return s, f"kenfrench-csv:{tag}"
    except Exception:  # noqa: BLE001 — a data-fetch error never breaks a run
        return None
