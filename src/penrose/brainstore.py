"""Native SQLite-backed brain store for penrose atoms.

It replaces an earlier external knowledge-store runtime with a pure-stdlib store.
Reads fail soft, writes are exposed only through ``PromotionClient`` in
``brain.py``.
"""
from __future__ import annotations

import hashlib
import builtins
import json
import math
import re
import sqlite3
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_ident(ident: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in ident.lower())


def _slug(kind: str, ident: str) -> str:
    return f"atoms/penrose/{kind}/{_safe_ident(str(ident))}"


def _frontmatter(d: dict) -> str:
    lines = ["---"]
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (builtins.list, dict)):
            lines.append(f"{k}: {json.dumps(v, sort_keys=True)}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return value
    return value


def _parse_content(content: str) -> tuple[dict, str]:
    text = content or ""
    if not text.startswith("---\n"):
        return {}, text
    try:
        _, rest = text.split("---\n", 1)
        fm_text, body = rest.split("\n---", 1)
    except ValueError:
        return {}, text
    fm: dict[str, Any] = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key:
            fm[key] = _parse_scalar(value)
    if body.startswith("\n"):
        body = body[1:]
    return fm, body


def _title(frontmatter: dict, body: str, slug: str) -> str:
    for key in ("title", "statement", "claim_id", "decision_id", "principle_id", "concept_id"):
        value = frontmatter.get(key)
        if str(value or "").strip():
            return str(value).strip()
    for line in (body or "").splitlines():
        line = line.strip().strip("#").strip()
        if line:
            return line[:160]
    return slug.rsplit("/", 1)[-1]


def _atom_search_text(row: sqlite3.Row | dict[str, Any]) -> str:
    return " ".join(str(x or "") for x in (
        row["slug"], row["title"], row["frontmatter_json"], row["body"],
    ))


def _format_score(score: float) -> str:
    return f"{score:.4f}".rstrip("0").rstrip(".")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value
    except Exception:  # noqa: BLE001
        return


class BrainStore:
    def __init__(self, db_path: str | Path | None = None, *, auto_rebuild: bool = True):
        self.db_path = Path(db_path) if db_path is not None else config.BRAINSTORE_DB
        self.auto_rebuild = auto_rebuild

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema(conn)
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS atoms (
                slug TEXT PRIMARY KEY,
                kind TEXT,
                ident TEXT,
                frontmatter_json TEXT,
                body TEXT,
                content TEXT,
                title TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                from_slug TEXT,
                to_slug TEXT,
                link_type TEXT,
                PRIMARY KEY (from_slug, to_slug, link_type)
            );
            CREATE TABLE IF NOT EXISTS vectors (
                slug TEXT PRIMARY KEY,
                dim INTEGER,
                vec_json TEXT,
                text_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )

    def _flat_signature(self) -> str | None:
        parts = []
        for path in (config.DECISIONS_LOG, config.CONCEPTS, config.PRINCIPLES_LOG):
            p = Path(path)
            try:
                stat = p.stat()
            except FileNotFoundError:
                parts.append(f"{p}:missing")
                continue
            except Exception:  # noqa: BLE001
                return None
            parts.append(f"{p}:{stat.st_size}:{stat.st_mtime_ns}")
        if all(part.endswith(":missing") for part in parts):
            return None
        return _text_hash("|".join(parts))

    def _atom_count(self, conn: sqlite3.Connection) -> int:
        try:
            return int(conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0])
        except Exception:  # noqa: BLE001
            return 0

    def _stored_flat_signature(self, conn: sqlite3.Connection) -> str | None:
        try:
            row = conn.execute("SELECT value FROM _meta WHERE key = ?", ("flat_sig",)).fetchone()
            return str(row["value"]) if row is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _mark_rebuilt(self, conn: sqlite3.Connection, flat_sig: str | None = None) -> None:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO _meta(key, value) VALUES (?, ?)",
                ("rebuilt_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )
            if flat_sig is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO _meta(key, value) VALUES (?, ?)",
                    ("flat_sig", flat_sig),
                )
        except Exception:  # noqa: BLE001
            return

    def _maybe_rebuild(self, conn: sqlite3.Connection) -> None:
        flat_sig = self._flat_signature()
        if self.auto_rebuild and flat_sig is not None and self._stored_flat_signature(conn) != flat_sig:
            rebuild_from_flat_files(db_path=self.db_path)
            self._mark_rebuilt(conn, flat_sig)

    def _put(self, slug: str, content: str) -> bool:
        try:
            fm, body = _parse_content(content)
            parts = slug.split("/")
            kind = str(fm.get("kind") or (parts[2] if len(parts) > 2 else "atom"))
            ident = str(parts[-1] if parts else slug)
            title = _title(fm, body, slug)
            frontmatter_json = json.dumps(fm, sort_keys=True)
            created_at = str(fm.get("created_at") or fm.get("logged_at") or time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO atoms(slug, kind, ident, frontmatter_json, body, content, title, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(slug) DO UPDATE SET
                        kind=excluded.kind,
                        ident=excluded.ident,
                        frontmatter_json=excluded.frontmatter_json,
                        body=excluded.body,
                        content=excluded.content,
                        title=excluded.title,
                        created_at=excluded.created_at
                    """,
                    (slug, kind, ident, frontmatter_json, body, content, title, created_at),
                )
                self._embed_atom(
                    conn,
                    slug,
                    _atom_search_text({
                        "slug": slug,
                        "title": title,
                        "frontmatter_json": frontmatter_json,
                        "body": body,
                    }),
                )
            return True
        except Exception:  # noqa: BLE001
            return False

    def _embed_atom(self, conn: sqlite3.Connection, slug: str, text: str) -> None:
        try:
            from . import llm

            if not getattr(llm, "embed_local_available", lambda: False)():
                return
            vec = llm.embed_local(text)
            if not isinstance(vec, builtins.list) or not vec:
                return
            vals = [float(x) for x in vec]
            conn.execute(
                """
                INSERT INTO vectors(slug, dim, vec_json, text_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    dim=excluded.dim,
                    vec_json=excluded.vec_json,
                    text_hash=excluded.text_hash
                """,
                (slug, len(vals), json.dumps(vals), _text_hash(text)),
            )
        except Exception:  # noqa: BLE001
            return

    def _link(self, from_slug: str, to_slug: str, link_type: str) -> bool:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO edges(from_slug, to_slug, link_type) VALUES (?, ?, ?)",
                    (from_slug, to_slug, link_type),
                )
            return True
        except Exception:  # noqa: BLE001
            return False

    def get(self, slug: str) -> Optional[str]:
        try:
            with self._connect() as conn:
                self._maybe_rebuild(conn)
                row = conn.execute("SELECT content FROM atoms WHERE slug = ?", (slug,)).fetchone()
            return str(row["content"]) if row and str(row["content"]).strip() else None
        except Exception:  # noqa: BLE001
            return None

    def _rows(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        try:
            return builtins.list(conn.execute(
                "SELECT slug, title, frontmatter_json, body FROM atoms ORDER BY slug"
            ).fetchall())
        except Exception:  # noqa: BLE001
            return []

    def _vector_scores(self, conn: sqlite3.Connection, query: str) -> dict[str, float]:
        try:
            from . import llm

            if not getattr(llm, "embed_local_available", lambda: False)():
                return {}
            q_vec = llm.embed_local(query)
            if not q_vec:
                return {}
            scores: dict[str, float] = {}
            for row in conn.execute("SELECT slug, vec_json FROM vectors ORDER BY slug").fetchall():
                try:
                    vec = json.loads(row["vec_json"])
                except Exception:  # noqa: BLE001
                    continue
                score = float(llm.cosine(q_vec, [float(x) for x in vec]))
                if score > 0:
                    scores[str(row["slug"])] = score
            return scores
        except Exception:  # noqa: BLE001
            return {}

    def _lexical_scores(self, query: str, rows: list[sqlite3.Row]) -> dict[str, float]:
        q_counts = Counter(_tokens(query))
        if not q_counts:
            return {}
        q_terms = set(q_counts)
        docs = {str(row["slug"]): _tokens(_atom_search_text(row)) for row in rows}
        n_docs = max(1, len(docs))
        df = Counter(term for toks in docs.values() for term in set(toks))
        scores: dict[str, float] = {}
        for slug, toks in docs.items():
            if not toks:
                continue
            counts = Counter(toks)
            score = 0.0
            for term in q_terms & set(counts):
                idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
                score += idf * counts[term] / (len(toks) ** 0.5)
            if score > 0:
                scores[slug] = score
        return scores

    @staticmethod
    def _rrf_scores(*score_maps: dict[str, float], k: int = 60) -> dict[str, float]:
        fused: dict[str, float] = {}
        for score_map in score_maps:
            ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
            for rank, (slug, _) in enumerate(ranked, start=1):
                fused[slug] = fused.get(slug, 0.0) + (1.0 / (k + rank))
        return fused

    def search(self, query: str, n: int = 10) -> str:
        try:
            q = (query or "").strip()
            if not q or n <= 0:
                return ""
            with self._connect() as conn:
                self._maybe_rebuild(conn)
                rows = self._rows(conn)
                by_slug = {str(row["slug"]): row for row in rows}
                vector_scores = self._vector_scores(conn, q)
                lexical_rows = [row for row in rows if str(row["slug"]) not in vector_scores]
                lexical_scores = self._lexical_scores(q, lexical_rows)
                scores = self._rrf_scores(vector_scores, lexical_scores)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:n]
            lines = []
            for slug, score in ranked:
                row = by_slug.get(slug)
                if row is None:
                    continue
                lines.append(f"{slug} :: {row['title']} :: {_format_score(float(score))}")
            return "\n".join(lines) + ("\n" if lines else "")
        except Exception:  # noqa: BLE001
            return ""

    def graph(self, slug: str, depth: int = 2) -> str:
        try:
            max_depth = max(0, int(depth))
            with self._connect() as conn:
                self._maybe_rebuild(conn)
                atom_slugs = {
                    str(row["slug"])
                    for row in conn.execute("SELECT slug FROM atoms").fetchall()
                }
                if slug not in atom_slugs:
                    return ""
                edges = [
                    (str(r["from_slug"]), str(r["to_slug"]), str(r["link_type"]))
                    for r in conn.execute(
                        "SELECT from_slug, to_slug, link_type FROM edges ORDER BY from_slug, to_slug, link_type"
                    ).fetchall()
                ]
            adj: dict[str, list[tuple[str, str, str]]] = {}
            for left, right, typ in edges:
                adj.setdefault(left, []).append((right, typ, "out"))
                adj.setdefault(right, []).append((left, typ, "in"))
            for values in adj.values():
                values.sort()
            seen = {slug}
            q = deque([(slug, 0)])
            lines: list[str] = []
            while q:
                current, d = q.popleft()
                if d >= max_depth:
                    continue
                for nxt, typ, direction in adj.get(current, []):
                    line = f"{current} -[{typ}/{direction}]- {nxt}"
                    if line not in lines:
                        lines.append(line)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, d + 1))
            return "\n".join(lines) + ("\n" if lines else "")
        except Exception:  # noqa: BLE001
            return ""

    def list(self, prefix: str = "atoms/penrose", n: int = 200) -> str:
        try:
            if n <= 0:
                return ""
            like = f"{prefix}%"
            with self._connect() as conn:
                self._maybe_rebuild(conn)
                rows = conn.execute(
                    "SELECT slug, title FROM atoms WHERE slug LIKE ? ORDER BY slug LIMIT ?",
                    (like, int(n)),
                ).fetchall()
            lines = [f"{row['slug']} :: {row['title']}" for row in rows]
            return "\n".join(lines) + ("\n" if lines else "")
        except Exception:  # noqa: BLE001
            return ""


def _put_structured(store: BrainStore, kind: str, ident: str, body: str, fm: dict) -> str:
    slug = _slug(kind, ident)
    content = _frontmatter({
        "type": "atom",
        "kind": kind,
        "scope": config.SCOPE,
        "source_id": config.SCOPE,
        "status": "active",
        **fm,
    }) + "\n\n" + body.strip() + "\n"
    store._put(slug, content)
    return slug


def _decision_body(row: dict) -> str:
    return (
        f"Verdict: **{row.get('verdict')}** (kill_reason={row.get('kill_reason')})\n\n"
        f"{row.get('rationale', '')}\n\n"
        f"Claim: {row.get('claim_statement', row.get('claim_id', ''))}\n\n"
        f"Metrics: {json.dumps(row.get('metrics', {}), sort_keys=True, default=str)}"
    )


def _concept_body(row: dict) -> str:
    fields = [
        row.get("statement", ""),
        "",
        f"Mechanism: {row.get('mechanism', '')}",
        f"Source claim: {row.get('source_claim_id', '')}",
        f"Kill reason: {row.get('kill_reason')}",
        f"Evidence: {json.dumps(row.get('evidence_strength', {}), sort_keys=True, default=str)}",
        f"Boundary: {json.dumps(row.get('boundary', {}), sort_keys=True, default=str)}",
        f"Data provenance: {json.dumps(row.get('data_provenance', {}), sort_keys=True, default=str)}",
    ]
    return "\n".join(fields)


def _principle_body(row: dict) -> str:
    return (
        f"{row.get('statement', '')}\n\n"
        f"Supporting kills: {row.get('supporting_kills', [])}\n"
        f"N={row.get('n_observations')} confidence={row.get('confidence')}"
    )


def rebuild_from_flat_files(
    *,
    db_path: str | Path | None = None,
    decisions_path: str | Path | None = None,
    concepts_path: str | Path | None = None,
    principles_path: str | Path | None = None,
) -> dict:
    store = BrainStore(db_path, auto_rebuild=False)
    counts = {"decisions": 0, "concepts": 0, "principles": 0, "edges": 0}
    decisions = Path(decisions_path) if decisions_path is not None else config.DECISIONS_LOG
    concepts = Path(concepts_path) if concepts_path is not None else config.CONCEPTS
    principles = Path(principles_path) if principles_path is not None else config.PRINCIPLES_LOG

    for row in _iter_jsonl(decisions):
        ident = str(row.get("decision_id") or "").strip()
        if not ident:
            continue
        s = _put_structured(store, "decision", ident, _decision_body(row), {
            "decision_id": ident,
            "claim_id": row.get("claim_id"),
            "verdict": row.get("verdict"),
            "kill_reason": row.get("kill_reason"),
            "trust": 0.7 if row.get("verified_by_human") else 0.5,
            "verified_by_human": row.get("verified_by_human"),
            "logged_at": row.get("logged_at"),
        })
        counts["decisions"] += 1
        claim_id = str(row.get("claim_id") or "").strip()
        if claim_id and store._link(s, _slug("claim", claim_id), "evaluated_in"):
            counts["edges"] += 1

    for row in _iter_jsonl(concepts):
        ident = str(row.get("concept_id") or "").strip()
        if not ident:
            continue
        s = _put_structured(store, "observation", ident, _concept_body(row), {
            "concept_id": ident,
            "source_claim_id": row.get("source_claim_id"),
            "abstraction_level": row.get("abstraction_level", "observation"),
            "kill_reason": row.get("kill_reason"),
            "created_at": row.get("created_at"),
            "trust": 0.5,
        })
        counts["concepts"] += 1
        claim_id = str(row.get("source_claim_id") or "").strip()
        if claim_id and store._link(s, _slug("claim", claim_id), "observed_from"):
            counts["edges"] += 1

    for row in _iter_jsonl(principles):
        ident = str(row.get("principle_id") or "").strip()
        if not ident:
            continue
        s = _put_structured(store, "principle", ident, _principle_body(row), {
            "principle_id": ident,
            "n_observations": row.get("n_observations"),
            "confidence": row.get("confidence"),
            "applicable_strategy_classes": row.get("applicable_strategy_classes"),
            "trust": 0.6,
        })
        counts["principles"] += 1
        for decision_id in row.get("supporting_kills", []) or []:
            if store._link(s, _slug("decision", str(decision_id)), "supports_principle"):
                counts["edges"] += 1
    try:
        with store._connect() as conn:
            store._mark_rebuilt(conn, store._flat_signature())
    except Exception:  # noqa: BLE001
        pass
    return counts


def _default_store() -> BrainStore:
    return BrainStore()


def _put(slug: str, content: str) -> bool:
    return _default_store()._put(slug, content)


def _link(from_slug: str, to_slug: str, link_type: str) -> bool:
    return _default_store()._link(from_slug, to_slug, link_type)


def get(slug: str) -> Optional[str]:
    return _default_store().get(slug)


def search(query: str, n: int = 10) -> str:
    return _default_store().search(query, n)


def graph(slug: str, depth: int = 2) -> str:
    return _default_store().graph(slug, depth)


def list(prefix: str = "atoms/penrose", n: int = 200) -> str:
    return _default_store().list(prefix, n)
