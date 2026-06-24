"""Portable, discovery-safe retrieval over the advisory corpus.

This module is read-only and inform-never-gate: it never writes to the corpus,
brain, pipeline, verdicts, or backtests. Embeddings are optional and in-process;
when unavailable, retrieval falls back to deterministic lexical scoring.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from . import config, corpus
from .brain import source_is_unanchored

_EMPTY = {"nodes": [], "edges": []}
_CACHE_PATH = config.ROOT / ".embed_cache" / "corpus_vectors.json"
_CONTEXT_CAP = 1200


def load_corpus(path: str | Path | None = None) -> dict:
    """Load the corpus JSON graph, returning an empty graph on any failure."""
    try:
        p = Path(path) if path is not None else config.CORPUS_JSON
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            return dict(_EMPTY)
        nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else []
        edges = data.get("edges") if isinstance(data.get("edges"), list) else []
        return {**data, "nodes": nodes, "edges": edges}
    except Exception:  # noqa: BLE001
        return dict(_EMPTY)


def _node_text(node: dict) -> str:
    parts = [
        node.get("statement"),
        node.get("direction"),
        node.get("level"),
        node.get("family"),
        node.get("domain"),
    ]
    return " ".join(str(x).strip() for x in parts if str(x or "").strip())


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _source_values(node: dict) -> list[str]:
    values: list[str] = []
    for key in ("source_type", "derivation", "source", "source_kind"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    for key in ("source_types", "derivations"):
        value = node.get(key)
        if isinstance(value, list):
            values.extend(str(x).strip() for x in value if str(x or "").strip())
    for container_key in ("data_provenance", "metadata"):
        container = node.get(container_key)
        if isinstance(container, dict):
            value = container.get("source_type")
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return values


def _reserve_footprints(node: dict) -> list[dict]:
    out: list[dict] = []
    for key in ("data_provenance", "provenance"):
        value = node.get(key)
        if isinstance(value, dict):
            out.append(value)
        elif isinstance(value, list):
            out.extend(x for x in value if isinstance(x, dict))
    return out


def _eligible(node: dict, reserve: dict | None) -> bool:
    try:
        for value in _source_values(node):
            if value == "confirmation":
                return False
            try:
                if source_is_unanchored(value):
                    return False
            except Exception:  # noqa: BLE001
                return False
        for footprint in _reserve_footprints(node):
            if corpus.reserve_intersects(footprint, reserve):
                return False
        return bool(node.get("node_id")) and bool(_node_text(node))
    except Exception:  # noqa: BLE001
        return False


def _node_id(node: dict) -> str:
    return str(node.get("node_id") or "")


def _provenance_ids(node: dict) -> list[str]:
    value = node.get("provenance")
    if not isinstance(value, list):
        return []
    return sorted(str(x).strip() for x in value if str(x or "").strip())


def _aggregate_provenance_eligible(node: dict, by_id: dict[str, dict]) -> bool:
    """Exclude aggregate nodes if any contributing observation is unanchored."""
    if _source_values(node):
        return True
    provenance = _provenance_ids(node)
    if not provenance:
        return True
    try:
        seen: set[str] = set()
        stack = list(reversed(provenance))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            source = by_id.get(current)
            if not isinstance(source, dict):
                continue
            values = _source_values(source)
            if values:
                for value in values:
                    try:
                        if source_is_unanchored(value):
                            return False
                    except Exception:  # noqa: BLE001
                        return False
                continue
            stack.extend(reversed(_provenance_ids(source)))
        return True
    except Exception:  # noqa: BLE001
        return False


def _read_vector_cache() -> dict:
    try:
        data = json.loads(_CACHE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _write_vector_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, sort_keys=True))
    except Exception:  # noqa: BLE001
        pass


def _vector_index(nodes: list[dict]) -> dict[str, list[float]]:
    try:
        from . import llm

        if not getattr(llm, "embed_local_available", lambda: False)():
            return {}
        cache = _read_vector_cache()
        changed = False
        vectors: dict[str, list[float]] = {}
        for node in nodes:
            node_id = str(node.get("node_id", ""))
            text = _node_text(node)
            h = _text_hash(text)
            entry = cache.get(node_id) if isinstance(cache.get(node_id), dict) else {}
            vec = entry.get("vector") if entry.get("hash") == h else None
            if not isinstance(vec, list):
                vec = llm.embed_local(text)
                if vec is not None:
                    cache[node_id] = {"hash": h, "vector": vec}
                    changed = True
            if isinstance(vec, list):
                vectors[node_id] = [float(x) for x in vec]
        if changed:
            _write_vector_cache(cache)
        return vectors
    except Exception:  # noqa: BLE001
        return {}


def _lexical_scores(query: str, nodes: list[dict]) -> dict[str, float]:
    q_counts = Counter(_tokens(query))
    if not q_counts:
        return {}
    q_terms = set(q_counts)
    docs = {str(n.get("node_id")): _tokens(_node_text(n)) for n in nodes}
    n_docs = max(1, len(docs))
    df = Counter(term for toks in docs.values() for term in set(toks))
    scores: dict[str, float] = {}
    for node_id, toks in docs.items():
        if not toks:
            continue
        counts = Counter(toks)
        score = 0.0
        for term in q_terms & set(counts):
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
            score += idf * counts[term] / (len(toks) ** 0.5)
        if score > 0:
            scores[node_id] = score
    return scores


def _rank_seeds(query: str, nodes: list[dict], k: int) -> dict[str, float]:
    try:
        vectors = _vector_index(nodes)
        if vectors:
            from . import llm

            q_vec = llm.embed_local(query)
            if q_vec is not None:
                scores = {
                    node_id: llm.cosine(q_vec, vec)
                    for node_id, vec in vectors.items()
                    if vec
                }
                return dict(sorted(
                    ((n, s) for n, s in scores.items() if s > 0),
                    key=lambda item: (-item[1], item[0]),
                )[:k])
    except Exception:  # noqa: BLE001
        pass
    scores = _lexical_scores(query, nodes)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k])


def _adjacency(edges: list[dict]) -> dict[str, list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        left, right = str(edge.get("from", "")), str(edge.get("to", ""))
        if left and right:
            graph[left].add(right)
            graph[right].add(left)
    return {node_id: sorted(neighbors) for node_id, neighbors in graph.items()}


def retrieve(query: str, *, k: int = 6, hops: int = 1,
             reserve: dict | None = None) -> list[dict]:
    """Return top discovery-safe corpus nodes for a query, with graph expansion."""
    try:
        q = (query or "").strip()
        if not q or k <= 0:
            return []
        data = load_corpus()
        reserve = reserve if reserve is not None else config.CONFIRMATION_RESERVE
        all_nodes = [n for n in data.get("nodes", []) if isinstance(n, dict)]
        all_by_id = {_node_id(n): n for n in all_nodes if _node_id(n)}
        eligible_nodes = [
            n for n in all_nodes
            if _eligible(n, reserve) and _aggregate_provenance_eligible(n, all_by_id)
        ]
        if not eligible_nodes:
            return []
        by_id = {str(n.get("node_id")): n for n in eligible_nodes}
        seed_scores = _rank_seeds(q, eligible_nodes, max(1, k))
        if not seed_scores:
            return []
        best = dict(seed_scores)
        if hops > 0:
            graph = _adjacency(data.get("edges", []))
            for seed_id, seed_score in seed_scores.items():
                seen = {seed_id}
                queue = deque([(seed_id, 0)])
                while queue:
                    current, depth = queue.popleft()
                    if depth >= hops:
                        continue
                    for neighbor in graph.get(current, []):
                        if neighbor in seen:
                            continue
                        seen.add(neighbor)
                        if neighbor in by_id and _eligible(by_id[neighbor], reserve):
                            score = seed_score * (0.5 ** (depth + 1))
                            best[neighbor] = max(best.get(neighbor, 0.0), score)
                            queue.append((neighbor, depth + 1))
        ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:k]
        out: list[dict] = []
        for node_id, score in ranked:
            node = dict(by_id[node_id])
            if _eligible(node, reserve):
                node["_score"] = round(float(score), 6)
                out.append(node)
        return out
    except Exception:  # noqa: BLE001
        return []


def _support_count(node: dict) -> int:
    value = node.get("support_count")
    if isinstance(value, int) and value > 0:
        return value
    provenance = node.get("provenance")
    if isinstance(provenance, list):
        return max(1, len(provenance))
    return 1


def format_context(nodes: list[dict]) -> str:
    """Format retrieved nodes as a compact prompt context block."""
    try:
        if not nodes:
            return ""
        lines: list[str] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            level = str(node.get("level") or "corpus")
            statement = str(node.get("statement") or "").strip()
            if not statement:
                continue
            direction = str(node.get("direction") or "unknown")
            line = (
                f"[{level}] {statement} ({direction}) "
                f"- supported by {_support_count(node)} observations"
            )
            if len("\n".join([*lines, line])) > _CONTEXT_CAP:
                remaining = _CONTEXT_CAP - len("\n".join(lines))
                if remaining > 20:
                    lines.append(line[:remaining].rstrip())
                break
            lines.append(line)
        return "\n".join(lines)[:_CONTEXT_CAP]
    except Exception:  # noqa: BLE001
        return ""
