#!/usr/bin/env python3
"""Smoke-test portable corpus retrieval without external embedding services."""
from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

from penrose import config, retrieval


def _synthetic_corpus(path: Path) -> None:
    path.write_text(json.dumps({
        "nodes": [
            {
                "node_id": "n1",
                "level": "observation",
                "direction": "positive",
                "statement": "Funding rate carry predicts forward crypto returns.",
                "source_type": "external_source",
            },
            {
                "node_id": "n2",
                "level": "family_principle",
                "direction": "positive",
                "statement": "Positive evidence recurs in funding carry.",
                "source_type": "external_source",
                "support_count": 3,
            },
        ],
        "edges": [{"from": "n1", "to": "n2", "type": "supports"}],
    }))


def main() -> None:
    touched_external = []
    original_urlopen = urllib.request.urlopen

    def guard(req, *args, **kwargs):
        url = getattr(req, "full_url", req)
        text = str(url)
        if "localhost:8088" in text or "127.0.0.1:8088" in text:
            touched_external.append(text)
            raise AssertionError(f"retrieval touched external embedding service: {text}")
        return original_urlopen(req, *args, **kwargs)

    urllib.request.urlopen = guard
    original_path = config.CORPUS_JSON
    try:
        if not config.CORPUS_JSON.exists():
            with tempfile.TemporaryDirectory() as td:
                config.CORPUS_JSON = Path(td) / "corpus.json"
                _synthetic_corpus(config.CORPUS_JSON)
                nodes = retrieval.retrieve("funding rate carry", k=3)
        else:
            nodes = retrieval.retrieve("funding rate carry", k=3)
        assert isinstance(nodes, list)
        retrieval.format_context(nodes)
        assert not touched_external
    finally:
        config.CORPUS_JSON = original_path
        urllib.request.urlopen = original_urlopen
    print(f"portable retrieval ok: {len(nodes)} nodes")


if __name__ == "__main__":
    main()
