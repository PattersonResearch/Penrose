"""Read-only MCP server for Penrose. Optional: ``pip install penrose[mcp]``.

EXPOSE OPERATIONS, NOT ESCAPE HATCHES. Every tool here is strictly READ-ONLY: it
reads data Penrose already wrote (verdicts, proposals, data-requests, status) or
computes read-only distillation. NOTHING here approves or promotes a verdict (P9
stays human), writes the approved brain / PRINCIPLES_LOG / trusted brainstore /
decisions, runs a paper or a module, or touches the Docker sandbox or the holdout.
There is deliberately no `run`/operate tool — that tier is a separate, guarded
future addition.

The ``mcp`` dependency is OPTIONAL and imported lazily here ONLY, so importing
``penrose`` / the CLI / the eval suite never requires it.
"""
from __future__ import annotations

import sys

_MCP_MISSING_MSG = (
    "penrose MCP requires the optional 'mcp' dependency.\n"
    "install with:  pip install penrose[mcp]"
)


def build_server():
    """Build the read-only FastMCP server. Raises ImportError (with a clear install
    message) if the optional ``mcp`` package is absent."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # optional dependency
        raise ImportError(_MCP_MISSING_MSG) from e

    from . import views

    server = FastMCP("penrose")

    @server.tool()
    def penrose_verdicts(limit: int = 20) -> list[dict]:
        """Recent backtested verdicts (kill / underpowered / watch / research-supported
        + kill_reason + key metrics). READ-ONLY; approval stays human (P9)."""
        return views.verdicts(limit)

    @server.tool()
    def penrose_proposals() -> list[dict]:
        """The propose-only principle proposals (status: proposed). READ-ONLY; promotion
        to the approved brain requires the human P9 review path, not this server."""
        return views.proposals()

    @server.tool()
    def penrose_principles() -> list[dict]:
        """Distilled cross-run advisory principle proposals from the full corpus.
        READ-ONLY compute; never writes the approved brain."""
        return views.principles()

    @server.tool()
    def penrose_data_requests() -> list[dict]:
        """Open `needs_data` blockers — the 'one dataset away from testable' list.
        READ-ONLY."""
        return views.data_requests()

    @server.tool()
    def penrose_status() -> dict:
        """Current pipeline status. READ-ONLY."""
        return views.status()

    return server


def main() -> int:
    try:
        server = build_server()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
