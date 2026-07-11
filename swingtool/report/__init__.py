"""Phase 4: plain-language fault report from measured metrics.

Deterministic rules engine (no LLM): metric -> cited threshold -> finding,
gated by the Phase 2/3 quality/confidence flags so approximate metrics are
hedged or suppressed, never laundered into confident sentences. Reference
ranges live in the versioned references_v1.json - every number cited or
absent.
"""

from swingtool.report.stage import run_report_stage

__all__ = ["run_report_stage"]
