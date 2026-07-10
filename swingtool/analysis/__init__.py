"""Phase 3A analysis: derive honest club/ball metrics from raw detections.

Pure geometry (no ML). Selects a club-head trajectory from noisy candidate
boxes using pose-derived hand priors + temporal continuity, handles gaps
without fabricating positions, and computes RELATIVE-only club speed. No
physical scale (no mph, no metres) is ever emitted.
"""

from swingtool.analysis.stage import run_derive_stage

__all__ = ["run_derive_stage"]
