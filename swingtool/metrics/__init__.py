"""Phase 2: biomechanical metrics from Phase-1 pose keypoints.

Pure geometry - no ML model, torch, or GPU. Reads output/keypoints.json
(the AnalysisResult contract) and writes output/metrics.json.
"""

from swingtool.metrics.engine import run_metrics_stage

__all__ = ["run_metrics_stage"]
