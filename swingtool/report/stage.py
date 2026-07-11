"""Report stage: metrics.json (+ optional analysis.json) -> report.json,
plus a self-comparison history (output/history.jsonl)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from swingtool.metrics.schema import MetricsResult
from swingtool.report.engine import build_report
from swingtool.report.schema import SwingReport


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def append_history(path: Path, report: SwingReport, metrics: MetricsResult) -> None:
    t = metrics.metrics.tempo
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": metrics.source.source_video,
        "tempo_ratio": t.tempo_ratio.value,
        "backswing_s": t.backswing_duration.value,
        "downswing_s": t.downswing_duration.value,
        "judged": report.coverage.judged,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def run_report_stage(metrics_path: Path, output_dir: Path,
                     analysis_path: Optional[Path] = None,
                     write_history: bool = True) -> SwingReport:
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    metrics = MetricsResult.model_validate(
        json.loads(metrics_path.read_text(encoding="utf-8")))

    # default: pick up analysis.json sitting next to the metrics file
    if analysis_path is None:
        candidate = metrics_path.parent / "analysis.json"
        analysis_path = candidate if candidate.exists() else None
    analysis = None
    if analysis_path is not None:
        analysis_path = Path(analysis_path)
        if not analysis_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {analysis_path}")
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history = read_history(history_path)

    report = build_report(metrics, analysis, history, str(metrics_path),
                          str(analysis_path) if analysis_path else None)

    (output_dir / "report.json").write_text(
        json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    if write_history:
        append_history(history_path, report, metrics)
    return report
