"""Command-line interface. Parsing and dispatch only - no model or video
logic lives here."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swingtool.config import AnalyzeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swingtool",
        description="Analyze a golf swing video.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run pose estimation on a video.")
    analyze.add_argument("video", type=Path, help="Path to the input video.")
    analyze.add_argument("--output-dir", type=Path, default=Path("output"))
    analyze.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                         help="Inference device (default: cuda; missing CUDA is an error).")
    analyze.add_argument("--frame-stride", type=int, default=1,
                         help="Process every Nth frame (default: 1).")
    analyze.add_argument("--max-dim", type=int, default=None,
                         help="Downscale so the longest side <= MAX_DIM before inference.")
    analyze.add_argument("--min-box-score", type=float, default=0.3,
                         help="Person detection confidence threshold.")
    analyze.add_argument("--overlay-min-score", type=float, default=0.3,
                         help="Hide keypoints below this score in the overlay.")

    metrics = sub.add_parser("metrics", help="Compute swing metrics from keypoints.json.")
    metrics.add_argument("keypoints", type=Path, help="Path to keypoints.json from `analyze`.")
    metrics.add_argument("--output-dir", type=Path, default=Path("output"))
    metrics.add_argument("--handed", choices=("right", "left"), default="right",
                         help="Golfer handedness; sets lead/trail side (default: right).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "analyze":
        from swingtool.ingest import VideoIngestError
        from swingtool.pose import run_pose_stage

        config = AnalyzeConfig(
            video_path=args.video,
            output_dir=args.output_dir,
            device=args.device,
            frame_stride=args.frame_stride,
            max_dim=args.max_dim,
            min_box_score=args.min_box_score,
            overlay_min_score=args.overlay_min_score,
        )
        try:
            result = run_pose_stage(config)
        except (VideoIngestError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        detected = len(result.frames)
        print(f"Done. {detected} frames with a detected golfer.")
        print(f"  keypoints: {config.keypoints_path}")
        print(f"  overlay:   {config.overlay_path}")
        return 0

    if args.command == "metrics":
        from swingtool.metrics import run_metrics_stage
        from swingtool.metrics.summary import format_summary

        try:
            result = run_metrics_stage(args.keypoints, args.output_dir, handed=args.handed)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(f"Wrote {args.output_dir / 'metrics.json'}")
        print(format_summary(result))
        return 0

    return 2
