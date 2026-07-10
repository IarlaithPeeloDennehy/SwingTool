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

    detect = sub.add_parser("detect", help="Zero-shot club/ball detection over the swing window.")
    detect.add_argument("video", type=Path, help="Path to the input video.")
    detect.add_argument("keypoints", type=Path, help="Path to keypoints.json from `analyze`.")
    detect.add_argument("--output-dir", type=Path, default=Path("output"))
    detect.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Detection device (default: cpu; see detect module docstring).")
    detect.add_argument("--box-threshold", type=float, default=0.25)
    detect.add_argument("--text-threshold", type=float, default=0.20)
    detect.add_argument("--frame-stride", type=int, default=1)
    detect.add_argument("--start-frame", type=int, default=None,
                        help="Override auto swing-window start (frame index).")
    detect.add_argument("--end-frame", type=int, default=None,
                        help="Override auto swing-window end (frame index).")

    derive = sub.add_parser("derive", help="Derive club/ball metrics from detections (pure geometry).")
    derive.add_argument("keypoints", type=Path, help="Path to keypoints.json.")
    derive.add_argument("detections", type=Path, help="Path to detections.json from `detect`.")
    derive.add_argument("--output-dir", type=Path, default=Path("output"))
    derive.add_argument("--handed", choices=("right", "left"), default="right")

    rclub = sub.add_parser("render-club", help="Render the club-path + ball overlay video.")
    rclub.add_argument("video", type=Path, help="Path to the input video.")
    rclub.add_argument("analysis", type=Path, help="Path to analysis.json from `derive`.")
    rclub.add_argument("--keypoints", type=Path, default=None,
                       help="Optional keypoints.json to underlay a faint skeleton.")
    rclub.add_argument("--output", type=Path, default=Path("output/overlay_club.mp4"))
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

    if args.command == "detect":
        from swingtool.detect import run_detect_stage

        try:
            detections = run_detect_stage(
                args.video, args.keypoints, args.output_dir, device=args.device,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                frame_stride=args.frame_stride, start_frame=args.start_frame,
                end_frame=args.end_frame)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        n = len(detections.frames)
        with_club = sum(1 for f in detections.frames if f.club)
        with_ball = sum(1 for f in detections.frames if f.ball)
        src = detections.source
        print(f"Wrote {args.output_dir / 'detections.json'}")
        print(f"  window frames {src.window_start}-{src.window_end} ({n} processed)")
        print(f"  club detected in {with_club}/{n} frames, ball in {with_ball}/{n}")
        return 0

    if args.command == "derive":
        from swingtool.analysis import run_derive_stage

        try:
            analysis = run_derive_stage(args.keypoints, args.detections,
                                        args.output_dir, handed=args.handed)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        cp = analysis.club_path
        detected = sum(1 for p in cp if p.detected and not p.interpolated)
        interp = sum(1 for p in cp if p.interpolated)
        gaps = sum(1 for p in cp if p.x is None)
        speed = analysis.relative_club_speed.peak
        print(f"Wrote {args.output_dir / 'analysis.json'}")
        print(f"  club path: {detected} detected, {interp} interpolated (short gaps), "
              f"{gaps} unfilled gaps of {len(cp)} frames")
        peak = f"{speed.value} {speed.unit}" if speed.value is not None else "n/a"
        print(f"  peak relative club speed: {peak} [{speed.quality}]")
        print(f"  ball at address: {analysis.ball.address.quality}; "
              f"launch direction: {analysis.ball.launch_direction.quality}")
        return 0

    if args.command == "render-club":
        from swingtool.ingest import VideoIngestError
        from swingtool.render import render_club_overlay

        try:
            out = render_club_overlay(args.video, args.analysis, args.output,
                                      keypoints_path=args.keypoints)
        except (FileNotFoundError, VideoIngestError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {out}")
        return 0

    return 2
