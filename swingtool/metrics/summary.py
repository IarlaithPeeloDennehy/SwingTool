"""Human-readable console summary of a MetricsResult."""

from __future__ import annotations

from swingtool.metrics.schema import MetricValue, MetricsResult, SwingEvent


def _fmt_event(name: str, e: SwingEvent | None) -> str:
    if e is None:
        return f"  {name:8s} not detected"
    flag = "  [flagged: low confidence / interpolated]" if e.interpolated else ""
    return f"  {name:8s} frame {e.frame_index:4d}  t={e.timestamp_s:6.2f}s  conf {e.confidence:.2f}{flag}"


def _val(m: MetricValue, digits: int = 1) -> str:
    if m.value is None:
        return f"n/a ({m.notes})"
    return f"{m.value:.{digits}f} {m.unit}"


def format_summary(r: MetricsResult) -> str:
    ev = r.events
    m = r.metrics
    lines = ["", "Swing events:",
             _fmt_event("address", ev.address),
             _fmt_event("top", ev.top),
             _fmt_event("impact", ev.impact),
             "", "Metrics:"]

    t = m.tempo
    ratio = t.tempo_ratio.value
    ratio_str = f"{ratio:.1f}:1" if ratio is not None else "n/a"
    lines.append(f"  Tempo         backswing {_val(t.backswing_duration, 2)} : "
                 f"downswing {_val(t.downswing_duration, 2)}  =  {ratio_str}  (benchmark ~3:1)")

    h = m.head_stability
    lines.append(f"  Head drift    lateral {_val(h.lateral_drift_px)} ({_val(h.lateral_drift_frac, 3)}),  "
                 f"vertical {_val(h.vertical_drift_px)} ({_val(h.vertical_drift_frac, 3)})")

    k = m.knee_flex
    lines.append(f"  Knee flex     lead {_val(k.lead_address)} -> {_val(k.lead_impact)}  "
                 f"(address->impact) [view-dependent]")
    lines.append(f"                trail {_val(k.trail_address)} -> {_val(k.trail_impact)}")

    lines.append(f"  Spine tilt    {_val(m.spine.tilt_from_vertical_address)} at address [view-dependent]")

    rot = m.rotation_2d
    lines.append(f"  Rotation(2D)  shoulder addr/top/impact "
                 f"{_val(rot.shoulder_angle_address)} / {_val(rot.shoulder_angle_top)} / "
                 f"{_val(rot.shoulder_angle_impact)}  [approximate 2D - X-factor needs Phase 3]")
    return "\n".join(lines)
