"""Deterministic rules engine: metric -> threshold -> finding.

No LLM anywhere. Every rule reads a measured MetricValue, applies a cited
reference from references_v1.json, and passes three gates before it may
produce a finding or highlight:

  1. quality gate      - reliable: eligible; view_dependent: directional rules
                         only, always hedged; everything more approximate
                         (approximate_2d, depth_assisted_approximate,
                         relative_only, coarse, low_confidence, not_detected)
                         is suppressed;
  2. confidence gate   - < 0.5 suppressed; 0.5-0.7 hedged;
  3. event-integrity   - tempo/duration findings are hedged if the underlying
                         swing events were interpolated.

Suppression is never silent: a suppressed metric appears in comparisons as
status=not_judged with the reason stated.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

from swingtool.analysis.schema import MetricValue as AnalysisMetricValue
from swingtool.metrics.schema import MetricsResult, MetricValue
from swingtool.report.schema import (
    Comparison,
    Coverage,
    Finding,
    Highlight,
    NumberFact,
    ProgressDelta,
    ReferenceInfo,
    ReportSource,
    SwingReport,
    TryThis,
)

CONF_SUPPRESS = 0.5
CONF_HEDGE = 0.7

_SUPPRESSED_QUALITIES = {
    "approximate_2d", "depth_assisted_approximate", "relative_only",
    "coarse", "low_confidence", "not_detected",
}


def load_references() -> dict:
    text = resources.files("swingtool.report").joinpath("references_v1.json").read_text(encoding="utf-8")
    return json.loads(text)


def _ref_info(entry: dict) -> ReferenceInfo:
    return ReferenceInfo(
        range=tuple(entry["range"]) if "range" in entry else None,
        center=entry.get("center"),
        direction=entry.get("direction"),
        min_change=entry.get("min_change"),
        unit=entry["unit"],
        source=entry["source"],
        source_type=entry["source_type"],
        tolerance_ours=entry.get("tolerance_ours", False),
        notes=entry.get("notes", ""),
    )


def gate(quality: str, confidence: float, directional: bool = False,
         interpolated_events: bool = False) -> tuple[str, str]:
    """Return (verdict, reason): verdict in {'ok', 'hedged', 'suppressed'}."""
    if quality in _SUPPRESSED_QUALITIES:
        return "suppressed", f"quality '{quality}' is too approximate to support a finding"
    if quality == "view_dependent" and not directional:
        return "suppressed", ("view-dependent 2D angle; only direction-of-change "
                              "rules are judged for this camera geometry")
    if quality not in ("reliable", "view_dependent"):
        return "suppressed", f"unknown quality '{quality}'"
    if confidence < CONF_SUPPRESS:
        return "suppressed", f"metric confidence {confidence:.2f} is below {CONF_SUPPRESS}"
    reasons = []
    if quality == "view_dependent":
        reasons.append("view-dependent projection")
    if confidence < CONF_HEDGE:
        reasons.append(f"confidence {confidence:.2f}")
    if interpolated_events:
        reasons.append("swing events partly interpolated")
    if reasons:
        return "hedged", "; ".join(reasons)
    return "ok", ""


def _not_judged(metric: str, m: Optional[MetricValue], reason: str,
                ref: Optional[ReferenceInfo] = None) -> Comparison:
    return Comparison(
        metric=metric,
        measured=None if m is None else m.value,
        unit="" if m is None else m.unit,
        quality="not_detected" if m is None else m.quality,
        confidence=0.0 if m is None else m.confidence,
        reference=ref, status="not_judged", reason=reason)


def _short_source(source: str) -> str:
    return source.split(":")[0].split(";")[0]


class _Rules:
    """Accumulates comparisons/findings/highlights while running each rule."""

    def __init__(self, refs: dict):
        self.entries = refs["entries"]
        self.comparisons: list[Comparison] = []
        self.findings: list[Finding] = []
        self.highlights: list[Highlight] = []
        self.try_this: list[TryThis] = []

    # -- range rule ------------------------------------------------------
    def range_rule(self, key: str, m: Optional[MetricValue], text_out: str,
                   text_in: str, interpolated_events: bool = False) -> None:
        ref = _ref_info(self.entries[key])
        if m is None or m.value is None:
            self.comparisons.append(_not_judged(key, m, "metric not measured", ref))
            return
        verdict, reason = gate(m.quality, m.confidence,
                               interpolated_events=interpolated_events)
        if verdict == "suppressed":
            self.comparisons.append(_not_judged(key, m, reason, ref))
            return
        lo, hi = ref.range
        inside = lo <= m.value <= hi
        hedged = verdict == "hedged"
        self.comparisons.append(Comparison(
            metric=key, measured=m.value, unit=m.unit, quality=m.quality,
            confidence=m.confidence, reference=ref,
            status="within_range" if inside else "outside_range",
            hedged=hedged, reason=reason))
        if inside:
            self.highlights.append(Highlight(text=text_in, metric=key,
                                             measured=m.value, reference=ref, hedged=hedged))
        else:
            deviation = (lo - m.value) if m.value < lo else (m.value - hi)
            severity = round(deviation / (hi - lo) * m.confidence, 3)
            self.findings.append(Finding(
                text=text_out, metric=key, measured=m.value, reference=ref,
                confidence_label="hedged" if hedged else "normal",
                severity=severity, tier="worth a look"))

    # -- directional rule --------------------------------------------------
    def direction_rule(self, key: str, before: Optional[MetricValue],
                       after: Optional[MetricValue], text_ok: str, text_missing: str) -> None:
        ref = _ref_info(self.entries[key])
        if (before is None or after is None or before.value is None or after.value is None):
            self.comparisons.append(_not_judged(key, None, "metric not measured", ref))
            return
        quality = before.quality if before.confidence <= after.confidence else after.quality
        conf = min(before.confidence, after.confidence)
        verdict, reason = gate(quality, conf, directional=True)
        change = after.value - before.value
        if verdict == "suppressed":
            m = MetricValue(value=round(change, 1), unit=ref.unit, quality=quality,
                            confidence=conf)
            self.comparisons.append(_not_judged(key, m, reason, ref))
            return
        hedged = verdict == "hedged"
        if change >= ref.min_change:
            self.comparisons.append(Comparison(
                metric=key, measured=round(change, 1), unit=ref.unit, quality=quality,
                confidence=conf, reference=ref, status="within_range",
                hedged=hedged, reason=reason))
            self.highlights.append(Highlight(text=text_ok, metric=key,
                                             measured=round(change, 1), reference=ref,
                                             hedged=hedged))
        elif change <= 0:
            self.comparisons.append(Comparison(
                metric=key, measured=round(change, 1), unit=ref.unit, quality=quality,
                confidence=conf, reference=ref, status="outside_range",
                hedged=hedged, reason=reason))
            self.findings.append(Finding(
                text=text_missing, metric=key, measured=round(change, 1), reference=ref,
                confidence_label="hedged" if hedged else "normal",
                severity=round(min(abs(change) / 45.0, 1.0) * conf, 3),
                tier="worth a look"))
        else:  # 0 < change < min_change: inside the noise floor
            m = MetricValue(value=round(change, 1), unit=ref.unit, quality=quality, confidence=conf)
            self.comparisons.append(_not_judged(
                key, m, f"change {change:+.1f} deg is within the +/-{ref.min_change:.0f} deg noise floor", ref))


def build_report(metrics: MetricsResult, analysis: Optional[dict],
                 history: list[dict], metrics_path: str,
                 analysis_path: Optional[str]) -> SwingReport:
    refs = load_references()
    rules = _Rules(refs)
    fps = metrics.source.fps or 30.0
    t = metrics.metrics.tempo
    ev = metrics.events
    interp = any(e is not None and e.interpolated for e in (ev.address, ev.top, ev.impact))

    # --- rule 1: tempo ratio (presented in Novosel's frame language) ------
    bs, ds = t.backswing_duration.value, t.downswing_duration.value
    bf = round(bs * fps) if bs else None
    df = round(ds * fps) if ds else None
    ratio = t.tempo_ratio.value
    tempo_out = (f"Your tempo: {bf}/{df} frames ({ratio:.1f}:1). Tour players: "
                 f"27/9, 24/8, 21/7 - all 3:1." if ratio is not None else "")
    tempo_in = (f"Tempo {ratio:.1f}:1 - inside the ~3:1 tour reference band."
                if ratio is not None else "")
    rules.range_rule("tempo_ratio", t.tempo_ratio, tempo_out, tempo_in,
                     interpolated_events=interp)

    # --- rule 2: downswing duration ---------------------------------------
    ds_out = (f"Downswing {ds:.2f}s is outside the published tour window (0.20-0.30s)."
              if ds is not None else "")
    ds_in = (f"Downswing {ds:.2f}s - inside the tour window (0.20-0.30s)."
             if ds is not None else "")
    rules.range_rule("downswing_duration", t.downswing_duration, ds_out, ds_in,
                     interpolated_events=interp)

    # --- rule 3: lead-knee extension (directional) -------------------------
    k = metrics.metrics.knee_flex
    chg = ((k.lead_impact.value - k.lead_address.value)
           if (k.lead_impact.value is not None and k.lead_address.value is not None) else None)
    knee_ok = (f"You post up onto your lead leg through impact ({chg:+.0f} deg extension "
               f"in this camera's view) - the move the biomechanics literature "
               f"describes in skilled golfers." if chg is not None else "")
    knee_missing = ("Lead knee did not extend through impact in this camera's view - "
                    "skilled golfers typically straighten ('post up') onto the lead leg.")
    rules.direction_rule("lead_knee_extension", k.lead_address, k.lead_impact,
                         knee_ok, knee_missing)

    # --- rule 4: X-factor (depth-assisted; expected to be gate-suppressed) -
    if analysis is not None and "depth_assisted" in analysis:
        xf = analysis["depth_assisted"]["xfactor"]
        m = AnalysisMetricValue(**xf)
        rules.range_rule("xfactor", m,
                         f"Hip-shoulder separation {m.value} deg vs published 40-60 deg."
                         if m.value is not None else "",
                         f"Hip-shoulder separation {m.value} deg - within published 40-60 deg."
                         if m.value is not None else "")

    # --- measured-but-not-judged metrics (visible honesty) -----------------
    mm = metrics.metrics
    rules.comparisons.append(_not_judged(
        "head_lateral_drift", mm.head_stability.lateral_drift_frac,
        "no citable reference range in these units (torso-fraction); refusing to guess"))
    rules.comparisons.append(_not_judged(
        "head_vertical_drift", mm.head_stability.vertical_drift_frac,
        "no citable reference range in these units (torso-fraction); refusing to guess"))
    rules.comparisons.append(_not_judged(
        "spine_tilt_address", mm.spine.tilt_from_vertical_address,
        "projection axis of this camera does not match the literature's measurement convention"))
    rules.comparisons.append(_not_judged(
        "trail_knee_flex", mm.knee_flex.trail_address,
        "no sourced threshold; measured value shown for reference only"))
    if analysis is not None:
        sp = analysis.get("relative_club_speed", {}).get("peak")
        if sp is not None:
            rules.comparisons.append(_not_judged(
                "relative_club_speed", AnalysisMetricValue(**sp),
                "relative units only (body-lengths/s); no physical scale exists to compare against"))
        pl = analysis.get("depth_assisted", {}).get("swing_plane_tilt")
        if pl is not None:
            rules.comparisons.append(_not_judged(
                "swing_plane_tilt", AnalysisMetricValue(**pl),
                "depth-assisted approximation; not accurate enough to judge against a standard"))

    # --- try-this: pure arithmetic on the ratio definition ------------------
    tempo_cmp = next((c for c in rules.comparisons if c.metric == "tempo_ratio"), None)
    ds_cmp = next((c for c in rules.comparisons if c.metric == "downswing_duration"), None)
    if (tempo_cmp is not None and tempo_cmp.status == "outside_range"
            and ds_cmp is not None and ds_cmp.status == "within_range"
            and bs is not None and ds is not None):
        needed = 3.0 * ds
        rules.try_this.append(TryThis(
            text=(f"Arithmetic, not a promise: your {ds:.2f}s downswing is tour-like - "
                  f"at 3:1 it wants a ~{needed:.2f}s backswing, about "
                  f"{needed - bs:+.2f}s vs your {bs:.2f}s."),
            metric="tempo_ratio", derivation="arithmetic",
            inputs={"downswing_s": ds, "backswing_s": bs, "ratio_center": 3.0}))

    # --- prioritise findings ------------------------------------------------
    rules.findings.sort(key=lambda f: f.severity, reverse=True)
    if rules.findings:
        rules.findings[0] = rules.findings[0].model_copy(update={"tier": "biggest opportunity"})

    # --- progress vs own history (self-comparison, no literature needed) ----
    progress: list[ProgressDelta] = []
    if history and ratio is not None:
        prev = history[-1].get("tempo_ratio")
        if prev is not None:
            closer = abs(3.0 - ratio) < abs(3.0 - prev)
            note = "moving toward 3:1" if closer else (
                "same distance from 3:1" if abs(3.0 - ratio) == abs(3.0 - prev)
                else "moved away from 3:1")
            progress.append(ProgressDelta(metric="tempo_ratio", previous=prev,
                                          current=ratio, note=note))

    # --- by the numbers ------------------------------------------------------
    numbers: list[NumberFact] = []
    if t.swing_duration.value is not None:
        numbers.append(NumberFact(label="Full swing (address to impact)",
                                  value=f"{t.swing_duration.value:.2f}s"))
    if bf is not None and df is not None:
        numbers.append(NumberFact(label="Tempo in frames (Novosel style)",
                                  value=f"{bf}/{df}"))
    if analysis is not None:
        sp = analysis.get("relative_club_speed", {}).get("peak", {})
        if sp.get("value") is not None:
            numbers.append(NumberFact(
                label="Peak club-head speed (relative)",
                value=f"~{sp['value']:.0f} body-lengths/s",
                note="relative units - monocular video has no physical scale"))
        cp = analysis.get("club_path", [])
        det = sum(1 for p in cp if p.get("detected"))
        if cp:
            numbers.append(NumberFact(label="Club tracked",
                                      value=f"{det}/{len(cp)} swing frames",
                                      note="see overlay_club.mp4 for the path trace"))

    judged = sum(1 for c in rules.comparisons if c.status != "not_judged")
    limitations = [
        "No metric club speed or distance - monocular video is relative-only; a launch monitor measures these.",
        "Swing plane and X-factor are depth-assisted approximations, not true 3D - not confident enough to coach from yet.",
        "Knee and spine angles are 2D projections; their magnitudes depend on where the camera stands.",
        "30 fps undersamples the downswing (~7 frames); impact-zone values are coarse.",
        "This is one swing - patterns need multiple swings to be believable.",
    ]

    return SwingReport(
        reference_version=refs["reference_version"],
        source=ReportSource(metrics_path=metrics_path, analysis_path=analysis_path,
                            video=metrics.source.source_video),
        coverage=Coverage(judged=judged, total=len(rules.comparisons)),
        progress=progress,
        highlights=rules.highlights,
        findings=rules.findings,
        try_this=rules.try_this,
        comparisons=rules.comparisons,
        by_the_numbers=numbers,
        limitations=limitations,
    )
