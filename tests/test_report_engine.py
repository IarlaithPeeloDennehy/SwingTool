"""Rules-engine tests. The important ones: low-confidence/approximate metrics
must be suppressed or hedged - for findings AND highlights (no criticism
laundering, no praise laundering)."""

import pytest

from swingtool.metrics.schema import (
    HeadStabilityMetrics,
    KneeFlexMetrics,
    Metrics,
    MetricsResult,
    MetricsSource,
    MetricValue,
    Rotation2DMetrics,
    SpineMetrics,
    SwingEvent,
    SwingEvents,
    TempoMetrics,
)
from swingtool.report.engine import build_report, gate


def mv(value, unit="s", quality="reliable", confidence=0.9):
    return MetricValue(value=value, unit=unit, quality=quality, confidence=confidence)


def make_metrics(ratio=3.0, backswing=0.75, downswing=0.25,
                 tempo_quality="reliable", tempo_conf=0.9,
                 knee_address=140.0, knee_impact=165.0,
                 knee_quality="view_dependent", knee_conf=0.8,
                 interpolated_impact=False):
    ev = SwingEvents(
        address=SwingEvent(frame_index=100, timestamp_s=3.3, confidence=0.95, interpolated=False),
        top=SwingEvent(frame_index=120, timestamp_s=4.0, confidence=0.9, interpolated=False),
        impact=SwingEvent(frame_index=128, timestamp_s=4.26, confidence=0.6 if interpolated_impact else 0.9,
                          interpolated=interpolated_impact),
    )
    na = MetricValue(value=None, unit="deg", quality="low_confidence", confidence=0.0)
    return MetricsResult(
        source=MetricsSource(keypoints_path="k.json", source_video="clip.mov", fps=30.0,
                             width=1080, height=1920, handed="right", frames_analyzed=400),
        events=ev,
        metrics=Metrics(
            tempo=TempoMetrics(
                backswing_duration=mv(backswing),
                downswing_duration=mv(downswing),
                swing_duration=mv(backswing + downswing),
                tempo_ratio=mv(ratio, unit="ratio", quality=tempo_quality, confidence=tempo_conf)),
            head_stability=HeadStabilityMetrics(
                lateral_drift_px=mv(20.0, "px"), vertical_drift_px=mv(15.0, "px"),
                lateral_drift_frac=mv(0.1, "torso_frac"), vertical_drift_frac=mv(0.08, "torso_frac")),
            knee_flex=KneeFlexMetrics(
                lead_address=mv(knee_address, "deg", knee_quality, knee_conf),
                trail_address=mv(160.0, "deg", knee_quality, knee_conf),
                lead_impact=mv(knee_impact, "deg", knee_quality, knee_conf),
                trail_impact=mv(150.0, "deg", knee_quality, knee_conf)),
            spine=SpineMetrics(tilt_from_vertical_address=mv(24.0, "deg", "view_dependent", 0.8)),
            rotation_2d=Rotation2DMetrics(
                shoulder_angle_address=na, shoulder_angle_top=na, shoulder_angle_impact=na,
                hip_angle_address=na, hip_angle_top=na, hip_angle_impact=na),
        ))


def run(m, analysis=None, history=None):
    return build_report(m, analysis, history or [], "metrics.json", None)


class TestGate:
    def test_reliable_high_conf_ok(self):
        assert gate("reliable", 0.9) == ("ok", "")

    def test_low_confidence_suppressed(self):
        verdict, reason = gate("reliable", 0.3)
        assert verdict == "suppressed" and "0.30" in reason

    def test_mid_confidence_hedged(self):
        assert gate("reliable", 0.6)[0] == "hedged"

    def test_approximate_suppressed_even_when_confident(self):
        assert gate("depth_assisted_approximate", 0.95)[0] == "suppressed"
        assert gate("approximate_2d", 0.95)[0] == "suppressed"

    def test_view_dependent_needs_directional(self):
        assert gate("view_dependent", 0.9, directional=False)[0] == "suppressed"
        assert gate("view_dependent", 0.9, directional=True)[0] == "hedged"


class TestRangeRules:
    def test_in_range_no_finding_but_highlight(self):
        r = run(make_metrics(ratio=3.0))
        assert not any(f.metric == "tempo_ratio" for f in r.findings)
        assert any(h.metric == "tempo_ratio" for h in r.highlights)

    def test_out_of_range_produces_sourced_finding(self):
        r = run(make_metrics(ratio=2.0, backswing=0.5, downswing=0.25))
        f = next(f for f in r.findings if f.metric == "tempo_ratio")
        assert "Novosel" in f.reference.source
        assert f.confidence_label == "normal"
        assert f.severity > 0
        assert f.tier == "biggest opportunity"          # top finding

    def test_frames_language_in_finding(self):
        r = run(make_metrics(ratio=2.0, backswing=0.5, downswing=0.25))
        f = next(f for f in r.findings if f.metric == "tempo_ratio")
        assert "15/8 frames" in f.text or "15/7 frames" in f.text  # 0.5*30/0.25*30

    def test_try_this_is_pure_arithmetic(self):
        r = run(make_metrics(ratio=2.0, backswing=0.5, downswing=0.25))
        tt = next(t for t in r.try_this if t.metric == "tempo_ratio")
        assert tt.derivation == "arithmetic"
        # 3.0 * 0.25 = 0.75 target backswing must appear
        assert "0.75" in tt.text
        assert tt.inputs["downswing_s"] == 0.25


class TestSuppression:
    """THE important tests: unreliable metrics never become confident sentences."""

    def test_low_confidence_out_of_range_is_suppressed(self):
        r = run(make_metrics(ratio=2.0, tempo_conf=0.3))
        assert not any(f.metric == "tempo_ratio" for f in r.findings)
        c = next(c for c in r.comparisons if c.metric == "tempo_ratio")
        assert c.status == "not_judged"
        assert "confidence" in c.reason

    def test_approximate_quality_out_of_range_is_suppressed(self):
        r = run(make_metrics(ratio=2.0, tempo_quality="approximate_2d"))
        assert not any(f.metric == "tempo_ratio" for f in r.findings)
        c = next(c for c in r.comparisons if c.metric == "tempo_ratio")
        assert c.status == "not_judged" and "approximate" in c.reason

    def test_praise_laundering_blocked(self):
        # In-range value with junk confidence must NOT become a highlight.
        r = run(make_metrics(ratio=3.0, tempo_conf=0.3))
        assert not any(h.metric == "tempo_ratio" for h in r.highlights)

    def test_interpolated_events_hedge_tempo(self):
        r = run(make_metrics(ratio=2.0, backswing=0.5, downswing=0.25,
                             interpolated_impact=True))
        f = next(f for f in r.findings if f.metric == "tempo_ratio")
        assert f.confidence_label == "hedged"

    def test_xfactor_depth_assisted_never_a_finding(self):
        analysis = {"depth_assisted": {
            "xfactor": {"value": 20.0, "unit": "deg",
                        "quality": "depth_assisted_approximate",
                        "confidence": 0.9, "notes": ""},   # even confident: quality gates it
            "swing_plane_tilt": {"value": 76.0, "unit": "deg",
                                 "quality": "depth_assisted_approximate",
                                 "confidence": 0.9, "notes": ""}}}
        r = run(make_metrics(), analysis=analysis)
        assert not any(f.metric == "xfactor" for f in r.findings)
        c = next(c for c in r.comparisons if c.metric == "xfactor")
        assert c.status == "not_judged"


class TestDirectionalKnee:
    def test_extension_is_hedged_highlight(self):
        r = run(make_metrics(knee_address=140, knee_impact=165))
        h = next(h for h in r.highlights if h.metric == "lead_knee_extension")
        assert h.hedged is True                      # view-dependent always hedged
        assert "Hume" in h.reference.source

    def test_no_extension_is_hedged_finding(self):
        r = run(make_metrics(knee_address=160, knee_impact=150))
        f = next(f for f in r.findings if f.metric == "lead_knee_extension")
        assert f.confidence_label == "hedged"

    def test_noise_floor_change_not_judged(self):
        r = run(make_metrics(knee_address=160, knee_impact=163))
        c = next(c for c in r.comparisons if c.metric == "lead_knee_extension")
        assert c.status == "not_judged" and "noise floor" in c.reason

    def test_low_confidence_knee_no_highlight(self):
        r = run(make_metrics(knee_address=140, knee_impact=165, knee_conf=0.2))
        assert not any(h.metric == "lead_knee_extension" for h in r.highlights)


class TestReportShape:
    def test_not_judged_metrics_visible_with_reasons(self):
        r = run(make_metrics())
        nj = {c.metric: c for c in r.comparisons if c.status == "not_judged"}
        assert "head_lateral_drift" in nj
        assert "refusing to guess" in nj["head_lateral_drift"].reason
        assert "spine_tilt_address" in nj

    def test_disclaimer_and_limitations_present(self):
        r = run(make_metrics())
        assert "not a substitute" in r.disclaimer.lower()
        assert any("launch monitor" in lim for lim in r.limitations)

    def test_progress_from_history(self):
        r = run(make_metrics(ratio=2.4, backswing=0.6, downswing=0.25),
                history=[{"tempo_ratio": 2.0}])
        p = next(p for p in r.progress if p.metric == "tempo_ratio")
        assert p.previous == 2.0 and p.current == 2.4
        assert p.note == "moving toward 3:1"

    def test_coverage_counts(self):
        r = run(make_metrics())
        assert r.coverage.judged >= 2                 # tempo + downswing (+ knee)
        assert r.coverage.total > r.coverage.judged   # not-judged entries exist
