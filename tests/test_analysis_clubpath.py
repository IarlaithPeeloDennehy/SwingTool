import numpy as np

from swingtool.analysis.clubpath import (
    cluster_balls,
    fill_short_gaps,
    fit_clean_path,
    relative_speed,
    select_address_ball,
    select_club_path,
)

BODY_SCALE = 100.0


def _frame(fi, club_boxes, ball_boxes=()):
    return {"frame_index": fi, "timestamp_s": fi / 30.0,
            "club": list(club_boxes), "ball": list(ball_boxes)}


def _box(cx, cy, score, s=10.0):
    return (cx - s, cy - s, cx + s, cy + s, score)


class TestSelectClubPath:
    def test_picks_candidate_near_hands(self):
        # Two candidates: one near the hands, one far background (higher score).
        frames = [_frame(0, [_box(500, 1100, 0.6), _box(50, 200, 0.9)])]
        hands = {0: (500, 1085)}
        path = select_club_path(frames, hands, BODY_SCALE)
        assert path[0]["detected"] is True
        assert abs(path[0]["x"] - 500) < 1     # chose the near one, not the far high-score

    def test_prefers_temporal_continuity(self):
        frames = [
            _frame(0, [_box(500, 1100, 0.6)]),
            # frame 1: a jumpy but IN-GATE higher-score candidate vs a continuous one
            _frame(1, [_box(600, 1160, 0.65), _box(505, 1105, 0.55)]),
        ]
        hands = {0: (500, 1085), 1: (500, 1085)}
        path = select_club_path(frames, hands, BODY_SCALE)
        assert path[1]["x"] < 560     # continuity beat the jumpy higher score (~505, not ~600)

    def test_no_candidate_is_a_gap_not_fabricated(self):
        frames = [_frame(0, [])]               # nothing detected
        hands = {0: (500, 1085)}
        path = select_club_path(frames, hands, BODY_SCALE)
        assert path[0]["detected"] is False
        assert path[0]["x"] is None and path[0]["y"] is None   # NOT invented

    def test_far_only_candidate_rejected_as_gap(self):
        frames = [_frame(0, [_box(50, 200, 0.9)])]   # only a far background box
        hands = {0: (500, 1085)}
        path = select_club_path(frames, hands, BODY_SCALE)
        assert path[0]["detected"] is False and path[0]["x"] is None


class TestGapHandling:
    def _detected(self, fi, x, y):
        return {"frame_index": fi, "timestamp_s": fi / 30.0, "x": x, "y": y,
                "confidence": 0.6, "detected": True, "interpolated": False}

    def _gap(self, fi):
        return {"frame_index": fi, "timestamp_s": fi / 30.0, "x": None, "y": None,
                "confidence": 0.0, "detected": False, "interpolated": False}

    def test_short_gap_interpolated_and_flagged(self):
        pts = [self._detected(0, 100, 100), self._gap(1), self._gap(2),
               self._detected(3, 400, 400)]
        filled = fill_short_gaps(pts, max_gap=3)
        assert filled[1]["interpolated"] is True and filled[2]["interpolated"] is True
        assert filled[1]["detected"] is False         # interpolated != real detection
        assert filled[1]["x"] == 200 and filled[2]["x"] == 300   # linear
        assert filled[1]["confidence"] < 0.6          # confidence reduced

    def test_long_gap_left_as_hole_not_fabricated(self):
        pts = [self._detected(0, 100, 100)] + [self._gap(i) for i in range(1, 6)] + \
              [self._detected(6, 700, 700)]
        filled = fill_short_gaps(pts, max_gap=3)      # gap of 5 > max_gap
        for i in range(1, 6):
            assert filled[i]["x"] is None             # NOT filled
            assert filled[i]["detected"] is False and filled[i]["interpolated"] is False

    def test_leading_gap_not_extrapolated(self):
        pts = [self._gap(0), self._detected(1, 100, 100)]
        filled = fill_short_gaps(pts, max_gap=3)
        assert filled[0]["x"] is None                 # no left anchor -> not invented


class TestRelativeSpeed:
    def _pt(self, fi, x, y, detected=True):
        return {"frame_index": fi, "timestamp_s": fi / 30.0, "x": x, "y": y,
                "confidence": 0.6, "detected": detected, "interpolated": False}

    def test_relative_speed_units_and_value(self):
        # move 100px (== 1 body length) in 1 frame at 30fps -> 30 body_len/s
        pts = [self._pt(10, 0, 0), self._pt(11, 100, 0)]
        steps, peak = relative_speed(pts, (10, 11), BODY_SCALE, 30.0)
        assert peak == 30.0

    def test_gap_step_is_none_not_guessed(self):
        pts = [self._pt(10, 0, 0),
               {"frame_index": 11, "timestamp_s": 11 / 30, "x": None, "y": None,
                "confidence": 0.0, "detected": False, "interpolated": False},
               self._pt(12, 200, 0)]
        steps, peak = relative_speed(pts, (10, 12), BODY_SCALE, 30.0)
        assert any(s["value"] is None for s in steps)   # step across the gap not fabricated


class TestCleanPath:
    def _pt(self, fi, x, y, conf):
        return {"frame_index": fi, "timestamp_s": fi / 30.0, "x": x, "y": y,
                "confidence": conf, "detected": True, "interpolated": False}

    def test_continuous_over_full_window(self):
        pts = [self._pt(i, float(i * 10), 100.0, 0.8) for i in range(10)]
        line = fit_clean_path(pts, (0, 9), bandwidth=2.0)
        assert [p["frame_index"] for p in line] == list(range(10))   # every frame, unbroken

    def test_low_confidence_outlier_is_damped(self):
        # A smooth line x=10*t, with one high-error but LOW-confidence spike.
        pts = [self._pt(i, float(i * 10), 100.0, 0.9) for i in range(11)]
        pts[5] = self._pt(5, 900.0, 100.0, 0.05)      # big spike, tiny confidence
        line = fit_clean_path(pts, (0, 10), bandwidth=2.0)
        at5 = next(p for p in line if p["frame_index"] == 5)
        assert at5["x"] < 200      # pulled back toward the true ~50, not up at 900

    def test_gap_is_bridged_for_continuous_line(self):
        pts = [self._pt(0, 0.0, 0.0, 0.8),
               {"frame_index": 1, "timestamp_s": 1 / 30, "x": None, "y": None,
                "confidence": 0.0, "detected": False, "interpolated": False},
               self._pt(2, 20.0, 0.0, 0.8)]
        line = fit_clean_path(pts, (0, 2), bandwidth=2.0)
        assert any(p["frame_index"] == 1 for p in line)   # visual line spans the gap


class TestBall:
    def test_cluster_prefers_persistent(self):
        frames = [_frame(i, [], [_box(760, 1330, 0.5)]) for i in range(5)]
        frames.append(_frame(5, [], [_box(100, 100, 0.9)]))   # one-off high score
        clusters = cluster_balls(frames)
        assert clusters[0]["count"] == 5                # persistent beats one-off
        assert abs(clusters[0]["x"] - 760) < 2

    def test_struck_ball_identified_by_disappearing_after_impact(self):
        # Teed ball (760,1330) present pre-impact then gone; a range ball
        # (500,1190) persists throughout and is even more confident.
        frames = []
        for i in range(10):                             # impact at frame 6
            balls = [_box(500, 1190, 0.6)]              # range ball: always present
            if i <= 6:
                balls.append(_box(760, 1330, 0.45))     # struck ball: vanishes after 6
            frames.append(_frame(i, [], balls))
        chosen = select_address_ball(frames, impact_fi=6)
        assert chosen["struck"] is True
        assert abs(chosen["x"] - 760) < 2               # the one that disappeared

    def test_ambiguous_when_nothing_disappears(self):
        # Two persistent balls, neither disappears -> flagged not-struck.
        frames = [_frame(i, [], [_box(760, 1330, 0.5), _box(500, 1190, 0.4)])
                  for i in range(10)]
        chosen = select_address_ball(frames, impact_fi=6)
        assert chosen["struck"] is False                # honest: can't confirm struck ball
