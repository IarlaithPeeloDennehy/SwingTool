import pytest

from swingtool.metrics.geometry import (
    angle_at_vertex,
    line_angle_from_horizontal,
    tilt_from_vertical,
)


class TestAngleAtVertex:
    def test_right_angle(self):
        # vertex at origin, arms along +x and -y (image coords)
        assert angle_at_vertex((100, 0), (0, 0), (0, 100)) == pytest.approx(90.0)

    def test_straight_leg(self):
        # hip, knee, ankle colinear -> 180 degrees
        assert angle_at_vertex((0, 0), (0, 100), (0, 200)) == pytest.approx(180.0)

    def test_known_bend(self):
        # 45-degree bend
        assert angle_at_vertex((100, 0), (0, 0), (100, 100)) == pytest.approx(45.0)

    def test_degenerate(self):
        assert angle_at_vertex((0, 0), (0, 0), (1, 1)) is None


class TestTiltFromVertical:
    def test_upright(self):
        # bottom below, top directly above (smaller y is higher on screen)
        assert tilt_from_vertical((0, 100), (0, 0)) == pytest.approx(0.0)

    def test_45_degrees(self):
        assert tilt_from_vertical((0, 100), (100, 0)) == pytest.approx(45.0)

    def test_horizontal(self):
        assert tilt_from_vertical((0, 0), (100, 0)) == pytest.approx(90.0)


class TestLineAngle:
    def test_horizontal(self):
        assert line_angle_from_horizontal((0, 100), (100, 100)) == pytest.approx(0.0)

    def test_tilted_up(self):
        # p2 higher on screen (smaller y) -> positive angle
        assert line_angle_from_horizontal((0, 100), (100, 0)) == pytest.approx(45.0)
