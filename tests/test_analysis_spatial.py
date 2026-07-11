import pytest

from swingtool.analysis.spatial import (
    fit_plane_tilt,
    rotation_angle_top_down,
    xfactor,
)


class TestFitPlaneTilt:
    def test_in_image_plane_is_zero_tilt(self):
        # all z == 0 -> the swing plane IS the image plane -> 0 tilt out of image
        pts = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0), (5, 5, 0)]
        tilt, planarity = fit_plane_tilt(pts)
        assert tilt == pytest.approx(0.0, abs=0.1)
        assert planarity > 0.99

    def test_tilted_plane(self):
        # plane z = x -> normal (-1,0,1)/sqrt2 -> 45 deg from the depth axis
        pts = [(0, 0, 0), (10, 0, 10), (0, 10, 0), (10, 10, 10), (5, 5, 5)]
        tilt, planarity = fit_plane_tilt(pts)
        assert tilt == pytest.approx(45.0, abs=1.0)

    def test_too_few_points(self):
        assert fit_plane_tilt([(0, 0, 0), (1, 1, 1)]) == (None, 0.0)


class TestRotationAngle:
    def test_flat_line(self):
        assert rotation_angle_top_down((0, 0), (10, 0)) == pytest.approx(0.0)

    def test_45(self):
        assert rotation_angle_top_down((0, 0), (10, 10)) == pytest.approx(45.0)

    def test_toward_camera(self):
        assert rotation_angle_top_down((0, 0), (0, 10)) == pytest.approx(90.0)


class TestXFactor:
    def test_difference(self):
        assert xfactor(40.0, 10.0) == pytest.approx(30.0)

    def test_none_when_missing(self):
        assert xfactor(None, 10.0) is None
        assert xfactor(40.0, None) is None
