"""Reference-config integrity: every threshold must carry a real citation."""

from swingtool.report.engine import load_references

ALLOWED_SOURCE_TYPES = {"literature", "coaching_standard", "pipeline_reference"}


def test_config_versioned():
    refs = load_references()
    assert refs["reference_version"]


def test_every_entry_has_citation_and_type():
    refs = load_references()
    assert refs["entries"], "reference config must not be empty"
    for key, entry in refs["entries"].items():
        assert entry.get("source", "").strip(), f"{key} has no citation"
        # a citation needs at least an author/title and a year
        assert any(ch.isdigit() for ch in entry["source"]), f"{key} citation has no year"
        assert entry.get("source_type") in ALLOWED_SOURCE_TYPES, f"{key} bad source_type"
        assert "tolerance_ours" in entry, f"{key} must state whether tolerances are ours"


def test_every_entry_has_range_or_direction():
    refs = load_references()
    for key, entry in refs["entries"].items():
        assert "range" in entry or "direction" in entry, f"{key} has no rule payload"


def test_pipeline_references_labeled_sample_size():
    refs = load_references()
    for key, entry in refs["entries"].items():
        if entry["source_type"] == "pipeline_reference":
            assert "sample_size" in entry, f"{key}: pipeline references must state sample_size"
