import pytest

from xgs.config import apply_overrides, deep_merge, load_config, parse_override


def test_deep_merge_nested():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4}) == {"a": {"b": 1, "c": 3}, "d": 4}


def test_parse_override_yaml_values():
    assert parse_override("select.max_keyframes=200") == (["select", "max_keyframes"], 200)
    assert parse_override("ingest.duration_s=null") == (["ingest", "duration_s"], None)
    assert parse_override("privacy.enabled=false") == (["privacy", "enabled"], False)
    with pytest.raises(ValueError):
        parse_override("nonsense")


def test_apply_overrides_creates_path():
    assert apply_overrides({}, ["x.y=1.5"]) == {"x": {"y": 1.5}}


def test_load_config_with_device_and_override():
    cfg = load_config(device="tum_fr1", overrides=["ingest.fps=10"])
    assert cfg["ingest"]["fps"] == 10
    assert cfg["device_profile"]["kind"] == "tum"
    assert cfg["device_profile"]["calibration"]["model"] == "OPENCV"


def test_all_device_profiles_load():
    for name in ("generic_video", "xreal_aura", "google_glass", "tum_fr1"):
        assert load_config(device=name)["device_profile"]["id"] == name
