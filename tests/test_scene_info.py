import json

from xgs.report.export import refresh_scene_info, scene_info


def test_scene_info_lists_attribution_for_published_scenes():
    for name in ("dc-glass", "london-walk", "tum-desk"):
        info = scene_info(name)
        assert info["attribution"] and info["licence"] and info["licence_url"].startswith("https://")
    assert scene_info("not-a-scene") == {}


def test_refresh_scene_info(tmp_path):
    (tmp_path / "dc-glass").mkdir()
    (tmp_path / "dc-glass" / "scene.json").write_text(json.dumps({"name": "dc-glass", "source": "Google Glass"}))
    assert refresh_scene_info(tmp_path) == ["dc-glass"]
    meta = json.loads((tmp_path / "dc-glass" / "scene.json").read_text())
    assert meta["info"]["licence"] == "CC BY 3.0"
    idx = json.loads((tmp_path / "scenes.json").read_text())
    assert idx["scenes"][0]["title"].startswith("Washington DC")
