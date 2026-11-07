from xgs.manifest import Stage, read_manifest


def test_stage_caching_and_invalidation(tmp_path):
    a = Stage(tmp_path, "a", {"x": 1})
    assert not a.is_cached()
    with a.running() as st:
        st.metrics["n"] = 3
    assert read_manifest(tmp_path / "a")["metrics"] == {"n": 3}
    assert Stage(tmp_path, "a", {"x": 1}).is_cached()
    assert not Stage(tmp_path, "a", {"x": 2}).is_cached()          # params changed

    b = Stage(tmp_path, "b", {}, upstream=["a"])
    with b.running():
        pass
    assert Stage(tmp_path, "b", {}, upstream=["a"]).is_cached()
    with Stage(tmp_path, "a", {"x": 2}).running():                  # upstream re-ran with new params
        pass
    assert not Stage(tmp_path, "b", {}, upstream=["a"]).is_cached()


def test_failed_stage_is_not_cached(tmp_path):
    st = Stage(tmp_path, "c", {})
    try:
        with st.running():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert read_manifest(tmp_path / "c")["status"] == "error"
    assert not Stage(tmp_path, "c", {}).is_cached()
