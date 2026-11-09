"""角色中心「序列帧动画」：产物必须落在**角色的生命阶段**下。

这条链路复用了画布项目的 `poseset` / `posebatch` / `godotexport`，只是把根目录
换成了 `assets/characters/<角色>/Sequence[/s_<阶段>]/`。这里钉住四件事：

1. 两个阶段互不可见（`s_<id>` 分目录，默认阶段用 `Sequence/` 根）；
2. 姿势集 / 成片集 / 候选 / 选中全部写在这一层，URL 是 `/files/characters/...`；
3. Godot 导出包也留在阶段目录里，不落到 `assets/projects/`；
4. 批量生成时的临时 scope 用完就摘掉（否则进程里会积一堆假项目路径）。
"""
from __future__ import annotations

import base64
import io
import shutil

import pytest
from PIL import Image

from workflows import generate, paths, posebatch, poseset, sequence

CHAR = "seq测试角色"
STAGE_A = "stage_aaaa1111"
STAGE_B = "stage_bbbb2222"


def _png_b64(level: int = 0, side: int = 8) -> str:
    buf = io.BytesIO()
    Image.new("RGBA", (side, side), (level, level, level, 255)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _frames(n: int = 2):
    return [{"t": i * 0.2, "yaw": 0, "pose": {"i": i}, "png": _png_b64(i * 20)}
            for i in range(1, n + 1)]


@pytest.fixture(autouse=True)
def _clean():
    d = paths.CHARACTERS_DIR / CHAR
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    yield
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def test_stage_dirs_are_separate():
    sequence.root(CHAR, STAGE_A)
    sequence.root(CHAR, "")
    base = paths.CHARACTERS_DIR / CHAR / "Sequence"
    assert (base / f"s_{STAGE_A}").is_dir()
    assert base.is_dir()


def test_pose_set_scoped_to_stage_and_hidden_from_others():
    poseset.save_pose_set("", "走路", _frames(3), character=CHAR, stage=STAGE_A)
    folder = paths.CHARACTERS_DIR / CHAR / "Sequence" / f"s_{STAGE_A}" / "poses" / "走路"
    assert (folder / "set.json").is_file()
    assert len(list(folder.glob("f_*.png"))) == 3

    assert [s["name"] for s in poseset.list_pose_sets("", character=CHAR, stage=STAGE_A)] == ["走路"]
    # 另一个阶段看不到
    assert poseset.list_pose_sets("", character=CHAR, stage=STAGE_B) == []

    detail = poseset.load_pose_set("", "走路", character=CHAR, stage=STAGE_A)
    assert detail["frames"][0]["url"].startswith("/files/characters/")


def test_default_stage_uses_sequence_root():
    poseset.save_pose_set("", "默认阶段", _frames(1), character=CHAR)
    assert (paths.CHARACTERS_DIR / CHAR / "Sequence" / "poses" / "默认阶段"
            / "set.json").is_file()


def test_render_candidates_stay_in_stage():
    poseset.save_pose_set("", "跑", _frames(2), character=CHAR, stage=STAGE_A)
    poseset.create_render("", "跑成片", "跑", character=CHAR, stage=STAGE_A)
    src = paths.CHARACTERS_DIR / CHAR / "Sequence" / f"s_{STAGE_A}" / "_scratch.png"
    src.write_bytes(base64.b64decode(_png_b64(200)))

    poseset.add_candidate("", "跑成片", 1, src, seed=7,
                          character=CHAR, stage=STAGE_A)
    one = poseset.load_render("", "跑成片", character=CHAR, stage=STAGE_A)
    assert one["done"] == 1
    cand_url = one["frames"][0]["candidates"][0]["url"]
    assert cand_url.startswith("/files/characters/")
    assert "Sequence" in cand_url

    two = poseset.select_candidate("", "跑成片", 1, 1,
                                   character=CHAR, stage=STAGE_A)
    assert two["frames"][0]["selectedIndex"] == 1


def test_godot_bundle_lands_in_stage_exports():
    from workflows import godotexport

    poseset.save_pose_set("", "跳", _frames(2), character=CHAR, stage=STAGE_A)
    poseset.create_render("", "跳成片", "跳", character=CHAR, stage=STAGE_A)
    for i in (1, 2):
        src = paths.CHARACTERS_DIR / CHAR / "Sequence" / f"s_{STAGE_A}" / f"_s{i}.png"
        src.write_bytes(base64.b64decode(_png_b64(i * 40)))
        poseset.add_candidate("", "跳成片", i, src, character=CHAR, stage=STAGE_A)

    r = godotexport.export_bundle("", "跳成片", character=CHAR, stage=STAGE_A)
    assert r["ok"] and r["frames"] == 2
    out = paths.CHARACTERS_DIR / CHAR / "Sequence" / f"s_{STAGE_A}" / "exports"
    assert (out / r["zip"]["name"]).is_file()
    assert not (paths.PROJECTS_DIR / "default" / "exports").exists()


def test_studio_and_mannequin_are_stage_scoped():
    assert sequence.load_studio(CHAR, STAGE_A) == {}
    sequence.save_studio(CHAR, STAGE_A, {"pose": {"x": 1}, "yaw": 90})
    assert sequence.load_studio(CHAR, STAGE_A)["yaw"] == 90
    # 另一个阶段读不到
    assert sequence.load_studio(CHAR, STAGE_B) == {}

    png = base64.b64decode(_png_b64(120))
    one = sequence.save_mannequin(CHAR, STAGE_A, png)
    assert one["name"].startswith(sequence.MANNEQUIN_PREFIX)
    assert one["url"].startswith("/files/characters/")
    names = [m["name"] for m in sequence.list_mannequins(CHAR, STAGE_A)]
    assert one["name"] in names


def test_batch_uses_stage_root_and_cleans_scratch(monkeypatch, tmp_path):
    poseset.save_pose_set("", "踢", _frames(2), character=CHAR, stage=STAGE_A)
    poseset.create_render("", "踢成片", "踢", character=CHAR, stage=STAGE_A)

    seen_scopes = []

    def fake_clone(eng, *, pose_image, char_image, seed=-1, scope="", tag="", **kw):
        out = tmp_path / f"g{len(seen_scopes)}.png"
        out.write_bytes(base64.b64decode(_png_b64(30)))
        seen_scopes.append(scope)
        return {"file": str(out), "meta": {"seed": seed}}

    monkeypatch.setattr(generate, "pose_clone", fake_clone)
    res = posebatch.run_batch(None, {
        "character": CHAR, "stage": STAGE_A,
        "render": "踢成片", "pose_set": "踢",
        "char_image": "char.png", "engine": "wpc",
    })
    assert res["images"] == 2 and res["failed"] == []
    assert all(s.startswith("seq_") for s in seen_scopes)
    # 临时 scope 用完必须摘掉
    assert not any(s in generate._SCOPE_ROOTS for s in seen_scopes)
    # 中间目录空了就删掉（阶段目录里不留 _tmp）
    assert not (paths.CHARACTERS_DIR / CHAR / "Sequence" / f"s_{STAGE_A}" / "_tmp").exists()
