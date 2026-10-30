"""**真跑 Qwen** 的端到端冒烟：十个工作流各走一遍最小尺寸。

为什么需要这一组：日常的 pytest 全是 monkeypatch（不加载模型），
它们能验「贴回/缩放/落盘」这条链路，但**验不了模型本身还能不能出图**——
改了采样器、改了条件图编码、改了 prompt 模板之后，单测照样全绿，
用户一跑才发现出不来图。

所以这里每个工作流都**真的调一次模型**，代价是每条 10~30s、整组约 2~3 分钟。
默认跳过，用环境变量开：

    MEOW_SLOW=1 pytest tests/test_e2e_qwen.py -v

**断言只查「有没有出图、meta 对不对」，不查画质**：低步数下画质当然差，
画质由 `tools/verify_qwen_workflows.py` 与产线判据负责，这条只管链路通不通。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.skipif(
    os.environ.get("MEOW_SLOW") != "1",
    reason="要真跑 Qwen 管线（约 2~3 分钟）：MEOW_SLOW=1 pytest tests/test_e2e_qwen.py")


ROOT = Path(__file__).resolve().parents[1]


# ---------------- 素材 ----------------

def _src(tmp_path: Path, size: int = 256) -> Path:
    im = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([size // 3, size // 4, size * 2 // 3, size * 3 // 4],
                fill=(200, 40, 40))
    p = tmp_path / "src.png"
    im.save(p)
    return p


def _mask(tmp_path: Path, size: int = 256) -> Path:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rectangle([size // 3, size // 4, size // 2, size // 2], fill=255)
    p = tmp_path / "mask.png"
    m.save(p)
    return p


def _skeleton(tmp_path: Path, n: int = 1) -> list[Path]:
    """用项目自己的骨架渲染器造条件图（不依赖真人照片素材）。"""
    from tools.make_pose import render, walk_pose

    out = []
    for i in range(n):
        p = tmp_path / f"pose_{i}.png"
        render(walk_pose(i / max(1, n)), size=(256, 256)).save(p)
        out.append(p)
    return out


def _char(tmp_path: Path) -> Path:
    im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([88, 60, 168, 140], fill=(40, 170, 200, 255))
    d.rectangle([104, 140, 152, 220], fill=(30, 110, 160, 255))
    p = tmp_path / "char.png"
    im.save(p)
    return p


def _sheet(tmp_path: Path) -> Path:
    im = Image.new("RGBA", (64 * 4, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(4):
        d.rectangle([i * 64 + 25, 30, i * 64 + 38, 48], fill=(120, 30, 40, 255))
    p = tmp_path / "sheet.png"
    im.save(p)
    return p


def _ok(res, tmp_path: Path, workflow: str) -> None:
    """通用断言：meta 里工作流对得上、产物真的在盘上。

    **不查耗时**：跑模型的几个工作流才有 seconds/seconds_total，
    纯后处理的（抠图、像素化）没有——查了就得给每种工作流写一套分支。
    """
    assert res["meta"]["workflow"] == workflow
    out = Path(res["file"])
    assert out.exists() and out.stat().st_size > 0


# ---------------- 十个工作流 ----------------

def test_e2e_txt2img(eng, tmp_path):
    from workflows import generate as g

    r = g.txt2img(eng, prompt="a blue slime monster", width=256, height=256,
                  steps=12, seed=11, preset="none", pixel=0, palette=0,
                  scope="e2e", tag="e2e")
    _ok(r, tmp_path, "w1")


def test_e2e_img2img(eng, tmp_path):
    from workflows import generate as g

    r = g.img2img(eng, init_image=_src(tmp_path), prompt="make it green",
                  steps=12, seed=12, preset="none", pixel=0, palette=0,
                  scope="e2e", tag="e2e")
    _ok(r, tmp_path, "w2")
    assert Path(r["meta"]["raw"]).exists()


def test_e2e_inpaint(eng, tmp_path):
    from workflows import generate as g

    r = g.inpaint(eng, image=_src(tmp_path), mask=_mask(tmp_path),
                  prompt="turn the masked area green", steps=12, seed=13,
                  scope="e2e", tag="e2e")
    _ok(r, tmp_path, "w3")
    # 硬承诺：选区外逐点不变
    assert r["meta"]["outside_max_diff"] == 0


def test_e2e_ai_repaint(eng, tmp_path):
    from workflows import generate as g

    r = g.ai_repaint(eng, image=_src(tmp_path), mask=_mask(tmp_path),
                     prompt="turn the masked area into a blue gem",
                     steps=12, seed=14, preset="none", scope="e2e", tag="e2e")
    _ok(r, tmp_path, "w5")


def test_e2e_poseswap(eng, tmp_path):
    from workflows import generate as g

    r = g.poseswap(eng, pose_images=_skeleton(tmp_path, 1),
                   prompt="a blue slime monster", steps=12, width=256, height=256,
                   matte=False, candidates=1, gate=False, scope="e2e", tag="e2e")
    # w4 的返回是「精灵图 + 帧目录」，不是单张 file
    assert r["meta"]["workflow"] == "w4"
    assert r["meta"]["frames"] == 1
    sheet = Path(r["sheet"])
    assert sheet.exists() and sheet.stat().st_size > 0
    frames = sorted(Path(r["frames_dir"]).glob("*.png"))
    assert len(frames) == 1


def test_e2e_pose_clone(eng, tmp_path):
    from workflows import generate as g

    r = g.pose_clone(eng, pose_image=_skeleton(tmp_path, 1)[0],
                     char_image=_char(tmp_path), prompt="a blue slime monster",
                     width=256, height=256, steps=12, ref_mode="skeleton",
                     matte=False, scope="e2e", tag="e2e")
    _ok(r, tmp_path, "wpc")


def test_e2e_sprite_swap(eng, tmp_path):
    from workflows import generate as g

    r = g.sprite_swap(eng, sheet_image=_sheet(tmp_path), char_image=_char(tmp_path),
                      cols=4, rows=1, mode="recolor", palette=8,
                      scope="e2e", tag="e2e")
    _ok(r, tmp_path, "wss")
    assert r["meta"]["frames"] == 4


def test_e2e_matte_bg(eng, tmp_path):
    from workflows import generate as g

    r = g.matte_bg(eng, init_image=_src(tmp_path), bg_color="#204080",
                   scope="e2e", tag="e2e")
    _ok(r, tmp_path, "wbg")


def test_e2e_pixelify(eng, tmp_path):
    from workflows import generate as g

    r = g.pixelify(eng, init_image=_src(tmp_path), target=(64, 64), colors=16,
                   scope="e2e", tag="e2e")
    _ok(r, tmp_path, "wpx")


def test_e2e_image_to_pixel(eng, tmp_path):
    """转像素的**生成路径**：没给源图时先出一张再像素化。

    这条以前是坏的：它给 txt2img 传了已经删除的 clip_skip 参数 →
    TypeError。参数删干净了才走得通，所以必须留一条真跑它的测试。
    """
    from workflows import generate as g

    r = g.image_to_pixel(eng, prompt="a blue slime monster",
                         source_width=256, source_height=256, steps=12,
                         seed=15, colors=16, scope="e2e", tag="e2e")
    _ok(r, tmp_path, "topixel")
    assert r["meta"]["generated"] is True
