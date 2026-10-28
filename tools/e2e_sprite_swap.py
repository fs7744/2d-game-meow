"""端到端验证：序列帧角色替换（WSS）+ 角色迁移的非真人参考模式（WPC）。

跑真实模型，产出实际图片供目视检查：

    python tools/e2e_sprite_swap.py

素材用仓库里的 assets/_probe/（像素骑士序列帧 + 像素史莱姆），
没有的话先跑 python tools/probe_nonhuman.py 造（已并入本脚本）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workflows.engine import Engine
from workflows import generate as g

OUT = ROOT / "assets" / "_probe"


def _make_assets() -> tuple[Path, Path]:
    """现场造素材：4 帧像素骑士序列 + 一张像素史莱姆角色图。

    像素骑士是**人体检测器认不出来**的那种素材——这正是要验证的场景。
    """
    from PIL import ImageDraw

    OUT.mkdir(parents=True, exist_ok=True)

    def knight(i: int) -> Image.Image:
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([26, 48, 30, 58], fill=(40, 40, 60, 255))
        d.rectangle([33, 48, 37, 58], fill=(40, 40, 60, 255))
        d.rectangle([25, 30, 38, 48], fill=(120, 30, 40, 255))
        d.rectangle([27, 18, 36, 30], fill=(235, 200, 160, 255))
        d.rectangle([26, 16, 37, 21], fill=(160, 165, 175, 255))
        d.rectangle([21, 31, 25, 42], fill=(120, 30, 40, 255))
        d.rectangle([38, 31, 42, 42], fill=(120, 30, 40, 255))
        d.rectangle([44, 20, 46, 40], fill=(200, 200, 210, 255))
        return im

    sheet = Image.new("RGBA", (64 * 4, 64), (0, 0, 0, 0))
    for i in range(4):
        sheet.paste(knight(i), (i * 64, 0))
    sheet_path = OUT / "knight_sheet.png"
    sheet.save(sheet_path)

    slime = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(slime)
    d.ellipse([10, 20, 54, 58], fill=(70, 190, 90, 255))
    d.ellipse([14, 24, 50, 54], fill=(110, 225, 120, 255))
    char_path = OUT / "slime_char.png"
    slime.save(char_path)
    return sheet_path, char_path


def subject_px(p: Path) -> int:
    a = np.asarray(Image.open(p).convert("RGBA"))
    return int((a[..., 3] > 127).sum())


def main() -> None:
    eng = Engine(ROOT / "models")
    sheet, char = _make_assets()

    print("=== WSS generate：4×1 序列帧 → 史莱姆 ===")
    r = g.sprite_swap(
        eng, sheet_image=sheet, char_image=char,
        prompt="green slime monster", negative=None,
        preset="sprite", cols=0, rows=0,
        width=768, height=768, steps=25, seed=20261028,
        cn_scale=1.0, ipa_scale=0.85, palette=16,
        matte=True, tag="wss_e2e",
    )
    m = r["meta"]
    print(f"  网格={m['grid']} 帧数={m['frames']} 格子={m['cell']} 模式={m['mode']}")
    print(f"  引擎={m.get('engine_mode')} 抠图={m['matte_used']} 色板={m['palette_size']} 色")
    print(f"  总={m['seconds_total']}s 显存={m['peak_vram_gb']}GB "
          f"分辨率={m.get('width')}²")
    if m.get("vram_note"):
        print(f"  显存规划={m['vram_note']}")
    if m.get("repaired_frames") is not None:
        print(f"  补提帧={m['repaired_frames']} 闸门={m.get('gate_passed')}/{m['frames']}")
    for p in sorted(Path(r["frames_dir"]).glob("frame_*.png")):
        print(f"    {p.name} 主体像素={subject_px(p)}")

    print("\n=== WSS recolor：确定性换皮 ===")
    r2 = g.sprite_swap(
        eng, sheet_image=sheet, char_image=char,
        prompt="", preset=None, cols=0, rows=0,
        palette=8, mode="recolor", tag="wss_recolor",
    )
    m2 = r2["meta"]
    print(f"  网格={m2['grid']} 帧数={m2['frames']} 色板={m2['palette_size']} 色 "
          f"确定性={m2['deterministic']} 耗时={m2['seconds_total']}s")
    for p in sorted(Path(r2["frames_dir"]).glob("frame_*.png")):
        print(f"    {p.name} 主体像素={subject_px(p)}")

    print("\n=== WPC：像素画参考（auto 降级）===")
    r3 = g.pose_clone(
        eng, pose_image=sheet, char_image=char,
        prompt="green slime monster", negative=None, preset="sprite",
        width=768, height=768, steps=25, seed=20261028,
        cn_scale=1.0, ipa_scale=0.85, ref_mode="auto", matte=True,
        tag="wpc_nonhuman",
    )
    m3 = r3["meta"]
    print(f"  请求={m3['ref_mode']} 实测={m3['ref_mode_used']} 降级自={m3['ref_fallback_from']}")
    print(f"  产物={r3['file']}")


if __name__ == "__main__":
    main()
