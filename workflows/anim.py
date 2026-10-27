"""游戏动效：播放预览（GIF/APNG）+ 引擎导出（Godot 4 / 通用 JSON）。

设计原则：
1. 导出是**确定性文本生成**，不跑模型——同一份帧出来的产物必须逐字节可复现。
2. 像素素材一律最近邻采样（Godot 侧 texture_filter = 1），否则引擎里会糊。
3. 原点对齐脚底：Sprite 的 offset.y = -cell_h/2，这样 set_position() 到地面线就站得住。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image

from . import postprocess as pp


# ---------------- 预览 ----------------

def save_gif(frames: Sequence[Image.Image], out: str | Path, fps: int = 10,
             loop: int = 0, scale: int = 1) -> Path:
    """GIF 预览。透明帧保留 alpha（GIF 只有 1-bit 透明，硬边素材没问题）。"""
    if not frames:
        raise ValueError("没有帧")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ims = list(frames)
    if scale != 1:
        ims = [f.resize((f.width * scale, f.height * scale), Image.NEAREST) for f in ims]
    duration = int(round(1000 / max(1, fps)))
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=duration,
                loop=loop, disposal=2, optimize=False)
    return out


def save_apng(frames: Sequence[Image.Image], out: str | Path, fps: int = 10) -> Path:
    """APNG：支持 8-bit alpha，比 GIF 更贴近实际效果（浏览器可直接播）。"""
    if not frames:
        raise ValueError("没有帧")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=list(frames)[1:],
                   duration=int(round(1000 / max(1, fps))), loop=0, format="PNG")
    return out


# ---------------- 引擎索引 ----------------

def engine_index(frames: Sequence[Image.Image], sheet: dict, *, name: str,
                 fps: int = 10, loop: bool = True, origin: str = "bottom_center") -> dict:
    """与引擎无关的描述文件，Unity / Phaser / 自研引擎都能直接读。"""
    return {
        "name": name,
        "sheet": Path(sheet["sheet"]).name,
        "cell": {"w": sheet["cell_w"], "h": sheet["cell_h"]},
        "grid": {"cols": sheet["cols"], "rows": sheet["rows"]},
        "count": len(frames),
        "fps": fps,
        "loop": loop,
        "origin": origin,
        "filter": "nearest",
        "frames": [
            {"i": i, "x": (i % sheet["cols"]) * sheet["cell_w"],
             "y": (i // sheet["cols"]) * sheet["cell_h"],
             "w": sheet["cell_w"], "h": sheet["cell_h"]}
            for i in range(len(frames))
        ],
    }


# ---------------- Godot 4 ----------------

_TRES_HEAD = '''[gd_resource type="SpriteFrames" load_steps={steps} format=3]

[ext_resource type="Texture2D" path="res://{texture}" id="1_sheet"]
'''


def _godot_uid(seed: str) -> str:
    """Godot 的 uid 只是引用标识，用文件名派生一个稳定的十六进制串即可。"""
    import hashlib

    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return "uid://" + "".join(c if c.isalnum() else "0" for c in h)


def export_godot(frames: Sequence[Image.Image], out_dir: str | Path, *, anim_name: str,
                 fps: int = 10, loop: bool = True, cols: int = 8,
                 sheet_name: str | None = None) -> dict:
    """导出一个可直接放进 Godot 4 工程的动画资源（.tres + .tscn + 图集）。

    产物：
      <dir>/<name>_sheet.png      图集
      <dir>/<name>.tres           SpriteFrames（每帧一个 AtlasTexture 子资源）
      <dir>/<name>.tscn           AnimatedSprite2D 场景，autoplay 已开、脚底对齐
      <dir>/<name>.json           引擎无关索引
      <dir>/<name>.zip            上述四项打包
    """
    frames = [f.convert("RGBA") for f in frames]
    if not frames:
        raise ValueError("没有帧")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = sheet_name or anim_name
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug) or "anim"

    sheet_path = out_dir / f"{safe}_sheet.png"
    sheet = pp.pack_sheet(frames, sheet_path, cols=cols)

    rel_dir = f"anim"          # Godot 内建议放在 res://anim/ 下
    sub_res = []
    for i in range(len(frames)):
        x = (i % sheet["cols"]) * sheet["cell_w"]
        y = (i // sheet["cols"]) * sheet["cell_h"]
        sub_res.append(
            f'[sub_resource type="AtlasTexture" id="AtlasTexture_{i:03d}"]\n'
            f'atlas = ExtResource("1_sheet")\n'
            f'region = Rect2({x}, {y}, {sheet["cell_w"]}, {sheet["cell_h"]})\n'
        )

    frame_entries = ",\n".join(
        '{\n"duration": 1.0,\n"texture": SubResource("AtlasTexture_%03d")\n}' % i
        for i in range(len(frames))
    )
    tres = (
        _TRES_HEAD.format(steps=len(sub_res) + 2, texture=f"{rel_dir}/{safe}_sheet.png")
        + "\n" + "\n".join(sub_res)
        + "\n[resource]\nanimations = [{\n"
        + f'"frames": [\n{frame_entries}\n],\n'
        + f'"loop": {"true" if loop else "false"},\n'
        + f'"name": &"{anim_name}",\n'
        + f'"speed": {float(fps)}\n'
        + "}]\n"
    )
    tres_path = out_dir / f"{safe}.tres"
    tres_path.write_text(tres, encoding="utf-8", newline="\n")

    # 脚底对齐：AnimatedSprite2D 居中绘制，向下偏移半格即让脚底落在节点原点
    tscn = (
        '[gd_scene load_steps=2 format=3]\n\n'
        f'[ext_resource type="SpriteFrames" path="res://{rel_dir}/{safe}.tres" id="1_sf"]\n\n'
        f'[node name="{anim_name}" type="AnimatedSprite2D"]\n'
        'texture_filter = 1\n'
        'sprite_frames = ExtResource("1_sf")\n'
        f'autoplay = "{anim_name}"\n'
        f"offset = Vector2(0, {-sheet['cell_h'] / 2:g})\n"
    )
    tscn_path = out_dir / f"{safe}.tscn"
    tscn_path.write_text(tscn, encoding="utf-8", newline="\n")

    index = engine_index(frames, sheet, name=anim_name, fps=fps, loop=loop)
    index["godot"] = {"tres": tres_path.name, "tscn": tscn_path.name,
                      "texture": sheet_path.name, "texture_filter": 1}
    idx_path = out_dir / f"{safe}.json"
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), "utf-8")

    gif_path = save_gif(frames, out_dir / f"{safe}_preview.gif", fps=fps)
    apng_path = save_apng(frames, out_dir / f"{safe}_preview.png", fps=fps)

    zip_path = out_dir / f"{safe}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (sheet_path, tres_path, tscn_path, idx_path, gif_path, apng_path):
            zf.write(p, arcname=p.name)

    return {
        "dir": str(out_dir), "sheet": str(sheet_path), "tres": str(tres_path),
        "tscn": str(tscn_path), "json": str(idx_path), "gif": str(gif_path),
        "apng": str(apng_path), "zip": str(zip_path), "index": index,
    }
