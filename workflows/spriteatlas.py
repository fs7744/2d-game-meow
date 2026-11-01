"""精灵图集导出 —— 把角色目录里攒的帧打成引擎能直接读的图集 + 元数据。

**为什么需要这一层**：`assets/characters/<角色>/Sprites/<服装>/<表情>/*.png`
是"给人和工具看"的目录结构——一套动画散在几十个文件里。游戏引擎要的是**一张
大图 + 一份坐标表**：一次上传、一次绑定、一帧一次 UV 偏移。中间这一步不做，
每个用这套资产的人都要自己写一遍打包脚本，而且十有八九会把脚底对齐搞错
（表现是"角色走路时上下抖"）。

## 对齐规则（最容易做错的地方）

同一套动画里帧的**画布尺寸可能不一样**（像素化、抠像、不同姿势的包围盒不同）。
直接按左上角拼会让角色在动画里上下跳。所以：

```
统一画布 = 所有帧里最大的 (w, h)
每帧在画布内：水平居中、底边对齐
```

底边对齐是因为精灵帧要站在地面上——脚底是唯一在所有姿势里都该固定的位置。
`atlas.json` 里同时给出 `canvas`（统一画布尺寸）与每帧的 `spriteSourceSize`
偏移，引擎按 `canvas` 建 sprite 就不会抖。

## 输出格式

`frames` 是 **TexturePacker JSON Hash**（Aseprite 也能直接导入），
`animations` 是我们自己加的一层——把「服装/表情」映射成帧序列，
免得引擎侧再自己分组。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from workflows import characters


class AtlasError(RuntimeError):
    """图集导出的可读错误（API 转 400）。"""


@dataclass
class Frame:
    """一帧的来处与几何信息。"""

    costume: str
    emotion: str
    index: int
    path: str
    size: Tuple[int, int]
    #: 统一画布内的偏移（水平居中 + 底边对齐），导出时填
    offset: Tuple[int, int] = (0, 0)


@dataclass
class AtlasResult:
    image_path: str
    meta_path: str
    frames: int = 0
    animations: int = 0
    size: Tuple[int, int] = (0, 0)
    canvas: Tuple[int, int] = (0, 0)
    bytes: int = 0
    notes: List[str] = field(default_factory=list)


def _frame_index(path: str) -> int:
    """从 `sprite_<表情>_0007.png` 里取 7；取不出来就按 0（排最后）。"""
    stem = Path(path).stem
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def collect_frames(
    name: str,
    costumes: Optional[Sequence[str]] = None,
    emotions: Optional[Sequence[str]] = None,
) -> List[Frame]:
    """扫角色目录，收集要打包的帧（按服装 → 表情 → 帧号排序）。

    `costumes` / `emotions` 为空表示"全要"。`Naked` 与 `Original` 是基础集，
    导出时**默认包含**——引擎里它们通常就是"默认外观"。
    """
    if not characters.character_exists(name):
        raise AtlasError(f"没有这个角色：{name}")
    want_c = {c.casefold() for c in costumes} if costumes else None
    want_e = {e.casefold() for e in emotions} if emotions else None

    out: List[Frame] = []
    for costume in sorted(characters.list_costumes(name)):
        if want_c is not None and costume.casefold() not in want_c:
            continue
        # 目录里可能有任意表情子目录，不能写死 neutral/smile
        base = os.path.dirname(characters.sprites_dir(name, costume, "neutral"))
        if not os.path.isdir(base):
            continue
        for emotion in sorted(os.listdir(base)):
            sub = os.path.join(base, emotion)
            if not os.path.isdir(sub):
                continue
            if want_e is not None and emotion.casefold() not in want_e:
                continue
            paths = characters.list_sprite_images(name, costume, emotion)
            for path in paths:
                try:
                    with Image.open(path) as im:
                        size = im.size
                except (OSError, ValueError):
                    continue
                out.append(Frame(costume=costume, emotion=emotion,
                                 index=_frame_index(path), path=path, size=size))
    if not out:
        raise AtlasError(
            f"「{name}」没有可导出的精灵帧：先在「出图」页生成一张，"
            "或跑一次多姿势批量")
    return out


def normalize_layout(frames: Sequence[Frame]) -> Tuple[int, int]:
    """给每帧算统一画布内的偏移（**底边对齐 + 水平居中**），返回画布尺寸。

    这个函数会**写回** `frames[i].offset`。底边对齐是精灵动画的硬要求：
    脚底固定在画面底部，角色走路才不会上下飘。
    """
    if not frames:
        return (0, 0)
    canvas_w = max(int(f.size[0]) for f in frames)
    canvas_h = max(int(f.size[1]) for f in frames)
    for f in frames:
        w, h = int(f.size[0]), int(f.size[1])
        f.offset = ((canvas_w - w) // 2, canvas_h - h)   # 水平居中、贴底
    return (canvas_w, canvas_h)


def _shelf_pack(
    sizes: Sequence[Tuple[int, int]], padding: int, max_width: int,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """货架式打包：按高度从高到矮，逐行铺开（够用、好懂、不追求最优）。

    `padding` 是**帧与帧之间**的间距。像素精灵图集必须留缝：相邻帧如果贴边，
    引擎用线性过滤采样时会把邻居的像素吸进来（表现为边缘一圈杂色）。
    """
    # 统一画布之后所有格子同尺寸，按**原顺序**铺就好：图集是给人也能看的，
    # 打乱顺序只会在排查问题时多一层困惑（坐标反正都记在 JSON 里）。
    order = list(range(len(sizes)))
    pad = max(0, int(padding))
    limit = max(1, int(max_width))

    xs: List[int] = [0] * len(sizes)
    ys: List[int] = [0] * len(sizes)
    x = y = 0
    shelf_h = 0
    used_w = 0
    for i in order:
        w, h = sizes[i]
        if x > 0 and x + w > limit:
            # 换行
            x = 0
            y += shelf_h + pad
            shelf_h = 0
        xs[i] = x
        ys[i] = y
        x += w + pad
        used_w = max(used_w, xs[i] + w)
        shelf_h = max(shelf_h, h)
    total_h = y + shelf_h
    return list(zip(xs, ys)), (max(1, used_w), max(1, total_h))


def build_atlas(
    frames: Sequence[Frame],
    *,
    padding: int = 2,
    max_width: int = 4096,
    background: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Tuple[Image.Image, List[Tuple[int, int]], Tuple[int, int]]:
    """把帧拼成一张大图，返回 `(atlas, positions, canvas_size)`。

    画布尺寸会先算 `normalize_layout`（调用方可能已经算过，重复算无害）。
    """
    canvas = normalize_layout(frames)
    positions, atlas_size = _shelf_pack(
        [(canvas[0], canvas[1]) for _ in frames], padding, max_width)
    atlas = Image.new("RGBA", atlas_size, background)
    for frame, (x, y) in zip(frames, positions):
        with Image.open(frame.path) as opened:
            img = opened.convert("RGBA")
        if img.size != canvas:
            # 放到统一画布里（透明底 + 偏移），而不是拉伸 —— 拉伸会毁掉像素网格
            cell = Image.new("RGBA", canvas, (0, 0, 0, 0))
            cell.paste(img, frame.offset)
            img = cell
        atlas.paste(img, (x, y))
    return atlas, positions, canvas


def export_character_atlas(
    name: str,
    *,
    out_dir: Optional[str] = None,
    costumes: Optional[Sequence[str]] = None,
    emotions: Optional[Sequence[str]] = None,
    padding: int = 2,
    max_width: int = 4096,
    frame_rate: float = 8.0,
) -> AtlasResult:
    """导出 `atlas.png` + `atlas.json`，返回落盘信息。

    默认输出到 `<角色目录>/Atlas/`（跟着角色走，删角色时一起删掉）。
    """
    frames = collect_frames(name, costumes, emotions)
    atlas, positions, canvas = build_atlas(
        frames, padding=padding, max_width=max_width)

    target = Path(out_dir) if out_dir else Path(characters.character_dir(name)) / "Atlas"
    target.mkdir(parents=True, exist_ok=True)
    image_path = target / "atlas.png"
    meta_path = target / "atlas.json"
    # 原子落盘：写一半的图集比没有更糟（引擎会读到坏文件）
    tmp_img = image_path.with_suffix(".png.tmp")
    atlas.save(tmp_img, format="PNG")
    os.replace(tmp_img, image_path)

    entries: Dict[str, Any] = {}
    groups: Dict[str, List[str]] = {}
    notes: List[str] = []

    for frame, (x, y) in zip(frames, positions):
        # 帧名用「服装/表情/原文件名」：引擎侧看名字就知道是哪一套的哪一帧
        key = f"{frame.costume}/{frame.emotion}/{Path(frame.path).name}"
        entries[key] = {
            "frame": {"x": x, "y": y, "w": canvas[0], "h": canvas[1]},
            "rotated": False,
            "trimmed": False,
            # 原图在统一画布里的位置（底边对齐后的偏移）
            "spriteSourceSize": {"x": frame.offset[0], "y": frame.offset[1],
                                 "w": frame.size[0], "h": frame.size[1]},
            "sourceSize": {"w": canvas[0], "h": canvas[1]},
        }
        groups.setdefault(f"{frame.costume}/{frame.emotion}", []).append(key)

    animations: Dict[str, Any] = {}
    for group, keys in groups.items():
        keys.sort(key=lambda k: _frame_index(k))
        animations[group] = {"frames": keys, "frameRate": frame_rate, "loop": True}
        if len(keys) == 1:
            notes.append(f"{group} 只有 1 帧（单帧动画，引擎里当静态图用）")

    meta = {
        "frames": entries,
        "animations": animations,
        "meta": {
            "app": "meow-sprite-atlas",
            "version": "1",
            "image": "atlas.png",
            "format": "RGBA8888",
            "size": {"w": atlas.width, "h": atlas.height},
            "scale": "1",
            "character": name,
            # 引擎侧建 sprite 时按这个尺寸，别按 atlas 里单帧的 frame.w/h 猜
            "canvas": {"w": canvas[0], "h": canvas[1]},
            # 轴心在底边中点：地面站位就是帧的底边
            "pivot": {"x": 0.5, "y": 1.0},
        },
    }
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    return AtlasResult(
        image_path=str(image_path), meta_path=str(meta_path),
        frames=len(frames), animations=len(animations),
        size=(atlas.width, atlas.height), canvas=canvas,
        bytes=image_path.stat().st_size, notes=notes,
    )
