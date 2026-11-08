"""方向 / 机位的**后端唯一定义**。

之前方向表只长在 ``playtest.py``（验收台）里；现在生成（posegen）、打包
（spriteatlas）、引擎导出（godotexport）都要按方向分组，各写一份必然对不上。
统一放这里，四个模块共用。

前端 ``web/src/store/playtest.ts`` 有一张对应的 viewMode → 方向集表，
**改这里要同步改那边**（跨语言没法共享，靠测试钉住）。

⚠️ 官方 VNCCS 没有原生八方向。八个机位是我们在 PoseStudio 里自己转相机
（``PoseStudio.tsx`` 的 ``YAWS``）得到的，别写成官方能力；角度与方向名的
对应首次真机出八方向帧时还要校准一次。见
``docs/角色中心-剩余工作与做法.md`` §8。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

#: 视角模式 → 需要的方向 key。顺序即试玩页 HUD 的展示顺序。
DIRECTIONS: Dict[str, List[str]] = {
    "topdown-4": ["down", "up", "left", "right"],
    "topdown-8": [
        "down", "down-right", "right", "up-right",
        "up", "up-left", "left", "down-left",
    ],
    "side-2": ["left", "right"],
}

#: 方向 → 导出机位（度）。与 PoseStudio.tsx 的 YAWS 一一对应。
YAW_MAP: Dict[str, int] = {
    "down": 0,
    "down-right": 45,
    "right": 90,
    "up-right": 135,
    "up": 180,
    "up-left": 225,
    "left": 270,
    "down-left": 315,
}

#: 横板只需要一份朝向，另一份靠 flip_h 派生（生成成本减半）。
MIRROR_FROM: Dict[str, str] = {"right": "left"}

DIRECTION_KEYS = frozenset(YAW_MAP)

#: ``sprite_<direction>_0001.png``；没有方向维的老帧是 ``sprite_pose_0001.png``。
_FRAME_DIR_RE = re.compile(r"^sprite_([a-z][a-z0-9-]*)_(\d+)\.png$", re.I)


def direction_keys(mode: str) -> List[str]:
    """该视角模式需要的方向；未知模式回落到四方向。"""
    return list(DIRECTIONS.get(mode, DIRECTIONS["topdown-4"]))


def validate_direction(value: Any) -> str:
    """把帧元数据里的方向收敛成已知 key（脏值一律当「无方向」，别写进文件名）。"""
    key = str(value or "").strip().lower()
    return key if key in DIRECTION_KEYS else ""


def direction_from_name(name: Any) -> str:
    """从 ``sprite_<direction>_NNNN.png`` 里取方向；不是方向帧返回空串。

    老帧 ``sprite_pose_0001.png`` 也会命中正则，但 ``pose`` 不在白名单里，
    所以安全地回落成「无方向」——这正是老角色目录该有的行为。
    """
    m = _FRAME_DIR_RE.match(Path(str(name or "")).name)
    return validate_direction(m.group(1)) if m else ""


def yaw_of(direction: Any, default: int = 0) -> int:
    """方向 → 机位角；无方向帧返回 default。"""
    return YAW_MAP.get(validate_direction(direction), int(default))


def frame_filename(direction: Any, index: int) -> str:
    """方向帧按方向**独立编号**：``sprite_down_0001.png``。

    无方向时保持官方的 ``sprite_pose_0001.png`` 一字不变（老存档、老测试、
    VNCCS 目录互通都依赖这个名字）。
    """
    key = validate_direction(direction)
    stem = f"sprite_{key}" if key else "sprite_pose"
    return f"{stem}_{max(1, int(index)):04d}.png"
