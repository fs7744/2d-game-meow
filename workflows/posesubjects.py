"""多角色人偶参考图 → 提示词契约修正。

**为什么需要**：导出的白底灰人偶图可以是双人/多人同框（互动姿势），但
`generate.py` 的姿态链路写死了「a single character」「Redraw the same
character」，更糟的是 `QWEN_FRAME_NEG` 把 **"two characters" 列为负向词**。
多角色参考图接上去，模型会被明确要求只画一个主体。

**主体数怎么来**：导出人偶图时，前端把 `scene.characters.length` 一起发给
`/upload`，服务端在图片旁写一个 `.json` sidecar（文件在磁盘，不进数据库，
符合项目约定）。生成时按参考图路径读 sidecar。

为什么不用图像识别去数：实测真机导出的三个人偶因轻微接触，连通域会合并成
一块；而 w4 常见的 OpenPose 骨架线图又全是断开的细线，反而会被数成多个主体。
**确定性元数据比猜可靠**，且老图没有 sidecar 时一律按 1 处理。

**保守原则**：count == 1 时 `adapt_instruction()` / `adapt_negative()` 原样
返回，所有既有单主体产线的提示词一个字节都不变。官方 VNCCS 模板
（`_VNCCS_TEMPLATE`）是配套 LoRA 的固定指令，一个字都不动。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MAX_SUBJECTS = 12


def normalize_count(value: Any) -> int:
    """把任意来源的主体数收敛到 1~12；拿不准就是 1（保持旧行为）。"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(MAX_SUBJECTS, n))


def sidecar_path(image: str | Path) -> Path:
    """`xxx.png` → `xxx.png.json`。跟其它产物 sidecar 同一套命名。"""
    return Path(str(image) + ".json")


def write_subjects(image: str | Path, count: int) -> None:
    """上传时写 sidecar。单主体不写，避免给每张普通图片都留一个垃圾文件。"""
    n = normalize_count(count)
    if n <= 1:
        return
    sidecar_path(image).write_text(
        json.dumps({"pose_subjects": n}, ensure_ascii=False), encoding="utf-8")


def subjects_for(image: Any) -> int:
    """读参考图旁的 sidecar；没有 / 坏了 / 传的是 PIL 图 → 1。"""
    if image is None or isinstance(image, (bytes, bytearray)):
        return 1
    try:
        p = sidecar_path(image)
        if not p.is_file():
            return 1
        doc = json.loads(p.read_text(encoding="utf-8"))
        return normalize_count(doc.get("pose_subjects") if isinstance(doc, dict) else None)
    except (OSError, ValueError, TypeError):
        return 1


def subject_phrase(count: int) -> str:
    """给英文指令用的主体短语。count<=1 保持原契约词。"""
    if count <= 1:
        return "a single character"
    words = {2: "two characters", 3: "three characters", 4: "four characters",
             5: "five characters", 6: "six characters"}
    return words.get(count, f"{count} characters")


def adapt_instruction(text: str, count: int) -> str:
    """把单主体契约换成多主体契约。count<=1 时原样返回（含空格/大小写）。"""
    if normalize_count(count) <= 1 or not text:
        return text
    phrase = subject_phrase(normalize_count(count))
    out = text
    out = out.replace("A single character", phrase[0].upper() + phrase[1:])
    out = out.replace("a single character", phrase)
    out = out.replace("The same character", "The " + phrase)
    out = out.replace("the same character", "the " + phrase)
    out = out.replace("The figure must fill", "The figures must fill")
    if "relative positions" not in out:
        out += (" Keep all figures and their relative positions; "
                "do not merge or drop any figure.")
    return out


def adapt_negative(negative: str, count: int) -> str:
    """多主体时去掉负向词里的 "two characters"（QWEN_FRAME_NEG 有这一条）。"""
    if normalize_count(count) <= 1 or not negative:
        return negative
    out = re.sub(r",\s*two characters\b", "", negative)
    out = re.sub(r"\btwo characters\s*,?\s*", "", out)
    return out.strip().strip(",").strip()
