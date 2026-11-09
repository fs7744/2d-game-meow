"""后端的中文显示层（提示词要用）。

**唯一真源是 `web/src/labels.ts`**，这里只读 `tools/build_labels_zh.py` 导出的
`workflows/labels_zh.json`（导出后加进仓库，运行时不做任何解析）。

为什么后端也要一份：属性标签在磁盘上存的是英文（`silver hair`），中文提示词
要把它写成「银发」。查不到就**原样返回**，绝不返回空串 —— 用户自己写的
自由文本本来就该原样进 prompt。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

_JSON = Path(__file__).with_name("labels_zh.json")


@lru_cache(maxsize=1)
def _maps() -> Dict[str, Dict[str, str]]:
    try:
        return json.loads(_JSON.read_text(encoding="utf-8"))
    except OSError:                               # 文件没生成 → 退回纯英文，不影响出图
        return {}


def _lookup(table: str, value: str) -> str:
    key = str(value or "").strip()
    if not key:
        return key
    return _maps().get(table, {}).get(key, key)


def tag_label(value: str) -> str:
    """属性标签（`silver hair` → 银发）。查不到原样返回。

    标签在磁盘上空格/下划线两种写法都出现过（`long_hair` vs `long hair`），
    两种都试一遍，别让小差异漏译。
    """
    key = str(value or "").strip()
    if not key:
        return key
    table = _maps().get("TAG_LABEL", {})
    for candidate in (key, key.replace(" ", "_"), key.replace("_", " ")):
        if candidate in table:
            return table[candidate]
    return key


def style_label(value: str) -> str:
    """画风名（`Shonen Anime` → 少年漫）。查不到原样返回。"""
    return _lookup("STYLE_LABEL", value)


def style_group_label(value: str) -> str:
    return _lookup("STYLE_GROUP_LABEL", value)


def translate_traits(value: str) -> str:
    """把 `blonde hair, blue eyes` 这种逗号串逐个翻译，未收录的保留原文。"""
    parts = [p.strip() for p in str(value or "").split(",")]
    out = [tag_label(p) for p in parts if p]
    return ", ".join(out)
