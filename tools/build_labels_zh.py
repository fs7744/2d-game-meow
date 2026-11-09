"""把 `web/src/labels.ts` 里的中文映射抽成后端能读的 JSON。

**为什么要这一步**：界面上显示中文靠 `labels.ts`，但提示词是**后端**拼的
（`workflows/charactergen.py`）。中文提示词要把 `silver hair` 写成「银发」，
就得让 Python 也能查同一张表 —— 否则两份翻译一定会漂移。

- 唯一真源仍是 `web/src/labels.ts`（前端和测试都在用）；
- 本脚本只做单向导出：`python tools/build_labels_zh.py` → `workflows/labels_zh.json`；
- `tests/test_labels_zh.py` 会重新解析 labels.ts 跟 JSON 比对，**改了表没重跑就会红**。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "web" / "src" / "labels.ts"
OUT = ROOT / "workflows" / "labels_zh.json"

#: 只导出后端真正要用的表（其余留在前端，别把两个仓库都塞满）
WANTED = ("TAG_LABEL", "STYLE_LABEL", "STYLE_GROUP_LABEL", "BG_COLOR_LABEL")

#: 表头：`export const X: Record<string, string> = {`
_CONST_RE = re.compile(
    r"^export const (?P<name>[A-Z_]+)\s*:\s*Record<string,\s*string>\s*=\s*\{",
    re.MULTILINE,
)
#: 表里的一项：键可以单/双引号或裸标识符，值可能折到下一行（长中文会换行）。
_ENTRY_RE = re.compile(
    r"^\s*(?:'(?P<q>[^']*)'|\"(?P<dq>[^\"]*)\"|(?P<bare>[A-Za-z0-9_]+))\s*:\s*"
    r"'(?P<val>(?:[^'\\]|\\.)*)'\s*,?\s*$",
    re.MULTILINE,
)


def parse_maps(text: str) -> dict[str, dict[str, str]]:
    """从 labels.ts 里抠出所有 `export const X: Record<string, string> = {...}`。

    正则跟 `tests/test_labels.py` 用的那套保持一致（同一个文件、两种键写法、
    值可能折行），否则导出结果会跟覆盖率测试对不上。
    """
    out: dict[str, dict[str, str]] = {}
    for m in _CONST_RE.finditer(text):
        name = m.group("name")
        start = text.index("{", m.start())
        end = text.index("\n}", start)
        body = text[start + 1:end]
        table: dict[str, str] = {}
        for e in _ENTRY_RE.finditer(body):
            key = e.group("q") or e.group("dq") or e.group("bare")
            if key is None:
                continue
            table[key] = e.group("val")
        out[name] = table
    return out


def build() -> dict[str, dict[str, str]]:
    maps = parse_maps(SRC.read_text(encoding="utf-8"))
    data = {name: maps.get(name, {}) for name in WANTED}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    return data


if __name__ == "__main__":
    data = build()
    print("写出", OUT.relative_to(ROOT))
    for name, entries in data.items():
        print(f"  {name}: {len(entries)} 条")
