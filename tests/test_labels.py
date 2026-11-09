"""翻译覆盖率校验：官方资源里的每一条，界面上都得有中文。

**为什么要有这个测试**：界面上的英文全是从官方资源里原样取出来的
（`emotions.json` / `character_styles.json` / `character_presets_v2.json` /
`PoseLibrary/`）。这些文件是官方仓库的拷贝，**我们不改它们**（改了以后
升级官方版本就得手工合并一遍），而是在 `web/src/labels.ts` 里维护一张
显示层映射表。

于是就有一种很隐蔽的漏法：官方加了新表情 / 新画风，我们没同步翻译，
界面上就悄悄冒出一个英文。人工看不出来，但用户看得见。

所以这个测试直接对着官方资源逐条核对：

| 官方资源 | 必须有译文的表 |
|---|---|
| `emotions.json` | `EMOTION_CATEGORY_LABEL` / `EMOTION_LABEL` |
| `character_styles.json` | `STYLE_GROUP_LABEL` / `STYLE_LABEL` |
| `character_presets_v2.json` | `TAG_CATEGORY_LABEL` / `TAG_GROUP_LABEL` / `TAG_LABEL` |
| `PoseLibrary/` | `POSE_CATEGORY_LABEL` / `POSE_NAME_LABEL` / `POSE_TAG_LABEL` |

映射表是 TS 文件（带注释、带类型标注），这里用正则解析出字面量 ——
比再维护一份 JSON 划算，代价是解析器要认 TS 的引号/标识符两种键写法。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows import poselib  # noqa: E402

LABELS_TS = ROOT / "web" / "src" / "labels.ts"
VNCCS = ROOT / "web" / "public" / "vnccs"
TEMPLATE = VNCCS / "character_template"

#: `export const NAME: Record<string, string> = {` 开头
_CONST_RE = re.compile(
    r"^export const (?P<name>[A-Z_]+)\s*:\s*Record<string,\s*string>\s*=\s*\{",
    re.MULTILINE,
)

#: 表里的一行：键可以带引号、可以是不带引号的标识符、也可以是数字
_ENTRY_RE = re.compile(
    r"^\s*(?:'(?P<q>[^']*)'|\"(?P<dq>[^\"]*)\"|(?P<bare>[A-Za-z0-9_]+))\s*:\s*"
    r"'(?P<val>(?:[^'\\]|\\.)*)'\s*,?\s*$",
    re.MULTILINE,
)


def parse_label_tables() -> Dict[str, Dict[str, str]]:
    """从 `web/src/labels.ts` 里解析出所有 `Record<string, string>` 常量。"""
    text = LABELS_TS.read_text(encoding="utf-8")
    tables: Dict[str, Dict[str, str]] = {}
    for m in _CONST_RE.finditer(text):
        name = m.group("name")
        # 从开头的 `{` 往后找配对的 `}`（表里没有嵌套花括号，安全）
        start = text.index("{", m.start())
        end = text.index("\n}", start)
        body = text[start + 1:end]
        table: Dict[str, str] = {}
        for e in _ENTRY_RE.finditer(body):
            key = e.group("q") or e.group("dq") or e.group("bare")
            if key is None:
                continue
            table[key] = e.group("val")
        tables[name] = table
    return tables


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_label_tables_parsed() -> None:
    """解析器本身要能用 —— 少解析出表名就等于整个测试空转。"""
    tables = parse_label_tables()
    expected = {
        "COSTUME_LABEL", "BG_COLOR_LABEL",
        "EMOTION_CATEGORY_LABEL", "EMOTION_LABEL",
        "STYLE_GROUP_LABEL", "STYLE_LABEL",
        "TAG_CATEGORY_LABEL", "TAG_GROUP_LABEL", "TAG_LABEL",
        "POSE_CATEGORY_LABEL", "POSE_NAME_LABEL", "POSE_TAG_LABEL",
    }
    missing = expected - set(tables)
    assert not missing, f"labels.ts 里少了这些表：{sorted(missing)}"
    for name in expected:
        assert tables[name], f"{name} 解析出来是空的（正则没匹配上？）"


def test_emotions_fully_translated() -> None:
    """官方表情库的每一个表情、每一个分类都要有中文。"""
    tables = parse_label_tables()
    data = _load(VNCCS / "emotions" / "emotions.json")
    missing_cat = [c for c in data if c not in tables["EMOTION_CATEGORY_LABEL"]]
    assert not missing_cat, f"这些表情分类没翻译：{missing_cat}"
    missing = []
    for items in data.values():
        for e in items:
            if e["safe_name"] not in tables["EMOTION_LABEL"]:
                missing.append(e["safe_name"])
    assert not missing, f"这些表情没翻译：{missing}"


def test_default_emotion_translated() -> None:
    """`neutral` 不在官方 157 条里，但它是**我们这边的默认表情**。

    后端 `EMOTIONS = ["neutral"]`，磁盘上就是 `Sprites/<服装>/neutral/`，
    素材页的表情下拉里一定有它 —— 官方表情库从 `angry` 开始，没有这一条，
    所以上面的「全量对齐官方」测不到它，得单独盯着。
    """
    tables = parse_label_tables()
    for key in ("neutral", "Neutral"):
        assert key in tables["EMOTION_LABEL"], f"默认表情 {key} 没有中文译文"
    assert tables["EMOTION_LABEL"]["neutral"]


def test_styles_fully_translated() -> None:
    """官方画风库的 358 条画风 + 13 个分组都要有中文。"""
    tables = parse_label_tables()
    data = _load(TEMPLATE / "character_styles.json")
    missing_group = [
        g["label"] for g in data["groups"]
        if g["label"] not in tables["STYLE_GROUP_LABEL"]
    ]
    assert not missing_group, f"这些画风分组没翻译：{missing_group}"
    missing = []
    for g in data["groups"]:
        for s in g["styles"]:
            if s["label"] not in tables["STYLE_LABEL"]:
                missing.append(s["label"])
    assert not missing, f"这些画风没翻译：{missing}"


def test_tags_fully_translated() -> None:
    """角色属性标签（270 条）全都要有中文。"""
    tables = parse_label_tables()
    data = _load(TEMPLATE / "character_presets_v2.json")
    missing_cat = [c for c in data["tags"] if c not in tables["TAG_CATEGORY_LABEL"]]
    assert not missing_cat, f"这些标签分类没翻译：{missing_cat}"
    missing_group: List[str] = []
    missing: List[str] = []
    for items in data["tags"].values():
        for item in items:
            if item["tag"] not in tables["TAG_LABEL"]:
                missing.append(item["tag"])
            grp = item.get("group")
            if grp and grp not in tables["TAG_GROUP_LABEL"]:
                missing_group.append(grp)
    assert not missing, f"这些属性标签没翻译：{missing}"
    assert not missing_group, f"这些标签分组没翻译：{missing_group}"


def test_pose_library_fully_translated() -> None:
    """官方姿势库（191 个姿势）的分类 / 名字 / 标签都要有中文。"""
    tables = parse_label_tables()
    items = poselib.list_presets(force=True)
    assert items, "官方姿势库是空的 —— 是不是路径不对？"

    missing_cat = [
        i["category"] for i in items
        if i["category"] and i["category"] not in tables["POSE_CATEGORY_LABEL"]
    ]
    assert not missing_cat, f"这些姿势分类没翻译：{sorted(set(missing_cat))}"

    missing_name = [
        i["name"] for i in items if i["name"] not in tables["POSE_NAME_LABEL"]
    ]
    assert not missing_name, f"这些姿势名没翻译：{sorted(set(missing_name))}"

    missing_tag: List[str] = []
    for i in items:
        for t in i.get("tags", []):
            if t not in tables["POSE_TAG_LABEL"]:
                missing_tag.append(t)
    assert not missing_tag, f"这些姿势标签没翻译：{sorted(set(missing_tag))}"


def test_translations_are_actually_chinese() -> None:
    """译文本身得是中文 —— 防止把英文原样抄进表里（等于没翻译）。"""
    tables = parse_label_tables()
    for name, table in tables.items():
        for key, val in table.items():
            assert val.strip(), f"{name}[{key!r}] 的译文是空的"
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in val)
            assert has_cjk, f"{name}[{key!r}] = {val!r} 没有中文"


def test_character_tags_fully_translated() -> None:
    """属性选择器用的第二份官方库 `character_tags.json` 也要全中文。

    这份库按 hair_color / hairstyles / races / details 分组，tag 是下划线写法。
    以前只校验 `character_presets_v2.json`，于是选择器里发色 / 发型 / 种族那一排
    悄悄全是英文（2026-11-08 用户直接看到的）。
    """
    tables = parse_label_tables()
    data = _load(TEMPLATE / "character_tags.json")
    missing: List[str] = []
    for items in (data.get("tags") or {}).values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("tag") not in tables["TAG_LABEL"]:
                missing.append(str(item.get("tag")))
    assert not missing, f"这些属性标签没翻译：{sorted(set(missing))}"


def test_backend_labels_json_in_sync() -> None:
    """`workflows/labels_zh.json` 必须跟 `labels.ts` 一致。

    后端拼中文提示词要靠这份 JSON（`tools/build_labels_zh.py` 导出）。
    改了 labels.ts 忘了重跑脚本，提示词里就会冒出英文 —— 这里直接拦住。
    同步办法：`python tools/build_labels_zh.py`
    """
    tables = parse_label_tables()
    exported = _load(ROOT / "workflows" / "labels_zh.json")
    for name in ("TAG_LABEL", "STYLE_LABEL", "STYLE_GROUP_LABEL", "BG_COLOR_LABEL"):
        assert exported.get(name) == tables.get(name), (
            f"{name} 与 labels.ts 不同步，跑 python tools/build_labels_zh.py")
