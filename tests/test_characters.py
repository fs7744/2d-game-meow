"""角色中心的磁盘存储（workflows/characters.py，移植自 VNCCS utils.py）。

**这里最重要的不是"能存"，而是"存的跟官方是同一套"**：目录结构、
config 字段名、服装字段白名单、旧版本归档规则，全部照搬。这样官方工作流
产出的角色目录我们能直接读，反之亦然——测试要钉死的就是这份契约。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from workflows import characters as ch
from workflows import paths


def _fresh(name: str = "Alice") -> str:
    """保证一个干净的角色（测试之间不互相污染）。"""
    if ch.character_exists(name):
        ch.delete_character(name)
    return ch.create_character(name)["character_info"]["name"]


def test_characters_dir_isolated():
    """测试期间必须写在 tests/_out 下的 assets 里，不能碰用户真实的角色库。"""
    out = Path(paths.TEST_OUT).resolve()
    assert out in Path(paths.CHARACTERS_DIR).resolve().parents


def test_create_makes_official_layout():
    """目录骨架 = 官方 `utils.py:443-487`：Sprites/Faces/Sheets + Naked 子集。"""
    _fresh()
    root = Path(ch.character_dir("Alice"))
    for main in ("Sprites", "Faces", "Sheets"):
        assert (root / main).is_dir(), f"缺主目录 {main}"
        assert (root / main / "Naked").is_dir(), f"缺基础集 {main}/Naked"
    assert (root / "Alice_config.json").is_file()


def test_config_fields_match_official():
    """config 顶层字段 = 官方 `character_creator_v2.py:2152-2156`。"""
    _fresh()
    cfg = ch.load_config("Alice")
    assert cfg["config_version"] == "2.0"
    assert cfg["folder_structure"]["main_directories"] == ["Sprites", "Faces", "Sheets"]
    assert cfg["folder_structure"]["emotions"] == ["neutral"]
    assert cfg["costumes"] == {}
    info = cfg["character_info"]
    # 官方字段一个都不能少（缺一个官方工作流就认不出这个角色）
    for key in ("name", "sex", "age", "race", "skin_color", "hair", "eyes", "face",
                "body", "breast_size", "additional_details", "nsfw", "aesthetics",
                "negative_prompt", "lora_prompt", "background_color", "framing",
                "style", "custom_style", "seed"):
        assert key in info, f"缺字段 {key}"


def test_create_conflict_and_bad_name():
    ch.create_character("Dup")
    try:
        ch.create_character("Dup")
    except ch.CharacterConflict:
        pass
    else:
        raise AssertionError("同名角色应该报冲突，不能静默覆盖")
    for bad in ("a/b", "../evil", "", "中文", "x" * 200):
        try:
            ch.create_character(bad)
        except ValueError:
            continue
        raise AssertionError(f"非法角色名没被挡住：{bad!r}")
    ch.delete_character("Dup")


def test_costume_crud_and_base_protection():
    _fresh()
    ch.save_costume_info("Alice", "Dress", {"top": "red dress", "evil": "drop me"})
    cfg = ch.load_config("Alice")
    # 白名单之外的字段必须被丢掉（不能往 config 里塞任意结构）
    assert cfg["costumes"]["Dress"]["top"] == "red dress"
    assert "evil" not in cfg["costumes"]["Dress"]
    # costume_id 是 ETag 乐观锁的身份
    assert cfg["costume_ids"]["Dress"]
    assert ch.list_costumes("Alice") == ["Naked", "Dress"]

    for base in ("Naked", "naked", "Original"):
        try:
            ch.delete_costume("Alice", base)
        except ValueError:
            continue
        raise AssertionError(f"基础集可以删，不该：{base}")

    ch.delete_costume("Alice", "Dress")
    assert ch.list_costumes("Alice") == ["Naked"]


def test_staged_publish_versions_old_images():
    """重生成时旧图搬进 V1（官方 `utils.py:252`）—— 用户能找回上一版。"""
    _fresh()
    d = ch.sprites_dir("Alice", "Naked", "neutral")
    os.makedirs(d, exist_ok=True)
    Path(d, "sprite_pose_0001.png").write_bytes(b"old")
    with ch.staged_image_batch(d) as stage:
        Path(stage, "sprite_pose_0001.png").write_bytes(b"new")
    assert Path(d, "sprite_pose_0001.png").read_bytes() == b"new"
    assert Path(d, "V1", "sprite_pose_0001.png").read_bytes() == b"old"


def test_rename_moves_config_and_updates_name():
    if ch.character_exists("RenameTo"):
        ch.delete_character("RenameTo")
    _fresh("RenameFrom")
    ch.rename_character("RenameFrom", "RenameTo")
    assert not ch.character_exists("RenameFrom")
    cfg = ch.load_config("RenameTo")
    assert cfg["character_info"]["name"] == "RenameTo"
    assert cfg["character_path"] == ch.character_dir("RenameTo")
    # 旧 config 文件名不能留在目录里（否则列表里会冒出两个同名角色）
    assert not Path(ch.character_dir("RenameTo"), "RenameFrom_config.json").exists()
    ch.delete_character("RenameTo")


def test_corrupted_config_reads_as_none():
    _fresh()
    p = Path(ch.config_path("Alice"))
    p.write_text("{ 半截 json", encoding="utf-8")
    assert ch.load_config("Alice") is None, "坏 config 要当作没有，不能把应用拖死"
    try:
        ch.load_config("Alice", strict=True)
    except OSError:
        pass
    else:
        raise AssertionError("strict 模式应该报错")


def test_summary_and_cover():
    _fresh()
    d = ch.sprites_dir("Alice", "Naked", "neutral")
    os.makedirs(d, exist_ok=True)
    Path(d, "sprite_pose_0001.png").write_bytes(b"png")
    s = ch.character_summary("Alice")
    assert s["name"] == "Alice"
    assert s["sprites"] == 1, "素材数量要能数出来（列表卡片要用）"
    # 没有 preview.png 时退回第一张 sprite，否则卡片全是空的
    assert s["cover"] == str(Path(d, "sprite_pose_0001.png"))


def test_safe_join_blocks_escape():
    root = ch.characters_root()
    for parts in (("..", "evil"), ("..",)):
        try:
            ch.safe_join_under(root, *parts)
        except ValueError:
            continue
        raise AssertionError(f"路径穿越没被挡住：{parts}")


def test_atomic_write_leaves_no_tmp():
    _fresh()
    ch.save_costume_info("Alice", "Dress", {"top": "x"})
    leftovers = list(Path(ch.character_dir("Alice")).glob("*.tmp"))
    assert not leftovers, f"原子写留下了临时文件：{leftovers}"


def test_config_is_valid_json_with_indent():
    """写出来的 config 要能被人手改（官方也是 indent=4）。"""
    _fresh()
    raw = Path(ch.config_path("Alice")).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["character_info"]["name"] == "Alice"
    assert "\n    " in raw, "缩进过的 JSON 才能手改"
