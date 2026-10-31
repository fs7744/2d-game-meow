"""角色中心的磁盘存储（workflows/characters.py，移植自 VNCCS utils.py）。

**这里最重要的不是"能存"，而是"存的跟官方是同一套"**：目录结构、
config 字段名、服装字段白名单、旧版本归档规则，全部照搬。这样官方工作流
产出的角色目录我们能直接读，反之亦然——测试要钉死的就是这份契约。
"""
from __future__ import annotations

import json
import os
import shutil
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


# ------------------------------------------------------------------ 出图


def test_prompt_is_official_structured():
    """官方造角色的 prompt 是**每个字段一行**，不是一句自然语言。

    风格 LoRA 是按这个格式调的（`character_creator_v2.py:1124`），
    格式一变风格就吃不准——这条用例钉住格式。
    """
    from workflows import charactergen as cg
    info = {"name": "Alice", "sex": "female", "age": 22, "race": "elf",
            "hair": "blonde hair", "eyes": "blue eyes", "background_color": "green",
            "framing": "full_body", "aesthetics": "masterpiece"}
    p = cg.character_prompt(info, "ghibli_miyazaki")
    for key in ("Draw an image of a character in the following style:",
                "Style reference:", "Style description:",
                "The image contains one character.", "Framing:", "Background:",
                "Gender:", "Age:", "Race:", "Facial features:", "Hair:", "Eyes:",
                "Skin color:", "Build:", "Clothes:", "Additional details:"):
        assert key in p, f"缺官方字段行：{key}"
    assert "22 years old" in p
    assert "solid green background" in p
    assert "head to toe" in p, "full_body 要有全身构图词"
    # 性别为女才出现胸围那一行（官方同样只在 female 时加）
    assert "Breast size:" in p
    male = cg.character_prompt({**info, "sex": "male"})
    assert "Breast size:" not in male


def test_prompt_transparent_background():
    from workflows import charactergen as cg
    p = cg.character_prompt({"background_color": "transparent"})
    assert cg.QI2_ALPHA_BACKGROUND_PROMPT in p


def test_resolution_uses_official_formula():
    """官方 `get_generation_resolution`：latent 9:16 × 步长 8，按总像素等比。"""
    from workflows import charactergen as cg
    # 1024 → sqrt(1024*1024 / (72*128)) ≈ 10.67 → 11 → 792×1408
    assert cg.generation_resolution(1024) == (792, 1408)
    w, h = cg.generation_resolution(768)
    assert w % 8 == 0 and h % 8 == 0
    assert abs((w / h) - (9 / 16)) < 0.01, "比例要锁在 9:16（官方 latent 宽高）"


def test_style_and_race_lookups_from_official_data():
    """风格与种族特征都从随项目发行的官方 json 里读，不能自己编。"""
    from workflows import charactergen as cg
    sp = cg.style_prompt_of("shonen_anime")
    assert sp.startswith("Style: Shonen Anime.")
    assert "Visual description:" in sp
    assert cg.style_prompt_of("不存在的风格") == ""
    feats = cg.race_features("elf")
    assert "Elf:" in feats and "pointed ears" in feats
    assert cg.race_features("") == ""


def test_publish_uses_official_naming_and_versions():
    """`sprite_pose_0001.png`（neutral）/ `sprite_<表情>_0001.png`，重生成旧图进 V1。"""
    import shutil
    from pathlib import Path
    if ch.character_exists("Pub"):
        ch.delete_character("Pub")
    ch.create_character("Pub")
    src = Path(paths.CHARACTERS_DIR, "_tmp_src.png")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"v1")
    try:
        r1 = ch.publish_image("Pub", str(src), "Naked", "neutral")
        assert Path(r1["path"]).name == "sprite_pose_0001.png", r1["file"]
        assert r1["index"] == 1
        # 封面：角色库卡片靠它认人
        assert Path(r1["cover"]).is_file()

        src.write_bytes(b"v2")
        r2 = ch.publish_image("Pub", str(src), "Naked", "neutral")
        assert Path(r2["path"]).name == "sprite_pose_0002.png"
        # 第二张不覆盖第一张
        assert Path(r1["path"]).read_bytes() == b"v1"

        r3 = ch.publish_image("Pub", str(src), "Naked", "angry")
        assert Path(r3["path"]).name == "sprite_angry_0001.png"
        assert Path(r3["path"]).parent.name == "angry"
    finally:
        src.unlink(missing_ok=True)
        ch.delete_character("Pub")


def test_publish_rejects_missing_source():
    if ch.character_exists("NoSrc"):
        ch.delete_character("NoSrc")
    ch.create_character("NoSrc")
    try:
        ch.publish_image("NoSrc", "/绝对不存在的图.png")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("源图不存在应该报错，不能建出一个空文件")
    ch.delete_character("NoSrc")


def test_publish_appends_without_archiving():
    """逐张发布是**追加**：官方目录里本来就是 sprite_pose_0001..N 一张张攒的。

    （`staged_image_batch` 的"整批替换 + 旧版进 V1"是给一次出一整批姿势帧用的，
    逐张发布用它就会把前面所有图都推进 V1。）
    """
    from pathlib import Path
    if ch.character_exists("Ver"):
        ch.delete_character("Ver")
    ch.create_character("Ver")
    src = Path(paths.CHARACTERS_DIR, "_tmp_src.png")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"one")
    try:
        a = ch.publish_image("Ver", str(src), "Naked", "neutral")
        src.write_bytes(b"two")
        b = ch.publish_image("Ver", str(src), "Naked", "neutral")
    finally:
        src.unlink(missing_ok=True)
    assert Path(a["path"]).read_bytes() == b"one", "先发的那张不能被后来的挤掉"
    assert Path(b["path"]).read_bytes() == b"two"
    assert Path(a["path"]).name == "sprite_pose_0001.png"
    assert Path(b["path"]).name == "sprite_pose_0002.png"
    assert not Path(ch.sprites_dir("Ver"), "V1").exists(), "逐张发布不该产生版本目录"
    ch.delete_character("Ver")


def test_staged_batch_versions_when_publishing_a_set():
    """一次出一整批时才走 staged：旧的整批进 V1（官方 `utils.py:252`）。"""
    from pathlib import Path
    d = Path(paths.CHARACTERS_DIR, "_batch_probe")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    Path(d, "sprite_pose_0001.png").write_bytes(b"old")
    with ch.staged_image_batch(str(d)) as stage:
        Path(stage, "sprite_pose_0001.png").write_bytes(b"new")
    assert Path(d, "sprite_pose_0001.png").read_bytes() == b"new"
    assert Path(d, "V1", "sprite_pose_0001.png").read_bytes() == b"old"
    shutil.rmtree(d)
