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

import pytest
from PIL import Image

from workflows import characters as ch
from workflows import paths
from workflows import charactergen as cg
from workflows import qwen_text


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
    # **先清掉残留**：这个用例以前不清，上一次跑到一半中断就会留下 Dup 目录，
    # 下一次直接在第一句就撞"角色已存在"（看起来像功能坏了，其实是测试不干净）。
    for leftover in ("Dup", "Alice"):
        if ch.character_exists(leftover):
            ch.delete_character(leftover)
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
            "framing": "full_body", "aesthetics": "masterpiece",
            # 空字段现在不占行，所以每个字段都要给值才能钉住所有官方行
            "skin_color": "fair skin", "face": "freckles", "body": "slim build",
            "breast_size": "medium breasts", "additional_details": "long hair",
            "lora_prompt": "vnccs_style",
            # 这条钉的是**官方英文模板**的字段行，所以显式要英文
            "prompt_language": "en"}
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


def test_framing_defaults_to_full_body():
    """默认取景必须是**全身**：官方默认 cowboy_shot（半身），切精灵图会缺脚。"""
    from workflows import charactergen as cg

    en = {"prompt_language": "en"}
    framing = cg.qi2_fields({"sex": "female", **en})["framing"]
    assert "head to toe" in framing, "没写 framing 时要按全身出图"
    assert "cowboy" not in framing.lower()
    # 中文（现在的默认）也是全身
    assert "从头到脚" in cg.qi2_fields({"sex": "female"})["framing"]
    # 显式选了就照办（半身 / 头像都留着给用户）
    assert "upper-thigh" in cg.qi2_fields({"framing": "cowboy_shot", **en})["framing"]
    assert "head and shoulders" in cg.qi2_fields(
        {"image_type": "portrait", **en})["framing"]


def test_chinese_prompt_mode():
    """中文模式：结构词中文，属性值原样（标签库就是英文 trait）。"""
    from workflows import charactergen as cg

    info = {"sex": "female", "age": 22, "hair": "blonde hair",
            "eyes": "blue eyes", "background_color": "green",
            "prompt_language": "zh"}
    p = cg.character_prompt(info, "ghibli_miyazaki")
    for key in ("请画一张角色立绘", "取景：", "背景：纯绿色背景",
                "性别：女", "年龄：22 岁", "头发：金发", "眼睛：蓝眼"):
        assert key in p, f"中文模式缺：{key}"
    assert "从头到脚" in p, "中文模式同样要全身构图词"
    assert "Draw an image" not in p and "Framing:" not in p
    # 语言跟角色走：显式选 en 时回到官方英文模板
    assert "Draw an image" in cg.character_prompt({**info, "prompt_language": "en"},
                                                  "ghibli_miyazaki")


def test_pe_runs_in_chinese_mode(monkeypatch):
    """中文模式**照样跑 PE**，只是要求它输出中文（不再因为「只出英文」而跳过）。"""
    from workflows import charactergen as cg, qwen_text

    # 1) 中文 system prompt：把「一律英文」换成「一律中文」，其余结构照官方
    zh = qwen_text.system_prompt_of("t2i", "zh")
    assert "简体中文" in zh
    assert "always in English" not in zh
    assert "Output format" in zh and "rewritten_prompt" in zh, "JSON 输出格式不能丢"
    en = qwen_text.system_prompt_of("t2i", "en")
    assert "always in English" in en, "英文模式仍是官方原文"

    # 2) enhance 把 lang 透到 system prompt
    seen = {}
    monkeypatch.setattr(qwen_text, "generate_text",
                        lambda pipe, req, sp="", **kw: (seen.update(sp=sp) or "x"))
    monkeypatch.setattr(qwen_text, "system_prompt_of",
                        lambda kind="t2i", lang="en": seen.update(lang=lang) or "SYS")
    qwen_text.enhance(None, "req", kind="t2i", lang="zh")
    assert seen["lang"] == "zh"

    # 3) 中文角色改写成功 → applied=1，正文是中文
    monkeypatch.setattr(cg.qwen_text, "enhance",
                        lambda pipe, req, kind="t2i", lang="en", **kw:
                        "一位银发红瞳的精灵女剑士站在纯绿色背景前，全身入画。")
    r = cg.enhance_character_prompt(_FakePipe(), {"prompt_language": "zh", "hair": "blonde"},
                                    "", enhance=True)
    assert r["applied"] == "1"
    assert "精灵女剑士" in r["prompt"]
    assert cg.pe_eligible({"prompt_language": "zh"}, True) is True, "中文模式不再跳过 PE"


def test_breast_size_reaches_prompt():
    """胸围选了就得进提示词。

    官方 `_qi2_character_fields` 的字段清单**漏了 breast_size**，但结构化模板里
    又打印 `Breast size:` —— 于是那一行永远是空的（官方 bug）。我们补上字段，
    并且空属性不再占行，避免界面上看到「胸围：」后面什么都没有。
    """
    from workflows import charactergen as cg

    zh = cg.character_prompt({"sex": "female", "breast_size": "medium breasts"})
    assert "胸围：中等胸型" in zh, zh
    en = cg.character_prompt({"sex": "female", "breast_size": "medium breasts",
                              "prompt_language": "en"})
    assert "Breast size: medium breasts" in en
    # 男性不出这一行（官方同样只在 female 时加）
    assert "胸围" not in cg.character_prompt({"sex": "male", "breast_size": "medium breasts"})
    # 没填就不占行
    assert "胸围" not in cg.character_prompt({"sex": "female"})


def test_prompt_transparent_background():
    from workflows import charactergen as cg
    assert "透明背景" in cg.character_prompt({"background_color": "transparent"})
    p_en = cg.character_prompt({"background_color": "transparent",
                                "prompt_language": "en"})
    assert cg.QI2_ALPHA_BACKGROUND_PROMPT in p_en


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


# ---------------------------------------------------------------- Prompt Enhancement


class _FakePipe:
    """占位：PE 只需要一个有 text_encoder / tokenizer 的对象，测试里用假货。"""


def _pe(monkeypatch, out: str):
    monkeypatch.setattr(cg.qwen_text, "enhance",
                        lambda pipe, req, kind="t2i", **kw: out)


def _tiny_text_model():
    """CPU 上的迷你「文本编码器」：带 embedding（有 weight）+ 一个空参数模块。"""
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.linear = torch.nn.Linear(8, 8)
            self.register_buffer("scale", torch.ones(4))

        def get_input_embeddings(self):
            return self.embed

    return Tiny().eval()


def test_pe_device_helpers():
    """自检用的小工具：设备清点 + 体积估算（meta 骨架要跳过）。"""
    import torch

    m = _tiny_text_model()
    assert qwen_text._off_device(m, "cpu") == []
    assert qwen_text._embed_device(m) == torch.device("cpu")
    assert qwen_text._weight_gb(m) > 0
    # fp8 加载中间会留下 meta 张量，那不是「混设备」，不能误报
    m._buffers["scale"] = torch.ones(4, device="meta")
    assert qwen_text._off_device(m, "cpu") == []


def test_pe_ondevice_unhooks_and_restores():
    """`_OnDevice` 进出要把 accelerate 的 offload 钩子摘掉再挂回来。

    这是 2026-11-08 那个 `wrapper_CUDA__index_select` 的根因：钩子只搬模块
    不搬输入。测试不依赖 GPU，用执行设备=cpu 的 AlignDevicesHook 验证生命周期。
    """
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    m = _tiny_text_model()
    add_hook_to_module(m, AlignDevicesHook(execution_device="cpu", io_same_device=True))
    assert getattr(m, "_hf_hook", None) is not None
    hook = m._hf_hook

    with qwen_text._OnDevice(m, device="cpu") as inner:
        assert inner is m
        assert getattr(m, "_hf_hook", None) is None, "generate 期间不该还有钩子搬来搬去"
        assert qwen_text._off_device(m, "cpu") == [], "搬完必须整块在同设备上"
    assert getattr(m, "_hf_hook", None) is hook, "出来要把原来的钩子挂回去"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="需要 GPU")
def test_pe_ondevice_moves_everything_to_cuda():
    """fp8 的 w_fp8 是 buffer：只搬参数不搬 buffer 就是混设备的源头。"""
    import torch

    m = _tiny_text_model()
    with qwen_text._OnDevice(m, device="cuda"):
        assert qwen_text._off_device(m, "cuda") == []
        assert all(str(t.device).startswith("cuda")
                   for t in list(m.parameters()) + list(m.buffers()))
    # 出来后回到原来的 cpu，别把显存留着
    assert all(str(t.device) == "cpu"
               for t in list(m.parameters()) + list(m.buffers()))


def test_pe_returns_raw_when_disabled():
    """不启用 PE 就是结构化原文，applied=0（前端要据此显示"未扩写"）。"""
    r = cg.enhance_character_prompt(None, {"name": "A"}, "", enhance=False)
    assert r["applied"] == "0"
    assert r["prompt"] == r["raw"]


def test_pe_rewrites_then_appends_style_and_lora(monkeypatch):
    """官方 `character_creator_v2.py:1271-1317` 的顺序：

    PE → 补回透明底 → 追加风格参考 → 追加 LoRA 触发词。
    **风格参考是在 PE 之后才追加的**（:1315），不是只在结构化 prompt 里。
    """
    _pe(monkeypatch, "A lone elven girl stands in a sunlit clearing.")
    info = {"name": "Alice", "background_color": "transparent",
            "lora_prompt": "vnccs_style", "prompt_language": "en"}
    r = cg.enhance_character_prompt(_FakePipe(), info, "ghibli_miyazaki")
    assert r["applied"] == "1"
    p = r["prompt"]
    assert p.startswith("A lone elven girl")
    # 透明底：PE 改写没了要补回来（:1308-1312）
    assert cg.QI2_ALPHA_BACKGROUND_PROMPT in p
    # 风格参考在 PE 之后原样追加
    assert cg.QI2_STYLE_REFERENCE_HEADING in p
    assert "Hayao Miyazaki" in p
    # LoRA 触发词放最后，且只出现一次（送进 PE 的那份要先切掉它，:1285）
    assert p.endswith("LoRA triggers: vnccs_style")
    assert p.count("LoRA triggers") == 1


def test_pe_falls_back_when_rewriting_fails(monkeypatch):
    """PE 是增强不是依赖：改写失败（空 / 带未闭合 think / 空 JSON / 代码块）就退回原文。"""
    for bad in ("", "   ", "<think>想一下</think>", "<think>没闭合",
                "```json\n{}\n```", '{"foo": 1}'):
        _pe(monkeypatch, bad)
        r = cg.enhance_character_prompt(_FakePipe(), {"name": "A"}, "")
        assert r["applied"] == "0", f"这种输出不该算改写成功：{bad!r}"
        assert r["prompt"] == r["raw"]


def test_pe_parses_official_json_output(monkeypatch):
    """官方 t2i 提示词要求回 JSON —— 必须解析，不能见到 `{` 就判失败。

    以前这里直接 reject，于是每次 PE 都白跑（没有报错，界面只显示「未扩写」）。
    官方 `character_creator_v2.py:1015 _qi2_json_result` 就是干这个的。
    """
    _pe(monkeypatch, '{"rewritten_prompt": "A lone elf stands in a clearing.", "wh_ratio": "3:2"}')
    r = cg.enhance_character_prompt(_FakePipe(), {"name": "A"}, "")
    assert r["applied"] == "1"
    assert r["prompt"] == "A lone elf stands in a clearing."

    # think 头 + 合法 JSON：只取 JSON
    _pe(monkeypatch, 'thinking...\n</think>\n{"rewritten_prompt": "Second version.", "wh_ratio": "1:1"}')
    r = cg.enhance_character_prompt(_FakePipe(), {"name": "A"}, "")
    assert r["prompt"] == "Second version."

    # 没有 JSON 的纯段落照样能用（老提示词 / 模型不听话）
    _pe(monkeypatch, "A plain prose rewrite, no braces.")
    r = cg.enhance_character_prompt(_FakePipe(), {"name": "A"}, "")
    assert r["prompt"] == "A plain prose rewrite, no braces."


def test_pe_request_excludes_lora_line(monkeypatch):
    """送进 PE 的 request 要切掉 `LoRA triggers:` 那一行（官方 :1285）。"""
    seen = {}

    def fake(pipe, req, kind="t2i", **kw):
        seen["req"] = req
        return "rewritten text here"
    monkeypatch.setattr(cg.qwen_text, "enhance", fake)
    info = {"lora_prompt": "secret_trigger"}
    cg.enhance_character_prompt(_FakePipe(), info, "")
    assert "secret_trigger" not in seen["req"], "LoRA 触发词不该送进 PE"


def test_field_expansion_excludes_literal_fields(monkeypatch):
    """字段扩写时 aesthetics / lora_prompt 不参与（官方 QI2_LITERAL_FIELDS，:262）。"""
    seen = {}
    monkeypatch.setattr(cg.qwen_text, "generate_text",
                        lambda pipe, req, sp="", **kw: (seen.update(req=req) or "expanded"))
    out = cg.expand_character_fields(_FakePipe(),
                                     {"aesthetics": "masterpiece", "lora_prompt": "t", "hair": "blonde"})
    assert out == "expanded"
    import json as _json
    payload = seen["req"].split("character_fields:\n", 1)[1]
    fields = _json.loads(payload)
    assert "aesthetics" not in fields and "lora_prompt" not in fields
    assert fields["hair"] == "blonde"


def test_pe_defaults_are_official():
    """采样参数照抄官方 `character_creator_v2.py:159-174`。"""
    d = cg.qwen_text.QI2_TEXT_DEFAULTS
    assert d["max_length"] == 2048
    assert d["temperature"] == 0.3 and d["top_k"] == 64 and d["top_p"] == 0.95
    assert d["min_p"] == 0.05 and d["repetition_penalty"] == 1.05


def test_official_system_prompts_are_bundled():
    """官方那三份提示词模板必须随项目发行（不能读不到就自己编）。"""
    for kind in ("t2i", "fields", "edit"):
        assert cg.qwen_text.system_prompt_of(kind), f"缺官方模板：{kind}"


def test_race_features_matches_tag_picker_output():
    """标签选择器写进字段的值必须能被后端认出来。

    选择器 toggle 写的是 `tag` 的**空格形式**（官方 `presetSelection.toggle`），
    多个 trait 用 `", "` 拼起来。后端 `race_features` 要能照官方那样逐个查出
    物种特征——官方 UI 上写着 "Combine species for hybrids"（允许多选混血），
    所以这儿必须是**多值的**。
    """
    from workflows import charactergen as cg
    # 选择器实际会产生的两种写法
    one = cg.race_features("elf")
    assert "Elf:" in one and "pointed ears" in one
    # 多选（混血）—— 官方明确支持
    two = cg.race_features("elf, demon")
    assert "Elf:" in two and "Demon:" in two
    # label / synonyms 别名也要认（官方给 "bunny girl" 之类留了同义词表）
    # 注意别写臆测的 tag —— 官方库里没有 catgirl 这种说法，只有 feline beastfolk
    for alias in ("elf", "dark elf", "feline beastfolk", "bunny girl"):
        assert cg.race_features(alias), f"认不出 {alias}"
    # 带权重的写法（用户在字段里随手加了 (...:1.2) 也能用）
    assert "Elf:" in cg.race_features("(elf:1.3)")
    # 认不出就别瞎编，返回空由上层决定要不要写这一行
    assert cg.race_features("zzz_not_a_race") == ""


def test_tag_picker_values_round_trip_through_prompt():
    """端到端：选择器写的值 → 拼进 prompt， species 特征与 eye/hair 字段都在。"""
    from workflows import charactergen as cg
    info = {"name": "A", "sex": "female", "age": 18, "race": "elf",
            "hair": "bob haircut, black hair", "eyes": "blue eyes, green eyes"}
    p = cg.character_prompt({**info, "prompt_language": "en"}, "")
    assert "Typical species features" in p, "选择器写的 elf 没被认出来"
    assert "Hair: bob haircut, black hair" in p
    assert "Eyes: blue eyes, green eyes" in p
    # 中文（默认）：属性值也走译表；没收录的（bob haircut）原样保留
    zh = cg.character_prompt(info, "")
    assert "头发：波波头, 黑发" in zh
    assert "眼睛：蓝眼, 绿眼" in zh


# ---------------------------------------------------------------- 名字校验（中文）
#
# 以前用官方那条 `[A-Za-z0-9 _-]` 白名单，中文角色名直接被拒。
# 而这个项目里中文命名才是最自然的（整套 prompt 都是中文），
# 白名单只是在给用户添麻烦。改成黑名单：只挡真正会出问题的。


def test_chinese_names_are_accepted():
    """中文 / 日文 / 带空格的名字都要能过。"""
    for name in ("白猫骑士", "アリス", "猫 骑士", "红魔女-2", "角色_01"):
        assert ch.ensure_safe_name(name) == name


def test_name_rejects_path_traversal():
    """路径分隔符与 `..` 必须挡住 —— 这个白名单换成黑名单后最容易漏掉。"""
    for bad in ("a/b", "a\\b", "..", "../etc", "x/../y"):
        with pytest.raises(ValueError):
            ch.ensure_safe_name(bad)


def test_name_rejects_windows_illegal_chars():
    for bad in ('a<b', 'a>b', 'a:b', 'a"b', 'a|b', 'a?b', "a*b"):
        with pytest.raises(ValueError):
            ch.ensure_safe_name(bad)


def test_name_rejects_invisible_and_controls():
    """零宽空格最阴：两个名字看起来一样，实际是两个目录。"""
    for bad in ("a\x00b", "a\u200bb", "a\nb", "a\tb"):
        with pytest.raises(ValueError):
            ch.ensure_safe_name(bad)


def test_name_rejects_trailing_dot_and_reserved():
    """Windows 会静默吃掉结尾的点；保留设备名建不出目录。"""
    for bad in ("x.", "CON", "con.txt", "nul", "COM1", "lpt9.log"):
        with pytest.raises(ValueError):
            ch.ensure_safe_name(bad)


def test_name_length_limit():
    assert ch.ensure_safe_name("角" * 120) == "角" * 120
    with pytest.raises(ValueError):
        ch.ensure_safe_name("角" * 121)


def test_chinese_character_end_to_end():
    """中文角色名要能真的建目录、存图、删掉（不是只过校验）。"""
    name = "白猫骑士"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    assert ch.character_exists(name)
    d = Path(ch.sprites_dir(name, "护士服", "中性"))
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(d / "sprite_pose_0001.png")
    assert Path(ch.sprites_dir(name, "护士服", "中性")).is_dir()
    assert "护士服" in ch.list_costumes(name)
    assert len(ch.list_sprite_images(name, "护士服", "中性")) == 1
    ch.delete_character(name)
    assert not ch.character_exists(name)


def test_chinese_name_survives_api(client):
    """接口层也要能用中文名（URL 编码往返不能把名字弄坏）。"""
    from urllib.parse import quote

    name = "猫娘 小玉"
    if ch.character_exists(name):
        ch.delete_character(name)
    r = client.post("/characters", json={"name": name})
    assert r.status_code == 200, r.text
    got = client.get(f"/characters/{quote(name)}")
    assert got.status_code == 200, got.text
    assert ch.character_exists(name)
    client.delete(f"/characters/{quote(name)}?remove_files=1")


def test_portraits_are_separate_from_action_frames():
    """立绘放 `Portraits/`，**不进** `Sprites/`。

    这是 2026-11-08 用户点出来的：立绘是「长什么样」的定妆图，动作帧是「怎么动」
    的素材。以前立绘被当第 1 帧塞进 Sprites/neutral，于是试玩台把一张站桩图
    当动画播、图集导出里也混着它。
    """
    import shutil
    from pathlib import Path
    if ch.character_exists("Port"):
        ch.delete_character("Port")
    ch.create_character("Port")
    src = Path(paths.CHARACTERS_DIR, "_tmp_portrait.png")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"p1")
    try:
        p1 = ch.publish_portrait("Port", str(src), "Naked", "neutral")
        src.write_bytes(b"p2")
        p2 = ch.publish_portrait("Port", str(src), "Knight", "neutral")
        assert "Portraits" in Path(p1["path"]).parts
        assert Path(p1["path"]).read_bytes() == b"p1", "立绘是追加，不覆盖"
        assert Path(p2["path"]).read_bytes() == b"p2"
        # 动作帧目录里一张都不该有
        assert ch.list_sprite_images("Port", "Naked", "neutral") == []
        assert ch.list_pose_preview_files("Port") == []
        # 全部立绘都能列出来（新的在前）
        items = ch.list_portraits("Port")
        assert len(items) == 2, items
        assert items[0]["mtime"] >= items[1]["mtime"]
        assert {i["costume"] for i in items} == {"Naked", "Knight"}
    finally:
        shutil.rmtree(Path(paths.CHARACTERS_DIR, "_tmp_portrait.png").parent
                      / "Port", ignore_errors=True)
        src.unlink(missing_ok=True)
        if ch.character_exists("Port"):
            ch.delete_character("Port")
