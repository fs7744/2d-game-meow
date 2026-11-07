"""LoRA 导入（格式检测 / 用户确认才转换 / 删除）与全局设置。

只测纯逻辑与文件落点，不加载任何模型、不碰 GPU。
"""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file

from workflows.engine import lora_format

DIFF_KEYS = {
    "transformer.transformer_blocks.0.attn.to_q.lora.down.weight": (2, 4),
    "transformer.transformer_blocks.0.attn.to_q.lora.up.weight": (4, 2),
}
COMFY_KEYS = {
    "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": (2, 4),
    "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": (4, 2),
}
OTHER_KEYS = {
    "unet.down_blocks.0.attn.to_q.lora_A.weight": (2, 4),
    "lora_te1.text_model.encoder.layers.0.lora_A.weight": (2, 4),
}


def _write(path, spec):
    save_file({k: torch.zeros(shape, dtype=torch.float16) for k, shape in spec.items()},
              str(path))
    return path


@pytest.fixture
def lora_env(tmp_path, monkeypatch):
    """把 LoRA 目录 / 暂存区 / 元数据库都指到临时目录（绝不写真实 models/）。"""
    from workflows import lorastore, metadb, paths

    monkeypatch.setattr(paths, "LORA_DIR", tmp_path / "lora")
    monkeypatch.setattr(paths, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(lorastore, "STAGE_DIR", tmp_path / "stage")
    monkeypatch.setattr(metadb, "DB_PATH", tmp_path / "meta.sqlite3")
    (tmp_path / "lora").mkdir()
    (tmp_path / "stage").mkdir()
    (tmp_path / "library").mkdir()
    return tmp_path


# ---------------- 格式判定（不需要文件） ----------------

def test_diffusers_format_is_accepted():
    info = lora_format(list(DIFF_KEYS))
    assert info["format"] == "diffusers"
    assert info["need_convert"] is False


def test_comfyui_format_needs_conversion():
    info = lora_format(list(COMFY_KEYS))
    assert info["format"] == "comfyui"
    assert info["need_convert"] is True
    assert info["convertible"] is True


def test_prefixless_comfy_variant_is_recognised():
    """第三种变体：没有 `diffusion_model.` 前缀，直接从 transformer_blocks. 开始。"""
    info = lora_format(["transformer_blocks.0.attn.to_q.lora_A.default.weight"])
    assert info["format"] == "comfyui"
    assert info["convertible"] is True


def test_other_architecture_is_not_convertible():
    info = lora_format(list(OTHER_KEYS))
    assert info["format"] == "unknown"
    assert info["convertible"] is False


# ---------------- 导入：必须用户确认才转换 ----------------

def test_comfyui_import_refused_without_conversion(lora_env):
    from workflows import lorastore

    src = _write(lora_env / "comfy.safetensors", COMFY_KEYS)
    token = lorastore.new_token()
    lorastore.stage_path(token).write_bytes(src.read_bytes())

    with pytest.raises(ValueError, match="需要转换"):
        lorastore.commit(token, convert=False, name="comfy_style")
    assert not list((lora_env / "lora").glob("*.safetensors")), \
        "用户没确认，文件不该落进 models/lora/"


def test_comfyui_import_after_confirmation_converts(lora_env):
    from workflows import lorastore

    src = _write(lora_env / "comfy.safetensors", COMFY_KEYS)
    token = lorastore.new_token()
    lorastore.stage_path(token).write_bytes(src.read_bytes())

    out = lorastore.commit(token, convert=True, name="comfy_style")
    assert out["converted"] is True
    assert out["item"]["format"] == "diffusers"

    dst = lora_env / "lora" / "comfy_style.safetensors"
    assert dst.is_file()
    keys = list(load_file(str(dst)))
    assert all(k.startswith("transformer.") for k in keys)
    assert all(".lora.down." in k or ".lora.up." in k for k in keys)
    # 暂存文件用完即清，不留垃圾
    assert not lorastore.stage_path(token).exists()


def test_diffusers_import_is_direct(lora_env):
    from workflows import lorastore

    src = _write(lora_env / "clean.safetensors", DIFF_KEYS)
    token = lorastore.new_token()
    lorastore.stage_path(token).write_bytes(src.read_bytes())

    out = lorastore.commit(token, convert=False, name="clean_style")
    assert out["converted"] is False
    assert out["item"]["tensors"] == len(DIFF_KEYS)
    assert (lora_env / "lora" / "clean_style.safetensors").is_file()


def test_other_architecture_import_fails_politely(lora_env):
    from workflows import lorastore

    src = _write(lora_env / "sdxl.safetensors", OTHER_KEYS)
    token = lorastore.new_token()
    lorastore.stage_path(token).write_bytes(src.read_bytes())

    with pytest.raises(ValueError, match="不是本底模|没找到"):
        lorastore.commit(token, convert=True, name="sdxl_style")


def test_file_name_is_sanitised(lora_env):
    """文件名要挡住路径穿越与非法字符（peft 的 adapter 名不能带点）。"""
    from workflows import lorastore

    assert lorastore.safe_name("../../etc/passwd") == "passwd"
    assert "/" not in lorastore.safe_name("..\\..\\x")
    # 版本号里的点不能被当成扩展名吃掉
    assert lorastore.safe_name("VNCCS_QI2_ClothesCoreV2.6.safetensors") == \
        "VNCCS_QI2_ClothesCoreV2_6"
    assert lorastore.safe_name("   ") == "lora"


def test_delete_removes_file(lora_env):
    from workflows import lorastore

    src = _write(lora_env / "x.safetensors", DIFF_KEYS)
    token = lorastore.new_token()
    lorastore.stage_path(token).write_bytes(src.read_bytes())
    lorastore.commit(token, convert=False, name="gone")
    assert lorastore.delete("gone") == "gone"
    assert not (lora_env / "lora" / "gone.safetensors").exists()
    with pytest.raises(FileNotFoundError):
        lorastore.delete("gone")


# ---------------- 全局设置 ----------------

def test_settings_roundtrip(lora_env):
    from workflows import metadb

    assert metadb.get_settings()["matte_backend"] == "auto"
    s = metadb.set_settings({
        "matte_backend": "u2net",
        "loras": [{"name": "a", "scale": 0.7}, {"name": "", "scale": 1}],
        "fast_mode": True,
    })
    assert s["matte_backend"] == "u2net"
    assert s["fast_mode"] is True
    assert [x["name"] for x in s["loras"]] == ["a"], "空名字要丢掉"
    assert s["loras"][0]["scale"] == 0.7
    # 重新读一遍：真的落盘了
    assert metadb.get_settings()["matte_backend"] == "u2net"


def test_settings_whitelist_and_bad_values(lora_env):
    from workflows import metadb

    metadb.set_settings({"bogus_key": 1, "matte_backend": "不存在的后端"})
    s = metadb.get_settings()
    assert "bogus_key" not in s
    assert s["matte_backend"] == "auto", "未知抠图后端要归一成 auto"


def test_engine_restore_drops_missing_files(lora_env):
    """设置里记着但磁盘上已经没有的 LoRA 要丢掉，否则每次生成都会抛 FileNotFound。"""
    from workflows.engine import Engine

    eng = Engine(lora_env)
    kept = eng.restore_loras([{"name": "not_there", "scale": 1.0}])
    assert kept == []
    assert eng.active_loras() == []


# ---------------- 目录扫描 / 就地转换 ----------------

def test_scan_lists_every_file(lora_env):
    """目录里格式不对的文件也要报出来，不能静默过滤（用户会以为丢了）。"""
    from workflows.engine import Engine

    _write(lora_env / "lora" / "good.safetensors", DIFF_KEYS)
    _write(lora_env / "lora" / "comfy.safetensors", COMFY_KEYS)
    _write(lora_env / "lora" / "other.safetensors", OTHER_KEYS)

    eng = Engine(lora_env)
    scan = eng.scan_loras()
    assert {x["name"]: x["format"] for x in scan} == {
        "good": "diffusers", "comfy": "comfyui", "other": "unknown"}
    assert [x["name"] for x in eng.available_loras()] == ["good"], \
        "可挂列表只能有 diffusers 格式的"


def test_convert_in_place_keeps_name_and_backs_up(lora_env):
    """把已经在目录里的 ComfyUI LoRA 就地转换：原名不变，原文件先备份。"""
    from workflows import lorastore, paths
    from workflows.engine import Engine, _lora_keys, lora_format

    target = _write(lora_env / "lora" / "comfy.safetensors", COMFY_KEYS)
    out = lorastore.convert_in_place("comfy")

    assert out["converted"] is True
    assert out["item"]["name"] == "comfy", "文件名不该被改掉"
    assert out["item"]["format"] == "diffusers"
    assert lora_format(_lora_keys(target))["format"] == "diffusers"
    backup = paths.LIBRARY_DIR / "lora_backup" / "comfy.safetensors"
    assert backup.is_file(), "转换前必须备份原文件"
    # 转换后引擎就能挂它了
    assert "comfy" in [x["name"] for x in Engine(lora_env).available_loras()]


def test_convert_in_place_is_noop_for_diffusers(lora_env):
    from workflows import lorastore

    _write(lora_env / "lora" / "already.safetensors", DIFF_KEYS)
    out = lorastore.convert_in_place("already")
    assert out["converted"] is False


def test_convert_in_place_refuses_other_architectures(lora_env):
    from workflows import lorastore

    _write(lora_env / "lora" / "sdxl.safetensors", OTHER_KEYS)
    with pytest.raises(ValueError, match="没找到|不是本底模"):
        lorastore.convert_in_place("sdxl")
    # 原文件一个字节都不该动
    assert (lora_env / "lora" / "sdxl.safetensors").is_file()
