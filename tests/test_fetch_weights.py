"""`tools/fetch_vnccs_weights.py` 的离线测试。

**不联网**（真下载要 4.4GB，且沙箱里 HuggingFace 是 502）。测的是那些
"错了就会很难排查"的东西：

1. **清单跟官方一致** —— 仓库、revision、文件名、落点。这些照抄官方代码，
   抄错一个字母就是"下下来但用不了"。
2. **校验器能挡住坏文件** —— 残文件/错格式被放行是最坏的情况。
3. **转换环节真的能把 ComfyUI 格式转成我们能用的**。
4. `--dry-run` 真的不写磁盘。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# 脚本在 tools/ 下（不是包），按文件路径加载
_spec = importlib.util.spec_from_file_location(
    "fetch_vnccs_weights", ROOT / "tools" / "fetch_vnccs_weights.py")
fw = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(fw)


# ------------------------------------------------------------------ 清单


def test_repo_and_revision_are_official():
    """官方 `nodes/qwen_vl.py:71,74`。

    revision 是**精确的 commit**（不是 main）—— 官方锁死它，因为 gguf 仓库
    随时会重新导出，同名文件的 chat template 可能变。
    """
    assert fw.LLM_REPO == "unsloth/Qwen3.5-4B-GGUF"
    assert fw.LLM_REVISION == "e966ccab6d3f3c91e94d858b4a01c921a5d7ef53"


def test_lora_filenames_match_official_manifest():
    """官方 `control_center.json` 里那两个 QI2 LoRA 的 `hf_path`。

    注意名字**不是** "VNCCS_Overhaul_QI2" 这种猜出来的写法 ——
    真实文件名带版本号（AnimeOverhaulV2 / ClothesCoreV2.6）。
    """
    items = {i.key: i for i in fw.build_items()}
    assert items["overhaul"].remote == (
        "models/loras/QI2.1/VNCCS/VNCCS_QI2_AnimeOverhaulV2.safetensors")
    assert items["clothes"].remote == (
        "models/loras/QI2.1/VNCCS/VNCCS_QI2_ClothesCoreV2.6.safetensors")
    for key in ("overhaul", "clothes"):
        assert items[key].repo == "MIUProject/VNCCS_v3.0"
        # **必须转格式**：官方是 ComfyUI 格式，我们的引擎只认 diffusers 原生
        assert items[key].needs_convert is True


def test_mmproj_uses_the_name_that_exists_in_the_repo():
    """**这条抓过一个真 bug**。

    官方 `qwen_vl.py:68` 写的是 `mmproj-Qwen3.5-4B-F16.gguf`，但仓库里**没有
    这个文件**（实际叫 `mmproj-F16.gguf`）—— 照官方那个名字下会 404。
    官方自己也不是只认那个名字：`_find_qwen_vl_mmproj`（:171-187）先找短名
    `mmproj-F16.gguf` / `-BF16` / `-F32`，带模型名的那两个只是兜底。
    """
    items = {i.key: i for i in fw.build_items()}
    assert items["mmproj"].remote == "mmproj-F16.gguf"
    assert items["mmproj"].dest.name == "mmproj-F16.gguf"


def test_yolo_comes_from_the_comfyui_bbox_repo():
    """官方只给了 `bbox/face_yolov8m.pt` 这个路径（character_generator.py:845），
    **没给下载地址**（它假定用户装了 Impact Pack）。仓库是用户提供的，
    已联网核对文件存在（49.6 MB）。"""
    items = {i.key: i for i in fw.build_items()}
    assert items["yolo"].repo == "Tenofas/ComfyUI"
    assert items["yolo"].remote == "ultralytics/bbox/face_yolov8m.pt"
    assert items["yolo"].dest.name == "face_yolov8m.pt"
    assert items["yolo"].dest.parent.name == "bbox"


def test_destinations_are_under_models(monkeypatch):
    """落点必须集中在 models 下（路径只有一处定义：workflows/paths.py）。"""
    from workflows import paths
    items = fw.build_items()
    for it in items:
        assert str(it.dest).startswith(str(paths.MODELS_DIR)), it.key


# ------------------------------------------------------------------ 校验


def test_gguf_validation_is_official(tmp_path):
    """官方 `_validate_gguf_file`（`qwen_vl.py:141`）：存在 + >1MB + magic。"""
    good = tmp_path / "a.gguf"
    good.write_bytes(b"GGUF" + b"x" * (2 << 20))
    fw.validate_gguf(good)                       # 不该抛

    missing = tmp_path / "nope.gguf"
    with pytest.raises(FileNotFoundError):
        fw.validate_gguf(missing)

    # 下到一半的残文件：小于 1MB
    tiny = tmp_path / "tiny.gguf"
    tiny.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="太小"):
        fw.validate_gguf(tiny)

    # 拿 HTML 错误页当权重的经典情况
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"<htm" + b"x" * (2 << 20))
    with pytest.raises(ValueError, match="不是 GGUF"):
        fw.validate_gguf(bad)


def test_lora_validation_rejects_comfyui_format(tmp_path):
    """ComfyUI 格式必须被挡下来 —— 挂上去**静默不生效**比报错更坏。"""
    import torch
    from safetensors.torch import save_file
    comfy = tmp_path / "comfy.safetensors"
    save_file({"diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight":
               torch.zeros(4, 8)}, str(comfy), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="ComfyUI"):
        fw.validate_lora(comfy)

    native = tmp_path / "native.safetensors"
    save_file({"transformer.transformer_blocks.0.attn.to_k.lora.down.weight":
               torch.zeros(4, 8)}, str(native), metadata={"format": "pt"})
    fw.validate_lora(native)                     # 不该抛


def test_torch_weight_validation(tmp_path):
    """YOLO 的 `.pt` 是 zip 包。**不 torch.load**（没装 ultralytics 时会炸），
    只看 zip magic 就够判断"下完整了没有"。"""
    good = tmp_path / "face_yolov8m.pt"
    good.write_bytes(b"PK\x03\x04" + b"x" * (2 << 20))
    fw.validate_torch(good)

    with pytest.raises(FileNotFoundError):
        fw.validate_torch(tmp_path / "nope.pt")

    tiny = tmp_path / "tiny.pt"
    tiny.write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError, match="太小"):
        fw.validate_torch(tiny)

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"<htm" + b"x" * (2 << 20))
    with pytest.raises(ValueError, match="不是 torch"):
        fw.validate_torch(bad)


def test_convert_turns_comfyui_into_usable(tmp_path, monkeypatch):
    """端到端：合成一个 ComfyUI 格式的 LoRA → 跑转换 → 校验通过。

    这是整条链路上最容易"看起来成功、实际没用"的一环，所以真跑一遍。
    """
    import torch
    from safetensors.torch import save_file
    from workflows import paths

    monkeypatch.setattr(paths, "LORA_DIR", tmp_path)
    src = tmp_path / "in.safetensors"
    save_file({
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight":
            torch.zeros(8, 16),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight":
            torch.zeros(16, 8),
    }, str(src), metadata={"format": "pt"})

    out = tmp_path / "out.safetensors"
    code = subprocess.run(                                    # noqa: S603
        [sys.executable, str(ROOT / "tools" / "convert_lora.py"), str(src), str(out)],
        cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert code.returncode == 0, code.stdout + code.stderr
    fw.validate_lora(out)                        # 转完必须能过我们自己的校验


# ------------------------------------------------------------------ 行为


def test_dry_run_touches_nothing(monkeypatch, tmp_path):
    """`--dry-run` 必须真的不写磁盘 —— 它是给人先看一眼要下多大的。"""
    from workflows import paths
    for attr in ("LORA_DIR", "LLM_DIR", "BBOX_DIR"):
        monkeypatch.setattr(paths, attr, tmp_path / attr.lower())
    code = subprocess.run(                                    # noqa: S603
        [sys.executable, str(ROOT / "tools" / "fetch_vnccs_weights.py"),
         "--dry-run", "--force"],
        cwd=str(ROOT), capture_output=True, text=True, check=False, timeout=180)
    assert code.returncode == 0, code.stdout + code.stderr
    assert not any((tmp_path / a.lower()).exists()
                   for a in ("LORA_DIR", "LLM_DIR", "BBOX_DIR"))
    assert "没动磁盘" in code.stdout


def test_endpoint_flag_is_supported(monkeypatch, tmp_path):
    """国内直连 huggingface.co 常常不通（这里是 502），要能指定镜像。"""
    from workflows import paths
    monkeypatch.setattr(paths, "BBOX_DIR", tmp_path / "bbox")
    code = subprocess.run(                                    # noqa: S603
        [sys.executable, str(ROOT / "tools" / "fetch_vnccs_weights.py"),
         "--only", "yolo", "--endpoint", "https://hf-mirror.com", "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, check=False, timeout=180)
    assert "用镜像" in code.stdout
    assert code.returncode == 0, code.stdout + code.stderr


def test_prune_only_removes_empty_dirs(tmp_path):
    """清理远端路径留下的空壳目录，但**里面有东西就一个都不动**。

    hf 的 `local_dir` 会按远端路径建子目录（`ultralytics/bbox/`），文件挪走后
    空目录会一直杵在那儿 —— 看着像"怎么还有个 ultralytics 目录"。
    """
    root = tmp_path / "bbox"
    (root / "ultralytics" / "bbox").mkdir(parents=True)
    fw._prune_empty_dirs(root, "ultralytics/bbox/face_yolov8m.pt")
    assert not (root / "ultralytics").exists(), "空的应该被清掉"

    # 里面有别的文件时，一个目录都不许动
    (root / "ultralytics" / "bbox").mkdir(parents=True)
    (root / "ultralytics" / "bbox" / "keep.pt").write_bytes(b"x")
    fw._prune_empty_dirs(root, "ultralytics/bbox/face_yolov8m.pt")
    assert (root / "ultralytics" / "bbox" / "keep.pt").is_file()


def test_convert_handles_prefixless_comfyui_variant(tmp_path):
    """ComfyUI 的**第三种变体**：没有 `diffusion_model.` 前缀，且键名带 LoRA 名
    （`.lora_A.default.weight`）。

    项目里 `nsfw-posture-v3` / `qwen21-nsfw-image-edit` 就是这种。转换工具原来
    只认带前缀的两种，遇到它**一个键都转不出来**（`skipped` 全是它），
    用户看到的现象是"LoRA 挂了但界面什么都不报"。
    """
    import torch
    from safetensors.torch import save_file
    src = tmp_path / "in.safetensors"
    save_file({
        "transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.zeros(8, 16),
        "transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.zeros(16, 8),
    }, str(src), metadata={"format": "pt"})

    out = tmp_path / "out.safetensors"
    code = subprocess.run(                                    # noqa: S603
        [sys.executable, str(ROOT / "tools" / "convert_lora.py"), str(src), str(out)],
        cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert code.returncode == 0, code.stdout + code.stderr
    fw.validate_lora(out)
    from workflows.engine import _lora_keys
    keys = _lora_keys(out)
    assert any(k.endswith(".lora.down.weight") for k in keys)
    assert not any(".default." in k for k in keys), "LoRA 名不该留在键里"
