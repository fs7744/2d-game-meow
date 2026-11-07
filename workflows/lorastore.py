"""LoRA 导入：暂存 → 格式检测 →（用户确认后）转换 → 落盘。

**为什么分两步**：用户把一个 ComfyUI 格式的 Qwen LoRA 拖进来时，直接挂上去
diffusers 只会打一句 "unexpected keys found in the model" 然后继续 —— 看起来
导入成功了，实际一点效果都没有，比直接报错更难查。所以先只做只读检测，
把「这是什么格式 / 为什么当前底模用不了 / 转换会做什么」摆给用户，
**用户确认之后**才真正转换并写进 `models/lora/`。

存储边界跟 `metadb.py` 一致：**文件只在磁盘上**。这里只操作
`models/lora/*.safetensors` 与暂存的原始字节，不往数据库写任何权重。

格式判定的唯一来源是 `engine.lora_format()`；转换复用
`tools/convert_lora.py::convert_weights()`（同一套键名映射与 img_mlp 拆分），
不在服务端另写一份，免得 CLI 与 UI 两条路慢慢分叉。
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from workflows import paths
from workflows.engine import _lora_keys, lora_format

#: 原始上传先落到这里；只有用户点了「确认导入」才会进 models/lora/。
#: 放在 library/ 下（已被 .gitignore 忽略），跟交付资产分开。
STAGE_DIR = paths.LIBRARY_DIR / "lora_staging"

#: LoRA 常有几百 MB，但 2GiB 已经是明显异常的文件了。
MAX_LORA_BYTES = 2 * 1024 ** 3

_SAFE = re.compile(r"[^0-9A-Za-z_\-]+")
_TOKEN = re.compile(r"^[0-9a-f]{12,32}$")


def safe_name(raw: str) -> str:
    """文件名清洗：只留 ASCII 字母数字与 -_，挡住路径穿越与 peft 的非法字符。

    **只剥 `.safetensors` 这一个后缀**：用 `Path(...).stem` 会把
    `VNCCS_QI2_ClothesCoreV2.6` 里的 `.6` 也当扩展名吃掉，版本号就丢了。
    """
    stem = str(raw or "").strip().replace("\\", "/")
    if stem.lower().endswith(".safetensors"):
        stem = stem[: -len(".safetensors")]
    stem = stem.rsplit("/", 1)[-1]          # 只要最后一段，别留任何分隔符
    cleaned = _SAFE.sub("_", stem).strip("_")[:60]
    return cleaned or "lora"


def stage_path(token: str) -> Path:
    if not _TOKEN.match(str(token or "")):
        raise ValueError("导入凭据无效，请重新选择文件")
    return STAGE_DIR / f"{token}.safetensors"


def new_token() -> str:
    return uuid.uuid4().hex[:16]


def read_keys(path: Path) -> List[str]:
    """只读 safetensors 头拿 key 列表；解析失败给明确错误，不静默当成空格式。"""
    try:
        return _lora_keys(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"不是有效的 safetensors 文件（读不出头部）：{exc}") from exc


def inspect(path: Path, original_name: str = "") -> Dict[str, Any]:
    """只读检测。不写 models/lora/，不改任何状态。"""
    keys = read_keys(path)
    if not keys:
        raise ValueError("文件里没有任何张量，可能不是 LoRA 权重")
    info = lora_format(keys)
    return {
        "name": safe_name(original_name or path.stem),
        "original_name": Path(original_name or path.name).name,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
        "tensors": info["tensors"],
        "format": info["format"],
        "need_convert": info["need_convert"],
        "convertible": info["convertible"],
        "reason": info["reason"],
        "sample_keys": keys[:4],
    }


def _converter():
    """按文件路径加载 `tools/convert_lora.py`，不走包导入（tools/ 不是包）。"""
    p = paths.ROOT / "tools" / "convert_lora.py"
    spec = importlib.util.spec_from_file_location("meow_convert_lora", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"找不到转换器：{p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _final_path(name: str) -> Path:
    d = paths.LORA_DIR
    d.mkdir(parents=True, exist_ok=True)
    target = (d / f"{name}.safetensors").resolve()
    if d.resolve() not in target.parents:
        raise ValueError("非法的 LoRA 文件名")
    return target


def item_of(path: Path) -> Dict[str, Any]:
    info = lora_format(read_keys(path))
    return {
        "name": path.stem,
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
        "tensors": info["tensors"],
        "format": info["format"],
    }


def commit(token: str, convert: bool = False, name: str = "") -> Dict[str, Any]:
    """确认导入：`convert=True` 时先转格式，再原子写入 models/lora/。

    - 检测为 diffusers 原生 → 直接落盘，不需要用户确认转换；
    - 检测为 ComfyUI → **必须** `convert=True`，否则明确拒绝；
    - 检测为未知格式 → 转换也救不了，明确报错并附上头几个键名。
    """
    src = stage_path(token)
    if not src.is_file():
        raise ValueError("导入凭据已失效（暂存文件不在了），请重新选择文件")
    keys = read_keys(src)
    info = lora_format(keys)

    if info["need_convert"] and not convert:
        raise ValueError(f"这个 LoRA 需要转换才能用于当前底模：{info['reason']}")
    if info["convertible"] is False and convert:
        raise ValueError(
            f"{info['reason']}，转换也生成不了可用权重。"
            f"头几个键：{', '.join(keys[:3])}")

    dst = _final_path(safe_name(name or src.stem))
    tmp = dst.with_name(f".{dst.stem}.{uuid.uuid4().hex[:8]}.tmp")
    skipped: List[str] = []
    try:
        if convert:
            from safetensors.torch import load_file, save_file
            conv = _converter()
            out, skipped = conv.convert_weights(load_file(str(src)))
            if not out:
                raise ValueError(
                    f"转换后没有任何可用张量，这个文件可能不是 Qwen-Image LoRA。"
                    f"头几个键：{', '.join(keys[:3])}")
            # 转换结果必须真的是 diffusers 格式，否则落盘只会制造下一个
            # 「挂上了但没生效」的哑弹。
            check = lora_format(list(out))
            if check["format"] != "diffusers":
                raise ValueError("转换结果仍不是 diffusers 格式，已中止导入")
            save_file(out, str(tmp), metadata={"format": "pt"})
        else:
            shutil.copyfile(src, tmp)
        # 原子替换：同名重导入不会留下半截文件。
        os.replace(tmp, dst)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    # 暂存文件只在成功后清掉；失败时留着，用户可以直接再点一次确认。
    try:
        src.unlink()
    except OSError:
        pass
    return {
        "item": item_of(dst),
        "converted": bool(convert),
        "skipped": len(skipped),
        "name": dst.stem,
    }


def delete(name: str) -> str:
    """删掉一个已导入的 LoRA 文件（只删文件，不碰数据库）。"""
    target = _final_path(safe_name(name))
    if not target.is_file():
        raise FileNotFoundError(f"没有这个 LoRA：{target.stem}")
    target.unlink()
    return target.stem


def convert_in_place(name: str) -> Dict[str, Any]:
    """把 `models/lora/` 里一个**已经在目录里**的 LoRA 就地转成 diffusers 格式。

    场景：用户把 ComfyUI 格式的权重直接拷进了 `models/lora/`——它不会出现在
    「可挂载」列表里，但也不该逼用户重新走一遍上传。这里直接就地转换，
    原名不变，`/loras` 刷新后就能挂。

    **先备份**：原始字节复制到 `library/lora_backup/`（不在 models/lora/ 下，
    不会被当成另一个 LoRA 扫出来）。转换失败时原文件一个字节都没动。
    """
    target = _final_path(safe_name(name))
    if not target.is_file():
        raise FileNotFoundError(f"没有这个 LoRA：{target.stem}")
    info = lora_format(read_keys(target))
    if info["format"] == "diffusers":
        return {"item": item_of(target), "converted": False, "backup": ""}
    if not info["convertible"]:
        raise ValueError(f"{info['reason']}，转换也生成不了可用权重")

    backup_dir = paths.LIBRARY_DIR / "lora_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / target.name
    shutil.copyfile(target, backup)

    tmp = target.with_name(f".{target.stem}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        from safetensors.torch import load_file, save_file
        conv = _converter()
        out, skipped = conv.convert_weights(load_file(str(target)))
        if not out:
            raise ValueError("转换后没有任何可用张量，这个文件可能不是 Qwen-Image LoRA")
        if lora_format(list(out))["format"] != "diffusers":
            raise ValueError("转换结果仍不是 diffusers 格式，已中止")
        save_file(out, str(tmp), metadata={"format": "pt"})
        os.replace(tmp, target)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return {"item": item_of(target), "converted": True,
            "backup": str(backup), "skipped": len(skipped)}
