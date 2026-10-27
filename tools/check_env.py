"""环境自检：打印硬件、推理栈、权重清单与缺失项。

用法：python tools/check_env.py
"""
from __future__ import annotations

import json
import os
import warnings
import platform
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = {
    "checkpoints/sd15/unet": "SD1.5 底座 UNet",
    "checkpoints/sd15/text_encoder": "SD1.5 文本编码器",
    "checkpoints/sd15-inpaint/unet": "SD1.5 局部重绘 UNet（W3）",
    "controlnet/control_v11p_sd15_openpose.pth": "ControlNet OpenPose（W4）",
    "controlnet/control_v11p_sd15_canny.pth": "ControlNet Canny（备用结构锁定）",
    "controlnet/control_v11f1e_sd15_tile.pth": "ControlNet Tile（高清放大）",
    "ipadapter/models": "IP-Adapter SD1.5（角色身份锁定）",
    "pose_keypoint/dwpose": "DWPose ONNX（骨骼提取）",
}


def gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        return {"name": parts[0], "total_mb": int(parts[1]), "used_mb": int(parts[2]),
                "driver": parts[3]}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def torch_info() -> dict:
    try:
        import torch
        ok = torch.cuda.is_available()
        info = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "available": ok}
        if ok:
            free, total = torch.cuda.mem_get_info()
            info.update(gpu_name=torch.cuda.get_device_name(0),
                        free_gb=round(free / 2**30, 2), total_gb=round(total / 2**30, 2),
                        capability=".".join(map(str, torch.cuda.get_device_capability(0))))
        return info
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def libs() -> dict:
    out = {}
    for name in ("diffusers", "transformers", "accelerate", "safetensors", "cv2",
                 "controlnet_aux", "fastapi", "PIL", "numpy"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "ok")
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def weights() -> dict:
    rows = {}
    for rel, desc in REQUIRED.items():
        p = ROOT / "models" / rel
        if p.is_file():
            files = [p]
        else:
            files = [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        big = [f for f in files if f.suffix in (".safetensors", ".bin", ".onnx", ".pth", ".ckpt")]
        size = sum(f.stat().st_size for f in big) / 2**30
        rows[rel] = {"desc": desc, "ok": bool(big), "gb": round(size, 2), "files": len(big)}
    return rows


def main() -> int:
    disk = shutil.disk_usage(str(ROOT))
    report = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "disk_free_gb": round(disk.free / 2**30, 1),
        "gpu": gpu(),
        "torch": torch_info(),
        "libs": libs(),
        "weights": weights(),
    }
    missing = [k for k, v in report["weights"].items() if not v["ok"]]
    report["missing"] = missing
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
