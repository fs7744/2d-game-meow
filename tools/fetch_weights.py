"""一次性下载 SD1.5 全链路权重（走 hf-mirror）。

用途：验证 SD1.5 能否覆盖文档 W1-W4 所需能力。
只拉必需文件，跳过 checkpoints / safety_checker，总下载量约 5 GB。
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = lambda *p: os.path.join(ROOT, "models", *p)  # noqa: E731

JOBS = [
    (
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        M("checkpoints", "sd15"),
        [
            "model_index.json",
            "*.json",
            "tokenizer/*",
            "scheduler/*",
            "text_encoder/*.fp16.safetensors",
            "text_encoder/*.json",
            "unet/*.fp16.safetensors",
            "unet/*.json",
            "vae/*.fp16.safetensors",
            "vae/*.json",
        ],
    ),
    (
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        M("checkpoints", "sd15-inpaint"),
        [
            "model_index.json",
            "*.json",
            "tokenizer/*",
            "scheduler/*",
            "text_encoder/*.fp16.safetensors",
            "text_encoder/*.json",
            "unet/*.fp16.safetensors",
            "unet/*.json",
            "vae/*.fp16.safetensors",
            "vae/*.json",
        ],
    ),
    (
        "lllyasviel/control_v11p_sd15_openpose",
        M("controlnet", "openpose"),
        ["*.json", "*.fp16.safetensors"],
    ),
    (
        "lllyasviel/control_v11p_sd15_canny",
        M("controlnet", "canny"),
        ["*.json", "*.fp16.safetensors"],
    ),
    (
        "lllyasviel/control_v11f1e_sd15_tile",
        M("controlnet", "tile"),
        ["*.json", "*.fp16.safetensors"],
    ),
    (
        "h94/IP-Adapter",
        M("ipadapter"),
        ["models/ip-adapter_sd15.bin", "models/image_encoder/*", "models/image_encoder/*/*"],
    ),
    (
        "yzd-v/DWPose",
        M("pose_keypoint", "dwpose"),
        ["*"],
    ),
]


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None  # 只拉某个 dst 目录名，便于分批重试
    jobs = [j for j in JOBS if only is None or j[1].endswith(only)]
    if not jobs:
        print(f"no job matches {only!r}")
        return 2

    for repo, local, patterns in jobs:
        print(f"[dl] {repo} -> {local}", flush=True)
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=local,
                allow_patterns=patterns,
                max_workers=4,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {repo}: {type(exc).__name__}: {exc}", flush=True)
            return 1
        print(f"[ok] {repo}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
