"""权重下载器：优先走 ModelScope（魔搭，本机实测 ~11MB/s）。

hf-mirror 对本机出口 IP 会触发 429 配额限流，因此默认源全部指向魔搭；
魔搭缺的仓库再用 --hf 回落到 hf-mirror。

用法：
    python tools/fetch_modelscope.py            # 拉全部
    python tools/fetch_modelscope.py sd15       # 只拉匹配的子集（按 dest 目录名）
    python tools/fetch_modelscope.py --list     # 只看清单

清单规则（每个仓库一个过滤器）：
- 跳过 safety_checker / *.ckpt / *.bin（有同名 .safetensors 时）
- 有 fp16 变体时优先 fp16，避免多占一倍磁盘
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MS = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
MS_LIST = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=True"


def listing(repo: str) -> list[tuple[str, int]]:
    url = MS_LIST.format(repo=repo)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    out = []
    for f in (data.get("Data") or {}).get("Files", []):
        if f.get("Type") == "blob":
            out.append((f["Path"], int(f.get("Size") or 0)))
    return out


def pick(files: list[tuple[str, int]], rules: dict) -> list[tuple[str, int]]:
    names = {p for p, _ in files}
    keep = []
    for path, size in files:
        if path.startswith("safety_checker/"):
            continue
        if path.endswith((".ckpt", ".msgpack")):
            continue
        # 一律只要 safetensors：.bin 与同名 safetensors 内容等价但大 2 倍
        if path.endswith(".bin"):
            continue
        if path.startswith("sdxl_models/") or path.startswith("sdxl_"):
            continue
        # 有 fp16 变体时，跳过同路径的 fp32 safetensors
        fp16 = path.replace(".safetensors", ".fp16.safetensors")
        if path.endswith(".safetensors") and "fp16" not in path and fp16 in names:
            continue
        for pat in rules.get("include", []):
            if pat in path:
                break
        else:
            if rules.get("include"):
                continue
        for pat in rules.get("exclude", []):
            if pat in path:
                break
        else:
            keep.append((path, size))
    return keep


JOBS = [
    {
        "repo": "AI-ModelScope/stable-diffusion-v1-5",
        "dest": "models/checkpoints/sd15",
        "include": ["model_index.json", "scheduler/", "tokenizer/", "text_encoder/",
                    "unet/", "vae/", "feature_extractor/"],
        "exclude": ["v1-5-pruned", "safety_checker"],
    },
    {
        "repo": "AI-ModelScope/stable-diffusion-inpainting",
        "dest": "models/checkpoints/sd15-inpaint",
        "include": ["model_index.json", "scheduler/", "tokenizer/", "text_encoder/",
                    "unet/", "vae/", "feature_extractor/"],
        "exclude": ["sd-v1-5-inpainting"],
    },
    {
        "repo": "AI-ModelScope/IP-Adapter",
        "dest": "models/ipadapter",
        "include": ["models/ip-adapter_sd15.bin", "models/ip-adapter_sd15.safetensors",
                    "models/image_encoder/"],
        "exclude": ["plus", "full-face", "light", "vit-G", "sdxl"],
    },
    {
        "repo": "AI-ModelScope/controlnet-v1-1",
        "dest": "models/controlnet",
        # 原始 .pth（LDM 格式），引擎侧用 ControlNetModel.from_single_file 直接加载
        "include": ["control_v11p_sd15_openpose.pth", "control_v11f1e_sd15_tile.pth",
                    "control_v11p_sd15_canny.pth"],
        "exclude": ["depth", "lineart", "mlsd", "normalbae", "seg", "scribble",
                    "softedge", "shuffle", "ip2p", "_inpaint"],
    },
]


def download(repo: str, rel: str, dest: Path, size: int) -> None:
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size >= size * 0.99:
        return
    url = MS.format(repo=repo, path=urllib.parse.quote(rel, safe=""))
    print(f"  [dl] {rel}  {size / 2**20:.0f}MB", flush=True)
    cmd = ["curl", "-L", "--retry", "4", "--retry-delay", "3", "-o", str(out), url]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    got = out.stat().st_size if out.exists() else 0
    if got < size * 0.99:
        print(f"  [warn] {rel} 只有 {got / 2**20:.0f}/{size / 2**20:.0f}MB "
              f"err={proc.stderr.strip()[:100]}", flush=True)
    else:
        print(f"  [ok] {rel}", flush=True)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--list"]
    only = args[0] if args else None
    selections = []
    for job in JOBS:
        if only and only not in job["dest"]:
            continue
        files = pick(listing(job["repo"]), job)
        total = sum(s for _, s in files) / 2**30
        selections.append((job, files, total))

    for job, files, total in selections:
        print(f"=== {job['repo']} -> {job['dest']}  ({len(files)} 文件 / {total:.1f}GB)",
              flush=True)
        if "--list" in sys.argv:
            for p, s in files:
                print(f"    {s / 2**20:8.1f}MB  {p}")
            continue
        for rel, size in files:
            download(job["repo"], rel, ROOT / job["dest"], size)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
