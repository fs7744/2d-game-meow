"""下载像素专用权重：PixelDream（Illustrious LoRA）+ RetroDiffusion（SD1.5 底模）。

**渠道是实测出来的**（2026-10）：
  · huggingface.co  直连不通（ProxyError）
  · civitai.com     直连不通（ProxyError）
  · hf-mirror.com   ✅ HF 镜像可用，但**只有 ~40KB/s**（2GB 要 13 小时，实测）
  · civitai.red     ⚠️ API 可用（能读元数据），但下载 307 到 b2.civitai.com，
                       而 b2 直连超时、走代理 502 —— **权重下不下来**
  · ModelScope      ❌ 这两个模型都没有（搜过）

所以：RetroDiffusion 慢到不可用，PixelDream 拿不到。真要跑实测，
需要能连 civitai CDN 的网络环境，然后：
    python tools/fetch_pixel_models.py --model pixeldream
（把 CIVITAI 换回 https://civitai.com 或让代理放行 b2.civitai.com）

用法：
    python tools/fetch_pixel_models.py --model pixeldream
    python tools/fetch_pixel_models.py --model retrodiffusion
    python tools/fetch_pixel_models.py --model all

RetroDiffusion 只下 fp16 那一套（2.0GB）。fp32 全套 13.9GB 没必要——
engine.detect_variant() 见到 unet/*.fp16.safetensors 就会用 fp16 variant。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

HF = "https://hf-mirror.com"          # HF 镜像（直连不通）
CIVITAI = "https://civitai.red"       # Civitai 镜像（直连不通）

# RetroDiffusion：diffusers 目录。只取 fp16 + 小文件，总计约 2.03GB。
RETRO_REPO = "LostMedia/RetroDiffusion"
RETRO_DEST = ROOT / "models" / "checkpoints" / "retrodiffusion"
RETRO_FILES: list[tuple[str, int]] = [
    ("model_index.json", 0),
    ("scheduler/scheduler_config.json", 0),
    ("text_encoder/config.json", 0),
    ("text_encoder/model.fp16.safetensors", 235),
    ("tokenizer/merges.txt", 0),
    ("tokenizer/special_tokens_map.json", 0),
    ("tokenizer/tokenizer_config.json", 0),
    ("tokenizer/vocab.json", 1),
    ("unet/config.json", 0),
    ("unet/diffusion_pytorch_model.fp16.safetensors", 1640),
    ("vae/config.json", 0),
    ("vae/diffusion_pytorch_model.fp16.safetensors", 160),
]

# PixelDream：Civitai 1528534 / version 1729424，217.9MB，底模 Illustrious。
# civitai 直链下不了（307 → b2.civitai.com 不通），改走 HF 上的第三方搬运版。
# Muapi / John6666 这类账号会把 Civitai 权重原样转到 HF，是国内唯一能走的渠道。
PIXELDREAM_URL = f"{CIVITAI}/api/download/models/1729424"
PIXELDREAM_DEST = ROOT / "models" / "lora" / "pixeldream_il.safetensors"
PIXELDREAM_MB = 218

# HF 镜像上的搬运版 LoRA：(repo, 文件名, 目标名, 预期 MB)
#
# 为什么走这里：hf-mirror **限速到 ~30-50KB/s**（实测；同一时刻 ModelScope
# 是 7~9MB/s，差 200 倍），所以只有几十~二百 MB 的 LoRA 值得下，
# 2GB 级的底模（RetroDiffusion）在这个网速下没有实际可行性（约 11 小时）。
MIRROR_LORAS: dict[str, tuple[str, str, str, int]] = {
    # 217.9MB 与 Civitai 上 PixelDream 的文件大小完全一致 → 就是它
    "pixeldream": ("Muapi/pixel-art-style-illustrious-by-skormino",
                   "pixel-art-style-illustrious-by-skormino.safetensors",
                   "pixeldream_il.safetensors", 218),
    # 同作者的另一支 Illustrious 像素 LoRA，体积小、下得快，可先验证路子
    "pixelxl15": ("Muapi/illustrious-pixel-art-xl-1.5-by-creativehotia",
                  "illustrious-pixel-art-xl-1.5-by-creativehotia.safetensors",
                  "pixel_xl15_il.safetensors", 55),
}


def _download(url: str, dest: Path, expect_mb: int = 0) -> None:
    """流式下载，带进度；已存在且大小对得上就跳过。"""
    if dest.exists():
        mb = dest.stat().st_size / 2**20
        if not expect_mb or mb > expect_mb * 0.9:
            print(f"  跳过 {dest.name}（已有 {mb:.1f}MB）")
            return
        print(f"  重下 {dest.name}（现有 {mb:.1f}MB 偏小）")
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        last = -1
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                mb = done / 2**20
                if total and int(mb) % 10 == 0 and int(mb) != last:
                    last = int(mb)
                    print(f"    {mb:7.0f} / {total / 2**20:.0f} MB", flush=True)
    got = tmp.stat().st_size / 2**20
    if expect_mb and got < expect_mb * 0.5:
        tmp.unlink()
        raise RuntimeError(f"{dest.name} 只下到 {got:.0f}MB（预期 {expect_mb}MB），"
                           f"可能下到了错误页面")
    tmp.replace(dest)
    print(f"  ✅ {dest.name}  {got:.0f}MB")


def fetch_retrodiffusion() -> None:
    print(f"RetroDiffusion → {RETRO_DEST.relative_to(ROOT)}")
    for rel, mb in RETRO_FILES:
        url = f"{HF}/{RETRO_REPO}/resolve/main/{rel}"
        print(f"  ↓ {rel}")
        _download(url, RETRO_DEST / rel, mb)
    # 完整性自检：缺 model_index 或 unet 就当失败
    need = [RETRO_DEST / "model_index.json",
            RETRO_DEST / "unet" / "diffusion_pytorch_model.fp16.safetensors",
            RETRO_DEST / "vae" / "diffusion_pytorch_model.fp16.safetensors",
            RETRO_DEST / "text_encoder" / "model.fp16.safetensors"]
    missing = [p for p in need if not p.exists()]
    if missing:
        raise RuntimeError("缺少文件：" + ", ".join(str(p) for p in missing))
    print("RetroDiffusion 就绪（Public Domain Mark 1.0，可商用）")


def fetch_mirror_lora(name: str) -> None:
    """从 HF 镜像的搬运仓库下 LoRA（国内唯一可行的渠道）。"""
    repo, fname, dest_name, mb = MIRROR_LORAS[name]
    dest = ROOT / "models" / "lora" / dest_name
    url = f"{HF}/{repo}/resolve/main/{fname}"
    print(f"{repo} → {dest.relative_to(ROOT)}（约 {mb}MB，镜像限速 ~40KB/s，"
          f"预计 {mb*1024/40/60:.0f} 分钟）")
    _download(url, dest, mb)


def fetch_pixeldream() -> None:
    print(f"PixelDream → {PIXELDREAM_DEST.relative_to(ROOT)}")
    _download(PIXELDREAM_URL, PIXELDREAM_DEST, PIXELDREAM_MB)
    print("PixelDream 就绪（底模 Illustrious，触发词 pixeldream）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["pixeldream", "pixelxl15", "retrodiffusion", "all"])
    ap.add_argument("--via", default="mirror", choices=["mirror", "civitai"],
                    help="mirror=走 HF 搬运版（国内可用）；civitai=直连（需代理）")
    a = ap.parse_args()
    if a.model in ("pixeldream", "pixelxl15", "all") and a.via == "mirror":
        for n in (["pixelxl15", "pixeldream"] if a.model == "all" else [a.model]):
            fetch_mirror_lora(n)
    elif a.model == "pixeldream":
        fetch_pixeldream()
    if a.model in ("retrodiffusion", "all"):
        fetch_retrodiffusion()


if __name__ == "__main__":
    main()
