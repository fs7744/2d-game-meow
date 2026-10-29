"""下载动漫底模：Illustrious XL v0.1（SDXL 架构）。

为什么用它：SDXL base 的训练数据里动漫占比低，画出来偏厚涂、手部/武器
衔接糊、背景不干净。Illustrious 是 Civitai 动漫类最受欢迎的底模，
专攻插画质量，danbooru tag 体系友好——对游戏角色立绘是刚需。

走 ModelScope：HF 在国内经常 502（本项目实测过多次）。

用法：
    python tools/fetch_anime_xl.py                # 下载 Illustrious（默认）
    python tools/fetch_anime_xl.py --model animagine
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

# diffusers 结构需要的文件。单文件 .safetensors 是 6.6GB，
# 而 unet+vae+tokenizer 分开下只有 5.1GB，且能直接被 from_pretrained 加载。
REPOS = {
    "illustrious": {
        "repo": "OnomaAIResearch/Illustrious-xl-early-release-v0",
        "dest": "models/checkpoints/sdxl-anime",
        "label": "Illustrious XL v0.1",
    },
    "animagine": {
        "repo": "AI-ModelScope/animagine-xl-3.1",
        "dest": "models/checkpoints/sdxl-anime",
        "label": "Animagine XL 3.1",
    },
}

# 必要文件（缺一不可）+ 可选文件（缺失则用 SDXL 现有的）
REQUIRED = [
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
OPTIONAL = [
    "tokenizer/merges.txt", "tokenizer/vocab.json",
    "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
    "tokenizer_2/merges.txt", "tokenizer_2/vocab.json",
    "tokenizer_2/tokenizer_config.json", "tokenizer_2/special_tokens_map.json",
    "model_index.json", "configuration.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
]


def download(repo: str, path: str, dest: Path) -> bool:
    out = dest / path
    if out.exists() and out.stat().st_size > 0:
        return True
    url = (f"https://modelscope.cn/api/v1/models/{repo}/repo"
           f"?Revision=master&FilePath={path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with requests.get(url, stream=True, timeout=180) as r:
            if r.status_code != 200:
                return False
            n = 0
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    fh.write(chunk)
                    n += len(chunk)
            if n == 0:
                tmp.unlink(missing_ok=True)
                return False
            tmp.rename(out)
        print(f"  {path:50s} {n/2**20:8.1f}MB  {time.time()-t0:.0f}s")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  跳过 {path}（{type(e).__name__}）")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="illustrious", choices=list(REPOS))
    args = ap.parse_args()

    cfg = REPOS[args.model]
    repo, dest = cfg["repo"], ROOT / cfg["dest"]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"下载 {cfg['label']}")
    print(f"  源  ModelScope/{repo}")
    print(f"  目标 {dest}")

    ok = True
    for f in REQUIRED:
        if not download(repo, f, dest):
            print(f"  [必需] {f} 下载失败")
            ok = False
    for f in OPTIONAL:
        download(repo, f, dest)

    print()
    if ok:
        print("完成。缺的 tokenizer/text_encoder 会从现有 SDXL 补上（架构相同）。")
    else:
        print("有必需文件缺失，管线可能加载失败。")


if __name__ == "__main__":
    main()
