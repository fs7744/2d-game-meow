"""精灵管线的单位成本基准：出**一张可用帧**要多少秒、多少显存。

**为什么不直接用 tools/bench.py**：那个是全工作流冒烟基准，每个用例只跑一次，
而单次计时的波动能到 70%（实测同一配置两次跑 1.33s / 1.47s，SDXL 1.77s / 3.07s）。
拿那种数字做选型结论会翻车——本仓库已经翻过一次车。

这里的做法：
  1. **预热**（首次调用含权重加载、CUDA 内核选择），预热样本丢弃；
  2. 稳态测 **N 次取中位数**（中位数抗单次抖动）；
  3. 显存同时记 **allocated（张量峰值）** 与 **reserved（分配器峰值）**——
     真实"占掉多少卡"看 reserved，allocated 会低估；
  4. 最后把耗时换算成 **每可用帧成本**：
        cost = 候选数 × 中位耗时 / P(至少一张可用)
     P = 1 - (1-p)^k，p 是单张可交付率（来自 tools/ab_base.py 的实测）。

**为什么用"每可用帧"而不是"每张图"**：精灵产线的成本不是"生成一张"，而是"拿到一张能进
序列的帧"。反正要挑，那就把候选一起算进去——这是 SD1.5 与 SDXL 最公平的比较口径。

用法：
  python tools/bench_sprite.py                 # 默认矩阵（约 2 分钟）
  python tools/bench_sprite.py --quick         # 只测 b1，冒烟用
  python tools/bench_sprite.py --p-sd15 0.33 --p-sdxl 0.92
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch                                    # noqa: E402
from PIL import Image, ImageDraw                # noqa: E402

from workflows import generate as g             # noqa: E402
from workflows import presets                   # noqa: E402
from workflows.engine import Engine             # noqa: E402

ASSETS = ROOT / "assets"
ROWS: list[dict] = []

# 单张可交付率：来自 tools/ab_base.py 的实测。
#   sd15 @768² = 33%，sdxl @768² = 92%，sdxl @1024² = 75%
# **p 与尺寸强相关**：SDXL 在 1024² 下角色占比更大，反而更容易顶边（92% → 75%）。
# 这个工具每个基座只收一个 p，所以跨尺寸对比时要分别传参：
#   python tools/bench_sprite.py --p-sdxl 0.75   # 测 1024² 那一组时
# 换成自己的基座后应重跑 ab_base 更新这些数，否则成本换算会失真。
P_DEFAULT = {"sd15": 0.33, "sdxl": 0.92}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(fn, *, warm: int = 1, n: int = 3) -> tuple[float, float, float, bool]:
    """(中位耗时, 张量峰值GB, 分配器峰值GB, 是否 OOM)。"""
    for _ in range(max(0, warm)):
        fn()
    _sync()
    times: list[float] = []
    peak_a = peak_r = 0.0
    for _ in range(max(1, n)):
        torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()
        try:
            fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan"), 0.0, 0.0, True
        _sync()
        times.append(time.perf_counter() - t0)
        peak_a = max(peak_a, torch.cuda.max_memory_allocated() / 2**30)
        peak_r = max(peak_r, torch.cuda.max_memory_reserved() / 2**30)
    return statistics.median(times), peak_a, peak_r, False


POST_SEC: dict[str, float] = {}          # 后处理的固定耗时（纯 CPU，按基座记录）


def record(case: str, t: float, pa: float, pr: float, oom: bool,
           cand: int = 1, p: float | None = None, model: str = "") -> None:
    """记一行；给了 p 就换算每可用帧成本。

    **公式**：一次批量出 k 张的耗时是 t（t 已包含 k 张），
        每可用帧成本 = t / P(至少一张可用)，P = 1-(1-p)^k
    早期版本写成 `k*t/P`，等于把批量耗时又乘了一遍 k —— 那会把 SD1.5 的
    批量优势算成劣势（x4 时虚报 4 倍），是本次实验里抓到的第二个自伤 bug。
    """
    cost = cost_all = None
    if p is not None and not oom:
        hit = 1.0 - (1.0 - p) ** max(1, cand)
        cost = t / hit if hit > 0 else float("nan")
        post = POST_SEC.get(model, 0.0)
        cost_all = (t + post) / hit if hit > 0 else float("nan")
    ROWS.append({"case": case, "model": model,
                 "seconds": round(t, 2) if not oom else None,
                 "peak_alloc_gb": round(pa, 2) if not oom else None,
                 "peak_reserved_gb": round(pr, 2) if not oom else None,
                 "candidates": cand, "deliverable": p,
                 "cost_per_usable_s": round(cost, 2) if cost else None,
                 "cost_with_post_s": round(cost_all, 2) if cost_all else None,
                 "oom": oom})
    if oom:
        print(f"  {case:42s}   OOM（显存不够）", flush=True)
    else:
        extra = f" → 每可用帧 {cost:.1f}s" if cost else ""
        print(f"  {case:42s} {t:6.2f}s  张量 {pa:5.2f}GB  保留 {pr:5.2f}GB{extra}",
              flush=True)


def t2i_case(eng: Engine, model: str, size: int, steps: int, batch: int,
             p: float | None) -> None:
    prompt, neg = presets.build("a knight in silver armor holding a katana", None,
                                "pixel", max_tokens=presets.CHUNK_BUDGET)
    pipe = eng.txt2img_pipe(model)
    kw, _ = g._text_kwargs(pipe, prompt, neg, 1, model)

    def run():
        args = dict(width=size, height=size, num_inference_steps=steps,
                    guidance_scale=6.5, **kw)
        if batch > 1:
            args["num_images_per_prompt"] = batch
            args["generator"] = [torch.Generator("cuda").manual_seed(100 + i)
                                 for i in range(batch)]
        else:
            args["generator"] = torch.Generator("cuda").manual_seed(100)
        pipe(**args)

    t, pa, pr, oom = measure(run, warm=1, n=2 if batch > 1 else 3)
    record(f"W1 {model} {size}² x{batch} / {steps}步", t, pa, pr, oom,
           cand=batch, p=p, model=model)
    eng.drop_others(())                        # 每个用例之间清场，峰值才可比


def w4_case(eng: Engine, model: str, size: int, steps: int, batch: int = 1) -> None:
    """W4 场景：ControlNet + IP-Adapter。这是显存压力的真实上限。"""
    cond = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(cond)
    d.ellipse([size // 3, size // 6, size * 2 // 3, size // 2], outline=(0, 0, 0), width=4)
    d.line([size // 2, size // 2, size // 2, size * 5 // 6], fill=(0, 0, 0), width=4)
    ref = Image.new("RGB", (size, size), (230, 60, 60))

    pipe = eng.pose_pipe(controlnet="openpose", with_ipadapter=True, model=model)
    prompt, neg = presets.build("a knight in silver armor", None, "pixel",
                                max_tokens=presets.CHUNK_BUDGET)
    kw, _ = g._text_kwargs(pipe, prompt, neg, 1, model)

    def run():
        args = dict(image=cond, ip_adapter_image=ref, width=size, height=size,
                    num_inference_steps=steps, guidance_scale=6.5, **kw)
        if batch > 1:
            args["num_images_per_prompt"] = batch
            args["generator"] = [torch.Generator("cuda").manual_seed(7 + i)
                                 for i in range(batch)]
        else:
            args["generator"] = torch.Generator("cuda").manual_seed(7)
        pipe(**args)

    t, pa, pr, oom = measure(run, warm=1, n=2)
    record(f"W4 {model} {size}² CN+IPA x{batch} / {steps}步", t, pa, pr, oom,
           cand=batch, model=model)
    eng.drop_others(())


def pixels_of(eng: Engine, model: str, size: int) -> None:
    """像素化+量化这一段是纯 CPU，但要看它在 1024² 上的耗时占比。

    **输入必须是"未像素化的原图"（__raw）**，不能用成品：
    成品已经是 ≤32 色的像素画，`pixelate(grid_align=True)` 会命中网格检测并跑
    refine_one（实测 0.78s）；而产线真实输入是连续调 AI 图，色数成千上万，
    在 0.5ms 内直接回退。拿成品测会把后处理成本虚报 10 倍以上。
    合成条纹图也不行：clean_isolated 会因为满屏孤立点跑满 passes。
    """
    src = None
    cands = sorted(ASSETS.glob("_ab_base*/**/*__raw.png"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    for p in cands:
        im = Image.open(p)
        if im.width == size:
            src = im.convert("RGB")
            break
    if src is None:
        src = Image.new("RGB", (size, size), (250, 250, 250))
        d = ImageDraw.Draw(src)
        d.ellipse([size // 4, size // 8, size * 3 // 4, size * 7 // 8], fill=(40, 90, 200))
    block = max(2, size // 128)
    t, _pa, _pr, _oom = measure(
        lambda: g._postprocess(src, eng, matte=False, matte_backend="color", denoise=0,
                               pixel=block, palette=32, clean_iso=1, info={}),
        warm=0, n=3)
    POST_SEC[model] = t
    record(f"后处理 {model} {size}²（pixel={block}/32色）", t, 0.0, 0.0, False,
           model=model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只测候选数 1")
    ap.add_argument("--p-sd15", type=float, default=P_DEFAULT["sd15"])
    ap.add_argument("--p-sdxl", type=float, default=P_DEFAULT["sdxl"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ASSETS.mkdir(exist_ok=True)
    eng = Engine(ROOT / "models")
    have = eng.available_models()
    P = {"sd15": a.p_sd15, "sdxl": a.p_sdxl}

    print(f"单张可交付率（来自 ab_base 实测）：{P}")
    print()

    # 后处理先测：它是固定成本，要在成本换算里算进去
    pixels_of(eng, "sd15", 768)
    if "sdxl" in have:
        pixels_of(eng, "sdxl", 768)
    print()

    # ---- W1 候选批量：sd15 768² vs sdxl 768²（同分辨率比权重）vs sdxl 1024²（甜点）
    if "sd15" in have:
        for b in ([1] if a.quick else [1, 2, 4]):
            t2i_case(eng, "sd15", 768, 20, b, P["sd15"])
    if "sdxl" in have:
        for b in ([1] if a.quick else [1, 2]):
            t2i_case(eng, "sdxl", 768, 20, b, P["sdxl"])
        print()
        for b in ([1] if a.quick else [1, 2]):
            t2i_case(eng, "sdxl", 1024, 24, b, P["sdxl"])

    # ---- W4：ControlNet + IP-Adapter，这才是精灵产线的真实显存压力
    print()
    if "sd15" in have:
        w4_case(eng, "sd15", 768, 20)
    if "sdxl" in have:
        w4_case(eng, "sdxl", 768, 20)
        w4_case(eng, "sdxl", 1024, 24)
        if not a.quick:
            # 精灵产线要候选，所以"能不能批量"比单帧峰值更有决策价值
            w4_case(eng, "sdxl", 1024, 24, batch=2)

    out = Path(a.out) if a.out else ASSETS / "_bench_sprite.json"
    out.write_text(json.dumps(ROWS, ensure_ascii=False, indent=2), "utf-8")

    print("\n| 用例 | 中位耗时 | 张量峰值 | 保留峰值 | 每可用帧(生成) | 每可用帧(含后处理) |")
    print("|---|---|---|---|---|---|")
    for r in ROWS:
        c1 = "—" if r["cost_per_usable_s"] is None else f"{r['cost_per_usable_s']}s"
        c2 = "—" if r["cost_with_post_s"] is None else f"{r['cost_with_post_s']}s"
        if r["oom"]:
            print(f"| {r['case']} | OOM | — | — | — | — |")
        else:
            print(f"| {r['case']} | {r['seconds']}s | {r['peak_alloc_gb']}GB | "
                  f"{r['peak_reserved_gb']}GB | {c1} | {c2} |")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
