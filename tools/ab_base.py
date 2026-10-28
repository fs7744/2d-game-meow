"""基座 A/B：同一套产线参数下，量出「换哪个底模权重更合适」。

**为什么需要这个工具**：画质的主因是**基座权重**，不是架构。原始 runwayml v1.5
在"角色 + 武器 + 铠甲"这类复合概念上本来就弱，换一个社区微调（同架构、同显存、
同步数）往往就够了；但"哪个更好"不能靠肉眼看几张图下结论——那正好是确认偏差
最容易生效的地方。

用法：
  # 1) 把候选权重放到 models/checkpoints/<name>/（带 model_index.json 即可）
  python tools/fetch_modelscope.py --list          # 看现成的下载清单
  # 2) 跑对照（默认用本机所有可用底模）
  python tools/ab_base.py
  python tools/ab_base.py --models sd15,dreamshaper --seeds 3
  python tools/ab_base.py --prompts knight,slime --limit 1     # 快速冒烟

输出到 assets/_ab_base/：
  <model>__<prompt>__s<seed>.png    每张产物 + 同名 .json（meta 与指标）
  report.md                          人看的对比表（含结论与偏差提醒）
  summary.csv                        程序读的明细
  sheet_<prompt>.png                 每个 prompt 一张对比图（列=基座，行=seed）

**指标怎么读**（全部客观、可复现，别只看一张图）：
  subjects  抠图后前景连通域数。>1 = 画面里挤了多个主体/道具，精灵管线直接废片。
            这是 base v1.5 最典型的问题，也是换基座收益最明显的一项。
  edge      被画布切断的边数（贴边 >2% 才算）。裁脚/裁头，对齐必然跳。
  fg        前景占比。过小=人物缩成一点，过大=半身/糊成一坨。
  uniform   像素块单色率。像素画应当接近 1.0。
  colors    量化后唯一色数，必须 <= palette。
  white     四边接近纯白的比例。精灵管线的第一硬指标：背景脏 → 只能靠 u2net，
            边缘必糊。低 white 往往连带高 subjects。
  points    姿态模型在产物上检出的关节点数（0~18）。**只对类人形 prompt 有意义**，
            怪物/道具恒为 0，别拿它横向比。

**别把这张表当成唯一判据**：它量的是"能不能进产线"（不废片），量不出"好不好看"。
所以每张 prompt 还出一张对比图，人眼打分要跟客观指标一起看——两者不一致时，
以"废片率"为准，因为那是硬约束。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows import generate as g            # noqa: E402
from workflows import matte as matte_mod        # noqa: E402
from workflows import presets                   # noqa: E402
from workflows.engine import Engine             # noqa: E402

OUT = ROOT / "assets" / "_ab_base"

# 产线真实用例，按「base v1.5 最容易翻车」的方向选：
#   slime   小体型怪物 → 多主体/复制粘贴（SD1.5 的经典病）
#   knight  角色+武器+铠甲 → 复合概念，最考验基座的知识覆盖
#   wuxia   长袍动态姿态 → 衣纹与水墨背景污染
#   mech    非人形机械 → 结构崩坏
#   prop    道具（宝箱）→ 单主体但非人形
#   portrait 立绘半身 → 构图控制
PROMPTS: list[dict] = [
    {"id": "slime", "text": "a green slime monster with big eyes", "preset": "pixel",
     "humanoid": False, "note": "多主体高发"},
    {"id": "knight", "text": "a knight in silver armor holding a katana, red cape",
     "preset": "pixel", "humanoid": True, "note": "复合概念 + 武器"},
    {"id": "wuxia", "text": "wuxia swordsman in long white robe, dynamic pose",
     "preset": "wuxia", "humanoid": True, "note": "衣纹 / 印章 / 纸纹"},
    {"id": "mech", "text": "a bipedal battle mech with cannon arms, steel plating",
     "preset": "pixel", "humanoid": False, "note": "非人形结构"},
    {"id": "prop", "text": "a wooden treasure chest with gold trim", "preset": "pixel",
     "humanoid": False, "note": "道具 / 单主体"},
    {"id": "portrait", "text": "a female mage with purple hair, staff, front view",
     "preset": "sprite", "humanoid": True, "note": "构图与裁切"},
]

DEFAULT_STEPS = {"sd15": 20, "sdxl": 24}
DEFAULT_SIZE = {"sd15": 512, "sdxl": 1024}


def deliverable(row: dict) -> bool:
    """这一张能不能直接进产线（精灵管线的三条硬约束同时满足）。

    为什么合成一个布尔：分开看容易自我安慰——"只是脚被裁了"在静态图上不致命，
    但精灵帧要对齐脚底锚点，被裁的帧一进序列就跳。合起来看才是真实废片率。
    """
    return (row["subjects"] == 1 and row["edges"] == 0
            and 0.05 <= row["fg"] <= 0.60)


def _block_for(size: int) -> int:
    """与产线一致：有效像素固定在 128（512÷4、768÷6、1024÷8）。"""
    return max(2, size // 128)


def _uniformity(img: Image.Image, block: int) -> float:
    arr = np.asarray(img.convert("RGBA"))
    h, w = arr.shape[:2]
    hh, ww = h // block * block, w // block * block
    if hh < block or ww < block:
        return -1.0
    cells = arr[:hh, :ww].reshape(hh // block, block, ww // block, block, 4)
    cells = cells.transpose(0, 2, 1, 3, 4).reshape(-1, block * block, 4)
    return round(float((cells == cells[:, :1, :]).all(axis=(1, 2)).mean()), 3)


def _white_ratio(raw: Image.Image, band: int = 24, tol: int = 18) -> float:
    bg = Image.new("RGBA", raw.size, (255, 255, 255, 255))
    rgb = np.asarray(Image.alpha_composite(bg, raw.convert("RGBA")).convert("RGB"))
    h, w = rgb.shape[:2]
    band = min(band, h // 2, w // 2)
    edge = np.concatenate([rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
                           rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)])
    return round(float(((255 - edge.astype(np.int16)).max(axis=1) <= tol).mean()), 3)


def run_one(eng: Engine, model: str, spec: dict, seed: int, *,
            size: int, steps: int, cfg: float, lora: str | None,
            lora_scale: float, palette: int = 32,
            extra_neg: str = "", hires: float = 0.0) -> dict:
    """按**产线同一套后处理**跑一张，返回产物与指标。"""
    prompt, negative = presets.build(spec["text"], extra_neg or None, spec["preset"],
                                     max_tokens=presets.CHUNK_BUDGET)
    pipe = eng.txt2img_pipe(model)
    eng.apply_lora(pipe, lora, lora_scale)
    kw, _ = g._text_kwargs(pipe, prompt, negative, 1, model)

    t0 = time.time()
    raw = g._infer(eng, pipe, dict(width=size, height=size, num_inference_steps=steps,
                                   guidance_scale=cfg,
                                   generator=torch.Generator("cuda").manual_seed(seed),
                                   **kw), label=f"ab:{model}").images[0]
    secs = time.time() - t0

    if hires and hires > 1.0:
        # 两段式：先在底座分辨率出结构，再低幅度放大补细节（见 generate._hires_fix）
        raw = g._hires_fix(eng, raw, scale=hires, strength=0.35,
                           steps=max(8, steps // 2), cfg=cfg, kw=kw, model=model,
                           seed=seed, lora=lora, lora_scale=lora_scale)
        size = raw.width
    block = _block_for(size)
    info: dict = {}
    out, edge, subjects = g._postprocess(
        raw, eng, matte=True, matte_backend="auto", denoise=1,
        pixel=block, palette=palette, clean_iso=1, info=info)

    # 关节数只在「类人形」prompt 上有意义（怪物/道具恒为 0）
    points = 0
    extractor = eng.pose_extractor() if spec.get("humanoid") else None
    if extractor is not None:
        try:
            points = len(extractor.visible_points(raw, thr=0.25))
        except Exception:                 # noqa: BLE001
            points = 0

    rgba = out.convert("RGBA")
    solid = np.asarray(rgba)[..., 3] > 127
    colors = len({tuple(p) for p in np.asarray(rgba)[solid][:, :3]}) if solid.any() else 0
    fg = round(float(solid.mean()), 4)

    return {
        "model": model, "prompt_id": spec["id"], "preset": spec["preset"],
        "seed": seed, "size": size, "steps": steps, "cfg": cfg,
        "lora": Path(lora).name if lora else None, "lora_scale": lora_scale if lora else None,
        "block": block, "seconds": round(secs, 2),
        "subjects": subjects if subjects is not None else -1,
        "edge": edge, "edges": len(edge), "fg": fg,
        "white": _white_ratio(raw), "uniform": _uniformity(out, block),
        "colors": colors, "points": points if spec.get("humanoid") else None,
        "raw": raw, "out": out,
    }


def set_out(path: str = "") -> Path:
    """切换输出目录（--out）。用函数包住是为了避免 main 里出现两处 global。"""
    global OUT
    if path:
        OUT = Path(path)
        if not OUT.is_absolute():
            OUT = ROOT / path
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def load_rows(out_dir: Path) -> list[dict]:
    """从磁盘产物重建 rows（给 --rebuild 用，不跑模型）。"""
    rows: list[dict] = []
    for jf in sorted(out_dir.glob("*.json")):
        if jf.stem.endswith("__raw"):
            continue
        try:
            meta = json.loads(jf.read_text("utf-8"))
        except json.JSONDecodeError:
            continue
        png = jf.with_suffix(".png")
        if not png.exists():
            continue
        meta["out"] = Image.open(png).convert("RGBA")
        rows.append(meta)
    return rows


def contact_sheet(rows: list[dict], models: list[str], prompt_id: str,
                  cell: int = 300) -> Path:
    """列=基座，行=seed。放在一起看才比得出「多主体/糊/裁切」这类结构差异。"""
    seeds = sorted({r["seed"] for r in rows if r["prompt_id"] == prompt_id})
    head, cap = 30, 24                     # 顶部基座名 / 每行图注
    canvas = Image.new("RGB", (cell * len(models), head + (cell + cap) * len(seeds)),
                       (26, 26, 29))
    d = ImageDraw.Draw(canvas)
    for j, m in enumerate(models):
        d.text((j * cell + 8, 8), m, fill=(240, 240, 240))
    for i, s in enumerate(seeds):
        for j, m in enumerate(models):
            hit = next((r for r in rows
                        if r["prompt_id"] == prompt_id and r["seed"] == s
                        and r["model"] == m), None)
            if hit is None:
                continue
            x = j * cell
            y = head + i * (cell + cap)          # 图从表头之下开始，逐行下移
            bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
            tile = hit["out"].convert("RGBA")
            tile.thumbnail((cell, cell), Image.LANCZOS)
            bg.alpha_composite(tile, ((cell - tile.width) // 2, (cell - tile.height) // 2))
            canvas.paste(bg.convert("RGB"), (x, y))
            d.text((x + 6, y + cell + 3),
                   f"s{s} {'OK' if deliverable(hit) else 'NG'} subj{hit['subjects']} "
                   f"edge{hit['edges']} fg{hit['fg']:.2f}",
                   fill=(150, 180, 210))
    path = OUT / f"sheet_{prompt_id}.png"
    canvas.save(path)
    return path


def summarize(rows: list[dict], models: list[str]) -> tuple[str, list[dict]]:
    """按基座聚合，输出 markdown 与逐行明细。"""
    lines: list[str] = []
    table: list[dict] = []
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            continue
        n = len(rs)
        human = [r for r in rs if r["points"] is not None]
        agg = {
            "model": m, "n": n,
            "可交付率": round(sum(1 for r in rs if deliverable(r)) / n, 3),
            "多主体率": round(sum(1 for r in rs if r["subjects"] > 1) / n, 3),
            "贴边率": round(sum(1 for r in rs if r["edges"] > 0) / n, 3),
            "背景脏率": round(sum(1 for r in rs if r["white"] < 0.5) / n, 3),
            "块单色率": round(float(np.mean([r["uniform"] for r in rs])), 3),
            "前景占比": round(float(np.mean([r["fg"] for r in rs])), 3),
            "人均关节": (round(float(np.mean([r["points"] for r in human])), 2)
                     if human else None),
            "均耗时s": round(float(np.mean([r["seconds"] for r in rs])), 2),
        }
        table.append(agg)

    lines.append("| 基座 | 样本 | 可交付率↑ | 多主体率↓ | 贴边率↓ | 背景脏率↓ | 块单色率↑ | 前景占比 | 人均关节↑ | 均耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in table:
        pts = "—" if a["人均关节"] is None else f"{a['人均关节']:.1f}"
        lines.append(
            f"| {a['model']} | {a['n']} | {a['可交付率']:.0%} | {a['多主体率']:.0%} | {a['贴边率']:.0%} | "
            f"{a['背景脏率']:.0%} | {a['块单色率']:.3f} | {a['前景占比']:.2f} | "
            f"{pts} | {a['均耗时s']:.2f}s |")

    # 逐 prompt 的多主体率：换基座的收益通常就体现在这一列
    pids = [p["id"] for p in PROMPTS if any(r["prompt_id"] == p["id"] for r in rows)]
    if len(models) > 1 and pids:
        lines.append("")
        lines.append("**逐 prompt 多主体率**（越低越好，0% 才是产线可用）")
        lines.append("")
        lines.append("| prompt | " + " | ".join(models) + " |")
        lines.append("|---" * (len(models) + 1) + "|")
        for pid in pids:
            cells = []
            for m in models:
                sub = [r for r in rows if r["prompt_id"] == pid and r["model"] == m]
                cells.append(f"{sum(1 for r in sub if r['subjects'] > 1)}/{len(sub)}"
                             if sub else "—")
            lines.append(f"| {pid} | " + " | ".join(cells) + " |")
    return "\n".join(lines), table


def write_report(rows: list[dict], models: list[str], specs: list[dict],
                 args, elapsed: float) -> Path:
    """生成 report.md / summary.csv / 每 prompt 对比图。抽出来是为了 `--rebuild` 能复用。"""
    md, table = summarize(rows, models)
    sheets = [contact_sheet(rows, models, p["id"]) for p in specs
              if any(r["prompt_id"] == p["id"] for r in rows)]

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()) if table else ["model"])
        w.writeheader()
        w.writerows(table)

    seed_count = len({r["seed"] for r in rows})
    header = [
        "# 基座 A/B 报告", "",
        f"- 底模：{', '.join(models)}",
        f"- LoRA：{args.lora or '无'}（scale {args.lora_scale}）",
        f"- 额外负向词：{args.extra_neg or '无'}",
        (f"- 两段式：底分辨率 ×{args.hires}（hires fix）" if args.hires > 1 else ""),
        f"- 采样：cfg {args.cfg}，" + (
            f"**统一 {args.size}² / {args.steps} 步**（口径一致，可直接比权重）"
            if args.size and args.steps else
            "尺寸与步数按各基座架构档位（sd15 512/20、sdxl 1024/24）"
            "——跨架构的多主体率/关节数/耗时会混入尺寸因素"),
        f"- 样本：{len(specs)} prompt × {seed_count} seed × {len(models)} 基座 = {len(rows)} 张",
        f"- 总耗时：{elapsed / 60:.1f} 分钟", "",
        "## 汇总（↑ 越高越好 / ↓ 越低越好）", "",
        md, "",
        "## 判读提醒", "",
        "- **可交付率**是唯一的总指标：单主体 + 无贴边 + 前景占比 5%~60% 同时满足。",
        "- **多主体率**是硬门槛：>0 的基座在精灵管线里会持续产出废片，除非改提示词/加负向词。",
        "- **块单色率**低于 0.98 说明像素化被后续步骤搅浑（或该基座画得太碎）。",
        "- **人均关节**只看类人形 prompt；怪物/道具恒 0，横向比会误伤。",
        "- 客观表只管「能不能进产线」，好不好看要配合 sheet_*.png 人眼打分。",
        "- 同架构基座之间比才有意义；跨架构（sd15 vs sdxl）时尺寸/步数不同，耗时不具可比性。",
        "- **尺寸不同会同时影响多主体率与关节数**（分辨率越高越容易画全、姿态模型也越容易检出）。",
        "  要严格比权重，用 `--size 768 --steps 20` 把两者钉在同一口径上再跑一次。",
        "", "## 对比图", "",
    ] + [f"- `{p.name}`" for p in sheets]

    (OUT / "report.md").write_text("\n".join(header), encoding="utf-8")
    print("\n" + md)
    print(f"\n报告: {OUT / 'report.md'}")
    return OUT / "report.md"


def main() -> None:
    ap = argparse.ArgumentParser(description="基座 A/B 对照")
    ap.add_argument("--models", default="", help="逗号分隔；默认本机全部可用底模")
    ap.add_argument("--prompts", default="all", help="逗号分隔的 prompt id，或 all")
    ap.add_argument("--seeds", type=int, default=2, help="每个 prompt 用几个 seed")
    ap.add_argument("--size", type=int, default=0, help="0=按架构（sd15 512 / sdxl 1024）")
    ap.add_argument("--steps", type=int, default=0, help="0=按架构档位")
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--palette", type=int, default=32)
    ap.add_argument("--lora", default="", help="LoRA 名字或路径（如 px_illustrious）")
    ap.add_argument("--lora-scale", type=float, default=0.75)
    ap.add_argument("--extra-neg", default="",
                    help="额外负向词（测某个词能否压住某类问题）")
    ap.add_argument("--hires", type=float, default=0.0,
                    help=">1 时走两段式：底分辨率出结构 → 放大补细节")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 prompt（冒烟用）")
    ap.add_argument("--out", default="", help="输出目录（默认 assets/_ab_base）")
    ap.add_argument("--rebuild", action="store_true",
                    help="不重新生成，只用已有产物重建报告与对比图")
    args = ap.parse_args()

    if args.rebuild:
        set_out(args.out)
        rows = load_rows(OUT)
        if not rows:
            raise SystemExit(f"{OUT} 里没有产物可重建")
        models = [m.strip() for m in args.models.split(",") if m.strip()] or \
            sorted({r["model"] for r in rows})
        specs = [p for p in PROMPTS if any(r["prompt_id"] == p["id"] for r in rows)]
        print("重建:", write_report(rows, models, specs, args, 0.0))
        return

    from workflows import lora as lora_reg

    eng = Engine(ROOT / "models")
    have = eng.available_models()
    models = [m.strip() for m in args.models.split(",") if m.strip()] or have
    for m in models:
        if m not in have:
            raise SystemExit(f"底模 {m} 不可用（本机有：{', '.join(have)}）")

    specs = PROMPTS if args.prompts == "all" else \
        [p for p in PROMPTS if p["id"] in {x.strip() for x in args.prompts.split(",")}]
    if args.limit:
        specs = specs[:args.limit]
    if not specs:
        raise SystemExit("没有匹配的 prompt")

    lora = lora_reg.resolve(ROOT / "models", args.lora) if args.lora else None
    if args.lora and lora is None:
        raise SystemExit(f"LoRA 找不到：{args.lora}")
    if lora:
        # 注册表里声明了 base 就直接按它过滤：SD1.5 的 LoRA 不能用于 SDXL，
        # 硬跑只会在加载时报一堆 size mismatch。
        entry = next((x for x in lora_reg.scan(ROOT / "models")
                      if x["path"].endswith(Path(lora).name)), {}) or {}
        want = (entry.get("base") or "").strip()
        if want:
            keep = [m for m in models if eng.arch(m) == want]
            skipped = [m for m in models if m not in keep]
            if skipped:
                print(f"[skip] {', '.join(skipped)}：LoRA {Path(lora).name} 是给 "
                      f"{want} 训的，不能跨架构")
            models = keep or models

    set_out(args.out)
    seeds = [1000 + i * 7919 for i in range(max(1, args.seeds))]
    rows: list[dict] = []
    t_all = time.time()

    for m in models:
        size = args.size or DEFAULT_SIZE.get(eng.arch(m), 512)
        steps = args.steps or DEFAULT_STEPS.get(eng.arch(m), 20)
        for spec in specs:
            for seed in seeds:
                r = run_one(eng, m, spec, seed, size=size, steps=steps, cfg=args.cfg,
                            lora=lora, lora_scale=args.lora_scale, palette=args.palette,
                            extra_neg=args.extra_neg, hires=args.hires)
                tag = f"{m}__{spec['id']}__s{seed}"
                r["out"].save(OUT / f"{tag}.png")
                side = {k: v for k, v in r.items() if k not in ("raw", "out")}
                (OUT / f"{tag}.json").write_text(
                    json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
                r["raw"].save(OUT / f"{tag}__raw.png")
                rows.append(r)
                print(f"[{m}] {spec['id']} s{seed}: subj={r['subjects']} "
                      f"edge={r['edges']} fg={r['fg']:.2f} uni={r['uniform']} "
                      f"white={r['white']} {r['seconds']:.2f}s")
        eng.free()                        # 换基座前释放，避免两份底座同时驻留

    write_report(rows, models, specs, args, time.time() - t_all)



if __name__ == "__main__":
    main()
