"""LoRA 注册表：扫描 models/lora/*.safetensors，合并人工元数据。

为什么需要元数据：Civitai 下下来的 safetensors 里只有权重，没有「触发词」和
「推荐强度」。触发词不知道就等于白装（实测 xuer 的触发词是 `xuer martial arts`，
不写这个词它只输出普通照片）。所以元数据落在同目录的 `loras.json`，
以文件名为键，缺失时用保守默认值。

清单是**可选**的：文件存在就能用，元数据只是改良。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

LORA_DIR = "lora"
META_FILE = "loras.json"

_DEFAULT: Dict[str, Any] = {
    "label": None,
    "trigger": "",
    "scale": 0.8,
    "preset": None,
    "note": "",
    "source": "",
    # 选了这个 LoRA 后节点该跟着改的开关。实测像素 LoRA 都会在背景上画
    # 彩色/纹理底，不抠图就没法用，所以允许注册表声明「必须抠图」。
    "matte": None,        # None=不改动用户的勾选；True/False=强制
    "clip_skip": None,    # None=用节点当前值；2=多数动漫/像素 LoRA 的训练设置
    # 这个 LoRA 是给哪个架构训的（sd15 / sdxl）。SD1.5 的 LoRA 装进 SDXL 会
    # 因维度不同（640 vs 1280）报 size mismatch——那不是"权重坏了"，
    # 是拿错了基座。声明了就能在选之前拦住。
    "base": "",
}


def scan(models_root: str | Path) -> List[Dict[str, Any]]:
    """列出可用 LoRA（按 label 排序）。路径相对于 models 根目录。"""
    root = Path(models_root)
    d = root / LORA_DIR
    if not d.is_dir():
        return []

    meta: Dict[str, Any] = {}
    mf = d / META_FILE
    if mf.exists():
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    out: List[Dict[str, Any]] = []
    for f in sorted(d.glob("*.safetensors")):
        info = dict(_DEFAULT)
        info.update(meta.get(f.stem) or {})
        info["name"] = f.stem
        info["label"] = info["label"] or f.stem
        info["path"] = f"{LORA_DIR}/{f.name}"
        info["size_mb"] = round(f.stat().st_size / 2**20, 1)
        out.append(info)

    return sorted(out, key=lambda x: (x["label"] or "").lower())


def resolve(models_root: str | Path, name: str | None) -> str | None:
    """把 LoRA 名（stem 或相对路径）解析成绝对路径；空值返回 None。"""
    if not name:
        return None
    root = Path(models_root)
    cand = root / name if "/" in name or "\\" in name else root / LORA_DIR / f"{name}.safetensors"
    if cand.exists():
        return str(cand)
    # 也接受直接给绝对路径
    p = Path(name)
    return str(p) if p.exists() else None

# ---------------- 自动识别（读 safetensors header，不加载权重） ----------------

#: 各架构的**文本上下文维度**。这是最可靠的判据：
#: SD1.5 的 CLIP context 是 768，SDXL 是 2048，Flux/anima 是 4096 以上，
#: Wan 是 5120。比 metadata 里的 `ss_base_model_version` 可靠——实测
#: octopath 的 metadata 明明写着 `stable-diffusion-v1/lora`，tensor 却是 DiT。
_CTX_DIM = {"sd15": 768, "sdxl": 2048}


def _read_header(path: Path) -> Dict[str, Any]:
    """只读 safetensors 头部（8 字节长度 + JSON），不碰权重数据。

    170MB 的文件读 header 是毫秒级——这是"扫描不加载模型"的关键。
    """
    import json as _json
    import struct

    with path.open("rb") as fh:
        head = fh.read(8)
        if len(head) < 8:
            raise ValueError("不是合法的 safetensors")
        n = struct.unpack("<Q", head)[0]
        if n > 100 * 2**20:                     # header 不会这么大，防坏文件
            raise ValueError("header 异常大，可能不是 safetensors")
        return _json.loads(fh.read(n))


def probe(path: str | Path) -> Dict[str, Any]:
    """推断一个 LoRA 的架构、rank、触发词。**不加载权重**。

    返回 `{"base": "sd15"|"sdxl"|"flux"|"wan"|"", "rank": int|None,
          "trigger": str, "note": str, "supported": bool}`。

    为什么必须自动判：架构拿错的报错是 `size mismatch for ... 640 vs 1280`，
    用户看到只会以为"文件坏了"，然后去换一个同样错的 LoRA。
    """
    p = Path(path)
    try:
        head = _read_header(p)
    except Exception as exc:                     # noqa: BLE001
        return {"base": "", "rank": None, "trigger": "",
                "note": f"读不了头部（{exc}）", "supported": False}

    md = head.get("__metadata__") or {}
    keys = [k for k in head if k != "__metadata__"]

    # --- 1. 架构：先看键名，再看维度 ---
    # DiT（Flux / anima / SD3）的 LoRA 键里有 adaln / double_blocks
    joined = " ".join(keys[:400])
    is_dit = ("adaln_modulation" in joined or "double_blocks" in joined
              or "single_blocks" in joined)
    # Wan 等视频模型：顶层就是 diffusion_model.，且带 cross_attn
    is_video = (any(k.startswith("diffusion_model.") for k in keys[:50])
                and "cross_attn" in joined)

    # 三种命名都要认：kohya `lora_down`、新版 kohya/diffusers `lora.down`、
    # PEFT `lora_A`。只认一种会让 mk_sp_pixel（diffusers 格式）识别失败。
    _DOWN = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight")
    dims = set()
    for k, v in head.items():
        if k == "__metadata__":
            continue
        if not k.endswith(_DOWN):
            continue
        shape = v.get("shape") or []
        if len(shape) == 2:
            dims.add(int(shape[1]))

    base = ""
    if is_video:
        base = "wan"
    elif is_dit:
        base = "flux"
    elif dims:
        # 文本上下文维度是硬指纹：SD1.5=768，SDXL=2048
        if _CTX_DIM["sd15"] in dims and _CTX_DIM["sdxl"] not in dims:
            base = "sd15"
        elif _CTX_DIM["sdxl"] in dims and _CTX_DIM["sd15"] not in dims:
            base = "sdxl"
        elif _CTX_DIM["sd15"] in dims:
            # 两个都有：多半是 SD1.5 的 UNet 通道恰好撞上 2048，按 768 算
            base = "sd15"

    if not base:
        # 没有标准 lora_down/lora_A 张量时可读的东西（LoKr/LoHa 变体就是这样），
        # 退回训练元数据。**只在这种时候用**：metadata 会说谎（octopath 的
        # modelspec.architecture 写着 stable-diffusion-v1，实际是 DiT）。
        hint = " ".join(str(md.get(k, "")) for k in
                        ("ss_sd_model_name", "modelspec.architecture",
                         "ss_base_model_version")).lower()
        if "sdxl" in hint or "stable-diffusion-xl" in hint:
            base = "sdxl"
        elif "stable-diffusion-v1" in hint or "_v1-" in hint or "sd_v1" in hint:
            base = "sd15"

    supported = base in ("sd15", "sdxl")

    # --- 2. rank：ss_network_dim 或从 shape 反推 ---
    rank = None
    dim = str(md.get("ss_network_dim", "")).strip()
    if dim.isdigit():
        rank = int(dim)

    # --- 3. 触发词：kohya 的 ss_tag_frequency 里最高频 tag ---
    trigger = ""
    raw = md.get("ss_tag_frequency")
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            best, best_n = "", 0
            for _bucket, tags in (data or {}).items():
                for tag, n in (tags or {}).items():
                    if int(n) > best_n and len(tag) <= 40:
                        best, best_n = tag, int(n)
            trigger = best
        except Exception:                        # noqa: BLE001
            trigger = ""

    note = ""
    if base == "wan":
        note = "这是视频模型（Wan）的 LoRA，不能用于 SD1.5/SDXL 出图"
    elif base == "flux":
        note = "这是 DiT 架构（Flux / anima 系）的 LoRA，不能用于 SD1.5/SDXL"
    elif base == "":
        note = "识别不出架构（既没有 768 也没有 2048 的上下文维度）"

    return {"base": base, "rank": rank, "trigger": trigger,
            "note": note, "supported": supported}


def _fingerprint(path: Path) -> str:
    """文件指纹：大小 + 头部内容。用来发现"同一个 LoRA 存了两份"。"""
    import hashlib

    try:
        head = _read_header(path)
    except Exception:                            # noqa: BLE001
        return ""
    body = json.dumps({k: v.get("shape") for k, v in head.items() if k != "__metadata__"},
                      sort_keys=True)
    return hashlib.sha1(f"{path.stat().st_size}:{body}".encode()).hexdigest()


def autoscan(models_root: str | Path, *, write: bool = True,
             fill_trigger: bool = False) -> Dict[str, Any]:
    """扫描 `models/lora/*.safetensors`，把缺的配置补进 `loras.json`。

    规则（很重要，决定了它能不能反复跑）：
      - **不覆盖**用户手写过的字段。只补空缺（base / trigger / label / note）。
      - 识别不出架构的文件仍会登记，但标 `hidden=True` 并写明原因——
        让它出现在下拉里等你选中再报错，是更差的体验。
      - 同一份权重存成两个文件名时，第二个标 hidden 并指向第一个。
      - **默认不自动填触发词**（`fill_trigger=False`）：空 trigger 有两种含义，
        "不知道"和"知道但故意不用"（实测像素 LoRA 加触发词后 3/3 多主体）。
        自动填会把后者覆盖掉，等于把一条实测结论抹了。要填得显式要求。
    """
    root = Path(models_root)
    d = root / LORA_DIR
    out: Dict[str, Any] = {"added": [], "filled": [], "hidden": [], "duplicates": [],
                           "stale": [], "same_content": [], "total": 0, "written": False}
    if not d.is_dir():
        return out

    mf = d / META_FILE
    config: Dict[str, Any] = {}
    if mf.exists():
        try:
            config = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}

    files = sorted(d.glob("*.safetensors"))
    out["total"] = len(files)

    # 先算全量指纹：**已有配置的优先当"正本"**，后加的同内容文件让位。
    # 否则先按文件名排序会把 pixel-art-xl 当正本、已有的 pixelart_xl 当副本，
    # 而后者才有完整的 label/note/preset。
    def _canon_score(name: str):
        """谁是"正本"：没被标 hidden 的、配置更完整的优先。

        不按文件名排序定正本会翻车：pixel-art-xl 排在 pixelart_xl 前面，
        而后者才有完整的实测配置——按名字定会把有用的那份标成副本。
        """
        e = config.get(name) or {}
        return (0 if e.get("hidden") else 1,
                1 if e.get("note") else 0,
                1 if e.get("preset") else 0)

    groups: Dict[str, list] = {}
    for f in files:
        fp = _fingerprint(f)
        if not fp:
            continue
        groups.setdefault(fp, []).append(f.stem)
    fps: Dict[str, str] = {}
    for fp, names in groups.items():
        best = sorted(names, key=lambda n: (-_canon_score(n)[0], -_canon_score(n)[1],
                                            -_canon_score(n)[2], n))[0]
        fps[fp] = best
        if len(names) > 1:
            out["same_content"].append({"kept": best,
                                        "also": sorted(n for n in names if n != best)})

    for f in files:
        name = f.stem
        info = probe(f)
        entry = config.get(name)
        fp = _fingerprint(f)
        dup_of = fps.get(fp) if fp else None
        if dup_of == name:
            dup_of = None

        if entry is None:
            # 新文件：生成一份保守但完整的条目
            entry = {
                "label": name,
                "trigger": info["trigger"] if fill_trigger else "",
                "scale": 0.8,
                "preset": None,
                "note": info["note"],
                "source": "auto",
                "base": info["base"],
            }
            if not info["supported"]:
                entry["hidden"] = True
            if dup_of:
                entry["hidden"] = True
                entry["note"] = f"与 {dup_of} 是同一份权重（内容指纹相同）"
                out["duplicates"].append({"name": name, "same_as": dup_of,
                                          "kept": False})
            if info["rank"]:
                entry["rank"] = info["rank"]
            config[name] = entry
            out["added"].append(name)
            if entry.get("hidden"):
                out["hidden"].append(name)
            continue

        # 已配置：**只补空缺，绝不改用户的判定**。
        # 重复只报告不动配置——用户可能有意留两份（比如一份改了精度），
        # 自动把 hidden 打开会让"本来能用的 LoRA 突然不见了"。
        if dup_of and not entry.get("hidden"):
            out["duplicates"].append({"name": name, "same_as": dup_of, "kept": True})
        filled = []
        if not entry.get("base") and info["base"]:
            entry["base"] = info["base"]
            filled.append("base")
        if fill_trigger and not entry.get("trigger") and info["trigger"]:
            entry["trigger"] = info["trigger"]
            filled.append("trigger")
        if filled:
            out["filled"].append({"name": name, "fields": filled})

    # 配置里有、文件已经删掉的条目：只报告不删（可能是暂时移走了权重）
    names = {f.stem for f in files}
    out["stale"] = sorted(k for k in config if k not in names)

    if write:
        mf.write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")
        out["written"] = True
    return out


def format_scan(result: Dict[str, Any]) -> str:
    """把 autoscan 的结果排成人话（CLI 与 API 共用同一套文案）。"""
    lines = [f"扫描 {result['total']} 个权重"]
    if result["added"]:
        lines.append(f"新增配置 {len(result['added'])} 条：{', '.join(result['added'])}")
    if result["filled"]:
        detail = "；".join(f"{x['name']}（{'+'.join(x['fields'])}）" for x in result["filled"])
        lines.append(f"补全字段：{detail}")
    for d in result["duplicates"]:
        if d.get("kept"):
            lines.append(f"重复（已保留原配置，未改动）：{d['name']} 与 {d['same_as']}")
        else:
            lines.append(f"重复（已隐藏）：{d['name']} 与 {d['same_as']} 是同一份权重")
    for d in result.get("same_content") or []:
        lines.append(f"同一份权重存了多份：{d['kept']}（另有 {', '.join(d['also'])}），可删多余的那份")
    hidden = [h for h in result["hidden"]]
    if hidden:
        lines.append(f"不可用（已隐藏）：{', '.join(hidden)}")
    if result["stale"]:
        lines.append(f"配置里有但文件已不在：{', '.join(result['stale'])}")
    if not (result["added"] or result["filled"] or result["duplicates"]):
        lines.append("没有需要改的，配置已是最新")
    return "\n".join(lines)


if __name__ == "__main__":              # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="扫描 models/lora 并补全 loras.json")
    ap.add_argument("--models", default="models", help="models 根目录（默认 models）")
    ap.add_argument("--dry-run", action="store_true", help="只看结果，不写回")
    ap.add_argument("--fill-trigger", action="store_true",
                    help="把训练元数据里的高频 tag 填成触发词（默认不填，见 autoscan 说明）")
    a = ap.parse_args()
    res = autoscan(a.models, write=not a.dry_run, fill_trigger=a.fill_trigger)
    print(format_scan(res))
