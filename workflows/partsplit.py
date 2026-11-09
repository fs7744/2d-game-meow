"""角色立绘 → 拆部件 → **业界通用分层 PSD** + 分层预览清单。

两种拆法（用户在拆部件页二选一）：

1. `seethrough` —— See-Through（shitagaki-lab）语义分层：把一张静态动漫立绘
   拆成 Live2D drawable 层（face / topwear / handwear-l / hair…），**每层都补全
   被遮挡的区域**，按单目深度排序。它是独立环境：torch 2.11+cu128 + 17G 权重，
   跟本项目（torch 2.6 / Qwen）冲突，只能子进程跑。
   现成实现在 pixelforge 的 `tools/see_through_export.py`（本模块只负责调用、
   收产物、写统一清单），默认路径可用 `MEOW_PARTS_EXPORT` 覆盖。

2. `explode` —— 爆炸图 / 几何切件：用 `rigbuild.split_image()` 按人形比例切成
   头 / 躯干 / 左右臂 / 左右腿 6 块（纯 PIL，秒出、不需要额外模型）。
   质量上限比 See-Through 低（长发大裙摆会裂、道具会丢），但不需要等 3 分钟。

两种方法的产物都整理成同一份 `manifest.json`（层名 / 文件 / 画布坐标 / 顺序），
前端「PSD 标准分层预览」只认这份清单，不关心底层用的是哪种拆法。

**PSD 为什么要在子进程里写**：主环境没装 psd_tools（只有只读需求时也不必装），
而 see-through 的 venv 里有 `psd_tools 1.26`（支持 `PSDImage.new` +
`create_pixel_layer`）。所以爆炸图切完 PNG 后，由 `tools/parts_to_psd.py`
在 see-through 的 venv 里合成标准分层 PSD；那台机器没装 see-through 时
优雅降级：只出分层 PNG + 预览，`psd` 字段为 null 并在 `notice` 里说明。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from . import paths
from . import stages as stages_mod
from . import rigbuild

#: See-Through 仓库（独立 venv + 权重）
ST_DIR = Path(os.environ.get(
    "MEOWCUTE_SEETHROUGH_DIR", r"D:/code/gen/others/see-through-main"))
ST_PY = ST_DIR / ".venv" / "Scripts" / "python.exe"
#: 现成的导出脚本在 pixelforge 里（不复制一份，避免两处维护）
ST_EXPORT = Path(os.environ.get(
    "MEOW_PARTS_EXPORT", r"D:/code/pixelforge/tools/see_through_export.py"))

#: 爆炸图 6 块的绘制顺序（从底到顶）。头在最上，臂在躯干之前。
EXPLODE_ORDER = ("leg_l", "leg_r", "arm_l", "arm_r", "torso", "head")

ROLE_CN = {"head": "头", "torso": "躯干", "arm_l": "左臂",
           "arm_r": "右臂", "leg_l": "左腿", "leg_r": "右腿"}


class PartsError(RuntimeError):
    """拆部件失败（缺环境 / 缺权重 / 子进程报错）。"""


# ---------------------------------------------------------------- 路径 / 探测


def parts_root(name: str, stage_id: str = "") -> str:
    """拆件产物目录，**按生命阶段分**（跟 `Portraits/` 同一条规则）。

    - 默认阶段（`""` / `"default"`）→ `Parts/` 根，老产物原地不动；
    - 其它阶段 → `Parts/s_<id>/`，这样「成年 / 幼年」的拆件不会混在一块。

    用户 2026-11-09 要求 PSD 归到生命阶段下：拆部件本来就是「这个阶段这套设定」
    的产物，混在角色根目录里既分不清来源，也没法在阶段页里列出本阶段的历史。
    """
    root = paths.CHARACTERS_DIR / name / "Parts"
    sub = stages_mod.stage_dir_name(stage_id) if stage_id else ""
    return str(root / sub if sub else root)


def _url(p: str | Path) -> str:
    try:
        rel = Path(p).resolve().relative_to(paths.ASSETS.resolve())
    except (ValueError, OSError):
        return ""
    return "/files/" + "/".join(rel.parts)


def seethrough_available() -> Dict[str, Any]:
    """子进程能不能跑 See-Through（venv + 脚本 + 两份权重）。"""
    if not ST_PY.is_file():
        return {"ok": False, "reason": f"没找到 see-through 的 venv：{ST_PY}"}
    if not ST_EXPORT.is_file():
        return {"ok": False, "reason": f"没找到导出脚本：{ST_EXPORT}"}
    for sub in ("layerdiff3d", "marigold"):
        d = ST_DIR / "models" / sub
        if not d.is_dir():
            return {"ok": False, "reason": f"缺少权重目录：{d}"}
    return {"ok": True, "reason": ""}


def psd_available() -> bool:
    """写 PSD 需要 psd_tools —— 主环境没装，借用 see-through 的 venv。"""
    return ST_PY.is_file()


def available() -> Dict[str, Any]:
    st = seethrough_available()
    return {
        "seethrough": st,
        # 爆炸图永远可用（纯 PIL）；PSD 另说
        "explode": {"ok": True, "reason": ""},
        "psd": {"ok": psd_available(),
                "reason": "" if psd_available() else
                "没装 psd_tools：只能出分层 PNG，PSD 需要在 see-through 的 venv 里合成"},
        "seeThroughDir": str(ST_DIR),
    }


# ---------------------------------------------------------------- 产物清单


def _manifest_path(root: str, rid: str) -> str:
    return str(Path(root) / rid / "manifest.json")


def load_manifest(name: str, rid: str,
                  stage_id: str = "") -> Optional[Dict[str, Any]]:
    p = _manifest_path(parts_root(name, stage_id), rid)
    if not os.path.isfile(p):
        return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_manifest_at(folder: str, rid: str) -> Optional[Dict[str, Any]]:
    p = _manifest_path(folder, rid)
    if not os.path.isfile(p):
        return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rids_in(folder: str, reverse: bool = True) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [n for n in sorted(os.listdir(folder), reverse=reverse)
            if os.path.isdir(os.path.join(folder, n))]


def list_results(name: str,
                 stage_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """已拆过的产物（新的在前），按生命阶段过滤。只读磁盘，不跑模型。

    `stage_id=None` 扫**全部阶段**（诊断用），给了就只看那一个。
    产物按阶段落 `Parts/`（默认阶段）与 `Parts/s_<id>/`（其它阶段），所以
    默认阶段要**跳过 `s_*` 子目录**，否则别的阶段的产物会混进来。
    """
    base = parts_root(name)
    if not os.path.isdir(base):
        return []
    prefix = stages_mod._STAGE_DIR_PREFIX
    pairs: List[tuple] = []            # (rid, 所在目录)
    if stage_id is None:
        for n in _rids_in(base):
            full = os.path.join(base, n)
            if n.startswith(prefix):    # 阶段目录 → 里面的才是产物
                pairs += [(rid, full) for rid in _rids_in(full)]
            else:
                pairs.append((n, base))
    else:
        folder = parts_root(name, stage_id)
        pairs = [(rid, folder) for rid in _rids_in(folder)]

    out: List[Dict[str, Any]] = []
    for rid, folder in pairs:
        mf = _read_manifest_at(folder, rid)
        if not mf:
            continue
        out.append({
            "id": rid,
            "method": mf.get("method") or "",
            "methodLabel": "See-Through 语义分层" if mf.get("method") == "seethrough"
            else "爆炸图 · 几何切件",
            "created": mf.get("created") or 0,
            "seconds": mf.get("seconds"),
            "layers": len(mf.get("layers") or []),
            "psd": mf.get("psd"),
            "psdName": mf.get("psdName"),
            "preview": mf.get("preview"),
            "canvas": mf.get("canvas"),
            "source": mf.get("source"),
            "notice": mf.get("notice"),
            "stage": mf.get("stage") or (stage_id or "default"),
        })
    # 目录名带时间戳，但跨阶段合并后仍按 created 排一次更稳
    out.sort(key=lambda x: (x["created"], x["id"]), reverse=True)
    return out


# ---------------------------------------------------------------- 两种拆法


def _run(cmd: List[str], *, cwd: Optional[Path] = None,
         timeout: int = 3600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, timeout=timeout, env=env)


def _progress(job: Optional[Dict[str, Any]], pct: int, text: str) -> None:
    if job is None:
        return
    job["progress"] = {"done": int(pct), "total": 100, "current": text}


def _split_seethrough(src: str, out_dir: str, params: Dict[str, Any],
                      job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    av = seethrough_available()
    if not av["ok"]:
        raise PartsError(av["reason"])
    res = max(512, min(2048, int(params.get("resolution", 1280))))
    steps = max(10, min(60, int(params.get("steps", 30))))
    seed = max(-1, min(2**31 - 1, int(params.get("seed", 42))))
    if seed < 0:
        seed = int(time.time()) % (2**31 - 1)

    _progress(job, 5, "启动 See-Through（独立环境，首次加载约 1 分钟）…")
    cmd = [str(ST_PY), str(ST_EXPORT), "--ref", src, "--out", out_dir,
           "--resolution", str(res), "--steps", str(steps), "--seed", str(seed)]
    if params.get("lrsplit"):
        cmd.append("--lrsplit")
    proc = _run(cmd, cwd=ST_DIR)
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-1200:] + (proc.stderr or "")[-1200:]
        raise PartsError(f"See-Through 退出码 {proc.returncode}：{tail}")

    mf_path = Path(out_dir) / "manifest.json"
    if not mf_path.is_file():
        tail = (proc.stdout or "")[-800:] + (proc.stderr or "")[-800:]
        raise PartsError(f"See-Through 没有产出 manifest.json：{tail}")
    raw = json.loads(mf_path.read_text(encoding="utf-8"))

    # 它的 order 是「0 = 最上层」；统一改成**绘制顺序（0 = 最底）**，
    # 这样 PSD 合成与前端 z-index 都按同一个方向读。
    src_layers = sorted(raw.get("layers") or [],
                        key=lambda l: float(l.get("depth")
                                            if l.get("depth") is not None else 1.0))
    _progress(job, 92, "整理分层清单…")
    layers = []
    for i, l in enumerate(reversed(src_layers)):
        f = Path(out_dir) / "layers" / str(l.get("file") or "")
        layers.append({
            "name": l.get("name") or Path(str(l.get("file"))).stem,
            "file": str(Path("layers") / str(l.get("file") or "")).replace("\\", "/"),
            "url": _url(f),
            "x": int(l.get("x") or 0), "y": int(l.get("y") or 0),
            "width": int(l.get("width") or 0), "height": int(l.get("height") or 0),
            "box": l.get("box"),
            "depth": l.get("depth"),
            "order": i,                    # 0 = 最底
        })
    frame = raw.get("frameSize") or [0, 0]
    psd = Path(out_dir) / str(raw.get("psd") or "model.psd")
    preview = Path(out_dir) / "model_preview.png"
    return {
        "canvas": [int(frame[0]), int(frame[1])],
        "psdPath": str(psd) if psd.is_file() else None,
        "preview": str(preview) if preview.is_file() else None,
        "layers": layers,
        "seconds": raw.get("seconds") or 0,
        "notice": "",
        "options": {"resolution": res, "steps": steps, "seed": seed,
                    "lrsplit": bool(params.get("lrsplit"))},
    }


def _split_explode(src: str, out_dir: str, params: Dict[str, Any],
                   job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    _progress(job, 10, "按人形比例切件…")
    with Image.open(src) as im:
        img = im.convert("RGBA")
    props = params.get("proportions")
    res = rigbuild.split_image(
        img, proportions=props if isinstance(props, dict) else None,
        overlap=max(0, min(32, int(params.get("overlap", 3)))))
    order = [r for r in EXPLODE_ORDER if r in res["parts"]]
    layer_dir = Path(out_dir) / "layers"
    layer_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    for i, role in enumerate(order):
        part: Image.Image = res["parts"][role]
        box = res["regions"][role]["box"]
        f = layer_dir / f"{role}.png"
        part.save(f, format="PNG")
        layers.append({
            # 预览里给中文；**PSD 层名必须 ASCII** —— psd-tools 写层名走
            # mac_roman 编码，中文会直接抛 charmap 异常（踩过）。
            "name": f"{ROLE_CN.get(role, role)}（{role}）",
            "psdName": role,
            "role": role,
            "file": f"layers/{role}.png",
            "url": _url(f),
            "x": int(box[0]), "y": int(box[1]),
            "width": part.width, "height": part.height,
            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "depth": None,
            "order": i,
        })
    _progress(job, 70, "预览合成…")
    # 一张整图预览（给历史列表当封面；PSD 里的层本身才是交付物）
    preview = Path(out_dir) / "model_preview.png"
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for l in layers:
        with Image.open(Path(out_dir) / l["file"]) as part:
            canvas.paste(part.convert("RGBA"), (l["x"], l["y"]), part)
    canvas.save(preview, format="PNG")

    psd_path = None
    notice = ""
    if psd_available():
        _progress(job, 80, "合成标准分层 PSD…")
        target = Path(out_dir) / "model.psd"
        partial = Path(out_dir) / "manifest.partial.json"
        partial.write_text(json.dumps({
            "canvas": [img.width, img.height],
            "layers": [{"name": l.get("psdName") or l["name"],
                        "file": l["file"], "x": l["x"], "y": l["y"]}
                       for l in layers],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        proc = _run([str(ST_PY), str(_to_psd_tool()), "--manifest",
                     str(partial), "--out", str(target)], cwd=Path(out_dir))
        if proc.returncode == 0 and target.is_file():
            psd_path = str(target)
        else:
            notice = ("PSD 合成失败，已保留分层 PNG："
                      + ((proc.stderr or proc.stdout or "")[-300:]))
    else:
        notice = "没装 psd_tools（see-through 的 venv 不可用），只出分层 PNG"
    return {
        "canvas": [img.width, img.height],
        "psdPath": psd_path,
        "layers": layers,
        "seconds": 0,
        "notice": notice,
        "preview": str(preview),
        "options": {"overlap": int(params.get("overlap", 3))},
    }


def _to_psd_tool() -> Path:
    return Path(__file__).resolve().parents[1] / "tools" / "parts_to_psd.py"


# ---------------------------------------------------------------- 入口


def split(name: str, src: str, method: str,
          params: Optional[Dict[str, Any]] = None,
          job: Optional[Dict[str, Any]] = None,
          stage_id: str = "") -> Dict[str, Any]:
    """拆一张立绘，产物落该阶段的 `Parts[/s_<id>]/<时间戳_方法>/`。"""
    method = str(method or "").strip().lower()
    if method not in ("seethrough", "explode"):
        raise PartsError(f"不支持的拆法：{method}")
    p = params or {}
    root = parts_root(name, stage_id)
    os.makedirs(root, exist_ok=True)
    rid = time.strftime("%Y%m%d_%H%M%S") + "_" + method
    out_dir = str(Path(root) / rid)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    if method == "seethrough":
        res = _split_seethrough(src, out_dir, p, job)
    else:
        res = _split_explode(src, out_dir, p, job)

    psd_path = res.get("psdPath")
    manifest: Dict[str, Any] = {
        "id": rid,
        "method": method,
        # 这份产物属于哪个生命阶段（列表页据此分组，拆件页也据此过滤）
        "stage": (str(stage_id)
                  if stages_mod.stage_dir_name(stage_id) else "default"),
        "source": _url(src),
        "sourceName": Path(src).name,
        "canvas": res["canvas"],
        "psd": _url(psd_path) if psd_path else None,
        "psdName": Path(psd_path).name if psd_path else None,
        "preview": _url(res.get("preview", "")) if res.get("preview") else None,
        "created": time.time(),
        "seconds": round(res.get("seconds") or (time.time() - started), 1),
        "notice": res.get("notice") or "",
        "options": res.get("options") or {},
        # 从底到顶；order = 索引（0 = 最底）
        "layers": res["layers"],
    }
    # 爆炸图的 PSD 合成工具要读一份 partial 清单（层文件路径 + 画布）
    partial = Path(out_dir) / "manifest.partial.json"
    if partial.is_file():
        try:
            partial.unlink()
        except OSError:
            pass
    Path(out_dir, "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(job, 100, "完成")
    return manifest


def split_job(_engine: Any = None, params: Optional[Dict[str, Any]] = None,
              job: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """worker 入口。`_run` 对 `wparts` 走特例分支（跟 posebatch 一样），
    这样才拿得到 `job` 用来上报阶段进度。"""
    p = dict(params or {})
    return split(
        name=str(p.get("character") or ""),
        src=str(p.get("src") or ""),
        method=str(p.get("method") or ""),
        params=p.get("params") or {},
        job=job,
        stage_id=str(p.get("stage") or ""),
    )


def delete_result(name: str, rid: str, stage_id: str = "") -> bool:
    """删一次拆件产物（整目录）。路径先做 realpath 校验，挡 `..`。"""
    root = os.path.realpath(parts_root(name, stage_id))
    target = os.path.realpath(str(Path(root) / rid))
    if not (target == root or target.startswith(root + os.sep)) or target == root:
        raise PartsError("只能删这个角色 Parts/ 下的产物")
    if not os.path.isdir(target):
        return False
    shutil.rmtree(target, ignore_errors=True)
    return True
