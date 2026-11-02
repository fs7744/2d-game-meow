"""多姿势批量生成（官方 `character_generator.py:2108-2223` 的 QI2 分支）。

**跟画布上那个「角色迁移」节点不是一回事**：那个是单张、带我们自己的画布/骨架
处理；这里是**一整批**（N 张姿势 → N 张精灵帧），而且发布方式不同。

三处必须照官方的：

| 项 | 官方 | 为什么不能改 |
|---|---|---|
| 指令 | `Replace the pose of <image 2> with the pose of <image 1>. Keep the character of <image 2>. {prompt}` | 顺序反了就是把姿势套到骨架上 |
| 种子 | **整批共用一个**（`pose_sampler.inherit_pipe=True`） | 这是姿势之间外观一致的关键；每张随机种子会让同一套动作的衣服/发型飘 |
| 采样 | `steps 20 / cfg 1.0 / denoise 1.0`（`:825-833`） | cfg=1 是单次前向，20 步是这套 LoRA 的训练点 |

**发布走整批替换**（`staged_image_batch`，旧图进 `V1`），不是逐张追加——
这正是官方 `_save_final_sprites:2740` 的语义（一次出一整批 = 一套新的姿势组）。
逐张追加是表情/换装那条路（`publish_image`）。

串行 for 循环而不是 batch：QI2 的 encode→sample→decode 是逐张的（`:2205`），
batch 只会把显存顶爆，速度不会快——每张的 latent 尺寸可能都不一样。
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from PIL import Image

from workflows import characters

#: 官方 `pose_sampler` 默认（`:825-833`）
POSE_DEFAULTS = {"steps": 20, "cfg": 1.0, "target_size": 1024}


class PoseGenError(RuntimeError):
    """多姿势生成的可读错误（API 转 400）。"""


def _resolve_pose_paths(poses: Sequence[str], pose_dir: str = "") -> List[str]:
    """把前端给的姿势来源收敛成一组**真实存在的图片路径**。

    支持两种：`poses` 直接给路径列表；或给一个 `pose_dir`（我们自己的
    `/pose_dirs` 扫出来的骨架帧目录、官方姿势库导出目录都行）。
    """
    from workflows import paths

    out: List[str] = []
    for raw in poses or []:
        p = str(raw or "").strip()
        if not p:
            continue
        cand = p
        if p.startswith("/files/"):
            cand = p[len("/files/"):]
        path = cand if os.path.isabs(cand) else str(paths.ASSETS / cand)
        if not os.path.isfile(path):
            # 退一步：相对 assets 的文件名；再退一步：相对当前工作目录
            # （tools/ 下的脚本与手工 curl 常直接给相对路径）
            for alt in (str(paths.ASSETS / os.path.basename(cand)), cand):
                if os.path.isfile(alt):
                    path = alt
                    break
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in characters.IMAGE_EXTS:
            out.append(path)
    if not out and pose_dir:
        d = str(pose_dir)
        if not os.path.isdir(d):
            d = str(paths.ASSETS / d)
        if os.path.isdir(d):
            out = [os.path.join(d, n) for n in sorted(os.listdir(d))
                   if os.path.splitext(n)[1].lower() in characters.IMAGE_EXTS
                   and os.path.isfile(os.path.join(d, n))]
    # 去重（同一个文件被前端重复塞进来时不要出两遍）
    seen = set()
    uniq = []
    for p in out:
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def resolve_pose_paths(poses: Sequence[str], pose_dir: str = "") -> List[str]:
    """公开入口（`app.py` / `pipeline.py` 都调它，别直接摸下划线版本）。"""
    return _resolve_pose_paths(poses, pose_dir)


#: 官方 `character_generator.py:988`（透明底那句，跟换装/表情共用同一句）
NATIVE_BACKGROUND_PROMPT = "Transparent background with alpha channel."


def prompt_with_solid_background(prompt: str, background: str,
                                 transparent: bool = False) -> str:
    """官方 `_prompt_with_solid_background:1827-1843`（**逐字**）。

    姿势生成这条会往 prompt 后面追一句"把背景改成纯 X 色"。**必须要**：
    姿势图和人偶图本身带着白底，不指定背景时出来就是白底——抠像没法定阈值，
    跟其它帧也拼不到一起。

    `transparent`（或 `background` 本身就是 alpha/transparent）时改追透明底那句。
    """
    text = str(prompt or "").strip()
    is_alpha = transparent or str(background or "").strip().lower() in {"alpha", "transparent"}
    if is_alpha:
        if NATIVE_BACKGROUND_PROMPT.lower() in text.lower():
            return text
        return f"{text}, {NATIVE_BACKGROUND_PROMPT}" if text else NATIVE_BACKGROUND_PROMPT
    bg = str(background or "").strip()
    if not bg:
        return text
    instruction = f"Change background to solid {bg} color"
    if instruction.lower() in text.lower():
        return text
    return f"{text}, {instruction}" if text else instruction


def _prompt_for(index: int, prompts: Any, total: int) -> str:
    """官方 `_pose_prompts:1240-1243`：可以是"每张一句话"的列表，也可以是一句话通吃。

    列表短了就重复最后一条（官方的 `min(index, len-1)`），不是报错——
    用户常常只想给前几张写不同的描述。
    """
    if isinstance(prompts, (list, tuple)):
        values = [str(v or "") for v in prompts] or [""]
        return values[min(index, len(values) - 1)]
    return str(prompts or "")


def run_batch(
    engine: Any,
    params: Dict[str, Any],
    *,
    job: Optional[Dict[str, Any]] = None,
    pose_fn: Optional[Callable[..., Image.Image]] = None,
    matte_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> Dict[str, Any]:
    """N 张姿势 → N 张精灵帧，整批替换 `Sprites/<服装>/neutral/`。

    `params`：character / costume / poses / pose_dir / prompts / steps / cfg /
    seed / transparent / matte / lora
    """
    name = characters.ensure_safe_name(str(params.get("character") or ""), "角色名")
    costume = characters.ensure_safe_name(str(params.get("costume") or "Naked"), "服装名")
    if not characters.character_exists(name):
        raise PoseGenError(f"没有这个角色：{name}")

    pose_paths = _resolve_pose_paths(params.get("poses") or [], str(params.get("pose_dir") or ""))
    if not pose_paths:
        raise PoseGenError("没有可用的姿势图：给几张姿势图，或指定一个骨架帧目录")

    # 角色底图：**显式指定 source_costume 时只认它**，否则按「这套服装 → Naked」退。
    #
    # `source_costume` 是给编排用的：一键出全套时先跑「换白内衣」得到 Underwear
    # 底图，再让姿势从**那张**出发（而不是从上一次遗留的服装底图出发）。
    # 官方的 `naked_pose_generation` 就是从 `remove_clothes` 的产物出发的
    # （`character_generator.py:3149-3175`）。
    source_costume = str(params.get("source_costume") or "").strip()
    char_path = None
    if source_costume:
        char_path = characters.latest_sprite_path(name, source_costume)
        if not char_path:
            raise PoseGenError(
                f"「{name}」的「{source_costume}」没有底图："
                "编排里先跑的那一步没产出图，或者服装名写错了")
    if not char_path:
        char_path = (characters.latest_sprite_path(name, costume)
                     or characters.latest_sprite_path(name, "Naked"))
    if not char_path:
        raise PoseGenError(
            f"「{name}」还没有角色底图：先在「出图」页生成一张，或者先换装")

    steps = int(params.get("steps") or POSE_DEFAULTS["steps"])
    cfg = float(params.get("cfg") or POSE_DEFAULTS["cfg"])
    seed = params.get("seed")
    # **整批一个种子**（官方 `inherit_pipe=True`）：每张随机会让同一套动作的
    # 衣服/发型/画风互相飘。
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(
        os.urandom(3), "big")
    # 背景：参数优先，否则用角色自己的 background_color（没有就不追这句）
    info = characters.load_config(name) or {}
    background = str(params.get("background_color")
                     or (info.get("character_info") or {}).get("background_color") or "")
    # **透明判定的优先级**：
    #   1. 显式传了 `transparent` → 听它；
    #   2. 没传 → 看背景色：实色幕布（绿/蓝/白/灰/品红）就出实色图，
    #      其余（空 / alpha / transparent）走 QI2 原生 alpha；
    #   3. 两边都没有 → 默认原生透明（不再默认出绿幕再抠）。
    # 以前这里默认 False，等于强制"绿幕 + 抠图"多一道工序，边缘还会被啃。
    _tr = params.get("transparent")
    if _tr is None:
        _bg = background.strip().lower()
        transparent = _bg not in {"green", "blue", "white", "grey", "magenta"}
    else:
        transparent = bool(_tr)
    do_matte = bool(params.get("matte"))

    if pose_fn is None:
        from workflows import generate as gen

        def _fn(pose: Image.Image, char: Image.Image, prompt: str) -> Image.Image:
            return gen.qwen_pose_image(
                engine, pose=pose, char=char, prompt=prompt, seed=seed,
                steps=steps, cfg=cfg, transparent=transparent,
                lora=str(params.get("lora") or "") or None)
        pose_fn = _fn
    if matte_fn is None:
        from workflows import matte_screen
        matte_fn = matte_screen.matte_pil

    total = len(pose_paths)
    with Image.open(char_path) as opened:
        char_img = opened.convert("RGB")

    results: List[Image.Image] = []
    items: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    started = time.time()
    for index, pose_path in enumerate(pose_paths):
        if job is not None:
            job["progress"] = {"done": index, "total": total,
                               "current": f"第 {index + 1}/{total} 张姿势"}
        t0 = time.time()
        try:
            with Image.open(pose_path) as opened:
                pose_img = opened.convert("RGB")
            prompt = prompt_with_solid_background(
                _prompt_for(index, params.get("prompts"), total),
                background, transparent)
            image = pose_fn(pose_img, char_img, prompt)
            if image is None:
                raise PoseGenError("出图没有返回图像")
            matte_error = ""
            if do_matte:
                try:
                    cleaned = matte_fn(image)
                    if cleaned is not None:
                        image = cleaned
                except Exception as exc:              # noqa: BLE001
                    # 抠像失败不丢图：图还在，只是没透明
                    matte_error = str(exc)
            results.append(image)
            items.append({"index": index + 1, "pose": pose_path,
                          "size": list(image.size), "matte_error": matte_error,
                          "seconds": round(time.time() - t0, 2)})
        except Exception as exc:                      # noqa: BLE001
            failed.append({"index": index + 1, "pose": pose_path,
                           "error": f"{type(exc).__name__}: {exc}"})
        if job is not None:
            job["progress"]["done"] = index + 1

    if not results:
        raise PoseGenError(f"一张都没生成出来：{failed[0]['error'] if failed else '未知原因'}")

    # **整批替换**：旧图进 V1（官方 `_save_final_sprites:2740` 的 staged_image_batch）
    target = characters.sprites_dir(name, costume, "neutral")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    written: List[str] = []
    with characters.staged_image_batch(target, version_existing=True) as stage:
        for i, image in enumerate(results, start=1):
            fname = f"sprite_pose_{i:04d}.png"
            image.save(os.path.join(stage, fname), format="PNG")
            written.append(os.path.join(target, fname))

    if job is not None:
        job["progress"] = {"done": total, "total": total, "current": ""}
    return {
        "character": name, "costume": costume,
        "source": char_path, "count": len(written),
        "sprites": written, "poses": pose_paths[:len(written)],
        "seed": seed, "steps": steps, "cfg": cfg, "background": background,
        "items": items, "failed": failed,
        "seconds": round(time.time() - started, 2),
    }
