"""换装（Clothes Designer）—— 移植 VNCCS 的「描述 → 换装底图」那一步。

**它跟"只存服装描述，下次拼 prompt"完全不是一回事**：官方是把五个字段（上装/
下装/头饰/鞋/脸部穿戴物）拼成一条图生图指令，拿**裸装或原始底图**当输入，
生成一张**换装后的底图**；之后所有姿势/表情都从这张底图出发。所以衣服是
"画进图里"的，不是"文字里提一句"。

来源（相对 `ComfyUI_VNCCS-main`）：

| 这里 | 官方 |
|---|---|
| `construct_clothes_prompt` | `clothes_designer.py:413-448`（在 charactergen，逐字） |
| `clothes_background_prompt` | `:124-135`（逐字） |
| `run_design` | `:493-790` 的 QI2 分支 |
| 底图选择 | `:191-232` + `:464-483`（Naked 优先，退回 Original） |
| Clothes Core LoRA | `:286,301,711-716`（强度 1.0，无触发词） |
| 采样 | `:555-556`（steps 25 / cfg 3.0）、`:163`（target_size 1024） |

**没搬的**：
- 官方的**预览缓存**（`cache/preview_<costume>.png` + 一个 info json 记录
  "prompt/seed/输入哈希"做命中判断）。我们照样存 preview，但**不做命中复用**：
  我们的出图是显式点一次跑一次，缓存反而让"改了描述却没变"更难排查。
- `qi2_prompt_enhancement`（用图像模型自己改写换装 prompt）。我们的
  `charactergen.enhance_character_prompt` 是给**造角色**用的；换装这条先用
  原样 prompt，等实际出图觉得不够再按官方 `_rewrite_qi2_clothes_prompt:138-159`
  接（那段还要求"改写完把 constraints 追回去"，不能丢背景约束）。
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable, Dict, Optional

from PIL import Image

from workflows import characters
from workflows import charactergen


class ClothesError(RuntimeError):
    """换装的可读错误（API 转 400）。"""


def _clean_costume(value: str) -> str:
    """官方 `get_cache_paths:450-452` 的 clean_name：只留字母数字与空格 _ -。"""
    return "".join(c for c in str(value or "")
                   if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")


def base_sprite(name: str) -> Optional[str]:
    """换装的输入底图：`Naked` 最新一张，没有就退 `Original`（官方 `:479`）。

    **不能拿目标服装自己的图当底图**：新服装本来就是空目录；而且换装的定义
    是"从裸装改成这套衣服"，拿已穿好的图再改会越改越偏。
    """
    return (characters.latest_sprite_path(name, "Naked")
            or characters.latest_sprite_path(name, "Original"))


def _save_preview(name: str, costume: str, image: Image.Image) -> str:
    """存官方那张预览：`cache/preview_<服装>.png`（前端换装页显示的就是它）。"""
    cache = characters.cache_dir(name)
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, f"preview_{_clean_costume(costume)}.png")
    image.save(out, format="PNG")
    return out


def _write_temp_png(image: Image.Image) -> str:
    fd, path = tempfile.mkstemp(suffix=".png", prefix="meow_cloth_")
    os.close(fd)
    image.save(path, format="PNG")
    return path


# ---------------------------------------------------------------- 内衣底图（remove_clothes）
#
# 官方 `character_generator.py:878-886` 把它当**预处理的一步**：把自己的角色图换成
# **白色内衣**，得到一张"干净底图"（代码里叫 `naked_character`），后面所有服装
# 都从它出发——这样每套衣服都是在同一个干净底子上画的，不会把上一套的袖子
# 残留在下一套里。
#
# **它是一套具体服装（白内衣），不是"去掉衣服变裸体"**。官方的实现就是一条
# 换装指令加 Clothes LoRA，跟换护士服走的是同一套代码路径。

#: 官方 `:879`（**逐字**，整条指令就这一句，没有背景段）
UNDERWEAR_PROMPT = "Dress character: White underwear"

#: 官方 `:880-893`：target_size 1024、背景 White、采样 20 步 / cfg 1.0
UNDERWEAR_DEFAULTS: Dict[str, Any] = {
    "target_size": 1024,
    "steps": 20,
    "cfg": 1.0,
    "background_color": "White",
    "costume": "Underwear",
}


def flatten_alpha(image: Image.Image, color: str = "White") -> Image.Image:
    """把透明区域压到纯色上（官方 `VNCCS_MaskExtractor.fill_alpha_with_color`）。

    remove_clothes 的输入是**一张 RGB 图**（QI2 encode 不吃 alpha）。不压平的话，
    透明处会以预乘黑的形式送进去，模型看到"角色周围一圈黑边"，经常会照着画一圈
    黑轮廓出来。
    """
    rgba = image.convert("RGBA")
    rgb = {"white": (255, 255, 255), "green": (0, 255, 0),
           "blue": (0, 0, 255), "black": (0, 0, 0)}.get(
               str(color or "White").strip().lower(), (255, 255, 255))
    canvas = Image.new("RGBA", rgba.size, rgb + (255,))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def remove_clothes(
    engine: Any,
    params: Dict[str, Any],
    *,
    job: Optional[Dict[str, Any]] = None,
    edit: Optional[Callable[..., Image.Image]] = None,
    matte_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> Dict[str, Any]:
    """角色 → 白色内衣底图（官方 `_run_remove_clothes:2282-2340`）。

    跟 `run_design` 的区别只在三处，其余完全一样（同一个 LoRA、同一条出图链路）：

    | 项 | `run_design`（换装） | 这里 |
    |---|---|---|
    | prompt | `Dress the character:` + 五个字段 + 背景段 | `Dress character: White underwear` |
    | 采样 | 25 步 / cfg 3.0 | **20 步 / cfg 1.0**（`:894-900`） |
    | 输入 | 裸装底图，RGBA 直接送 | 先压到白底（encode 不吃 alpha） |

    产物落到 `Sprites/<costume>/neutral/sprite_pose_0001.png`，默认 `costume`
    是 `Underwear`。之后出姿势/表情都从这张图走。
    """
    name = characters.ensure_safe_name(str(params.get("character") or ""), "角色名")
    costume = characters.ensure_safe_name(
        str(params.get("costume") or UNDERWEAR_DEFAULTS["costume"]), "服装名")
    if not characters.character_exists(name):
        raise ClothesError(f"没有这个角色：{name}")

    source = base_sprite(name)
    if not source:
        raise ClothesError(
            f"「{name}」还没有裸装/原始底图：先把衣服换成内衣也是从底图改的，"
            "先在「出图」页生成一张（Naked 或 Original 服装）")

    if costume.casefold() not in {c.casefold() for c in characters.list_costumes(name)}:
        characters.save_costume_info(name, costume, {})

    steps = int(params.get("steps") or UNDERWEAR_DEFAULTS["steps"])
    cfg = float(params.get("cfg") or UNDERWEAR_DEFAULTS["cfg"])
    seed = params.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(
        os.urandom(3), "big")
    lora_scale = float(params.get("lora_scale") or 1.0)
    background = str(params.get("background_color")
                     or UNDERWEAR_DEFAULTS["background_color"])

    lora_name = ""
    if edit is None:
        from workflows import generate as gen
        from workflows.engine import Engine

        lora_name = Engine.CLOTHES_LORA

        def _edit(image: Image.Image, p: str, n: str, sd: int, st: int, cf: float):
            return gen.qwen_edit_image(
                engine, image=image, prompt=p, negative=n, seed=sd, steps=st,
                cfg=cf, label="wunderwear",
                lora=Engine.CLOTHES_LORA, lora_scale=lora_scale)

        edit = _edit

    if job is not None:
        job["progress"] = {"done": 0, "total": 1, "current": "换内衣底图中"}
    started = time.time()
    with Image.open(source) as opened:
        base_rgba = opened.convert("RGBA")
    base = flatten_alpha(base_rgba, background)
    image = edit(base, UNDERWEAR_PROMPT, "", seed, steps, cfg)
    if image is None:
        raise ClothesError("出图没有返回图像")
    if image.size != base.size:
        image = image.resize(base.size, Image.LANCZOS)

    matte_error = ""
    if params.get("matte"):
        try:
            from workflows import matte_screen
            matte_fn = matte_fn or matte_screen.matte_pil
            out = matte_fn(image)
            if out is not None:
                image = out
        except Exception as exc:                      # noqa: BLE001
            matte_error = str(exc)

    preview = _save_preview(name, costume, image)
    tmp = _write_temp_png(image)
    try:
        published = characters.publish_image(
            name, tmp, costume=costume, emotion="neutral", set_cover=False)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    if job is not None:
        job["progress"] = {"done": 1, "total": 1, "current": ""}
    return {
        "character": name, "costume": costume,
        "prompt": UNDERWEAR_PROMPT, "negative": "",
        "background": background, "transparent": False,
        "source": source, "sprite": published["path"], "preview": preview,
        "seed": seed, "steps": steps, "cfg": cfg,
        "lora": lora_name,
        "matte_error": matte_error,
        "seconds": round(time.time() - started, 2),
    }


def run_design(
    engine: Any,
    params: Dict[str, Any],
    *,
    job: Optional[Dict[str, Any]] = None,
    edit: Optional[Callable[..., Image.Image]] = None,
    matte_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> Dict[str, Any]:
    """按服装描述生成一张换装底图，并把它记进这套服装的 `Sprites/<服装>/neutral/`。

    `params`（都来自 API）：
      character / costume / background_color / clone / steps / cfg / seed /
      matte / lora_scale
    """
    name = characters.ensure_safe_name(str(params.get("character") or ""), "角色名")
    costume = characters.ensure_safe_name(str(params.get("costume") or ""), "服装名")
    if not characters.character_exists(name):
        raise ClothesError(f"没有这个角色：{name}")
    # 官方要求"这套服装已存在"（`:523` require_existing_costume）。我们放宽成
    # "不存在就先建"：前端「新建服装」只是把名字填进输入框，用户点换装时
    # 目录其实还没建，这时报错只会让人先去点一次保存。
    if costume.casefold() not in {c.casefold() for c in characters.list_costumes(name)}:
        characters.save_costume_info(name, costume, {})

    source = base_sprite(name)
    if not source:
        raise ClothesError(
            f"「{name}」还没有裸装/原始底图：换装是从底图改的，"
            "先在「出图」页生成一张（Naked 或 Original 服装）")

    info = characters.load_config(name) or {}
    costume_info = (info.get("costumes") or {}).get(costume) or {}
    if not isinstance(costume_info, dict):
        costume_info = {}
    background = (params.get("background_color")
                  or (info.get("character_info") or {}).get("background_color") or "green")
    clone = bool(params.get("clone"))
    prompt, negative = charactergen.construct_clothes_prompt(
        costume_info, background, clone=clone)
    bg_effective = charactergen.effective_background_color(background)
    transparent = bg_effective == charactergen.TRANSPARENT_BACKGROUND

    steps = int(params.get("steps") or charactergen.CLOTHES_DEFAULTS["steps"])
    cfg = float(params.get("cfg") or charactergen.CLOTHES_DEFAULTS["cfg"])
    seed = params.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(
        os.urandom(3), "big")
    lora_scale = float(params.get("lora_scale") or 1.0)

    lora_name = ""
    if edit is None:
        from workflows import generate as gen
        from workflows.engine import Engine

        lora_name = Engine.CLOTHES_LORA

        def _edit(image: Image.Image, p: str, n: str, sd: int, st: int, cf: float):
            return gen.qwen_edit_image(
                engine, image=image, prompt=p, negative=n, seed=sd, steps=st,
                cfg=cf, transparent=transparent, label="wcloth",
                lora=Engine.CLOTHES_LORA, lora_scale=lora_scale)

        edit = _edit

    if job is not None:
        job["progress"] = {"done": 0, "total": 1, "current": "换装中"}
    started = time.time()
    with Image.open(source) as opened:
        base = opened.convert("RGBA")
    image = edit(base, prompt, negative, seed, steps, cfg)
    if image is None:
        raise ClothesError("出图没有返回图像")
    if image.size != base.size:
        image = image.resize(base.size, Image.LANCZOS)

    matte_error = ""
    if params.get("matte"):
        try:
            from workflows import matte_screen
            matte_fn = matte_fn or matte_screen.matte_pil
            out = matte_fn(image)
            if out is not None:
                image = out
        except Exception as exc:                      # noqa: BLE001
            # 抠像失败不判失败：图还在，只是没透明
            matte_error = str(exc)

    # 存两处：官方那张预览（换装页直接看）+ 这套服装的第一张底图
    preview = _save_preview(name, costume, image)
    tmp = _write_temp_png(image)
    try:
        published = characters.publish_image(
            name, tmp, costume=costume, emotion="neutral", set_cover=False)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    if job is not None:
        job["progress"] = {"done": 1, "total": 1, "current": ""}
    return {
        "character": name, "costume": costume,
        "prompt": prompt, "negative": negative,
        "background": bg_effective, "transparent": transparent,
        "source": source, "sprite": published["path"], "preview": preview,
        "seed": seed, "steps": steps, "cfg": cfg,
        "lora": lora_name,
        "matte_error": matte_error,
        "seconds": round(time.time() - started, 2),
    }
