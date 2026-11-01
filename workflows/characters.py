"""角色中心：角色的磁盘存储（移植自 VNCCS 的 `utils.py`）。

**为什么照搬而不是自己设计**：角色库是全项目的**共享契约**——造角色、换装、
表情、出图、精灵表打包都按同一套目录与 config 落盘。自己另起一套的话，
后面每一条产线都要维护一份"我的路径规则"，且永远对不上官方工作流产出的素材。

来源与对应（路径相对 `ComfyUI_VNCCS-main`）：

| 这里 | 官方 | 说明 |
|---|---|---|
| `character_dir` 等 | `utils.py:405-438` | 目录布局 |
| `ensure_character_structure` / `ensure_costume_structure` | `utils.py:443-487` | 建骨架 |
| `list_characters` / `list_costumes` | `utils.py:490, 908` | 扫目录枚举（**没有索引文件**） |
| `load_config` / `save_config` | `utils.py:650-700` | config 读写（原子） |
| `save_costume_info` / `delete_costume` | `utils.py:832-900` | 服装 CRUD |
| `staged_image_batch` | `utils.py:252` | 原子发布 + 旧版归档到 V1/V2 |

**改动的两处**（都是我们项目自身的约定要求）：

1. 根目录是 `paths.CHARACTERS_DIR`（`assets/characters/`），不是
   `<ComfyUI output>/VNCCS/Characters` —— 官方那个 `base_output_dir()`
   （`utils.py:55`）依赖 `folder_paths`，我们没装 ComfyUI。放 assets 下还有个
   硬理由：角色预览图要能走 `/files/`，而那个端点只服务 assets 内。
2. 去掉 `X-VNCCS-CSRF` 写保护（`utils.py:27-28, 355-392`）：我们是本机单用户
   FastAPI，同源调用，那个头只防浏览器跨站，没有意义。

**路径安全**：角色名 / 服装名 / 表情名都是用户输入的路径片段，`ensure_safe_name`
限制成 `[A-Za-z0-9 _-]{1,120}`、`safe_join_under` 再做一次 realpath 校验，
两道都过才允许落盘（`utils.py:66, 190`）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from workflows import paths

#: 默认表情与主目录（官方 `utils.py:23-24`）
EMOTIONS = ["neutral"]
MAIN_DIRS = ["Sprites", "Faces", "Sheets"]
#: 路径片段白名单（官方 `utils.py:25`）
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,120}$")
#: 旧版本目录（重生成时旧图归档进去，官方 `utils.py:26`）
ARCHIVE_DIR_RE = re.compile(r"^V[0-9]+$", re.IGNORECASE)
#: 服装可编辑字段（官方 `utils.py:30`）
COSTUME_TEXT_FIELDS = {"top", "bottom", "shoes", "head", "face", "negative_prompt"}
#: 服装里**不可删除**的两套基础集（官方 `character_cloner.py` 产出 Original/Naked）
BASE_COSTUMES = {"naked", "original"}
#: 认得出的图片后缀（官方 `utils.py:960` 的同款过滤）
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
#: `list_pose_preview_files` 的兜底目录名（官方 `utils.py:947-950`）。
#: 官方写的是首字母大写（Naked/Original/Neutral），我们用小写 neutral 归档；
#: **两种都列着**——否则官方工作流产出的角色（`Neutral/`）在这边一张都翻不出来。
POSE_PREVIEW_COSTUMES = ["Naked", "Original", "naked", "original"]
POSE_PREVIEW_EMOTIONS = ["Neutral", "neutral"]

#: 写锁，按路径哈希分片（官方 `utils.py:29`）
_LOCKS = tuple(threading.RLock() for _ in range(64))


class CharacterConflict(ValueError):
    """角色/服装冲突：要么已经存在，要么磁盘上的版本比调用方手里的新。"""


# ------------------------------------------------------------------ 路径安全


def ensure_safe_name(value: str, field: str = "name") -> str:
    """校验用户输入的路径片段。"""
    if value is None:
        raise ValueError(f"{field} 不能为空")
    value = str(value).strip()
    if not SAFE_NAME_RE.match(value):
        raise ValueError(
            f"{field} 只能用字母/数字/空格/下划线/连字符（1~120 字符）：{value!r}")
    return value


def is_path_under(root: str, target: str) -> bool:
    """target 是否真的在 root 里（realpath 比较，挡 `..` 与软链）。"""
    try:
        r = os.path.realpath(root)
        t = os.path.realpath(target)
        return os.path.commonpath([r, t]) == r
    except ValueError:
        return False


def safe_join_under(root: str, *parts: str) -> str:
    """拼接并确认结果没跳出 root。"""
    out = os.path.abspath(os.path.join(os.path.abspath(root), *parts))
    if not is_path_under(root, out):
        raise ValueError(f"路径越界：{parts}")
    return out


@contextmanager
def storage_lock(key: str) -> Iterator[None]:
    """按路径哈希取一把锁（不同角色互不阻塞）。"""
    with _LOCKS[hash(key) % len(_LOCKS)]:
        yield


# ------------------------------------------------------------------ 目录布局


def characters_root() -> str:
    return str(paths.CHARACTERS_DIR)


def character_dir(name: str) -> str:
    return safe_join_under(characters_root(), ensure_safe_name(name, "角色名"))


def sprites_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    return safe_join_under(character_dir(name), "Sprites",
                           ensure_safe_name(costume, "服装名"),
                           ensure_safe_name(emotion, "表情名"))


def faces_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    return safe_join_under(character_dir(name), "Faces",
                           ensure_safe_name(costume, "服装名"),
                           ensure_safe_name(emotion, "表情名"))


def sheets_dir(name: str, costume: str = "Naked", emotion: str = "neutral") -> str:
    return safe_join_under(character_dir(name), "Sheets",
                           ensure_safe_name(costume, "服装名"),
                           ensure_safe_name(emotion, "表情名"))


def cache_dir(name: str) -> str:
    return safe_join_under(character_dir(name), "cache")


def config_path(name: str) -> str:
    n = ensure_safe_name(name, "角色名")
    return safe_join_under(character_dir(n), f"{n}_config.json")


def ensure_character_structure(
    name: str, emotions: Optional[List[str]] = None,
    main_dirs: Optional[List[str]] = None,
) -> str:
    """建角色骨架。**表情目录不在这里建**——等真的存图时再建（官方同款注释）。"""
    dirs = main_dirs or MAIN_DIRS
    root = characters_root()
    char = character_dir(name)
    os.makedirs(root, exist_ok=True)
    os.makedirs(char, exist_ok=True)
    for main in dirs:
        os.makedirs(safe_join_under(char, main), exist_ok=True)
        os.makedirs(safe_join_under(char, main, "Naked"), exist_ok=True)
    return char


def ensure_costume_structure(name: str, costume: str) -> None:
    char = character_dir(name)
    c = ensure_safe_name(costume, "服装名")
    for main in MAIN_DIRS:
        os.makedirs(safe_join_under(char, main), exist_ok=True)
        os.makedirs(safe_join_under(char, main, c), exist_ok=True)


# ------------------------------------------------------------------ 原子写


@contextmanager
def atomic_output_path(path: str) -> Iterator[str]:
    """先写临时文件再 replace：半截 JSON 会让整个角色打不开。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",
                               suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


@contextmanager
def staged_image_batch(target_dir: str, version_existing: bool = True) -> Iterator[str]:
    """把一批图**整目录**发布过去：先在临时目录攒齐，成功才替换。

    旧图不删，搬进 `V1` / `V2`…（`utils.py:252`）。用户在"重新生成"之后还能
    找回上一版，这个回滚能力是产线上最常被需要的。
    """
    parent = os.path.dirname(os.path.abspath(target_dir))
    os.makedirs(parent, exist_ok=True)
    with storage_lock(target_dir):
        stage = tempfile.mkdtemp(prefix=".sprites-", dir=parent)
        try:
            yield stage
            old = []
            if os.path.isdir(target_dir):
                old = [n for n in os.listdir(target_dir)
                       if os.path.isfile(os.path.join(target_dir, n))
                       and os.path.splitext(n)[1].lower() in IMAGE_EXTS]
            if version_existing and old:
                num = 1
                while os.path.lexists(os.path.join(target_dir, f"V{num}")):
                    num += 1
                archive = os.path.join(stage, f"V{num}")
                os.makedirs(archive, exist_ok=True)
                for n in old:
                    shutil.move(os.path.join(target_dir, n), os.path.join(archive, n))
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
            os.replace(stage, target_dir)
        finally:
            if os.path.isdir(stage):
                shutil.rmtree(stage, ignore_errors=True)


# ------------------------------------------------------------------ config


def _default_character_info(name: str) -> Dict[str, Any]:
    """角色属性默认值（官方 `character_creator_v2.py:2138-2146`）。

    字段清单照抄，这样官方工作流产出的 config 我们能直接读，反之亦然。
    """
    return {
        "name": name,
        "sex": "female",
        "age": 18,
        "race": "human",
        "skin_color": "",
        "hair": "",
        "eyes": "",
        "face": "",
        "body": "",
        "breast_size": "",
        "additional_details": "",
        "nsfw": False,
        "aesthetics": "masterpiece",
        "negative_prompt": "bad quality,worst quality,low quality",
        "lora_prompt": "",
        "background_color": "green",
        "framing": "",
        "style": "ghibli_miyazaki",
        "custom_style": "",
        "seed": 0,
    }


def load_config(name: str, strict: bool = False) -> Optional[Dict[str, Any]]:
    """读角色 config。文件不在或坏了都返回 None（**不抛**）——一个角色坏了，
    其余角色照样要能列出来。"""
    path = config_path(name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeDecodeError):
        if strict:
            raise OSError(f"读不了角色配置：{path}")
        return None
    if not isinstance(data, dict):
        if strict:
            raise OSError(f"角色配置不是对象：{path}")
        return None
    if strict:
        for key in ("character_info", "costumes"):
            if key in data and not isinstance(data[key], dict):
                raise OSError(f"角色配置的 {key} 不是对象：{path}")
    return data


def save_config(name: str, data: Dict[str, Any]) -> str:
    ensure_character_structure(name)
    path = config_path(name)
    with atomic_output_path(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return path


# ------------------------------------------------------------------ 角色 CRUD


def list_characters() -> List[str]:
    root = characters_root()
    try:
        return sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
    except OSError:
        return []


def character_exists(name: str) -> bool:
    try:
        return os.path.isdir(character_dir(name))
    except ValueError:
        return False


def create_character(
    name: str, info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """建一个角色：建目录 + 写 config。

    **已存在就报冲突**（`CharacterConflict`），不静默覆盖——两个窗口同秒建
    同名角色时，覆盖会让用户毫无察觉地丢东西。
    """
    n = ensure_safe_name(name, "角色名")
    if character_exists(n):
        raise CharacterConflict(f"角色已存在：{n}")
    cfg = {
        "character_info": {**_default_character_info(n), **(info or {})},
        "costumes": {},
        "costume_ids": {},
        "character_path": character_dir(n),
        "folder_structure": {"main_directories": MAIN_DIRS, "emotions": list(EMOTIONS)},
        "config_version": "2.0",
    }
    ensure_character_structure(n)
    save_config(n, cfg)
    return cfg


def delete_character(name: str, remove_files: bool = True) -> bool:
    """删角色。`remove_files=False` 只删 config，素材留在盘上。"""
    n = ensure_safe_name(name, "角色名")
    if not character_exists(n):
        return False
    if remove_files:
        shutil.rmtree(character_dir(n), ignore_errors=True)
    else:
        try:
            os.unlink(config_path(n))
        except OSError:
            return False
    return True


def rename_character(name: str, new_name: str) -> Dict[str, Any]:
    """改名：目录 + config 一起改（config 里的 name / character_path 也要跟着）。"""
    old = ensure_safe_name(name, "角色名")
    new = ensure_safe_name(new_name, "新角色名")
    if not character_exists(old):
        raise CharacterConflict(f"角色不存在：{old}")
    if character_exists(new):
        raise CharacterConflict(f"角色已存在：{new}")
    src = character_dir(old)
    dst = character_dir(new)
    with storage_lock(src):
        os.replace(src, dst)
        cfg = load_config(new) or {}
        cfg.setdefault("character_info", {})["name"] = new
        cfg["character_path"] = dst
        # 旧 config 文件名也换掉
        try:
            os.unlink(safe_join_under(dst, f"{old}_config.json"))
        except OSError:
            pass
        save_config(new, cfg)
    return cfg


# ------------------------------------------------------------------ 服装


def validate_costume_info(data: Dict[str, Any]) -> Dict[str, str]:
    """只留白名单字段，其余一律丢弃（防止往 config 里塞任意结构）。"""
    out = {k: "" for k in COSTUME_TEXT_FIELDS}
    for k, v in (data or {}).items():
        if k in COSTUME_TEXT_FIELDS and isinstance(v, str):
            out[k] = v
    return out


def list_costumes(name: str) -> List[str]:
    """config 里的 costumes + 扫 `Sprites/` 子目录；**恒含 Naked**（官方 `utils.py:908`）。"""
    out = ["Naked"]
    cfg = load_config(name) or {}
    for c in (cfg.get("costumes") or {}).keys():
        if c not in out:
            out.append(c)
    sp = safe_join_under(character_dir(name), "Sprites")
    if os.path.isdir(sp):
        for item in sorted(os.listdir(sp)):
            if os.path.isdir(os.path.join(sp, item)) and item not in out:
                out.append(item)
    return out


def save_costume_info(name: str, costume: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """存/改一套服装。新建时给它一个 `costume_id`（ETag 乐观锁用）。"""
    c = ensure_safe_name(costume, "服装名")
    with storage_lock(character_dir(name)):
        cfg = load_config(name, strict=True) or {"character_info": {}, "costumes": {}}
        cfg.setdefault("costumes", {})
        ids = cfg.setdefault("costume_ids", {})
        if c not in cfg["costumes"] or c not in ids:
            ids[c] = uuid.uuid4().hex
            ensure_costume_structure(name, c)
        cfg["costumes"][c] = validate_costume_info(data)
        save_config(name, cfg)
    return cfg


def delete_costume(name: str, costume: str) -> None:
    c = ensure_safe_name(costume, "服装名")
    if c.casefold() in BASE_COSTUMES:
        raise ValueError("基础精灵集（Naked / Original）不能删")
    with storage_lock(character_dir(name)):
        cfg = load_config(name, strict=True) or {}
        cfg.setdefault("costumes", {}).pop(c, None)
        cfg.get("costume_ids", {}).pop(c, None)
        save_config(name, cfg)
        for main in MAIN_DIRS:
            p = safe_join_under(character_dir(name), main, c)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------------ 素材枚举


def _images_in(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, n) for n in os.listdir(d)
        if os.path.isfile(os.path.join(d, n))
        and os.path.splitext(n)[1].lower() in IMAGE_EXTS
    )


def list_sprite_images(name: str, costume: str = "Naked",
                       emotion: str = "neutral") -> List[str]:
    return _images_in(sprites_dir(name, costume, emotion))


def source_sprite_paths(name: str, costume: str = "Naked") -> List[str]:
    """一套服装的**源帧**（表情生成从这些帧上改脸）。

    官方 `emotion_generator_v2.py:336` `list_costume_sprite_paths`：

    - 候选目录按 `Neutral` → `neutral` → 服装根目录，**前一组非空就只用它**；
    - 排序后**序号从 1 开始**（`sprite_pose_0001` 就是第 1 帧，表情产物也用同一个
      序号，见 `:4541-4543` 的 `save_index`）；
    - 用 `(st_dev, st_ino)` 去重目录：Windows 文件系统大小写不敏感，
      `Neutral` 与 `neutral` 常常是**同一个目录**，不去重就会把每帧收两遍。
    """
    n = ensure_safe_name(name, "角色名")
    c = ensure_safe_name(costume, "服装名")
    base = safe_join_under(character_dir(n), "Sprites", c)
    neutral: List[str] = []
    direct: List[str] = []
    seen_roots = set()
    for sub in ("Neutral", "neutral", ""):
        d = os.path.join(base, sub) if sub else base
        if not os.path.isdir(d):
            continue
        try:
            key = (os.stat(d).st_dev, os.stat(d).st_ino)
            names = sorted(os.listdir(d))
        except OSError:
            continue
        if key in seen_roots:
            continue
        seen_roots.add(key)
        bucket = neutral if sub else direct
        for fn in names:
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                bucket.append(p)
    out: List[str] = []
    seen_paths = set()
    for p in (neutral or direct):
        k = os.path.normcase(os.path.abspath(p))
        if k in seen_paths:
            continue
        seen_paths.add(k)
        out.append(p)
    return out


def latest_sprite_path(name: str, costume: str = "Naked") -> Optional[str]:
    """一套服装**最新**的一张图（换装的底图就是它）。

    官方 `clothes_designer.py:191-232` `get_latest_sprite_path`，三级回退：

    1. `Sprites/<服装>/Neutral` 递归找（**跳过 V1/V2 归档目录**）；
    2. 服装目录下的**直接**文件（官方有些角色是把 png 平铺的）；
    3. 服装目录递归找（同样跳过归档）。

    判"最新"用的是 `(mtime, path)` 而不是文件名——用户可能把别人的角色库
    拷进来，文件名序号不一定是本机生成的顺序。

    **必须跳过 `V*` 归档目录**：重生成时旧图会被推进 `V1`，不跳过的话
    "最新"会指到旧版本上去（官方专门写了 `is_sprite_archive_dir`）。
    """
    n = ensure_safe_name(name, "角色名")
    c = ensure_safe_name(costume, "服装名")
    base = safe_join_under(character_dir(n), "Sprites", c)

    def newest(files: List[str]) -> Optional[str]:
        usable = [p for p in files
                  if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTS]
        if not usable:
            return None
        return max(usable, key=lambda p: (os.path.getmtime(p), p))

    def walk(dirpath: str) -> List[str]:
        found: List[str] = []
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if not ARCHIVE_DIR_RE.match(d)]
            found.extend(os.path.join(root, f) for f in files)
        return found

    if not os.path.isdir(base):
        return None
    # 1) Neutral / neutral（大小写两种都认，见 source_sprite_paths 的说明）
    for sub in ("Neutral", "neutral"):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            best = newest(walk(d))
            if best:
                return best
    # 2) 直接文件
    try:
        direct = [os.path.join(base, f) for f in os.listdir(base)
                  if os.path.isfile(os.path.join(base, f))]
    except OSError:
        direct = []
    best = newest(direct)
    if best:
        return best
    # 3) 整个服装目录递归
    return newest(walk(base))


def face_filename(emotion: str, index: int) -> str:
    """官方 `_face_prefix_from_sprite_prefix:3982`：`sprite_` → `face_`。"""
    e = ensure_safe_name(emotion, "表情名")
    stem = "pose" if e.casefold() == "neutral" else e
    return f"face_{stem}_{max(1, int(index)):04d}.png"


def publish_face_image(name: str, src: str, costume: str, emotion: str,
                       index: int) -> str:
    """把**生成的裁切块**存进 `Faces/<服装>/<表情>/face_<表情>_%04d.png`。

    官方 `:4633-4643`：Sprites 存贴回后的整图，Faces 存单独那张脸
    （`record["face_crop"]`）。两者序号一致，方便对着翻。
    """
    n = ensure_safe_name(name, "角色名")
    c = ensure_safe_name(costume, "服装名")
    e = ensure_safe_name(emotion, "表情名")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"要存的图不存在：{src}")
    target = faces_dir(n, c, e)
    os.makedirs(target, exist_ok=True)
    out = os.path.join(target, face_filename(e, index))
    shutil.copy2(src, out)
    return out


def cover_image(name: str) -> Optional[str]:
    """角色封面：`cache/preview.png`（官方 `character_creator_v2.py:1657`）。

    没有预览图就退回第一张 sprite——否则角色库的卡片全是空的，
    用户只能靠名字猜谁是谁。
    """
    for p in [
        os.path.join(cache_dir(name), "preview.png"),
        os.path.join(cache_dir(name), "preview.webp"),
    ]:
        if os.path.isfile(p):
            return p
    for costume in list_costumes(name):
        for emotion in ["neutral"] + EMOTIONS:
            imgs = list_sprite_images(name, costume, emotion)
            if imgs:
                return imgs[0]
    return None


def next_sprite_index(name: str, costume: str, emotion: str) -> int:
    """下一个序号（官方 `%04d`，从 1 开始）。旧版本目录 V* 不算数。"""
    d = sprites_dir(name, costume, emotion)
    if not os.path.isdir(d):
        return 1
    n = 0
    for f in os.listdir(d):
        if os.path.isfile(os.path.join(d, f)) and \
                os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            n += 1
    return n + 1


def sprite_filename(emotion: str, index: int) -> str:
    """官方命名规则（`character_generator.py:2754-2758` / `4053-4059`）：
    neutral 用 `sprite_pose_0001.png`，表情用 `sprite_<表情>_0001.png`。"""
    e = ensure_safe_name(emotion, "表情名")
    stem = "pose" if e.casefold() == "neutral" else e
    return f"sprite_{stem}_{max(1, int(index)):04d}.png"


def publish_image(
    name: str, src: str, costume: str = "Naked",
    emotion: str = "neutral", set_cover: bool = True,
    matte: bool = False, index: Optional[int] = None,
) -> Dict[str, Any]:
    """把一张生成出来的图**追加**进角色目录（不覆盖已有素材）。

    **为什么是追加而不是替换**：官方的 `staged_image_batch`（`utils.py:252`）是
    给"一次出一整批（一组姿势帧）"用的——整批替换、旧的进 V1。我们这里是**逐张**
    发布（官方目录里本来也是 `sprite_pose_0001..N` 一张张攒起来的），
    一张新图把前面全推进 V1 显然是错的。等做批量姿势生成时再用 staged 那条路。

    `index` 显式给了就用它（表情生成要用**源帧的序号**：
    `sprite_pose_0003.png` 的表情版必须是 `sprite_angry_0003.png`，
    两张对着翻才不会错位，官方 `:4541-4543` 的 `save_index` 就是干这个的）。
    """
    n = ensure_safe_name(name, "角色名")
    c = ensure_safe_name(costume, "服装名")
    e = ensure_safe_name(emotion, "表情名")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"要收编的图不存在：{src}")
    target = sprites_dir(n, c, e)
    os.makedirs(target, exist_ok=True)
    index = int(index) if index else next_sprite_index(n, c, e)
    fname = sprite_filename(e, index)
    ext = os.path.splitext(src)[1] or ".png"
    if ext.lower() != ".png":
        fname = os.path.splitext(fname)[0] + ext
    out = os.path.join(target, fname)
    # 抠像（可选）：官方 `bg_remove` 阶段干的事。抠完存成另一份 `_matte.png`，
    # **不覆盖原图**——抠错了还能用原图重来。
    matte_path, matte_err = None, None
    if matte:
        try:
            from workflows import matte_screen
            matte_path, matte_err = matte_screen.matte_image_file(src, out=os.path.join(
                target, os.path.splitext(fname)[0] + "_matte.png"))
        except Exception as exc:                   # noqa: BLE001
            matte_err = str(exc)
    shutil.copy2(src, out)
    cover = ""
    if set_cover:
        cd = cache_dir(n)
        os.makedirs(cd, exist_ok=True)
        cover = os.path.join(cd, "preview.png")
        try:
            shutil.copy2(out, cover)
        except OSError:
            cover = ""
    return {
        "character": n, "costume": c, "emotion": e,
        "path": out, "file": fname, "index": index, "cover": cover,
        "matte": matte_path or "", "matte_error": matte_err or "",
    }


_POSE_PREVIEW_EXTS = IMAGE_EXTS


def list_pose_preview_files(name: str, costume: Optional[str] = None,
                            emotion: Optional[str] = None) -> List[str]:
    """一个「可翻看的姿势序列」——官方 `utils.py:936-981`。

    候选根目录**有优先级**（先指定服装的 neutral、再指定服装、再 Naked、
    Original），**第一个非空就停**。所以不给服装名时退到最基础的裸装帧，
    而不是把所有服装的图混在一起翻。

    `emotion` 是我们加的一维：官方只有 Neutral。给它就只看那一个表情目录。
    """
    n = ensure_safe_name(name, "角色名")
    base = character_dir(n)
    roots: List[str] = []
    c = ensure_safe_name(costume, "服装名") if costume else ""
    e = ensure_safe_name(emotion, "表情名") if emotion else ""
    if c:
        for does_emo in ([e] if e else POSE_PREVIEW_EMOTIONS):
            roots.append(os.path.join(base, "Sprites", c, does_emo))
        roots.append(os.path.join(base, "Sprites", c))
    for base_costume in POSE_PREVIEW_COSTUMES:
        for does_emo in ([e] if e else POSE_PREVIEW_EMOTIONS):
            roots.append(os.path.join(base, "Sprites", base_costume, does_emo))
        roots.append(os.path.join(base, "Sprites", base_costume))
    files: List[str] = []
    for d in roots:
        try:
            d = safe_join_under(base, os.path.relpath(d, base))
        except ValueError:                                 # noqa: PERF203
            continue
        if not os.path.isdir(d):
            continue
        found = []
        for fn in os.listdir(d):
            if os.path.splitext(fn)[1].lower() not in _POSE_PREVIEW_EXTS:
                continue
            try:
                p = safe_join_under(base, os.path.relpath(os.path.join(d, fn), base))
            except ValueError:
                continue
            if os.path.isfile(p):
                found.append(p)
        if found:
            files = sorted(found)
            break
    return files


def character_summary(name: str) -> Dict[str, Any]:
    """列表用的一条：名字 + 封面 + 服装 + 表情 + 各类图的数量。"""
    cfg = load_config(name) or {}
    info = cfg.get("character_info") or {}
    costumes = list_costumes(name)
    sprites = 0
    for c in costumes:
        d = safe_join_under(character_dir(name), "Sprites", c)
        if not os.path.isdir(d):
            continue
        for sub in os.listdir(d):
            if os.path.isdir(os.path.join(d, sub)) and not ARCHIVE_DIR_RE.match(sub):
                sprites += len(_images_in(os.path.join(d, sub)))
    cover = cover_image(name)
    return {
        "name": name,
        "cover": cover or "",
        "sex": info.get("sex") or "",
        "style": info.get("style") or "",
        "nsfw": bool(info.get("nsfw")),
        "costumes": costumes,
        "sprites": sprites,
        "config_version": cfg.get("config_version") or "",
    }


def list_character_summaries() -> List[Dict[str, Any]]:
    return [character_summary(n) for n in list_characters()]
