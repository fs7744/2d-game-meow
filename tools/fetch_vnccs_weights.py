"""下载 VNCCS 移植剩下的四个权重（以官方 ComfyUI_VNCCS 的清单为准）。

**每一项都来自官方代码，不是我编的**：

| 权重 | 官方出处 |
|---|---|
| `Qwen3.5-4B-Q8_0.gguf` | `nodes/qwen_vl.py:66`（文件名）、:71（repo）、:74（revision） |
| `mmproj-*.gguf` → `mmproj-F16.gguf` | `nodes/qwen_vl.py:68,74`（两个候选名 + 下载后重命名） |
| `VNCCS_QI2_AnimeOverhaulV2.safetensors` | `control_center.json` lora[6] 的 `hf_path` |
| `VNCCS_QI2_ClothesCoreV2.6.safetensors` | `control_center.json` lora[8] 的 `hf_path` |
| `face_yolov8m.pt` | `nodes/character_generator.py:845`（**只给了路径，没给来源**） |

两个不能省的步骤：

1. **LoRA 下载完必须转格式**。官方是 ComfyUI 格式（`diffusion_model.` +
   `lora_A/B`），我们的引擎只认 diffusers 原生（`transformer.` + `lora.down/up`）；
   直接挂上去会**静默不生效**（`load_lora_weights` 一个层都匹配不到，
   比报错更坏 —— 见 `engine.available_loras` 的注释）。所以下完自动跑
   `tools/convert_lora.py`。
2. **校验照官方做**。官方 `_validate_gguf_file`（`qwen_vl.py:141`）：文件存在、
   大于 1MB、头四字节是 `GGUF`。少了这步，下到一半的残文件会被当成好的。

`face_yolov8m.pt` 官方**没给下载地址**（它是 ComfyUI Impact Pack 生态里用户
自备的文件）。这条只检查在不在，不在就说明去哪找、放哪，**不猜 URL**。

用法：

    python tools/fetch_vnccs_weights.py --dry-run     # 只看要下什么、多大
    python tools/fetch_vnccs_weights.py --only lora   # 只下两个 LoRA
    python tools/fetch_vnccs_weights.py --only llm    # 只下 gguf（4.4GB）
    python tools/fetch_vnccs_weights.py --force       # 已存在也重新下
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workflows import paths           # noqa: E402

#: 官方 `nodes/qwen_vl.py:71,74`
LLM_REPO = "unsloth/Qwen3.5-4B-GGUF"
LLM_REVISION = "e966ccab6d3f3c91e94d858b4a01c921a5d7ef53"

#: 官方 `control_center.json` 里那两个 QI2 专用 LoRA
VNCCS_REPO = "MIUProject/VNCCS_v3.0"
VNCCS_LORA_SUBDIR = "models/loras/QI2.1/VNCCS"

GGUF_MAGIC = b"GGUF"


class Item:
    """一件要下的东西。"""

    def __init__(
        self,
        key: str,
        group: str,
        name: str,
        dest: Path,
        *,
        repo: str = "",
        revision: Optional[str] = None,
        remote: str = "",
        note: str = "",
        needs_convert: bool = False,
        manual: bool = False,
    ) -> None:
        self.key = key
        self.group = group
        self.name = name
        self.dest = dest
        self.repo = repo
        self.revision = revision
        self.remote = remote
        self.note = note
        self.needs_convert = needs_convert
        self.manual = manual


def build_items() -> List[Item]:
    return [
        Item(
            "llm", "llm", "Qwen3.5-4B-Q8_0.gguf",
            paths.LLM_DIR / "Qwen3.5-4B-Q8_0.gguf",
            repo=LLM_REPO, revision=LLM_REVISION,
            remote="Qwen3.5-4B-Q8_0.gguf",
            note="AI 向导 / 看图反推属性用（官方 qwen_vl.py:66）",
        ),
        Item(
            # **文件名用仓库里的真实名字**（`mmproj-F16.gguf`），不是
            # `qwen_vl.py:68` 那个 `mmproj-Qwen3.5-4B-F16.gguf` —— 后者在仓库里
            # **不存在，照它下会 404**。官方自己的查找顺序也是先找短名
            # （`_find_qwen_vl_mmproj`，qwen_vl.py:171-187），那两个带模型名的
            # 只是兜底。
            "mmproj", "llm", "mmproj-F16.gguf",
            paths.LLM_DIR / "mmproj-F16.gguf",
            repo=LLM_REPO, revision=LLM_REVISION,
            remote="mmproj-F16.gguf",
            note="看图反推属性要它做视觉投影（仓库里另有 BF16/F32 版可选）",
        ),
        Item(
            "overhaul", "lora", "VNCCS_QI2_AnimeOverhaulV2.safetensors",
            paths.LORA_DIR / "VNCCS_QI2_AnimeOverhaulV2.safetensors",
            repo=VNCCS_REPO,
            remote=f"{VNCCS_LORA_SUBDIR}/VNCCS_QI2_AnimeOverhaulV2.safetensors",
            note="造角色的二次元味道（官方 control_center.json lora[6]）",
            needs_convert=True,
        ),
        Item(
            "clothes", "lora", "VNCCS_QI2_ClothesCoreV2.6.safetensors",
            paths.LORA_DIR / "VNCCS_QI2_ClothesCoreV2.6.safetensors",
            repo=VNCCS_REPO,
            remote=f"{VNCCS_LORA_SUBDIR}/VNCCS_QI2_ClothesCoreV2.6.safetensors",
            note="换装的服装一致性（官方 control_center.json lora[8]）",
            needs_convert=True,
        ),
        Item(
            # 官方只给了路径 `bbox/face_yolov8m.pt`（character_generator.py:845），
            # **没给下载地址**（它假定用户装了 Impact Pack）。来源由用户提供：
            # Tenofas/ComfyUI 是 ComfyUI 生态那套 bbox/segm 权重的仓库。
            # 已联网核对过：ultralytics/bbox/face_yolov8m.pt 存在，49.6 MB。
            "yolo", "yolo", "face_yolov8m.pt",
            paths.BBOX_DIR / "face_yolov8m.pt",
            repo="Tenofas/ComfyUI",
            remote="ultralytics/bbox/face_yolov8m.pt",
            note="表情批量生成要定位人脸框（官方只有路径，来源由你提供）",
        ),
    ]


# ------------------------------------------------------------------ 校验


def validate_gguf(path: Path) -> None:
    """官方 `_validate_gguf_file`（`nodes/qwen_vl.py:141`），一行不差。

    下到一半的残文件没这个检查就会被当成好的，然后 llama-cpp 报一堆看不懂的错。
    """
    if not path.is_file():
        raise FileNotFoundError(f"没写出来：{path}")
    size = path.stat().st_size
    if size < 1024 * 1024:
        raise ValueError(f"太小了，不像一个 GGUF（{size} 字节）")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != GGUF_MAGIC:
        raise ValueError(f"不是 GGUF 文件（magic={magic!r}）")


def validate_torch(path: Path) -> None:
    """YOLO 的 `.pt` 是个 zip 包（torch 序列化格式）。

    **不 torch.load**：那会 unpickle 出 YOLO 的类定义，机器上没装 ultralytics
    就直接报错，而我们只想确认文件下完整了。看 zip magic 就够。
    """
    if not path.is_file():
        raise FileNotFoundError(f"没写出来：{path}")
    size = path.stat().st_size
    if size < 1024 * 1024:
        raise ValueError(f"太小了，不像一个权重文件（{size} 字节）")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] != b"PK":
        raise ValueError(f"不是 torch/zip 权重文件（magic={magic!r}）")


def validate_any(item: "Item", path: Path) -> None:
    """按后缀挑校验器。"""
    if path.suffix == ".gguf":
        validate_gguf(path)
    elif path.suffix == ".safetensors":
        validate_lora(path)
    elif path.suffix == ".pt":
        validate_torch(path)


def validate_lora(path: Path) -> None:
    """LoRA 必须是 **diffusers 原生格式**（`transformer.` 前缀）。

    官方发的是 ComfyUI 格式，不转就挂上去会静默不生效 —— 这是最容易漏的一步，
    因为界面上什么都不会报，只是"出图跟没挂 LoRA 一样"。
    """
    from workflows.engine import _lora_keys
    keys = _lora_keys(path)
    if not any(k.startswith("transformer.") for k in keys):
        raise ValueError(
            "这是 ComfyUI 格式（键名应该以 diffusion_model. 开头），"
            "我们的引擎挂上去不会生效，必须先转："
            f"python tools/convert_lora.py {path} <out>")


# ------------------------------------------------------------------ 运行时依赖


def qwen_runtime_installed() -> bool:
    """官方 `install.py:8-18`：装的必须是 **JamePeng fork** 且 >= 0.3.46。

    **为什么必须是这个 fork**：官方的 Qwen35ChatHandler（多模态投影那个 chat
    handler）还没进上游 llama-cpp-python。装错 fork 的表现是
    `ImportError: Qwen35ChatHandler`，很容易被当成"我们代码写错了"。
    """
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        package = distribution("llama-cpp-python")
    except Exception:                                        # noqa: BLE001
        return False
    from pip._vendor.packaging.version import Version
    try:
        if Version(package.version) < Version("0.3.46"):
            return False
    except Exception:                                        # noqa: BLE001
        return False
    sources = package.metadata.get_all("Project-URL", []) or []
    return any("github.com/JamePeng/llama-cpp-python" in x for x in sources)


def qwen_wheel_url(system: str, machine: str, python_version: tuple,
                   cuda_version: str) -> str:
    """官方 `install.py:21-47`，逐行照搬。

    官方**不是** `pip install llama-cpp-python` —— 那样会拉上游源码编译，
    既没有 Qwen35ChatHandler，还可能顺手把 torch/CUDA 换掉。他们发的是按
    Python 版本 / CUDA 版本 / 平台拼好的预编译 wheel。
    """
    if python_version not in {(3, m) for m in range(10, 15)}:
        raise ValueError("wheel 只支持 CPython 3.10 ~ 3.14")
    tag = f"cp{python_version[0]}{python_version[1]}"
    version, date = "0.3.46", "20260808"
    if system == "Darwin" and machine.lower() == "arm64":
        release = f"v{version}-Metal-macos-{date}"
        filename = f"llama_cpp_python-{version}-{tag}-{tag}-macosx_11_0_arm64.whl"
    elif system in {"Windows", "Linux"} and machine.lower() in {"amd64", "x86_64"}:
        if cuda_version not in {"12.4", "12.6", "12.8", "13.0", "13.1"}:
            raise ValueError(
                f"没有对应 PyTorch CUDA {cuda_version!r} 的 wheel")
        # Linux + CUDA 13.0 的第一版是从 0.4.2 才开始发的（官方注释）
        if system == "Linux" and cuda_version == "13.0":
            version, date = "0.4.2", "20261003"
        cuda_tag = "cu" + cuda_version.replace(".", "")
        release_os, wheel_platform = (
            ("win", "win_amd64") if system == "Windows" else ("linux", "linux_x86_64"))
        release = f"v{version}-{cuda_tag}-{release_os}-{date}"
        filename = (f"llama_cpp_python-{version}+{cuda_tag}-{tag}-{tag}"
                    f"-{wheel_platform}.whl")
    else:
        raise ValueError(f"这个平台没有现成的 wheel：{system} {machine}")
    return ("https://github.com/JamePeng/llama-cpp-python/releases/download/"
            f"{release}/{filename}")


def report_runtime() -> None:
    """打印还缺哪些运行时依赖（**不自动装**：装 wheel 有换掉 torch 的风险）。"""
    print("\n运行时依赖（下载权重之外还要装）")
    if qwen_runtime_installed():
        print("  ✓ llama-cpp-python（JamePeng fork）已就位")
        return
    try:
        import platform
        import torch
        url = qwen_wheel_url(platform.system(), platform.machine(),
                             sys.version_info[:2], torch.version.cuda)
    except Exception as exc:                                 # noqa: BLE001
        print(f"  · llama-cpp-python：拼不出 wheel 地址（{exc}）")
        return
    print("  ✗ llama-cpp-python 缺失或版本不对（必须是 JamePeng fork ≥0.3.46，"
          "上游没有 Qwen35ChatHandler）")
    print(f"    pip install --only-binary=:all: {url}")


# ------------------------------------------------------------------ 下载


def hf_size(repo: str, filename: str, revision: Optional[str]) -> int:
    """问 HuggingFace 这个文件多大（`--dry-run` 用）。

    **问不到要说明原因**，不能悄悄返回 0 —— 否则 `--dry-run` 会打印一排
    "0 MB"，用户以为这几个权重是空的。通常是网络/代理不通，或者仓库是 gated
    （MIUProject/VNCCS_v3.0 有这个可能）。
    """
    global _size_warned
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        for s in (info.siblings or []):
            if s.rfilename == filename:
                return int(getattr(s, "size", 0) or 0)
        return 0
    except Exception as exc:                                # noqa: BLE001
        if not _size_warned:
            print(f"  （问不到体积：{type(exc).__name__}：{str(exc)[:80]}"
                  "—— 不影响下载，只是这里显示不出大小）")
            _size_warned = True
        return 0


_size_warned = False


def download(item: Item, force: bool) -> bool:
    """下一件。返回 True 表示这一步真的动了磁盘。"""
    from huggingface_hub import hf_hub_download

    item.dest.parent.mkdir(parents=True, exist_ok=True)
    # 已存在的先校验：坏了才重新下（官方 `qwen_vl.py:224` 同样先校验再决定
    # force_download）。没这一步的话，每次跑都要重下 4.4GB。
    if item.dest.is_file() and not force:
        try:
            validate_any(item, item.dest)
            print(f"  ✓ 已存在且校验通过：{item.dest.name}")
            return False
        except (ValueError, FileNotFoundError) as exc:
            print(f"  · 现有的不对（{exc}），重新下")

    print(f"  ↓ {item.repo} @ {item.revision or 'main'}")
    print(f"    {item.remote}")
    got = hf_hub_download(
        repo_id=item.repo,
        filename=item.remote,
        revision=item.revision,
        local_dir=str(item.dest.parent),
    )
    # hf_hub_download 带 local_dir 时会按远端路径建子目录，这里只要那个文件本身
    src = Path(got)
    if src.resolve() != item.dest.resolve():
        if item.dest.exists():
            item.dest.unlink()
        src.replace(item.dest)
    _prune_empty_dirs(item.dest.parent, item.remote)
    print(f"  ✓ 下载完成（{item.dest.stat().st_size / 1048576:.1f} MB）")
    return True


def _prune_empty_dirs(root: Path, remote: str) -> None:
    """清掉 `local_dir` 留下的空壳目录。

    远端路径带子目录时（如 `ultralytics/bbox/face_yolov8m.pt`），hf 会在落点
    下先建出 `ultralytics/bbox/` 再放文件；文件被我们挪走之后，这两个空目录
    就一直杵在那儿。留着只会让人以为"怎么还有个 ultralytics 目录"。
    **只删空的**——里面真有东西就一个都不动。
    """
    parts = remote.split("/")[:-1]
    while parts:
        d = root.joinpath(*parts)
        parts.pop()
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def convert(item: Item) -> None:
    """ComfyUI 格式 → diffusers 原生格式（我们的引擎只认后者）。"""
    raw = item.dest.with_name(item.dest.stem + ".comfyui" + item.dest.suffix)
    if item.dest.exists():
        item.dest.replace(raw)
    code = subprocess.run(                                   # noqa: S603
        [sys.executable, str(ROOT / "tools" / "convert_lora.py"),
         str(raw), str(item.dest)],
        cwd=str(ROOT), check=False,
    ).returncode
    if code != 0:
        # 转失败就把原件放回去，别让用户既没成品又丢了源文件
        if raw.exists() and not item.dest.exists():
            raw.replace(item.dest)
        raise RuntimeError(f"转格式失败（退出码 {code}）：{raw.name}")
    raw.unlink(missing_ok=True)
    print(f"  ✓ 已转成 diffusers 原生格式：{item.dest.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["lora", "llm", "yolo"],
                    help="只处理某一组（默认全部）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印要做什么、每份多大，不真下")
    ap.add_argument("--force", action="store_true",
                    help="已存在且校验通过也重新下")
    ap.add_argument("--endpoint", default="",
                    help="HuggingFace 镜像地址（国内常用 https://hf-mirror.com）。"
                         "等价于设环境变量 HF_ENDPOINT")
    ap.add_argument("--proxy", default="",
                    help="HTTP 代理（如 http://127.0.0.1:7897）。"
                         "**实测镜像 1.2MB/s、走代理 11.7MB/s**——下 4GB 时差 10 倍，"
                         "所以留了这个口子")
    args = ap.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        print(f"用代理：{args.proxy}")
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
        print(f"用镜像：{args.endpoint}")

    items = [i for i in build_items() if not args.only or i.group == args.only]

    print("VNCCS 移植所需的额外权重（来源全部照官方清单）\n")
    total = 0
    for it in items:
        size = 0 if it.manual else hf_size(it.repo, it.remote, it.revision)
        total += size
        kind = "手动" if it.manual else f"{size / 1048576:.0f} MB"
        print(f"  [{it.group}] {it.name}  ({kind})")
        print(f"        → {it.dest}")
        print(f"        {it.note}")
    print(f"\n合计约 {total / 1048576 / 1024:.2f} GB"
          + ("（不含手动那一项）" if any(i.manual for i in items) else ""))

    if args.dry_run:
        print("\n--dry-run：没动磁盘。")
        return 0

    print()
    ok, failed, skipped = 0, [], []
    for it in items:
        print(f"── {it.name}")
        if it.manual:
            if it.dest.is_file():
                print(f"  ✓ 已就位：{it.dest}")
                ok += 1
            else:
                print(f"  · 需要你手动放：{it.dest}")
                print("    —— 官方只给了路径 bbox/face_yolov8m.pt"
                      "（character_generator.py:845），没给下载地址，所以不代下。")
                skipped.append(it.key)
            continue
        try:
            if download(it, args.force):
                if it.needs_convert:
                    convert(it)
                validate_any(it, it.dest)
            ok += 1
        except Exception as exc:                             # noqa: BLE001
            print(f"  ✗ {exc}")
            failed.append(it.key)

    print()
    if failed:
        print(f"失败的：{', '.join(failed)}")
    if skipped:
        print(f"要手动补的：{', '.join(skipped)}")
    print(f"完成 {ok}/{len(items)}")
    if not args.only or args.only == "llm":
        report_runtime()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
