"""让 ComfyUI 的 TRELLIS.2 实现能在 torch 2.6 / Python 3.13 下 import。

    python tools/patch_comfy_kitchen.py

**为什么需要这个补丁**：

`comfy_kitchen`（ComfyUI 的量化算子包，TRELLIS.2 的 int8 模型要用）在注册
torch custom op 时用了 PEP 585 的内建泛型注解 `list[int]`：

    def conv3d(x: Tensor, ..., stride: list[int], out: Tensor) -> None: ...

而 torch 2.6 的 `torch.library.infer_schema()` 只认 `typing.List[int]`，
遇到 `list[int]` 直接抛：

    ValueError: infer_schema(func): Parameter stride has unsupported type list[int]

这个错误发生在 **import 阶段**，所以整条链路根本起不来（`import comfy.ops`
就炸）。torch 2.7+ 才支持内建泛型。

**修法**：把这些注解换成 `typing.List[...]`（语义完全等价，只影响注解解析，
不改任何算子逻辑）。当前实测包内 16 个文件、共 40 处。

**这个补丁是幂等的**，已经改过的文件会跳过——重复跑没有副作用。
升级 `comfy_kitchen` 之后要重跑一次（新版本的文件会覆盖回去）。

**长期方案**：升级 torch 到 2.7+ 就不需要这个补丁了。但升级会牵动 Qwen 产线
（本项目 2D 生成的主力），所以先在注解层解决，不动 torch。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: 只在这几个包里找（不要全盘扫 site-packages，会误伤）
TARGET_PACKAGES = ("comfy_kitchen",)

_PATTERN = re.compile(r"(?<![\w.])list\[")


def patch_file(path: Path) -> int:
    """→ 替换的处数（0 表示不需要改）。"""
    s = path.read_text(encoding="utf-8")
    if not _PATTERN.search(s):
        return 0
    n = len(_PATTERN.findall(s))
    s2 = _PATTERN.sub("List[", s)
    if not re.search(r"^from typing import .*\bList\b", s2, re.M):
        if re.search(r"^import torch$", s2, re.M):
            s2 = s2.replace("import torch", "import torch\nfrom typing import List", 1)
        elif re.search(r"^from typing import ", s2, re.M):
            s2 = re.sub(r"^from typing import ", "from typing import List, ",
                        s2, count=1, flags=re.M)
        else:
            s2 = "from typing import List\n" + s2
    path.write_text(s2, encoding="utf-8", newline="")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="patch comfy_kitchen 的 list[...] 注解")
    ap.add_argument("--site-packages", default="", help="site-packages 路径（默认自动找）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.site_packages:
        root = Path(args.site_packages)
    else:
        root = Path(sys.prefix) / "Lib" / "site-packages"
        if not root.is_dir():                       # 非 Windows
            root = Path(sys.prefix) / "lib"
            cands = sorted(root.glob("python3.*/site-packages"))
            if not cands:
                print(f"找不到 site-packages（试过 {root}）")
                return 2
            root = cands[-1]

    total_files = 0
    total_sites = 0
    for pkg in TARGET_PACKAGES:
        base = root / pkg
        if not base.is_dir():
            print(f"跳过 {pkg}：没装（{base} 不存在）")
            continue
        for f in sorted(base.rglob("*.py")):
            if args.dry_run:
                s = f.read_text(encoding="utf-8")
                n = len(_PATTERN.findall(s))
                if n:
                    print(f"  待改 {f.relative_to(root)}  {n} 处")
                    total_files += 1
                    total_sites += n
                continue
            n = patch_file(f)
            if n:
                print(f"  已改 {f.relative_to(root)}  {n} 处")
                total_files += 1
                total_sites += n

    if total_files == 0:
        print("没有需要改的地方（已经 patch 过了，或者包不在）")
    else:
        print(f"{'需要' if args.dry_run else '已'} 修改 {total_files} 个文件、共 {total_sites} 处")
    if not args.dry_run and total_files:
        print("\n验证：python -c \"import sys; sys.path.insert(0, '<ComfyUI 根>'); "
              "from comfy.ldm.trellis2.vae import SparseTensor; print('ok')\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
