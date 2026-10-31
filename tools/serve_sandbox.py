"""起一个**沙盒实例**：产物、上传、项目、模板全落在 `tests/_out/ui/`。

**为什么需要这个脚本**：做界面验证（截图、点按钮、跑 demo 模式）时必须有一个
能跑的服务，但**不能用真 `assets/`**。前端的首屏逻辑是「一个项目都没有就新建
一个」，于是每次用无痕的 headless 浏览器连过来，都会在用户的项目列表里多出一个
空项目，并把 `workspace.json` 的 `current` 指过去——用户下次打开就发现自己的
画布被切走了。实测在 20:58 和 21:05 各发生一次。

**为什么必须用脚本、不能直接 `MEOW_ASSETS_DIR=... uvicorn app:app`**：
`workflows.paths.ASSETS` 是 **import 时求值**的模块级常量。`uvicorn app:app`
会在导入 `app` 时连锁导入 `workflows`，环境变量必须在那之前设好。写成脚本可以
保证顺序（先 `os.environ`，再 `import app`）；在命令行里手写前缀也行，但容易漏
一个（比如只设 ASSETS 没设 TEMPLATES），漏了就往真目录里写。

用法：

    python tools/serve_sandbox.py                 # 沙盒，端口 8123
    python tools/serve_sandbox.py --port 8200     # 换端口
    python tools/serve_sandbox.py --reset         # 先清空沙盒目录再起
    python tools/serve_sandbox.py --real          # 明确要用真 assets/（别在验证时用）

沙盒里的东西可以随便删——它不在交付路径上。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 沙盒根。放在 tests/_out 下，和 pytest 的产物是邻居但不共用
#: （pytest 会清/建自己的夹具，混在一起会让 UI 验证看到测试的残留项目）。
SANDBOX = ROOT / "tests" / "_out" / "ui"


def sandbox_dirs() -> dict[str, str]:
    """沙盒要用的两个目录（**只算不设**，方便测试）。

    返回的就是要写进环境变量的内容。抽成纯函数是为了让测试能验证
    「它指的确实是 `tests/_out/ui/`，而不是用户真实的 `assets/`」，
    又不必真的去改测试进程的环境变量（改了会影响后续用例的路径）。
    """
    return {
        "MEOW_ASSETS_DIR": str(SANDBOX / "assets"),
        "MEOW_TEMPLATES_DIR": str(SANDBOX / "templates"),
    }


def _prepare(reset: bool) -> None:
    if reset and SANDBOX.exists():
        # 只删沙盒自己。这句断言是防手滑：任何时候都不该删到 tests/_out 之外。
        assert (ROOT / "tests" / "_out") in SANDBOX.parents, SANDBOX
        shutil.rmtree(SANDBOX)
    (SANDBOX / "assets").mkdir(parents=True, exist_ok=True)
    (SANDBOX / "templates").mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="起一个不动真实 assets/ 的沙盒实例")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--reset", action="store_true", help="先清空沙盒目录")
    ap.add_argument("--real", action="store_true",
                    help="用真实 assets/ 与 templates/（做发布前验证时才需要）")
    args = ap.parse_args()

    if args.real:
        print("[!] --real：这个实例会读写真实的 assets/ 与 templates/", file=sys.stderr)
    else:
        _prepare(args.reset)
        # **必须在 import app 之前**：paths 模块在 import 时就把这两个值求出来了。
        for key, val in sandbox_dirs().items():
            os.environ[key] = val

    import uvicorn

    import app as app_module
    from workflows import paths

    print(f"  资产根  {paths.ASSETS}")
    print(f"  模板根  {paths.TEMPLATES_DIR}")
    print(f"  地址    http://{args.host}:{args.port}")

    uvicorn.run(app_module.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
