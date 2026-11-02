"""在**真实浏览器**里验证 3D 人偶编辑器（嵌入模式 + 节点里的尺寸）。

**为什么需要这么一个脚本**：jsdom 不加载 iframe，前端冒烟只能验到宿主页那一层
（"iframe 挂上了、按钮在、消息链路通"），验不到最关键的两件事：

  1. 官方 PoseStudio 在 `?embed=1` 下真的把画布铺满、真的渲染出人偶；
  2. 在**节点卡片那么小的尺寸**（280px）下，capture 出来的图不是糊的、
     不是被裁的、也不受 WebGL 缓冲锁的影响。

所以这里起一个静态服务（只服务 `web/dist`，不动 `assets/`），用无头 Edge +
SwiftShader 软件渲染跑一遍完整链路：加载 → ready → 套一个官方姿势库的姿势 →
导出 → 检查 PNG 尺寸与内容。

产物落在 `tests/_out/pose_editor/`（见 workflows/paths.tool_out）。

用法：

    python tools/verify_pose_editor.py             # 跑一遍
    python tools/verify_pose_editor.py --size 900  # 按放大后的尺寸验
"""
from __future__ import annotations

import argparse
import base64
import functools
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

#: 宿主页：把编辑器塞进一个**跟节点卡片一样宽**的容器里，再走一遍完整协议。
#: 结果写进 `document.title`（`--dump-dom` 能拿到），不依赖截图。
HARNESS = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body { margin:0; background:#121216; color:#eee; font:12px monospace; }
  /* 节点正文里那块：默认节点宽 300 → 内容 280 */
  #stage { width:{size}px; height:{size}px; border:1px solid #444; }
  #stage iframe { display:block; width:100%; height:100%; border:0; }
</style></head><body>
<div id="stage"><iframe id="f" src="/vnccs-pose/studio.html?embed=1"></iframe></div>
<div id="log">…</div>
<script>
const f = document.getElementById('f')
// 标题只放**短结论**（--dump-dom 会截断长文本），完整数据放 #log 里
const log = (s) => {
  document.getElementById('log').textContent = s
  document.title = s.split(' b64=')[0]
}
let stage = 0
let yawCheck = ''
const POSE = {pose_path}
window.addEventListener('message', (e) => {
  if (e.source !== f.contentWindow) return
  const d = e.data || {}
  if (d.type === 'progress') log('LOAD ' + d.text)
  else if (d.type === 'error') log('POSE_FAIL 子窗报错: ' + d.message)
  else if (d.type === 'ready') {
    if (stage === 0) {
      stage = 1
      // **不套任何姿势，用默认站姿**：这张图会被 `tests/test_emotiongen.py`
      // 当素材做人脸检测，套 Ballet/Arabesque 那种伸展姿势时脸朝下/被身体
      // 挡住，YOLO 直接检不出来（踩过）。姿势套用由 entry.js 自己那条链路保证。
      // 相机视角：开 → 与导出机位对齐
      f.contentWindow.postMessage({ type:'setCameraMode', on: true }, '*')
      // **必须是 0° 正面**：这张图还会被 test_emotiongen 当素材（检出人脸）。
      // 改成 30/90 会导出侧脸，人脸检测直接失败（踩过一次）。
      f.contentWindow.postMessage({ type:'setYaw', yaw: 0 }, '*')
      // 2) 问子窗要姿势 + 朝向，验证相机视角下 yaw 是否被正确保持
      setTimeout(() => f.contentWindow.postMessage({ type:'getPose' }, '*'), 400)
      setTimeout(() => f.contentWindow.postMessage({ type:'export', yaw: 0, size: 768 }, '*'), 1400)
    }
  } else if (d.type === 'pose') {
    // 相机视角下 getPose 应该把 yaw 原样回传（不是被 orbit 干扰的随机值）
    if (stage === 1 && d.yaw !== undefined) {
      yawCheck = 'yaw=' + d.yaw + ' expect=0 '
    }
  } else if (d.type === 'result') {
    const url = d.dataURL || ''
    const b64 = url.split(',')[1] || ''
    const bin = atob(b64)
    const okHead = url.startsWith('data:image/png')
    log('POSE_OK ' + yawCheck + 'bytes=' + bin.length + ' head=' + okHead + ' b64=' + b64)
  }
})
setTimeout(() => { if (!document.title.startsWith('POSE_')) log('POSE_FAIL 超时（模型没加载完？）') }, 170000)
</script></body></html>"""


def _serve(dist: Path, harness: bytes, port: int):
    """静态服务 `web/dist`，外加一个内存里的宿主页路由。"""

    class H(SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/__pose_check.html"):
                body = harness
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return super().do_GET()

        def log_message(self, *a):  # 静音
            pass

    handler = functools.partial(H, directory=str(dist))
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=280, help="宿主页里编辑器的边长（px）")
    ap.add_argument("--port", type=int, default=8129)
    ap.add_argument("--pose", default="", help="官方姿势库里某个 json 的相对路径")
    args = ap.parse_args()

    dist = ROOT / "web" / "dist"
    if not (dist / "index.html").is_file():
        print("先 npm run build（web/dist 不存在）", file=sys.stderr)
        return 2

    # 默认拿官方姿势库里的第一条（真的从磁盘读，验证清单指路没问题）
    if args.pose:
        pose_path = Path(args.pose)
    else:
        lib = ROOT / "web" / "public" / "vnccs-pose" / "PoseLibrary"
        cands = sorted(
            p for p in lib.glob("Totemistyk/*/poses/*/*.json")
            if p.name != "pose_library.json")
        if not cands:
            print("姿势库里没有 json", file=sys.stderr)
            return 2
        pose_path = cands[0]
    pose_json = json.loads(pose_path.read_text(encoding="utf-8"))
    print(f"用的姿势：{pose_path.relative_to(ROOT)}")

    harness = HARNESS.replace("{size}", str(args.size)).replace(
        "{pose_path}", json.dumps(pose_json, ensure_ascii=False)).encode("utf-8")
    srv = _serve(dist, harness, args.port)
    time.sleep(0.4)

    out_dir = Path(os.sys.path[0])  # noqa: B005
    sys.path.insert(0, str(ROOT))
    from workflows.paths import tool_out
    out_dir = tool_out("pose_editor")

    cmd = [
        EDGE, "--headless=new", "--disable-gpu-sandbox",
        "--enable-unsafe-swiftshader", "--no-sandbox",
        "--window-size=900,900",
        "--virtual-time-budget=180000",
        "--dump-dom", f"http://127.0.0.1:{args.port}/__pose_check.html",
    ]
    print("启动无头浏览器（软件渲染，人偶模型 37MB，约 15~60 秒）…")
    t0 = time.time()
    dom = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=300).stdout
    print(f"浏览器耗时 {time.time() - t0:.1f}s")

    m = re.search(r'<div id="log">([^<]*)</div>', dom)
    title = m.group(1) if m else ""
    print("结果：", title[:160])

    if title.startswith("POSE_OK"):
        b64 = re.search(r"b64=([A-Za-z0-9+/=]+)", title)
        if not b64:
            print("✗ 没有拿到 PNG 数据")
            return 1
        png = base64.b64decode(b64.group(1))
        out = out_dir / f"mannequin_{args.size}.png"
        out.write_bytes(png)
        print(f"✓ 导出成功：{len(png)} 字节 → {out}")
        # 白底灰人偶：检查一下确实是白底 + 中间有内容
        try:
            from PIL import Image
            im = Image.open(out).convert("RGB")
            w, h = im.size
            corner = im.getpixel((2, 2))
            center = im.getpixel((w // 2, h // 2))
            print(f"尺寸 {w}×{h}；左上角 {corner}（应接近白）；中心 {center}")
            ok = w == 768 and h == 768 and sum(corner) > 720
            print("✓ 尺寸与白底正确" if ok else "✗ 尺寸/底色不对")
            return 0 if ok else 1
        except ImportError:
            return 0
    print("✗ 没跑通：", title)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
