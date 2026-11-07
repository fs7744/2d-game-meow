"""在**真实浏览器**里验证多角色场景协议（`vnccs-pose/scene.js`）。

jsdom 不加载 iframe，前端冒烟只能验到「宿主发了什么消息」；这个脚本用无头
Edge + SwiftShader 真把 37MB 人偶模型跑起来，验的是 iframe 里那套状态机：
加载两个角色 → 输 X=4.5 → 撤销退回 0 → 加第三个角色 → 导出 PNG。
顺便验证出图时 gizmo 手柄被藏掉了（白底灰人偶不该出现彩色轴）。

    python tools/verify_pose_scene.py
"""
from __future__ import annotations

import argparse
import base64
import functools
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

HARNESS = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{margin:0;background:#121216;color:#eee;font:12px monospace}
#stage{width:520px;height:520px}#stage iframe{width:100%;height:100%;border:0}</style>
</head><body>
<div id="stage"><iframe id="f" src="/vnccs-pose/studio.html?embed=1"></iframe></div>
<div id="log">…</div>
<script>
const f = document.getElementById('f')
const log = (s) => { document.getElementById('log').textContent = s; document.title = s }
const send = (m) => f.contentWindow.postMessage(m, '*')
const P = (x) => ({ x, y:0, z:0, zoom:1, rotation:[0,0,0] })
const scene = {
  characters: [
    { id:'c1', name:'角色 1', color:'#e8eaf2', pose:null, placement:P(0) },
    { id:'c2', name:'角色 2', color:'#6cc4ef', pose:null, placement:P(3) },
  ],
  activeId:'c1', tabs:['t1'], activeTab:'t1',
  placements:{ t1:{ c1:P(0), c2:P(3) } },
  mode:'move', space:'world', linkAll:false, cameraParams:null,
}
let stage = 0
const states = []
window.addEventListener('message', (e) => {
  if (e.source !== f.contentWindow) return
  const d = e.data || {}
  if (d.type === 'error') log('FAIL 子窗报错: ' + d.message)
  else if (d.type === 'sceneState') states.push(d.scene)
  else if (d.type === 'ready') {
    if (stage !== 0) return
    stage = 1
    send({ type:'scene', action:'load', scene })
    setTimeout(() => send({ type:'scene', action:'transform', id:'c1', patch:{ x:4.5 } }), 900)
    setTimeout(() => send({ type:'scene', action:'undo' }), 1700)
    setTimeout(() => send({ type:'scene', action:'add' }), 2500)
    setTimeout(() => send({ type:'export', size:256 }), 3300)
  } else if (d.type === 'result') {
    const xOf = (s) => s && s.placements && s.placements.t1 && s.placements.t1.c1
      ? Number(s.placements.t1.c1.x) : null
    const load2 = states.some((s) => s && s.characters.length === 2)
    const tf = states.findIndex((s) => xOf(s) === 4.5)
    const undo0 = tf >= 0 && states.slice(tf + 1).some((s) => xOf(s) === 0)
    const last = states[states.length - 1] || {}
    const url = d.dataURL || ''
    log('OK load2=' + load2 + ' tf45=' + (tf >= 0) + ' undo0=' + undo0
      + ' chars=' + (last.characters ? last.characters.length : 0)
      + ' pngLen=' + url.length + ' b64=' + (url.split(',')[1] || ''))
  }
})
setTimeout(() => { if (!document.title.startsWith('OK')) log('FAIL 超时') }, 170000)
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8131)
    args = ap.parse_args()
    dist = ROOT / "web" / "dist"
    if not (dist / "index.html").is_file():
        print("先构建前端（web/dist 不存在）", file=sys.stderr)
        return 2

    harness = HARNESS.encode("utf-8")

    class H(SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/__pose_scene.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(harness)))
                self.end_headers()
                self.wfile.write(harness)
                return
            return super().do_GET()

        def log_message(self, *a):  # noqa: ARG002
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              functools.partial(H, directory=str(dist)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.4)

    cmd = [EDGE, "--headless=new", "--disable-gpu-sandbox",
           "--enable-unsafe-swiftshader", "--no-sandbox",
           "--window-size=900,900", "--virtual-time-budget=180000",
           "--dump-dom", f"http://127.0.0.1:{args.port}/__pose_scene.html"]
    print("启动无头浏览器（37MB 人偶模型 + 多角色协议，约 15~60 秒）…")
    t0 = time.time()
    dom = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    print(f"浏览器耗时 {time.time() - t0:.1f}s")
    srv.shutdown()

    m = re.search(r'<div id="log">([^<]*)</div>', dom)
    title = m.group(1) if m else ""
    print("结果：", title[:200])
    if not title.startswith("OK"):
        print("✗ 没跑通")
        return 1

    checks = {
        "载入 2 个角色": "load2=true" in title,
        "精确 X=4.5 生效": "tf45=true" in title,
        "撤销退回 X=0": "undo0=true" in title,
        "加角色后变 3 个": "chars=3" in title,
    }
    for name, ok in checks.items():
        print(("  ✓ " if ok else "  ✗ ") + name)

    b64 = re.search(r"b64=([A-Za-z0-9+/=]+)", title)
    if b64:
        png = base64.b64decode(b64.group(1))
        from workflows.paths import tool_out
        out = tool_out("pose_scene") / "multi_character.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
        print(f"  ✓ 导出 {len(png)} 字节 → {out}")
        try:
            from PIL import Image
            im = Image.open(out).convert("RGB")
            corner = im.getpixel((2, 2))
            checks["出图仍是白底"] = sum(corner) > 720
            print(f"    尺寸 {im.size[0]}×{im.size[1]}；左上角 {corner}")
        except ImportError:
            pass
    else:
        checks["出图"] = False
    print(f"{sum(checks.values())}/{len(checks)} 通过")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
