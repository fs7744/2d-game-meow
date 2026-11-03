"""ComfyUI 客户端：把「单图 → 3D 模型」跑在外部 ComfyUI 进程上。

**为什么是外部进程而不是本项目内嵌**：TRELLIS.2 官方仓库明说"只在 Linux 测过"，
依赖里 `nvdiffrast` / `cumesh` / `o-voxel` / `flexgemm` 全要编译 CUDA 扩展。
ComfyUI 把这些依赖的 Windows 轮子都解决好了，我们只需要一个 HTTP 客户端。

代价是多一个进程 + 一套模型（10~12GB），所以：

  · 服务地址走环境变量 `MEOW_COMFY_URL`（默认 `http://127.0.0.1:8188`）；
  · **显存串行**：Qwen 和 TRELLIS.2 在 24GB 卡上不能同时常驻。本项目 engine 有
    卸载接口，调用方要先 `engine.unload()` 再跑 3D（见 `generate()` 的注释）。

ComfyUI 的四步协议（官方 API，无认证）：

    POST /upload/image   表单上传输入图 → 返回 {name, subfolder, type}
    POST /prompt         {prompt: {id: {class_type, inputs}}, client_id} → {prompt_id}
    GET  /history/{id}   轮询，直到 outputs 出现（或 status 出错）
    GET  /view           按 {filename, subfolder, type} 取产物字节

**队列是全局的**：ComfyUI 单实例只有一个执行队列，所以本模块不并发提交，
且要配合 `timeout`，否则一个坏任务会把后面全部堵住。
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_URL = "http://127.0.0.1:8188"

#: 3D 生成是重活：TRELLIS.2 在 4090 上 1~3 分钟，留足余量
GENERATE_TIMEOUT = 900.0
POLL_INTERVAL = 1.5


class ComfyError(RuntimeError):
    """ComfyUI 不可用 / 任务失败（调用方转 503 或 400）。"""


def base_url() -> str:
    return (os.environ.get("MEOW_COMFY_URL") or DEFAULT_URL).rstrip("/")


# ---------------------------------------------------------------- 传输


def _request(url: str, *, data: Optional[bytes] = None,
             headers: Optional[Dict[str, str]] = None,
             timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers or {},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()[:400].decode("utf-8", "replace")
        raise ComfyError(f"ComfyUI 返回 {exc.code}：{body}") from exc
    except urllib.error.URLError as exc:
        raise ComfyError(
            f"连不上 ComfyUI（{base_url()}）：{exc.reason}。"
            "先启动 ComfyUI，或用 MEOW_COMFY_URL 指定地址") from exc
    except TimeoutError as exc:
        raise ComfyError(f"ComfyUI 超时（{timeout:.0f}s）") from exc


def _get_json(path: str, timeout: float = 30.0) -> Any:
    return json.loads(_request(base_url() + path, timeout=timeout) or b"{}")


def available(timeout: float = 3.0) -> bool:
    """探活。**不要用它当"能不能生成"的判据**——只说明服务在。"""
    try:
        _get_json("/system_stats", timeout=timeout)
        return True
    except ComfyError:
        return False


def info() -> Dict[str, Any]:
    """服务信息（前端用来显示"3D 后端在线/离线 + 显存"）。"""
    try:
        stats = _get_json("/system_stats", timeout=5.0)
    except ComfyError as exc:
        return {"online": False, "error": str(exc)}
    dev = (stats.get("devices") or [{}])[0]
    return {
        "online": True,
        "url": base_url(),
        "device": dev.get("name", ""),
        "vramTotal": dev.get("vram_total", 0),
        "vramFree": dev.get("vram_free", 0),
        "comfyVersion": (stats.get("system") or {}).get("comfyui_version", ""),
    }


def upload_image(path: Path, *, subfolder: str = "meow3d") -> str:
    """上传输入图，返回 ComfyUI 里的文件名（`/prompt` 里 LoadImage 要用它）。

    走 multipart/form-data —— 手拼的，因为不想为这一个请求引 requests 依赖。
    边界串必须够随机且不出现在内容里，用 uuid4 就够。
    """
    path = Path(path)
    if not path.is_file():
        raise ComfyError(f"找不到输入图：{path}")
    boundary = "----meow" + uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts: List[bytes] = []
    for name, value in (("type", "input"), ("subfolder", subfolder),
                        ("overwrite", "true")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="image"; filename="{path.name}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    raw = _request(base_url() + "/upload/image", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=120.0)
    doc = json.loads(raw or b"{}")
    name = doc.get("name")
    if not name:
        raise ComfyError(f"上传返回里没有 name：{doc}")
    sub = doc.get("subfolder") or ""
    return f"{sub}/{name}" if sub else str(name)


def submit(prompt: Dict[str, Any], *, client_id: str = "") -> str:
    """提交 API 格式的工作流，返回 prompt_id。"""
    payload = json.dumps({"prompt": prompt,
                          "client_id": client_id or uuid.uuid4().hex}).encode()
    raw = _request(base_url() + "/prompt", data=payload,
                  headers={"Content-Type": "application/json"}, timeout=60.0)
    doc = json.loads(raw or b"{}")
    # node_errors 是"工作流本身不合法"，比 prompt_id 更有用
    if doc.get("node_errors"):
        msg = json.dumps(doc["node_errors"], ensure_ascii=False)[:600]
        raise ComfyError(f"工作流有错误节点：{msg}")
    pid = doc.get("prompt_id")
    if not pid:
        raise ComfyError(f"提交没拿到 prompt_id：{doc}")
    return str(pid)


def wait(prompt_id: str, *, timeout: float = GENERATE_TIMEOUT,
         on_progress=None) -> Dict[str, Any]:
    """轮询直到任务结束，返回 history 里那一条。"""
    t0 = time.time()
    last_note = ""
    while time.time() - t0 < timeout:
        try:
            hist = _get_json(f"/history/{prompt_id}", timeout=15.0)
        except ComfyError:
            # 单次失败不算失败：ComfyUI 在重活时会短暂不响应
            time.sleep(POLL_INTERVAL)
            continue
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                msgs = status.get("messages") or []
                raise ComfyError(f"任务失败：{json.dumps(msgs[-2:], ensure_ascii=False)[:600]}")
            if entry.get("outputs"):
                return entry
            if status.get("completed"):
                return entry
        if on_progress:
            note = _queue_note(prompt_id)
            if note and note != last_note:
                last_note = note
                on_progress(note)
        time.sleep(POLL_INTERVAL)
    raise ComfyError(f"超过 {timeout:.0f} 秒还没出结果（模型第一次跑要下载权重，"
                     "也可能是显存不足）")


def _queue_note(prompt_id: str) -> str:
    """从队列状态里拼一句人话（"排队中 / 正在跑第 3 个节点"）。"""
    try:
        q = _get_json("/queue", timeout=5.0)
    except ComfyError:
        return ""
    running = (q.get("queue_running") or [])
    pending = (q.get("queue_pending") or [])
    for item in running:
        if len(item) > 1 and str(item[1]) == prompt_id:
            return "生成中"
    if any(len(i) > 1 and str(i[1]) == prompt_id for i in pending):
        return f"排队中（前面还有 {len(pending)} 个）"
    return ""


def outputs(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """把 history 条目里的产物拍平成 [{filename, subfolder, type}]。"""
    out: List[Dict[str, str]] = []
    for _node_id, node_out in (entry.get("outputs") or {}).items():
        for key in ("3d", "glb", "result", "meshes", "images"):
            for item in (node_out.get(key) or []):
                if isinstance(item, dict) and item.get("filename"):
                    out.append({"filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    return out


def fetch(item: Dict[str, str]) -> bytes:
    q = urllib.parse.urlencode({"filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    return _request(f"{base_url()}/view?{q}", timeout=180.0)


# ---------------------------------------------------------------- 工作流


def node_signature(class_type: str) -> Dict[str, Any]:
    """查一个节点的输入定义（调试用）。

    **为什么留这个函数**：ComfyUI 的 API prompt 字段名必须和节点定义完全一致，
    写错不会报错、只会让节点用默认值或直接失败。做适配时先 `--inspect` 打一遍。
    """
    doc = _get_json("/object_info/" + urllib.parse.quote(class_type), timeout=20.0)
    node = doc.get(class_type) or {}
    return {
        "input": node.get("input") or {},
        "output": node.get("output") or [],
        "output_name": node.get("output_name") or [],
        "input_order": node.get("input_order") or {},
    }


#: TRELLIS.2 三阶段的关键节点。**顺序就是执行顺序**，不要按字母排。
TRELLIS2_STAGES = ("Trellis2Conditioning", "EmptyTrellis2LatentStructure",
                   "Trellis2ShapeStage", "Trellis2UpsampleStage", "Trellis2TextureStage")


def trellis_available() -> Dict[str, Any]:
    """检查 ComfyUI 里有没有 TRELLIS.2 的核心节点（决定能不能跑）。"""
    try:
        doc = _get_json("/object_info", timeout=60.0)
    except ComfyError as exc:
        return {"online": False, "error": str(exc), "missing": list(TRELLIS2_STAGES)}
    have = set(doc)
    missing = [n for n in TRELLIS2_STAGES if n not in have]
    has_3d_save = any(n in have for n in ("Save3DAdvanced", "MeshToFile3D"))
    return {
        "online": True,
        "nodes": len(have),
        "missing": missing,
        "canSave3D": has_3d_save,
        "ready": not missing and has_3d_save,
    }


def generate(image_path: Path, out_path: Path, *, timeout: float = GENERATE_TIMEOUT,
             on_progress=None) -> Dict[str, Any]:
    """单图 → GLB。

    **前置**：调用方必须先把 Qwen 从显存里卸掉（`engine.unload()`）——
    TRELLIS.2 在 4090 上要 16~24GB，和 Qwen 共存必然 OOM。
    本函数不自己卸，因为卸载策略属于 engine 的职责。

    prompt 目前只做**能力检查**，不构造实际工作流：
    ComfyUI 的节点字段名必须对着 `/object_info` 写，不能凭猜（猜错的后果是
    节点用默认值，生成出莫名其妙的结果，而且不报错）。见 `node_signature()`。
    """
    raise ComfyError(
        "3D 工作流还没接上：需要先装 ComfyUI 并确认 TRELLIS.2 核心节点可用"
        f"（检查结果：{json.dumps(trellis_available(), ensure_ascii=False)}）。"
        "接法见 docs/3D模型生成-Godot工作流.md")
