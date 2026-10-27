"""下载器：Range 分块、续传、拒绝非 206 响应（防止 429 错误体污染文件）。"""
from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path

import pytest

PAYLOAD = bytes(range(256)) * 400          # 102400 字节，内容可校验


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """支持 Range 的最小实现；/norange 故意忽略 Range 并回 200。"""

    def log_message(self, *a):  # 静音
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        if self.path.startswith("/ok") and rng:
            start_s, _, end_s = rng.replace("bytes=", "").partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(PAYLOAD) - 1
            end = min(end, len(PAYLOAD) - 1)
            body = PAYLOAD[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)       # 非 206：下载器必须丢弃
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _RangeHandler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_total_size_reads_content_range(server):
    from tools.fetch_ranged import total_size

    assert total_size(f"{server}/ok") == len(PAYLOAD)


def test_ranged_download_reassembles_exact_bytes(server, tmp_path):
    from tools.fetch_ranged import fetch

    out = tmp_path / "blob.bin"
    rc = fetch(f"{server}/ok", out, expect=len(PAYLOAD), tries=30, block=8192)
    assert rc == 0
    assert out.stat().st_size == len(PAYLOAD)
    assert out.read_bytes() == PAYLOAD, "分块拼接内容与原始不一致"


def test_resume_continues_from_existing(server, tmp_path):
    from tools.fetch_ranged import fetch

    out = tmp_path / "partial.bin"
    out.write_bytes(PAYLOAD[:5000])                 # 预置断点
    rc = fetch(f"{server}/ok", out, expect=len(PAYLOAD), tries=30, block=8192)
    assert rc == 0
    assert out.read_bytes() == PAYLOAD


def test_non_206_is_rejected_not_appended(server, tmp_path):
    """核心回归：服务器回 200（例如 429 错误页）时，绝不能把响应当数据追加。"""
    from tools.fetch_ranged import fetch

    out = tmp_path / "bad.bin"
    rc = fetch(f"{server}/norange", out, expect=len(PAYLOAD), tries=2, block=4096)
    assert rc == 1, "非 206 必须判定失败"
    assert (not out.exists()) or out.stat().st_size == 0, "失败时不应留下污染数据"


def test_oversized_stale_file_is_reset(server, tmp_path):
    """比目标还大的旧文件（被污染过）应当被清掉重下。"""
    from tools.fetch_ranged import fetch

    out = tmp_path / "stale.bin"
    out.write_bytes(PAYLOAD + b"garbage")
    rc = fetch(f"{server}/ok", out, expect=len(PAYLOAD), tries=30, block=8192)
    assert rc == 0
    assert out.read_bytes() == PAYLOAD
