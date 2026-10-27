"""按 Range 分块下载：抗限流、抗断流、逐块校验。

为什么不用 curl -C -：镜像返回 429 时会**把错误 JSON 当正文追加进文件**，
续传几次文件就废了（实测 317 字节的 "rate limit" 文本混进了 .onnx）。
这里每次只请求剩余区间，非 206 一律丢弃重试，长度核对通过才落盘。

用法：
    python tools/fetch_ranged.py <url> <out_path> [--expect 字节数] [--tries 60]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def total_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1])
    except Exception:  # noqa: BLE001
        return None
    return None


def fetch(url: str, out: Path, expect: int | None, tries: int, block: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and expect and out.stat().st_size > expect:
        out.unlink()                      # 之前被污染的文件，重置
    for attempt in range(1, tries + 1):
        have = out.stat().st_size if out.exists() else 0
        if expect and have >= expect:
            print(f"[ok] {out.name} {have} 字节", flush=True)
            return 0
        end = (have + block - 1) if (expect is None) else min(have + block - 1, expect - 1)
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={have}-{end}", "User-Agent": "curl/8"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"HTTP {r.status}（不是 206，丢弃）")
                chunk = r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, OSError) as exc:
            wait = min(3 * attempt, 20)
            print(f"[{attempt}/{tries}] 失败 {type(exc).__name__}: {str(exc)[:70]}，{wait}s 后重试",
                  flush=True)
            time.sleep(wait)
            continue
        if not chunk:
            time.sleep(2)
            continue
        with out.open("ab") as fh:
            fh.write(chunk)
        done = out.stat().st_size
        pct = f"{done / expect * 100:.1f}%" if expect else f"{done} 字节"
        print(f"[{attempt}/{tries}] +{len(chunk) / 2**20:.1f}MB -> {done / 2**20:.1f}MB ({pct})",
              flush=True)
    have = out.stat().st_size if out.exists() else 0
    if expect and have >= expect:
        return 0
    print(f"[FAIL] 未完成：{have}/{expect}", flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--block", type=int, default=8 * 2**20)
    a = ap.parse_args()

    out = Path(a.out)
    expect = a.expect or total_size(a.url)
    print(f"目标 {out}，总长 {expect}", flush=True)
    return fetch(a.url, out, expect, a.tries, a.block)


if __name__ == "__main__":
    sys.exit(main())
