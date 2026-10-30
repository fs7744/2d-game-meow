"""并行分段下载（断点续传）。用于 hf-mirror 单连接被限速的大权重。

单连接实测 ~475 KB/s，8 并发可到 ~2.2 MB/s；每段独立 part 文件，
中断后重跑会跳过已完成的段，全部下完后按序拼接。

用法:
    python tools/fetch_parallel.py <url> <目标文件> [--jobs 16] [--chunk-mb 24] [--proxy ...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PARTS_SUFFIX = ".parts"


def head_size(url: str, proxy: str | None) -> int:
    cmd = ["curl", "-sIL", "-m", "60"]
    if proxy:
        cmd += ["-x", proxy]
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    sizes = []
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                sizes.append(int(line.split(":", 1)[1].strip()))
            except ValueError:
                pass
    big = [s for s in sizes if s > 1024 * 1024]
    if not big:
        raise SystemExit(f"拿不到 Content-Length，原始输出:\n{out}")
    return max(big)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dst")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=24)
    ap.add_argument("--proxy", default=os.environ.get("HTF_PROXY", "http://127.0.0.1:7897"))
    ap.add_argument("--timeout", type=int, default=180, help="单段超时秒数")
    args = ap.parse_args()

    total = head_size(args.url, args.proxy)
    dst = os.path.abspath(args.dst)
    parts_dir = dst + PARTS_SUFFIX
    os.makedirs(parts_dir, exist_ok=True)

    # 已有前缀（之前用 curl -C - 顺序续传下来的连续字节）当作第 0 段
    prefix = os.path.getsize(dst) if os.path.exists(dst) else 0
    if prefix > total:
        print(f"已有文件 {prefix} 比远端 {total} 还大，可能下坏了，请人工确认")
        return 1

    chunk = args.chunk_mb * 1024 * 1024
    spans = []
    start = prefix
    while start < total:
        end = min(start + chunk, total) - 1
        spans.append((start, end))
        start = end + 1

    print(f"总大小 {total/1024/1024:.0f} MB，已有前缀 {prefix/1024/1024:.0f} MB，"
          f"待下 {len(spans)} 段 x {args.chunk_mb} MB，并发 {args.jobs}")

    lock = threading.Lock()
    done_bytes = 0
    done_cnt = 0
    t0 = time.time()

    def part_path(i: int) -> str:
        return os.path.join(parts_dir, f"{i:05d}.part")

    def fetch(i: int) -> int:
        nonlocal done_bytes, done_cnt
        a, b = spans[i]
        want = b - a + 1
        pp = part_path(i)
        if os.path.exists(pp) and os.path.getsize(pp) == want:
            with lock:
                done_bytes += want
                done_cnt += 1
            return want
        tmp = pp + ".tmp"
        cmd = ["curl", "-sL", "-m", str(args.timeout),
               "-r", f"{a}-{b}", "-o", tmp, args.url]
        if args.proxy:
            cmd[1:1] = ["-x", args.proxy]
        for attempt in range(3):
            subprocess.run(cmd, capture_output=True)
            sz = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if sz == want:
                os.replace(tmp, pp)
                with lock:
                    done_bytes += want
                    done_cnt += 1
                    pct = done_bytes / (total - prefix) * 100
                    spd = done_bytes / max(time.time() - t0, 1e-6) / 1024
                    eta = (total - prefix - done_bytes) / max(done_bytes / max(time.time() - t0, 1e-6), 1)
                    print(f"[{done_cnt}/{len(spans)}] {pct:5.1f}%  {spd:6.0f} KB/s  "
                          f"ETA {eta/60:5.1f} min", flush=True)
                return want
            time.sleep(1 + attempt * 2)
        with lock:
            print(f"第 {i} 段失败（拿到 {sz}/{want}），保留 tmp 供下次重试", flush=True)
        return 0

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(fetch, range(len(spans))))

    missing = [i for i in range(len(spans)) if not os.path.exists(part_path(i))]
    if missing:
        print(f"还有 {len(missing)} 段没下完，重跑本脚本即可续传")
        return 2

    # 拼接：保留已有前缀，按序追加各段
    print("开始拼接…", flush=True)
    with open(dst, "r+b" if prefix else "wb") as f:
        f.truncate(prefix)
        f.seek(prefix)
        for i in range(len(spans)):
            with open(part_path(i), "rb") as p:
                shutil.copyfileobj(p, f, 8 * 1024 * 1024)
    got = os.path.getsize(dst)
    if got != total:
        print(f"拼接后大小不对: {got} != {total}")
        return 1
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"完成: {dst}  {got/1024/1024:.0f} MB，用时 {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
