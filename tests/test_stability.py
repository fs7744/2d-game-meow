"""稳定性：连续推理不涨显存、跨进程同 seed 逐像素一致。"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

_GEN_SNIPPET = """
import hashlib, sys
sys.path.insert(0, r"{root}")
from workflows.engine import Engine
from workflows import generate as g
eng = Engine(r"{root}/models")
r = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
              seed=1234, pixel=4, palette=16, preset="sprite", tag="repro")
print(hashlib.sha256(open(r["file"], "rb").read()).hexdigest())
"""


def test_no_vram_leak_over_20_runs(eng):
    from workflows import generate as g

    g.txt2img(eng, prompt="warmup", width=256, height=256, steps=6, seed=1, tag="soak")
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated() / 2**20
    base_reserved = torch.cuda.memory_reserved() / 2**20

    for i in range(20):
        g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
                  seed=i, tag="soak")
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 2**20
    reserved = torch.cuda.memory_reserved() / 2**20

    assert alloc - base_alloc < 300, f"已分配显存涨了 {alloc - base_alloc:.0f}MB"
    assert reserved - base_reserved < 600, f"保留显存涨了 {reserved - base_reserved:.0f}MB"


def test_free_releases_pipeline_and_controlnet(root):
    """free() 必须同时释放 pipeline 与 ControlNet（两者是独立缓存）。

    用独立 Engine 实例测量增量：全局显存里有别的引擎（API 测试里的 app.engine），
    只看绝对值会被污染。判据是「加载后显著上涨、free 后回到基线附近」。
    """
    from workflows.engine import Engine

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated() / 2**20

    e = Engine(root / "models")
    e.txt2img_pipe()
    e.pose_pipe(controlnet="openpose", with_ipadapter=False)
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated() / 2**20
    assert loaded - base > 1500, f"加载后只涨了 {loaded - base:.0f}MB，加载可能没生效"

    e.free()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated() / 2**20

    assert e.stats()["loaded"] == []
    assert after - base < 128, f"free 后仍残留 {after - base:.0f}MB（ControlNet 未释放？）"


def test_same_seed_identical_across_processes(tmp_path):
    """换进程重跑（模型重新加载）后，同 seed 必须逐像素一致——产线可复现的前提。"""
    code = _GEN_SNIPPET.format(root=str(ROOT))
    hashes = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=900, cwd=str(ROOT), env={"HF_HUB_OFFLINE": "1",
                                                            "PATH": "", "SYSTEMROOT": "C:\\Windows"})
        assert r.returncode == 0, r.stderr[-600:]
        hashes.append(r.stdout.strip().splitlines()[-1])
    assert hashes[0] == hashes[1], "跨进程同 seed 结果不一致"


def test_output_image_is_valid_png(eng, tmp_path):
    from workflows import generate as g

    r = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
                  seed=5, tag="png")
    im = Image.open(r["file"])
    im.verify()                                   # 结构校验
    assert im.size == (256, 256)


def test_sidecar_matches_output(eng):
    import json
    from pathlib import Path as P

    from workflows import generate as g

    r = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
                  seed=77, tag="sidecar")
    side = json.loads(P(r["sidecar"]).read_text("utf-8"))
    assert side["seed"] == 77 and side["width"] == 256 and side["workflow"] == "w1"
    assert sha256(r["file"]) == sha256(side["file"]) or side["file"] == r["file"]


def sha256(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
