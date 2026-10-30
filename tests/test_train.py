"""LoRA 训练：数据集、步数规划、导出格式、API 端点。"""
from __future__ import annotations

import json

import pytest

# SD 权重已归档（models/_archive/sd/）：这条用例要真跑 SD 管线 / 训练 SD LoRA，
# 权重不在就跳过；搬回来自动恢复（不是删测试）。
from pathlib import Path as _P

_SD_WEIGHTS = (_P(__file__).resolve().parents[1] / "models" / "checkpoints"
               / "sd15" / "model_index.json").is_file()
requires_sd = pytest.mark.skipif(
    not _SD_WEIGHTS,
    reason="SD 权重已归档到 models/_archive/sd/（恢复后自动恢复）")
from PIL import Image

from workflows import lora as L
from workflows import train as T


def _imgs(root, n=2, size=128):
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = root / f"a{i}.png"
        Image.new("RGB", (size, size), (20 * i, 60, 120)).save(p)
        out.append(p)
    return out


# ---------------- 数据集 ----------------

def test_dataset_reads_sidecar_caption(tmp_path):
    imgs = _imgs(tmp_path / "ds", 2)
    (imgs[0].with_suffix(".txt")).write_text("mk_knight, armored", encoding="utf-8")

    ds = T.Dataset(tmp_path / "ds", caption="fallback, style")
    caps = {p.name: c for p, c in ds.items}
    assert caps["a0.png"] == "mk_knight, armored"     # 有 txt 用它
    assert caps["a1.png"] == "fallback, style"        # 没有就退回统一 caption


def test_dataset_skips_images_without_any_caption(tmp_path):
    _imgs(tmp_path / "ds", 2)
    with pytest.raises(ValueError, match="没有可用样本"):
        T.Dataset(tmp_path / "ds", caption="")


def test_dataset_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        T.Dataset(tmp_path / "nope")


def test_dataset_tensor_shape_and_range(tmp_path):
    imgs = _imgs(tmp_path / "ds", 1, size=300)
    (imgs[0].with_suffix(".txt")).write_text("x", encoding="utf-8")
    ds = T.Dataset(tmp_path / "ds", resolution=256)
    t, cap = ds.tensor(0)
    assert t.shape == (3, 256, 256) and cap == "x"
    assert -1.05 <= float(t.min()) and float(t.max()) <= 1.05


# ---------------- 步数规划 ----------------

def test_plan_steps_scales_with_samples():
    assert T.plan_steps(4, epochs=8) >= 100          # 下限，太少学不到东西
    assert T.plan_steps(200, epochs=8) <= 2000       # 上限，防过拟合
    assert T.plan_steps(6, epochs=8) == 100          # 6*8=48 -> 抬到下限
    assert T.plan_steps(20, epochs=10) == 200


# ---------------- 端到端（真训练，短步数） ----------------

@pytest.mark.slow
def test_train_end_to_end_small(tmp_path, root):
    """跑 6 步验证整条链路：建模、训练、导出、能被 lora.scan 认出来。"""
    from workflows.engine import Engine

    if not (root / "models" / "checkpoints" / "sd15" / "unet").exists():
        pytest.skip("底模未就绪")

    ds = tmp_path / "ds"
    for i, p in enumerate(_imgs(ds, 2, size=256)):
        p.with_suffix(".txt").write_text(f"test_tiny, subject {i}", encoding="utf-8")

    events = []
    info = T.train(root / "models", ds, "pytest_tiny",
                   rank=4, lr=1e-3, steps=6, resolution=256, progress=events.append)

    out = root / "models" / "lora" / "pytest_tiny.safetensors"
    try:
        assert out.exists(), f"权重没落地: {out}"
        # 只该存 LoRA 层（几 MB），不能把整个 UNet（3.3GB）写进去
        assert out.stat().st_size < 60 * 2**20, f"文件过大 {out.stat().st_size/2**20:.0f}MB"
        assert info["trainable_params"] > 0
        assert any(e.get("stage") == "train" and e.get("step") == 6 for e in events)

        names = {x["name"] for x in L.scan(root / "models")}
        assert "pytest_tiny" in names

        # 能被真正加载（这一步才会暴露 key 名不对之类的问题）
        eng = Engine(root / "models")
        pipe = eng.txt2img_pipe()
        eng.apply_lora(pipe, out)
        assert eng.stats()["lora"].startswith("pytest_tiny")
        eng.free()
    finally:
        out.unlink(missing_ok=True)


# ---------------- API ----------------

@requires_sd
def test_train_api_roundtrip(client, tmp_path, root):
    """/train 提交 → 轮询 → 失败或完成都要有明确状态。"""
    ds = tmp_path / "ds"
    for i, p in enumerate(_imgs(ds, 2, size=256)):
        p.with_suffix(".txt").write_text(f"api_tiny, s{i}", encoding="utf-8")

    r = client.post("/train", json={
        "name": "api_tiny", "data_dir": str(ds), "rank": 4,
        "steps": 4, "resolution": 256, "trigger": "api_tiny",
    })
    assert r.status_code == 200
    tid = r.json()["train_id"]

    import time

    st = {}
    for _ in range(240):
        st = client.get(f"/train/{tid}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert st["status"] == "done", st.get("error")
    out = root / "models" / "lora" / "api_tiny.safetensors"
    try:
        assert out.exists()
        # 训练完应把触发词写进清单，否则前端不知道怎么写提示词
        meta = json.loads((root / "models" / "lora" / "loras.json").read_text(encoding="utf-8"))
        assert meta["api_tiny"]["trigger"] == "api_tiny"
    finally:
        out.unlink(missing_ok=True)


def test_train_api_bad_dataset_fails_cleanly(client):
    r = client.post("/train", json={"name": "x", "data_dir": "no/such/dir", "steps": 1})
    tid = r.json()["train_id"]

    import time

    st = {}
    for _ in range(40):
        st = client.get(f"/train/{tid}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert st["status"] == "failed"
    assert "FileNotFoundError" in (st.get("error") or "")


def test_train_list_and_404(client):
    assert client.get("/train").json()["count"] >= 0
    assert client.get("/train/deadbeef").status_code == 404
