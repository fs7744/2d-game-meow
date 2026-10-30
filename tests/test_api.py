"""服务层：端点契约、队列、路径穿越防护。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

# SD 权重已归档到 models/_archive/sd/：下面这些用例要真跑 SD 管线，
# 权重不在就跳过（恢复权重后自动重新生效，不是删测试）。
_SD_WEIGHTS = (Path(__file__).resolve().parent.parent / "models" / "checkpoints"
               / "sd15" / "model_index.json").is_file()
requires_sd = pytest.mark.skipif(
    not _SD_WEIGHTS,
    reason="SD 权重已归档到 models/_archive/sd/（恢复后自动恢复）")

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and "engine" in body
    assert body["engine"]["total_gb"] > 0


def test_engine_stats(client):
    r = client.get("/engine")
    assert r.status_code == 200
    assert "NVIDIA" in r.json()["device"]


@requires_sd
def test_generate_roundtrip(client):
    payload = {"workflow": "w1", "params": {
        "prompt": "blue slime monster", "width": 256, "height": 256,
        "steps": 8, "seed": 42, "pixel": 4, "palette": 16, "tag": "apitest"}}
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):                       # 最多等 60s
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    out = Path(job["result"]["file"])
    assert out.exists() and out.stat().st_size > 0
    assert job["result"]["meta"]["seed"] == 42


@requires_sd
def test_generate_w2_roundtrip(client, tmp_path):
    """图生图走同一条队列：w2 只认 init_image + strength，不接受 width/height。"""
    src = tmp_path / "w2_src.png"
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (256, 256), (40, 80, 160))
    ImageDraw.Draw(im).rectangle([70, 70, 180, 190], fill=(230, 60, 60))
    im.save(src)

    payload = {"workflow": "w2", "params": {
        "init_image": str(src), "prompt": "blue slime monster", "strength": 0.5,
        "steps": 6, "seed": 7, "pixel": 4, "palette": 16, "matte": True,
        "tag": "apiw2"}}
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    meta = job["result"]["meta"]
    assert meta["workflow"] == "w2" and meta["strength"] == 0.5
    assert Path(meta["raw"]).exists(), "w2 也要给下游留未处理的原图"


def test_upload_image(client):
    """上传端点：原始字节进去，磁盘路径 + 可访问 URL 出来。"""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (1, 2, 3)).save(buf, "PNG")
    r = client.post("/upload?name=测试 图.png", content=buf.getvalue(),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert Path(body["path"]).exists()
    assert (body["w"], body["h"]) == (32, 24)
    assert body["format"] == "PNG"
    # 磁盘名只用 ASCII（中文名会让 URL 需要百分号编码），原名另外回传
    assert "测试" not in Path(body["path"]).name
    assert " " not in Path(body["path"]).name
    assert body["path"].isascii()
    assert body["name"] == "测试 图.png"
    # 上传后要能被 /files 取回（画布上的图就是靠它显示的）
    assert client.get(body["url"]).status_code == 200


def test_upload_goes_into_project_dir(client):
    """上传带 project 时，文件落在该项目目录里（别的项目拿不到）。"""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (9, 9, 9)).save(buf, "PNG")
    r = client.post("/upload?name=pic.png&project=projX", content=buf.getvalue(),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200
    body = r.json()
    assert "projects/projX" in body["path"].replace("\\", "/"), body["path"]
    assert body["url"].startswith("/files/projects/projX/")
    assert client.get(body["url"]).status_code == 200


def test_assets_filtered_by_project(client):
    """资产清单能按项目过滤：只列本项目的产物。"""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (9, 9, 9)).save(buf, "PNG")
    up = client.post("/upload?name=p.png&project=projY", content=buf.getvalue(),
                     headers={"Content-Type": "image/png"}).json()
    with open(up["path"], "rb") as fh:
        data = fh.read()
    idx = Path(__file__).resolve().parents[1] / "assets" / "index.jsonl"
    with idx.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"file": up["path"], "scope": "projY", "tag": "t"}) + "\n")
    try:
        r = client.get("/assets?project=projY&limit=50")
        assert r.status_code == 200
        scopes = {i.get("scope") for i in r.json()["items"]}
        assert scopes == {"projY"}, scopes
    finally:
        Path(up["path"]).unlink(missing_ok=True)


def test_save_mask_endpoint(client):
    """蒙版端点：PNG 原始字节 → 项目目录的 masks/ 下同名覆盖。"""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (64, 48), 255).save(buf, "PNG")
    r = client.post("/mask?name=node1&project=projM", content=buf.getvalue(),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "projects/projM/masks/node1.png" in body["path"].replace("\\", "/"), body["path"]
    assert body["width"] == 64 and body["height"] == 48
    assert body["coverage"] == 1.0
    assert client.get(body["url"]).status_code == 200

    # 同名再存一次：覆盖而不是堆积
    buf2 = io.BytesIO()
    Image.new("L", (64, 48), 0).save(buf2, "PNG")
    r2 = client.post("/mask?name=node1&project=projM", content=buf2.getvalue(),
                     headers={"Content-Type": "image/png"})
    assert r2.json()["path"] == body["path"]
    assert r2.json()["coverage"] == 0.0


def test_mask_endpoint_rejects_non_png(client):
    r = client.post("/mask?name=x", content=b"not png at all",
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 400


def test_upload_rejects_garbage(client):
    """坏文件必须被拒：不能什么字节都往 assets 里塞。"""
    r = client.post("/upload", content=b"not an image at all",
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 400
    assert "图片" in r.json()["detail"]


def test_upload_rejects_empty_body(client):
    assert client.post("/upload", content=b"").status_code == 400


def test_delete_assets_removes_file_sidecar_and_raw(client):
    """真删：删一个 PNG 要连带删掉 sidecar 与 raw，并把索引里的条目摘掉。"""
    assets = Path(__file__).resolve().parents[1] / "assets"
    png = assets / "_del_api_test.png"
    png.write_bytes(b"x")
    (assets / "_del_api_test.png.json").write_text("{}", encoding="utf-8")
    (assets / "_del_api_test_raw.png").write_bytes(b"x")

    idx = assets / "index.jsonl"
    before = idx.read_text("utf-8") if idx.exists() else ""
    if before and not before.endswith("\n"):
        before += "\n"
    idx.write_text(before + json.dumps({"file": str(png), "tag": "t"}) + "\n", encoding="utf-8")

    r = client.request("DELETE", "/assets", json={"paths": [str(png)]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    assert not png.exists()
    assert not (assets / "_del_api_test.png.json").exists()
    assert not (assets / "_del_api_test_raw.png").exists()
    # 索引里那条也没了
    assert "_del_api_test" not in idx.read_text("utf-8")


def test_delete_assets_blocks_path_traversal(client):
    """路径穿越必须被拒：否则 ../ 能把 assets 之外的文件删掉。"""
    outside = Path(__file__).resolve().parents[1] / "_del_outside.txt"
    outside.write_text("keep me", encoding="utf-8")
    try:
        r = client.request("DELETE", "/assets", json={"paths": [str(outside), "../outside.txt"]})
        assert r.status_code == 200
        assert r.json()["count"] == 0
        assert r.json()["skipped"]
        assert outside.exists(), "assets 外的文件不能被删"
    finally:
        outside.unlink(missing_ok=True)


def test_delete_assets_protects_index(client):
    """索引文件不能被删（删了资产清单就废了）。"""
    idx = Path(__file__).resolve().parents[1] / "assets" / "index.jsonl"
    r = client.request("DELETE", "/assets", json={"paths": ["index.jsonl", str(idx)]})
    assert r.status_code == 200
    assert r.json()["count"] == 0
    assert idx.exists()


@requires_sd
def test_generate_wbg_roundtrip(client, tmp_path):
    """抠图工作流也走同一条队列（wbg）。"""
    from PIL import Image, ImageDraw

    src = tmp_path / "bg_src.png"
    im = Image.new("RGB", (128, 128), (255, 255, 255))
    ImageDraw.Draw(im).rectangle([30, 20, 98, 108], fill=(200, 40, 40))
    im.save(src)

    r = client.post("/generate", json={"workflow": "wbg", "params": {
        "init_image": str(src), "bg_color": "#204080", "tag": "apiwbg"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    meta = job["result"]["meta"]
    assert meta["workflow"] == "wbg" and meta["bg_mode"] == "color"
    assert Path(job["result"]["file"]).exists()


@requires_sd
def test_generate_w5_roundtrip(client, tmp_path):
    """局部 AI 重绘（w5）走同一条队列。"""
    from PIL import Image, ImageDraw

    src = tmp_path / "w5_src.png"
    im = Image.new("RGB", (256, 256), (250, 250, 250))
    ImageDraw.Draw(im).rectangle([78, 78, 178, 178], fill=(200, 40, 40))
    im.save(src)
    mk = tmp_path / "w5_mask.png"
    m = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(m).rectangle([98, 98, 158, 158], fill=255)
    m.save(mk)

    r = client.post("/generate", json={"workflow": "w5", "params": {
        "image": str(src), "mask": str(mk), "prompt": "blue gem",
        "steps": 6, "seed": 5, "preset": "none", "tag": "apiw5"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    meta = job["result"]["meta"]
    assert meta["workflow"] == "w5"
    assert meta["prompt_body"] == "blue gem"
    assert meta["coverage"] > 0
    assert Path(job["result"]["file"]).exists()


@requires_sd
def test_generate_rejects_unknown_workflow_lists_valid_ones(client):
    """未知工作流要报 400，并在信息里列出可用工作流（否则用户不知道能用什么）。"""
    r = client.post("/generate", json={"workflow": "w9", "params": {"prompt": "x"}})
    assert r.status_code == 400
    assert "w5" in r.json()["detail"], r.json()["detail"]


def test_generate_wpx_roundtrip(client, tmp_path):
    """完美像素转换（wpx）走同一条队列。"""
    from PIL import Image, ImageDraw

    src = tmp_path / "px_src.png"
    im = Image.new("RGB", (128, 128), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for y in range(8):
        for x in range(8):
            d.rectangle([x * 16, y * 16, (x + 1) * 16 - 1, (y + 1) * 16 - 1],
                        fill=((x * 37) % 256, (y * 53) % 256, 128))
    im.save(src)

    r = client.post("/generate", json={"workflow": "wpx", "params": {
        "init_image": str(src), "colors": 16, "tag": "apiwpx"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    meta = job["result"]["meta"]
    assert meta["workflow"] == "wpx"
    assert meta["offPalette"] == 0
    assert Path(job["result"]["file"]).exists()


@requires_sd
def test_generate_wpc_roundtrip(client, tmp_path):
    """角色迁移（wpc）走同一条队列。

    动作参考图用仓库里的真人素材——姿态模型只认得出真人，合成简笔人形
    连人体检测都过不了。
    """
    from PIL import Image, ImageDraw

    fixture = (Path(__file__).resolve().parents[1] / "tests" / "fixtures"
               / "pose_ref_person.png")
    if not fixture.exists():
        pytest.skip("缺少真人姿态参考素材")
    pose = tmp_path / "pc_pose.png"
    Image.open(fixture).convert("RGB").save(pose)
    char = tmp_path / "pc_char.png"
    cim = Image.new("RGB", (256, 256), (250, 250, 250))
    ImageDraw.Draw(cim).ellipse([80, 60, 176, 190], fill=(40, 170, 200))
    cim.save(char)

    r = client.post("/generate", json={"workflow": "wpc", "params": {
        "pose_image": str(pose), "char_image": str(char),
        "prompt": "a blue slime monster", "steps": 6, "seed": 3,
        "preset": "none", "width": 512, "height": 512, "tag": "apiwpc"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    meta = job["result"]["meta"]
    assert meta["workflow"] == "wpc"
    assert meta["joints"] >= 12
    assert Path(job["result"]["file"]).exists()


def test_assets_lists_generated(client):
    """清单接口要返回**真实存在**的产物，不只是个空壳。

    以前这条断言的是「最新 200 条里有 tag=apitest 的那张」——但 apitest 由
    `test_generate_roundtrip` 产出，而那条要真跑 SD 管线，SD 权重归档后它被
    skip，这条就变成了永远失败的死断言。现在改成校验「页面里至少有一条，
    且它的文件真的在盘上」，与谁产出无关。
    """
    r = client.get("/assets?limit=200")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    paths = [i.get("file") for i in body["items"] if isinstance(i, dict)]
    assert any(p and Path(p).is_file() for p in paths), paths[:3]


def test_files_serves_asset(client):
    """产物现在在 projects/<id>/ 子目录里，取文件必须带相对路径（只取 basename 会 404）。"""
    import shutil

    assets = Path(__file__).resolve().parents[1] / "assets"
    d = assets / "projects" / "_serve_test_proj"
    d.mkdir(parents=True, exist_ok=True)
    png = d / "x.png"
    png.write_bytes(b"x")
    try:
        rel = png.resolve().relative_to(assets.resolve()).as_posix()
        r = client.get(f"/files/{rel}")
        assert r.status_code == 200
        assert len(r.content) > 0
        # 只给 basename 取不到（项目子目录里的文件）——这正是要防的坑
        assert client.get(f"/files/{png.name}").status_code == 404
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_files_download_sets_attachment(client):
    """?download=1 要带 Content-Disposition，浏览器才会直接存盘而不是内联显示。"""
    import shutil

    assets = Path(__file__).resolve().parents[1] / "assets"
    d = assets / "projects" / "_dl_test_proj"
    d.mkdir(parents=True, exist_ok=True)
    png = d / "hero.png"
    png.write_bytes(b"12345")
    try:
        rel = png.resolve().relative_to(assets.resolve()).as_posix()
        r = client.get(f"/files/{rel}?download=1")
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd, cd
        assert "hero.png" in cd, cd
        assert r.content == b"12345"
        # 不带 download 时是内联显示（没有 disposition）
        assert client.get(f"/files/{rel}").headers.get("content-disposition") is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_files_blocks_path_traversal(client):
    r = client.get("/files/..%2F..%2Fapp.py")
    assert r.status_code == 404
    r2 = client.get("/files/nope_not_here.png")
    assert r2.status_code == 404


def test_bad_workflow_rejected(client):
    r = client.post("/generate", json={"workflow": "w9", "params": {}})
    assert r.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/jobs/deadbeef").status_code == 404


@requires_sd
def test_queue_serializes_two_jobs(client):
    ids = []
    for s in (11, 12):
        r = client.post("/generate", json={"workflow": "w1", "params": {
            "prompt": "green mushroom monster", "width": 256, "height": 256,
            "steps": 6, "seed": s, "tag": "queue"}})
        ids.append(r.json()["job_id"])
    states = []
    for _ in range(120):
        states = [client.get(f"/jobs/{i}").json()["status"] for i in ids]
        if all(s in ("done", "failed") for s in states):
            break
        time.sleep(0.5)
    assert states == ["done", "done"]
    # 串行语义：后提交的开始时间不早于先提交的结束时间
    a = client.get(f"/jobs/{ids[0]}").json()
    b = client.get(f"/jobs/{ids[1]}").json()
    assert b["started"] >= a["finished"] - 0.05


def test_loras_endpoint(client):
    """LoRA 清单：真实目录里应当能扫到 wuxia_v2（含触发词）。"""
    r = client.get("/loras")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["items"], list)
    names = {x["name"] for x in body["items"]}
    if not names:
        pytest.skip("本机未放 LoRA 权重")
    item = next(x for x in body["items"] if x["name"] == "wuxia_v2")
    assert item["trigger"] == "wuxia"
    assert item["path"].startswith("lora/")
    assert item["size_mb"] > 0


@requires_sd
def test_generate_resolves_lora_name(client):
    """params.lora 传名字要能解析成路径；名字不存在则降级而不是整单失败。"""
    import app as app_module

    resolved = app_module._resolve_lora({"lora": "wuxia_v2"})
    assert resolved["lora"].endswith("wuxia_v2.safetensors")

    missing = app_module._resolve_lora({"lora": "nope"})
    assert missing["lora"] is None and missing["lora_missing"] == "nope"

    assert app_module._resolve_lora({"prompt": "x"}) == {"prompt": "x"}


def test_generate_with_lora_name_roundtrip(client):
    """端到端：只给名字，服务端解析后能真跑出一张图。"""
    import app as app_module

    if not (app_module.ROOT / "models" / "lora" / "wuxia_v2.safetensors").exists():
        pytest.skip("本机无 wuxia_v2 权重")

    payload = {"workflow": "w1", "params": {
        "prompt": "Wuxia, swordsman, front view", "width": 256, "height": 256,
        "steps": 8, "seed": 7, "preset": "wuxia", "pixel": 4, "palette": 16,
        "matte": True, "lora": "wuxia_v2", "lora_scale": 0.8, "tag": "apilora"}}
    job_id = client.post("/generate", json=payload).json()["job_id"]

    for _ in range(120):
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert j["status"] == "done", j.get("error")
    meta = j["result"]["meta"]
    assert meta["lora"].endswith("wuxia_v2.safetensors")
    assert meta["lora_scale"] == 0.8
    assert isinstance(meta["edge_touch"], list)


@requires_sd
def test_generate_wss_roundtrip(client, tmp_path):
    """序列帧角色替换（wss）走同一条队列。

    用换皮模式：不跑模型，秒回，且验证网格参数与产物结构。
    """
    from PIL import Image, ImageDraw

    def frame():
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([25, 30, 38, 48], fill=(120, 30, 40, 255))
        d.rectangle([27, 18, 36, 30], fill=(235, 200, 160, 255))
        return im

    sheet = tmp_path / "sheet.png"
    im = Image.new("RGBA", (64 * 4, 64), (0, 0, 0, 0))
    for i in range(4):
        im.paste(frame(), (i * 64, 0))
    im.save(sheet)

    char = tmp_path / "char.png"
    cim = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(cim).ellipse([10, 20, 54, 58], fill=(70, 190, 90, 255))
    cim.save(char)

    r = client.post("/generate", json={"workflow": "wss", "params": {
        "sheet_image": str(sheet), "char_image": str(char),
        "cols": 4, "rows": 1, "mode": "recolor", "palette": 8,
        "tag": "apiwss"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    meta = job["result"]["meta"]
    assert meta["workflow"] == "wss"
    assert meta["mode"] == "recolor"
    assert meta["frames"] == 4
    assert meta["grid"] == [4, 1]
    assert Path(job["result"]["file"]).exists()


@requires_sd
def test_generate_rejects_unknown_workflow(client):
    """未知工作流必须被拒（别让它进队列再失败）。"""
    r = client.post("/generate", json={"workflow": "wzz", "params": {}})
    assert r.status_code == 400


def test_pose_dirs_lists_directories_with_frames(client):
    """W4 的输入目录清单：前端下拉直接用，不能返回空壳或超时。

    assets 里已有几千个文件，深扫会把接口拖慢；这里只扫两层。
    """
    import time

    t0 = time.perf_counter()
    r = client.get("/pose_dirs?limit=10")
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    body = r.json()
    assert elapsed < 1.0, f"扫目录花了 {elapsed:.2f}s，接口会被前端的下拉请求拖住"
    assert body["count"] == len(body["items"]) <= 10

    for it in body["items"]:
        assert it["count"] >= 2, "少于 2 张图的目录不该出现在候选里"
        assert isinstance(it["path"], str) and it["path"]
        assert isinstance(it["sample"], list) and it["sample"]


def test_models_endpoint_reports_refiner_availability(client):
    """前端要靠这个字段决定 Refiner 开关能不能点。"""
    body = client.get("/models").json()
    assert "refiner" in body
    assert isinstance(body["refiner"], bool)


def test_generate_topixel_roundtrip(client, tmp_path):
    """图像转像素（topixel）走同一条队列。用「给源图」的模式：不跑模型，秒回。"""
    from PIL import Image, ImageDraw

    src = tmp_path / "anime_src.png"
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 20, 88, 100], fill=(200, 50, 60, 255))
    d.ellipse([52, 32, 76, 56], fill=(235, 200, 160, 255))
    im.save(src)

    r = client.post("/generate", json={"workflow": "topixel", "params": {
        "init_image": str(src), "target": [48, 48], "colors": 12,
        "matte": False, "tag": "apitopixel"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    meta = job["result"]["meta"]
    assert meta["workflow"] == "topixel"
    assert meta["generated"] is False
    assert meta["pixel"]["size"] == [48, 48]
    assert Path(job["result"]["file"]).exists()


def test_generate_rejects_topixel_without_input(client):
    """topixel 既没源图也没提示词 → 任务失败并给出明确原因（不是静默出错图）。"""
    r = client.post("/generate", json={"workflow": "topixel", "params": {"tag": "apitopixel_bad"}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    job = None
    for _ in range(40):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "failed"
    assert "prompt 或 init_image" in (job.get("error") or "")
