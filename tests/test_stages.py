"""角色**生命阶段**：一个角色多套设定，各自有立绘（用户 2026-11-09 新架构）。

钉住的契约：
  · 阶段数组存在 `config.json` 里，**数组顺序就是展示顺序**；
  · `active_stage` 的 info 始终是顶层 `character_info` 的镜像 —— 现有全部出图
    API 不用带 stage 参数就自动跟着当前阶段走；
  · 立绘按阶段分目录，默认阶段仍用 `Portraits/` 老位置；
  · 老角色第一次读 stages 会补一个「初始设定」阶段，不丢原设定。
"""
from __future__ import annotations

from urllib.parse import quote

from workflows import characters as ch
from workflows import stages


def _fresh(name: str = "StageProbe") -> str:
    if ch.character_exists(name):
        ch.delete_character(name)
    return ch.create_character(name)["character_info"]["name"]


# ------------------------------------------------------------ 纯函数

def test_normalize_builds_default_stage_for_legacy_config():
    cfg = stages.normalize({"character_info": {"hair": "silver hair"}})
    assert len(cfg["stages"]) == 1
    assert cfg["stages"][0]["id"] == stages.DEFAULT_ID
    assert cfg["stages"][0]["title"] == stages.DEFAULT_TITLE
    assert cfg["stages"][0]["info"]["hair"] == "silver hair"
    assert cfg["active_stage"] == stages.DEFAULT_ID
    assert cfg["character_info"]["hair"] == "silver hair", "顶层镜像不能丢"


def test_sanitize_dedupes_ids_and_limits_count():
    raw = [{"id": "dup", "title": "A", "info": {"hair": "a"}},
           {"id": "dup", "title": "B", "info": {"hair": "b"}},
           "not a dict"]
    out = stages.sanitize(raw)
    assert len(out) == 2
    assert out[0]["id"] != out[1]["id"]
    assert out[0]["title"] == "A" and out[1]["title"] == "B"
    assert stages.sanitize([{"id": f"x{i}"} for i in range(80)]) and \
        len(stages.sanitize([{"id": f"x{i}"} for i in range(80)])) == stages.MAX_STAGES


def test_set_active_switches_top_level_info():
    cfg = stages.normalize({"character_info": {"hair": "blonde hair"},
                            "stages": [
                                {"id": "default", "title": "初始", "info": {"hair": "blonde hair"}},
                                {"id": "adult", "title": "成年", "info": {"hair": "black hair"}},
                            ]})
    stages.set_active(cfg, "adult")
    assert cfg["character_info"]["hair"] == "black hair"
    stages.set_active(cfg, "default")
    assert cfg["character_info"]["hair"] == "blonde hair"


def test_update_info_keeps_active_stage_in_sync():
    cfg = stages.normalize({"character_info": {"hair": "blonde hair"}})
    stages.update_info(cfg, {"eyes": "blue eyes"})
    assert cfg["character_info"]["eyes"] == "blue eyes"
    assert stages.active(cfg)["info"]["eyes"] == "blue eyes", \
        "只改顶层的话，下次 normalize 会把改动吞回去"


# ------------------------------------------------------------ 磁盘：立绘按阶段分目录

def test_portraits_dir_is_scoped_per_stage():
    name = _fresh()
    legacy = ch.portraits_dir(name, "Naked", "neutral")
    adult = ch.portraits_dir(name, "Naked", "neutral", stage_id="adult")
    assert adult != legacy
    assert "/s_adult/" in adult.replace("\\", "/")
    assert ch.portraits_dir(name, "Naked", "neutral",
                           stage_id=stages.DEFAULT_ID) == legacy


# ------------------------------------------------------------ API

def test_create_character_takes_description_and_builds_default_stage(client):
    name = "StageDescProbe"
    if ch.character_exists(name):
        ch.delete_character(name)
    r = client.post("/characters", json={
        "name": name, "description": "幼年与成年是两套设定"})
    assert r.status_code == 200, r.text
    cfg = ch.load_config(name)
    assert cfg["description"] == "幼年与成年是两套设定"
    assert len(cfg["stages"]) == 1 and cfg["active_stage"] == stages.DEFAULT_ID


def test_stages_crud_order_and_active_switch(client):
    name = _fresh("StageCrudProbe")
    r = client.get(f"/characters/{quote(name)}/stages")
    assert r.status_code == 200
    body = r.json()
    assert body["active_stage"] == stages.DEFAULT_ID
    assert len(body["stages"]) == 1

    old = body["stages"][0]
    adult = {"id": "adult", "title": "成年", "description": "黑发、铠甲",
             "info": {**old["info"], "hair": "black hair"},
             "portrait_count": 0, "portraits": []}
    child = {"id": "child", "title": "幼年", "description": "短发",
             "info": {**old["info"], "hair": "brown hair"},
             "portrait_count": 0, "portraits": []}

    # 新建两条 + 顺序：初始设定 → 幼年 → 成年
    r = client.put(f"/characters/{quote(name)}/stages", json={
        "stages": [old, child, adult], "active_stage": "adult"})
    assert r.status_code == 200, r.text
    assert [x["id"] for x in r.json()["stages"]] == ["default", "child", "adult"]

    # 切换激活阶段后，顶层 character_info 要跟着变（现有出图链路的入口）
    cfg = ch.load_config(name)
    assert cfg["character_info"]["hair"] == "black hair"

    # 拖拽排序：把「成年」挪到最前
    reordered = [adult, old, child]
    r = client.put(f"/characters/{quote(name)}/stages", json={
        "stages": reordered, "active_stage": "child"})
    assert [x["id"] for x in r.json()["stages"]] == ["adult", "default", "child"]
    assert ch.load_config(name)["character_info"]["hair"] == "brown hair"

    # 删除：只从列表移除，函数本身不删磁盘文件
    r = client.put(f"/characters/{quote(name)}/stages", json={
        "stages": [old, child], "active_stage": "child"})
    assert [x["id"] for x in r.json()["stages"]] == ["default", "child"]


def test_stages_rejects_empty_list(client):
    name = _fresh("StageEmptyProbe")
    r = client.put(f"/characters/{quote(name)}/stages",
                   json={"stages": [], "active_stage": ""})
    assert r.status_code == 400


def test_portraits_endpoint_accepts_stage_filter(client):
    name = _fresh("StagePortraitProbe")
    r = client.get(f"/characters/{quote(name)}/portraits?stage=adult")
    assert r.status_code == 200
    assert "items" in r.json()


def test_generate_publishes_portrait_into_active_stage(client, monkeypatch):
    """立绘生成要把当前阶段塞进 publish，worker 才会按阶段目录收编。"""
    import app as appmod
    from workflows import generate as gen

    name = _fresh("StageGenProbe")
    cfg = ch.load_config(name)
    cfg["stages"] = [
        {"id": "default", "title": "初始", "description": "", "info": dict(cfg["character_info"])},
        {"id": "adult", "title": "成年", "description": "",
         "info": {**cfg["character_info"], "hair": "black hair"}},
    ]
    cfg["active_stage"] = "adult"
    ch.save_config(name, stages.normalize(cfg))

    class FakeQ:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

        def qsize(self):
            return len(self.items)

    fake = FakeQ()
    monkeypatch.setattr(appmod, "_q", fake)
    monkeypatch.setattr(gen, "txt2img", lambda eng, **kw: {"file": "fake.png"})

    r = client.post(f"/characters/{quote(name)}/generate",
                    json={"workflow": "w1", "prompt": "x"})
    assert r.status_code == 200, r.text
    assert fake.items
    pub = fake.items[0][2]["publish"]
    assert pub["stage"] == "adult"
    assert pub["kind"] == "portrait"
    assert fake.items[0][2]["preset"] == "none", "画风中立那条契约不能被阶段化弄丢"
