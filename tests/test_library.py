"""用户资料库（提示词收藏 / 姿势库 / 偏好）与 VNCCS 官方姿势库清单。

**为什么这两块要落盘**：浏览器存储（localStorage）在换浏览器、清站点数据、
无痕窗口时会凭空消失，而收藏和姿势是用户攒了很久的资产。这里断言的是：
写入真的落到 `library/*.json`、坏了的文件不会把整个应用拖死、
官方姿势库的清单能扫出姿势本体与预览图的路径。
"""
from __future__ import annotations

import json
from pathlib import Path

from workflows import library as lb
from workflows import poselib
from workflows import paths


def test_library_dir_isolated():
    """测试期间必须写 tests/_out 下的库目录，不能碰用户真实的 library/。"""
    out = Path(paths.TEST_OUT).resolve()
    assert out in Path(paths.LIBRARY_DIR).resolve().parents


def test_prompt_add_dedupe_remove():
    lb._write(lb.PROMPTS_FILE, [])
    first = lb.add_prompt("白猫骑士", "no text")
    assert len(first) == 1 and first[0]["text"] == "白猫骑士"

    # 正向 + 负向都一样才算重复；负向不同是另一条收藏
    again = lb.add_prompt("白猫骑士", "no text")
    assert len(again) == 1, "同正向同负向应去重"
    other = lb.add_prompt("白猫骑士", "blurry")
    assert len(other) == 2, "负向不同应算两条"

    # 真的落盘了（不是只在内存里）
    on_disk = json.loads(
        (paths.LIBRARY_DIR / lb.PROMPTS_FILE).read_text(encoding="utf-8"))
    assert len(on_disk) == 2

    left = lb.remove_prompt(other[0]["id"])
    assert len(left) == 1
    assert lb.has_prompt("白猫骑士", "no text")
    assert not lb.has_prompt("白猫骑士", "blurry")


def test_pose_add_overwrites_same_name():
    lb._write(lb.POSES_FILE, [])
    a = lb.add_pose("挥剑", {"bones": {"upperarm_r": [1, 2, 3]}}, 90, "")
    assert len(a) == 1 and a[0]["yaw"] == 90

    # **同名覆盖**：用户改完胳膊想原地更新，不该翻出两条
    b = lb.add_pose("挥剑", {"bones": {"upperarm_r": [4, 5, 6]}}, 45, "")
    assert len(b) == 1
    assert b[0]["yaw"] == 45
    assert b[0]["pose"]["bones"]["upperarm_r"] == [4, 5, 6]

    # 角度归一化：负数/浮点没有意义
    c = lb.add_pose("背刺", {"bones": {}}, -90, "")
    assert c[0]["yaw"] == 270

    renamed = lb.rename_pose(c[0]["id"], "背刺·改")
    assert [p["name"] for p in renamed] == ["背刺·改", "挥剑"]
    # 名字被别人占了 → 不动（硬改名会凭空造出两条同名，删的时候分不清）
    assert lb.rename_pose(c[0]["id"], "挥剑")[0]["name"] == "背刺·改"

    assert len(lb.remove_pose(c[0]["id"])) == 1


def test_pose_rejects_bad_body():
    lb._write(lb.POSES_FILE, [])
    try:
        lb.add_pose("坏姿势", "not-a-pose")
    except ValueError:
        pass
    else:
        raise AssertionError("pose 不是对象应该报错")
    assert lb.load_poses() == []


def test_pose_add_batch_single_write():
    """批量存姿势（一条动画抽的 N 帧）：一次落盘、按顺序在最前、同名覆盖。

    **一次落盘**是这个接口存在的理由：逐条 add_pose 会重写 N 次整个 poses.json
    （每帧还带一张预览图）。这里用"文件 mtime/inode 不变 + 内容正确"以外的方式
    验不了写入次数，所以直接数 `_write` 的调用。
    """
    lb._write(lb.POSES_FILE, [])
    lb.add_pose("已有的", {"bones": {}}, 0, "")

    calls = []
    real_write = lb._write

    def spy(name, doc):
        calls.append(name)
        return real_write(name, doc)

    lb._write = spy
    try:
        out = lb.add_poses([
            {"name": "行走 01", "pose": {"bones": {"a": 1}}, "yaw": 90, "preview": "data:x"},
            {"name": "行走 02", "pose": {"bones": {"a": 2}}, "yaw": 90, "preview": ""},
            {"name": "行走 03", "pose": {"bones": {"a": 3}}, "yaw": 90, "preview": ""},
        ])
    finally:
        lb._write = real_write

    assert calls == [lb.POSES_FILE], f"应该只写一次盘，实际 {calls}"
    assert [p["name"] for p in out] == ["行走 03", "行走 02", "行走 01", "已有的"]
    assert len(out) == 4, "三条新姿势要全进去，不能互相覆盖"
    assert out[0]["pose"]["bones"]["a"] == 3, "最后插入的排在最前（at 递增）"
    assert out[2]["preview"] == "data:x", "预览图要原样保留"
    # 落盘的也确实是四条（不是只在返回值里）
    assert len(json.loads((paths.LIBRARY_DIR / lb.POSES_FILE)
                          .read_text(encoding="utf-8"))) == 4

    # 同名 → 原地覆盖，不新增
    again = lb.add_poses([{"name": "行走 02", "pose": {"bones": {"a": 9}}}])
    assert len(again) == 4
    assert next(p for p in again if p["name"] == "行走 02")["pose"]["bones"]["a"] == 9

    # 坏数据 → 明确报错，且**不落盘**（不能写进一半就退出留下残局）
    before = (paths.LIBRARY_DIR / lb.POSES_FILE).read_text(encoding="utf-8")
    try:
        lb.add_poses([{"name": "坏", "pose": "not-a-pose"}])
    except ValueError:
        pass
    else:
        raise AssertionError("pose 不是对象应该报错")
    assert (paths.LIBRARY_DIR / lb.POSES_FILE).read_text(encoding="utf-8") == before


def test_prefs_merge_and_snapshot():
    lb._write(lb.PREFS_FILE, {})
    lb.save_prefs({"maskColor": "#ff3b6b"})
    lb.save_prefs({"maskAlpha": 0.6})
    got = lb.load_prefs()
    assert got["maskColor"] == "#ff3b6b", "只覆盖传进来的键，已有的不能丢"
    assert got["maskAlpha"] == 0.6
    assert set(lb.snapshot().keys()) == {"prompts", "poses", "prefs"}


def test_seed_only_adds_never_overwrites():
    lb._write(lb.PROMPTS_FILE, [])
    lb._write(lb.POSES_FILE, [])
    lb._write(lb.PREFS_FILE, {})
    lb.add_prompt("本机已有", "x")
    lb.save_prefs({"maskColor": "#111111"})

    snap = lb.seed_library(
        prompts=[{"text": "浏览器里的", "negative": "y"}],
        poses=[{"name": "旧姿势", "pose": {"bones": {}}, "yaw": 180}],
        prefs={"maskColor": "#000000", "maskAlpha": 0.5},
    )
    assert {p["text"] for p in snap["prompts"]} == {"本机已有", "浏览器里的"}
    assert [p["name"] for p in snap["poses"]] == ["旧姿势"]
    # **已有的偏好不动**：当前这份是用户在用的，旧的是别处漂来的备份
    assert snap["prefs"]["maskColor"] == "#111111"
    # 没有的键补进来
    assert snap["prefs"]["maskAlpha"] == 0.5


def test_corrupted_file_reads_as_empty():
    """库文件坏了要当空库，不能把整个应用拖到 500。"""
    for name in (lb.PROMPTS_FILE, lb.POSES_FILE, lb.PREFS_FILE):
        (paths.LIBRARY_DIR / name).write_text("{ 半截的 json", encoding="utf-8")
    assert lb.load_prompts() == []
    assert lb.load_poses() == []
    assert lb.load_prefs() == {}


def test_atomic_write_leaves_no_tmp():
    lb._write(lb.PROMPTS_FILE, [{"id": "a", "text": "x", "negative": "", "at": 1}])
    leftovers = list(paths.LIBRARY_DIR.glob("*.json.tmp"))
    assert not leftovers, f"原子写留下了临时文件：{leftovers}"


# ---------------------------------------------------------------- 官方姿势库


def test_pose_library_lists_bodies_and_previews():
    items = poselib.list_presets(force=True)
    # 随项目发行的 191 个姿势必须在仓库里（复制进 web/public 才算数）
    assert len(items) >= 190, f"只扫到 {len(items)} 个姿势"
    for it in items:
        assert it["name"], f"缺名字：{it['id']}"
        assert it["json"].startswith(poselib.URL_PREFIX)
        assert it["json"].endswith(".json")
        # 每个条目都能对应到磁盘上的真实文件（URL 指路错就是点了没反应）
        rel = it["json"][len(poselib.URL_PREFIX) + 1:].replace("%20", " ")
        assert (poselib.pose_library_dir() / rel).is_file(), f"姿势本体不存在：{rel}"
        if it["preview"]:
            relp = it["preview"][len(poselib.URL_PREFIX) + 1:].replace("%20", " ")
            assert (poselib.pose_library_dir() / relp).is_file(), f"预览图不存在：{relp}"


def test_pose_library_yaw_guess_and_categories():
    items = poselib.list_presets()
    by_tag = {}
    for it in items:
        tags = " ".join(t.lower() for t in it["tags"])
        by_tag.setdefault(tags, it)
    # 标签里写着拍摄角度，默认机位要跟得上（侧面 90°、背面 180°）
    for tags, it in by_tag.items():
        if "rear view" in tags:
            assert it["yaw"] == 180, f"{it['name']} 朝后却给了 {it['yaw']}°"
        elif "side view" in tags or "profile view" in tags:
            assert it["yaw"] == 90, f"{it['name']} 朝侧面却给了 {it['yaw']}°"
    cats = poselib.categories()
    assert cats, "应该有分类"
    assert len(cats) == len(set(c.lower() for c in cats)), f"分类大小写没去重：{cats}"


def test_pose_library_cache_is_used():
    first = poselib.list_presets(force=True)
    # 第二次调用走缓存（同一个对象），改了目录内容用 refresh 才重扫
    assert poselib.list_presets() is first
    assert poselib.list_presets(force=True) is not first
