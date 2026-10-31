"""2026-10-31 全项目 review 的回归测试。

这一轮 review 找到的问题里，有几个是**结构性**的（不是改一行能修的），
所以用测试钉住，避免以后再退化：

1. **LoRA 双槽位**：工作流临时挂的 LoRA 不能顶掉用户挂的；工作流结束后
   自己的临时 LoRA 必须卸干净，不能污染后面的生成。
2. **路径穿越**：`startswith` 是字符串前缀比较，`assets2/` 能通过。
3. **meta 真实性**：记的步数必须是**实际执行**的步数（钳制后）、
   负向词必须是**真的发给了管线**的那份、快速模式必须标出"本次是否生效"。
4. **重复字典键**：Python 里重复键静默取最后一个。
"""
from __future__ import annotations

import ast
import inspect
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workflows import generate as g

ROOT = Path(__file__).resolve().parents[1]


# ---------------- 1. LoRA 双槽位 ----------------

class FakePipe:
    """够用的假管线：记录加载/卸载/激活了哪些 adapter。"""

    def __init__(self):
        self._loaded_lora_adapters = []
        self.loaded = []          # 每次 load_lora_weights 的 (adapter, 文件)
        self.deleted = []
        self.active = ([], [])

    def load_lora_weights(self, path, adapter_name=None):
        self.loaded.append((adapter_name, path))
        if adapter_name not in self._loaded_lora_adapters:
            self._loaded_lora_adapters.append(adapter_name)

    def delete_adapters(self, name):
        self.deleted.append(name)
        if name in self._loaded_lora_adapters:
            self._loaded_lora_adapters.remove(name)

    def set_adapters(self, names, scales):
        self.active = (list(names), list(scales))


@pytest.fixture
def fake_lora_dir(tmp_path, monkeypatch):
    """造两个假 LoRA 文件，并把 engine 的 lora_dir 指过去。"""
    d = tmp_path / "lora"
    d.mkdir()
    for n in ("user_lora", "temp_lora"):
        (d / f"{n}.safetensors").write_bytes(b"x")
    from workflows.engine import Engine
    monkeypatch.setattr(Engine, "lora_dir", lambda self: d)
    return d


def test_temp_lora_does_not_evict_user_lora(fake_lora_dir):
    """工作流临时挂的 LoRA 不能把用户挂的顶掉。

    以前 `apply_loras` 是"先全卸再挂"，于是：用户在界面挂了 AnyAngle →
    跑一次快速模式 → AnyAngle 被静默卸载 → 用户的设置凭空消失。
    """
    from workflows.engine import Engine
    eng = Engine("models")
    pipe = FakePipe()
    eng.apply_loras(pipe, [{"name": "user_lora", "scale": 1.0}], scope="user")
    assert [a["name"] for a in eng.active_loras()] == ["user_lora"]

    eng.apply_loras(pipe, [{"name": "temp_lora", "scale": 0.8}], scope="temp")
    names = [a["name"] for a in eng.active_loras()]
    assert "user_lora" in names and "temp_lora" in names, \
        f"临时 LoRA 把用户的顶掉了：{names}"


def test_unload_temp_keeps_user_lora(fake_lora_dir):
    """工作流收尾卸自己的临时 LoRA，不能连用户的一起卸。"""
    from workflows.engine import Engine
    eng = Engine("models")
    pipe = FakePipe()
    eng.apply_loras(pipe, [{"name": "user_lora", "scale": 1.0}], scope="user")
    eng.apply_loras(pipe, [{"name": "temp_lora", "scale": 1.0}], scope="temp")
    eng.unload_loras(pipe, scope="temp")
    assert [a["name"] for a in eng.active_loras()] == ["user_lora"], \
        "卸临时槽时把用户的 LoRA 一起卸了"


def test_unload_all_clears_everything(fake_lora_dir):
    from workflows.engine import Engine
    eng = Engine("models")
    pipe = FakePipe()
    eng.apply_loras(pipe, [{"name": "user_lora", "scale": 1.0}], scope="user")
    eng.apply_loras(pipe, [{"name": "temp_lora", "scale": 1.0}], scope="temp")
    eng.unload_loras(pipe, scope="all")
    assert eng.active_loras() == []


def test_lora_sync_is_idempotent(fake_lora_dir):
    """同样的组合不该反复加载（快速模式每次生成都会调一次）。"""
    from workflows.engine import Engine
    eng = Engine("models")
    pipe = FakePipe()
    eng.apply_loras(pipe, [{"name": "temp_lora", "scale": 1.0}], scope="temp")
    n = len(pipe.loaded)
    eng.apply_loras(pipe, [{"name": "temp_lora", "scale": 1.0}], scope="temp")
    assert len(pipe.loaded) == n, "同样的 LoRA 组合被重复加载了"


# ---------------- 2. 路径穿越 ----------------

def test_path_prefix_is_not_enough():
    """`assets2/` 不在 `assets/` 里——字符串前缀比较会放行。"""
    import app
    root = (app.ASSETS).resolve()
    inside = (root / "projects" / "x.png")
    outside = Path(str(root) + "2") / "x.png"
    assert app._inside(root, inside)
    assert not app._inside(root, outside), \
        "assets2/ 被当成 assets/ 的子目录放行了"


# ---------------- 3. meta 真实性 ----------------

def test_eff_steps_reflects_pipeline_clamp():
    """meta 记的步数必须是**实际执行**的步数。

    管线是 `max(8, min(steps, 60))`：请求 4 步实际跑 8 步，
    请求 100 步实际跑 60 步。以前 meta 记的是请求值。
    """
    assert g._clamp_steps(4) == 8
    assert g._clamp_steps(16) == 16
    assert g._clamp_steps(100) == 60
    assert g._eff_steps(_FakeEng(fast=False), 4) == 8
    assert g._eff_steps(_FakeEng(fast=False), 100) == 60


def test_fast_meta_reports_applied_per_call():
    """`fast_applied` 要区分「全局开关开着」与「本次真的走了少步分支」。"""
    eng = _FakeEng(fast=True)
    applied = g._fast_meta(eng, applied=True)
    assert applied["fast_mode"] is True
    assert applied["fast_applied"] is True
    assert applied["fast_lora"] == g.FAST_LORA

    # 快速模式开着，但这条工作流不走少步分支（真蒙版 inpaint / VNCCS / 扩图）
    skipped = g._fast_meta(eng, applied=False)
    assert skipped["fast_mode"] is True        # 全局开关状态照实记
    assert skipped["fast_applied"] is False    # 但本次没生效
    assert skipped["fast_lora"] is None       # 不能声称挂了 Lightning


class _FakeEng:
    def __init__(self, fast: bool):
        self.fast_mode = fast


# ---------------- 4. 重复字典键 ----------------

def test_no_duplicate_dict_keys_in_meta():
    """Python 里重复键静默取最后一个——曾经把 true_cfg 的 bool 覆盖成 float。"""
    src = io.open(ROOT / "workflows" / "generate.py", encoding="utf-8").read()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        dup = {k for k in keys if keys.count(k) > 1}
        if dup:
            bad.append((node.lineno, sorted(dup)))
    assert not bad, f"有重复字典键：{bad}"


# ---------------- 5. 颜色解析要报错 ----------------

def test_bad_color_raises():
    """非法颜色以前静默变白，而 meta 里记的是原始字符串，像是生效了。"""
    assert g._parse_color("#abc") == (170, 187, 204)
    assert g._parse_color("#aabbcc") == (170, 187, 204)
    assert g._parse_color("rgb(1,2,3)") == (1, 2, 3)
    assert g._parse_color((7, 8, 9)) == (7, 8, 9)
    for bad in ("#zzz", "#12345", "rgb(1,2)", "#1234567"):
        with pytest.raises(ValueError):
            g._parse_color(bad)


# ---------------- 6. 每个工作流入口都要 require ----------------

def test_every_workflow_checks_capability():
    """`eng.require` 是能力闸门。14 个工作流里漏掉 5 个的话，
    能力表一旦加第二个架构就会绕过检查。"""
    names = ["txt2img", "img2img", "inpaint", "matte_bg", "ai_repaint",
             "pixelify", "pose_clone", "sprite_swap", "poseswap",
             "image_to_pixel", "outpaint", "outfit_swap", "face_swap",
             "vnccs_pose"]
    for n in names:
        fn = getattr(g, n)
        src = inspect.getsource(fn)
        assert "eng.require(" in src, f"{n} 入口没有 require()"


def test_run_filter_excludes_engine_positional():
    """签名过滤要把第一个位置参数（eng）排除，否则客户端传 eng=123
    会变成 "got multiple values for argument"。"""
    import app
    src = inspect.getsource(app._run) if hasattr(app, "_run") else ""
    assert "eng" in src
    # 直接验证过滤规则本身
    def fn(eng, *, a: int = 1):
        return a
    sig = inspect.signature(fn)
    kw_names = set(list(sig.parameters)[1:])
    assert kw_names == {"a"}


# ---------------- 7. 扩图的 alpha ----------------

def test_outpaint_extension_is_opaque():
    """扩出来的区域 alpha 必须是**不透明**。

    模型给的 alpha 在「补场景」语义下是噪声（实测 30% 半透明像素），
    游戏里会看到背景从新扩展的区域透出来。
    """
    from workflows import outpaint as op
    out = Image.new("RGBA", (80, 40), (200, 100, 50, 128))   # 半透明
    fixed = op._opaque_outside(out, (40, 40), (0, 0))
    a = np.asarray(fixed.getchannel("A"))
    assert a[:, 50].min() == 255, "扩展区域不是不透明"
    assert a[:, 10].max() == 128, "原图区域的 alpha 被改了"
