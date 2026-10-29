"""LoRA 注册表 + 去碎屑 + 贴边检测。"""
from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

from workflows import lora as L
from workflows import matte as M
from workflows import presets as P


# ---------------- LoRA 注册表 ----------------

def _fake_lora(root, name: str, size: int = 1024):
    d = root / "lora"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.safetensors"
    p.write_bytes(b"\0" * size)
    return p


def test_scan_empty_when_no_dir(tmp_path):
    assert L.scan(tmp_path) == []


def test_scan_reads_metadata(tmp_path):
    _fake_lora(tmp_path, "wuxia_v2")
    _fake_lora(tmp_path, "other")
    (tmp_path / "lora" / "loras.json").write_text(
        json.dumps({"wuxia_v2": {"label": "武侠", "trigger": "wuxia", "scale": 0.75}}),
        encoding="utf-8")

    items = {x["name"]: x for x in L.scan(tmp_path)}
    assert items["wuxia_v2"]["label"] == "武侠"
    assert items["wuxia_v2"]["trigger"] == "wuxia"
    assert items["wuxia_v2"]["scale"] == 0.75
    assert items["wuxia_v2"]["path"] == "lora/wuxia_v2.safetensors"
    # 没写元数据的用默认，但仍可用
    assert items["other"]["label"] == "other"
    assert items["other"]["scale"] == 0.8


def test_scan_survives_broken_json(tmp_path):
    _fake_lora(tmp_path, "x")
    (tmp_path / "lora" / "loras.json").write_text("{ broken", encoding="utf-8")
    items = L.scan(tmp_path)
    assert len(items) == 1 and items[0]["name"] == "x"


def test_resolve_accepts_name_and_path(tmp_path):
    p = _fake_lora(tmp_path, "wuxia_v2")
    assert L.resolve(tmp_path, "wuxia_v2") == str(p)
    assert L.resolve(tmp_path, "lora/wuxia_v2.safetensors") == str(p)
    assert L.resolve(tmp_path, str(p)) == str(p)
    assert L.resolve(tmp_path, None) is None
    assert L.resolve(tmp_path, "nope") is None


# ---------------- 去碎屑 ----------------

def _figure_with_debris():
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 150, 180], fill=(30, 30, 30, 255))     # 主体，面积 111*141
    d.rectangle([180, 120, 192, 140], fill=(200, 20, 20, 255))  # 印章（小）
    d.point([(20, 190)], fill=(1, 2, 3, 255))                   # 一粒屑
    return im


def test_despeckle_drops_seal_and_dust():
    out = M.despeckle(_figure_with_debris())
    a = out.getchannel("A")
    assert a.getpixel((190, 130)) == 0, "印章应被清掉"
    assert a.getpixel((20, 190)) == 0, "碎屑应被清掉"
    assert a.getpixel((100, 100)) == 255, "主体必须保留"


def test_despeckle_keeps_single_subject_untouched():
    im = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([10, 10, 40, 40], fill=(9, 9, 9, 255))
    assert M.despeckle(im).getchannel("A").getbbox() == (10, 10, 41, 41)


def test_despeckle_noop_when_empty():
    im = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    assert M.despeckle(im).getchannel("A").getbbox() is None


# ---------------- 贴边检测 ----------------

def test_edge_stats_clean_frame_is_zero():
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([20, 20, 80, 80], fill=(1, 1, 1, 255))
    assert M.edge_touch(im) == []
    assert max(M.edge_stats(im).values()) == 0.0


def test_edge_stats_flags_cut_feet():
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    # 前景一直画到最后一行的 60%，等价于脚被裁
    ImageDraw.Draw(im).rectangle([20, 20, 80, 99], fill=(1, 1, 1, 255))
    st = M.edge_stats(im)
    assert st["bottom"] > 0.5
    assert "bottom" in M.edge_touch(im)


def test_edge_touch_ignores_hairline_contact():
    """发丝蹭到边一两个像素是正常构图，不该算「被裁」。"""
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 60, 60], fill=(1, 1, 1, 255))
    d.point([(50, 0)], fill=(1, 1, 1, 255))       # 顶部单点接触，占比 1%
    assert M.edge_touch(im) == []
    assert M.edge_stats(im)["top"] == pytest.approx(0.01, abs=1e-6)


# ---------------- 武侠契约 ----------------

def test_wuxia_preset_exists_and_differs_from_sprite():
    pos, neg = P.build("swordsman", None, "wuxia")
    assert "ink painting" in pos
    # 武侠契约刻意不写 pixel art：实测 LoRA 会压过风格词，两头不到位
    assert "pixel art" not in pos
    assert "seal" in neg and "signature" in neg, "武侠 LoRA 会盖印章，必须压"
    assert "cut off head" in neg

    sprite_pos, _ = P.build("swordsman", None, "sprite")
    assert "pixel art" in sprite_pos


def test_wuxia_preset_keeps_budget():
    pos, neg = P.build("Wuxia, swordsman, holding sword, front view", None, "wuxia")
    assert P.rough_tokens(pos) <= 77
    assert P.rough_tokens(neg) <= 77


def test_unknown_preset_still_raises():
    with pytest.raises(ValueError):
        P.build("x", None, "nope")


# ---------------- 架构自动识别（读 header，不加载权重） ----------------

def _fake_st(root, name: str, tensors: dict, metadata: dict | None = None):
    """造一个**只有合法 header** 的 safetensors：probe 只读头部，不碰权重。"""
    import struct

    d = root / "lora"
    d.mkdir(parents=True, exist_ok=True)
    head = {k: {"dtype": "F16", "shape": list(v), "data_offsets": [0, 0]}
            for k, v in tensors.items()}
    if metadata:
        head["__metadata__"] = metadata
    blob = json.dumps(head).encode()
    p = d / f"{name}.safetensors"
    p.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return p


def test_probe_distinguishes_architectures(tmp_path):
    """靠上下文维度（768/2048）和键名（adaLN/cross_attn）分架构。

    这是最要紧的一条：架构拿错的报错是 size mismatch，用户会以为文件坏了，
    然后换一个同样错的 LoRA。
    """
    p = _fake_st(tmp_path, "sd15_lora",
                 {"lora_unet_down_blocks_0_attn.to_k.lora_down.weight": (32, 768)})
    assert L.probe(p)["base"] == "sd15"

    p = _fake_st(tmp_path, "xl_lora",
                 {"lora_unet_input_blocks_4_1_proj_in.lora_down.weight": (32, 2048)})
    assert L.probe(p)["base"] == "sdxl"

    # DiT（Flux/anima）：键里有 adaLN，上下文维度也不是 768/2048
    p = _fake_st(tmp_path, "flux_lora",
                 {"lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_down.weight": (32, 3072)})
    info = L.probe(p)
    assert info["base"] == "flux" and not info["supported"]

    # 视频模型（Wan）：顶层就是 diffusion_model.
    p = _fake_st(tmp_path, "wan_lora",
                 {"diffusion_model.blocks.0.cross_attn.k.lora_A.weight": (32, 5120)})
    info = L.probe(p)
    assert info["base"] == "wan" and not info["supported"]


def test_probe_recognises_diffusers_naming(tmp_path):
    """diffusers 格式是 `unet.*.lora.down.weight`，不是 kohya 的 `lora_down`。

    实测 mk_sp_pixel 就是这种：只认 kohya 命名会让它识别失败。
    """
    p = _fake_st(tmp_path, "diffusers_style",
                 {"unet.down_blocks.0.attentions.0.to_k.lora.down.weight": (16, 768)})
    assert L.probe(p)["base"] == "sd15"


def test_probe_falls_back_to_metadata_for_lokr(tmp_path):
    """LoKr/LoHa 变体没有 lora_down 张量，只能读训练元数据。"""
    p = _fake_st(
        tmp_path, "xl_lokr",
        {"lora_unet_input_blocks_1_0.lokr_w1": (64, 64)},
        {"ss_base_model_version": "sdxl_base_v1-0",
         "modelspec.architecture": "stable-diffusion-xl-v1-base/lora"})
    assert L.probe(p)["base"] == "sdxl"


def test_autoscan_adds_missing_and_is_idempotent(tmp_path):
    _fake_st(tmp_path, "good_sd15",
              {"lora_unet_down_blocks_0_attn.to_k.lora_down.weight": (32, 768)})
    r1 = L.autoscan(tmp_path)
    assert r1["added"] == ["good_sd15"]
    cfg = json.loads((tmp_path / "lora" / "loras.json").read_text("utf-8"))
    assert cfg["good_sd15"]["base"] == "sd15"

    # 再跑一次：不该重复"新增"，也不该改任何东西
    r2 = L.autoscan(tmp_path)
    assert r2["added"] == [] and r2["filled"] == []
    assert json.loads((tmp_path / "lora" / "loras.json").read_text("utf-8")) == cfg


def test_autoscan_hides_unsupported_architectures(tmp_path):
    _fake_st(tmp_path, "wan_lora",
             {"diffusion_model.blocks.0.cross_attn.k.lora_A.weight": (32, 5120)})
    r = L.autoscan(tmp_path)
    assert "wan_lora" in r["hidden"]
    cfg = json.loads((tmp_path / "lora" / "loras.json").read_text("utf-8"))
    assert cfg["wan_lora"]["hidden"] is True
    assert "视频" in cfg["wan_lora"]["note"]


def test_autoscan_never_overwrites_handwritten_fields(tmp_path):
    """手写的 trigger / scale / note 一律不动，尤其是**故意留空的 trigger**。

    实测像素 LoRA 加触发词后 3/3 多主体，所以配置里那条 trigger 是故意留空的。
    自动扫描把 metadata 的高频 tag 填进去，等于把一条实测结论抹了。
    """
    _fake_st(tmp_path, "px",
             {"lora_unet_down_blocks_0_attn.to_k.lora_down.weight": (32, 768)},
             {"ss_tag_frequency": json.dumps({"1": {"pixelart": 50, "game": 20}})})
    (tmp_path / "lora" / "loras.json").write_text(json.dumps({
        "px": {"label": "手写标签", "trigger": "", "scale": 0.75, "note": "故意不加触发词"},
    }), "utf-8")

    L.autoscan(tmp_path)
    cfg = json.loads((tmp_path / "lora" / "loras.json").read_text("utf-8"))["px"]
    assert cfg["trigger"] == "", "空 trigger 是刻意留的，不能被自动填"
    assert cfg["label"] == "手写标签" and cfg["scale"] == 0.75
    assert cfg["base"] == "sd15", "base 是空缺，应该补上"


def test_autoscan_detects_duplicate_weights(tmp_path):
    """同一份权重存两个文件名：后加的标 hidden，**已有的不动**。"""
    tensors = {"lora_unet_down_blocks_0_attn.to_k.lora_down.weight": (32, 768)}
    _fake_st(tmp_path, "a_original", tensors)
    _fake_st(tmp_path, "b_copy", tensors)
    (tmp_path / "lora" / "loras.json").write_text(json.dumps({
        "a_original": {"label": "正本", "scale": 0.8, "note": "手工配置"},
    }), "utf-8")

    r = L.autoscan(tmp_path)
    cfg = json.loads((tmp_path / "lora" / "loras.json").read_text("utf-8"))
    assert cfg["a_original"].get("hidden") is None, "已有的正本不能被标隐藏"
    assert cfg["b_copy"]["hidden"] is True
    assert any(d["kept"] is False for d in r["duplicates"])
    assert any(x["kept"] == "a_original" for x in r["same_content"])


# ---------------- 静默空挂检测（apply_lora 的真实行为） ----------------

def _make_engine(root):
    """建一个只用来测 apply_lora 的引擎：不加载权重，只用它的 adapter 状态池。"""
    from workflows.engine import Engine

    return Engine(root)


class _Mod:
    """最小的模块替身：只需要 named_parameters / setattr。"""

    def __init__(self) -> None:
        self._params: dict = {}
        self._seq = 0

    def named_parameters(self):
        return iter(list(self._params.items()))

    def add_lora(self, n: int = 1) -> None:
        # 键名必须每次不同：固定键名的话第二次调用不会增加参数个数，
        # 于是"逐个校验注入"会误判成空挂（假管线自己的 bug 伪装成产品 bug）
        for _ in range(n):
            self._seq += 1
            self._params[f"blk{self._seq}.lora_A.weight"] = object()


class _FakePipe:
    """按 mode 模拟 diffusers 的三种行为，覆盖实测到的三条分支。"""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.unet = _Mod()
        self.text_encoder = None
        self.text_encoder_2 = None
        self.transformer = None

    def load_lora_weights(self, path, adapter_name=None):
        if self.mode == "inject":            # 正常的 SD LoRA
            self.unet.add_lora(3)
        elif self.mode == "silent":          # Wan/视频 LoRA：只 warning，不报错
            pass
        elif self.mode == "nomatch":         # DiT：PEFT 找不到目标模块
            raise RuntimeError(
                "Target modules {'q.proj', 'v.proj', 'mlp.1'} not found in the base model")

    def unload_lora_weights(self):
        pass

    def set_adapters(self, names, adapter_weights=None):
        pass

    def get_active_adapters(self):
        return []


def test_apply_lora_rejects_silent_noop(tmp_path):
    """加载"成功"但一个键都没匹配上时必须报错。

    实测 Wan 的视频 LoRA 挂 SDXL 就是这样：diffusers 只打三条 warning，
    unet 里 LoRA 参数增量是 0。界面显示已挂、出图毫无变化，用户会以为是
    强度不够，一路把 0.8 调到 1.0——全是白费。
    """
    lora = tmp_path / "wan.safetensors"
    lora.write_bytes(b"\0" * 16)
    eng = _make_engine(tmp_path)
    pipe = _FakePipe("silent")

    with pytest.raises(ValueError) as exc:
        eng.apply_lora(pipe, lora, 0.8)
    msg = str(exc.value)
    assert "没起作用" in msg
    assert "warning" in msg, "要说清这是「只给了 warning 不报错」的那种情况"
    assert eng._pool["lora"] is None, "失败后不能留脏状态"


def test_apply_lora_reports_module_mismatch(tmp_path):
    """PEFT 的 NoMatchingPeftModuleError 是"架构不对"，不是"LoKr 变体"。"""
    lora = tmp_path / "octo.safetensors"
    lora.write_bytes(b"\0" * 16)
    eng = _make_engine(tmp_path)

    with pytest.raises(ValueError) as exc:
        eng.apply_lora(_FakePipe("nomatch"), lora, 0.8)
    msg = str(exc.value)
    assert "结构对不上" in msg
    assert "LoKr" not in msg, "别把架构问题说成变体不支持"


def test_apply_lora_accepts_real_injection(tmp_path):
    lora = tmp_path / "ok.safetensors"
    lora.write_bytes(b"\0" * 16)
    eng = _make_engine(tmp_path)
    pipe = _FakePipe("inject")

    eng.apply_lora(pipe, lora, 0.8)
    assert eng._pool["lora"] and eng._pool["lora"].startswith("ok.safetensors")


# ---------------- 多个 LoRA 叠加 ----------------

class _StackPipe:
    """按文件名决定行为：`good` 真注入，`bad` 静默空挂（什么都不做）。

    用来验证"叠加时逐个校验"——只看总数会漏：一个空挂、另一个正常，
    总数照样上涨，但空挂那个永远不会生效。
    """

    def __init__(self) -> None:
        self.unet = _Mod()
        self.text_encoder = None
        self.text_encoder_2 = None
        self.transformer = None
        self.adapters: tuple[list[str], list[float]] = ([], [])
        self.unloaded = 0

    def load_lora_weights(self, path, adapter_name=None):
        if 'bad' not in str(path):
            self.unet.add_lora(2)

    def unload_lora_weights(self):
        self.unloaded += 1
        self.unet._params.clear()

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = (list(names), list(adapter_weights or []))

    def get_active_adapters(self):
        return []


def _mk(tmp_path, *names):
    out = []
    for n in names:
        p = tmp_path / f'{n}.safetensors'
        p.write_bytes(b'\0' * 16)
        out.append(p)
    return out


def test_apply_lora_stacks_two_with_own_scales(tmp_path):
    """两个 LoRA 各自一个 adapter，各自一个强度。"""
    eng = _make_engine(tmp_path)
    a, b = _mk(tmp_path, 'good_a', 'good_b')
    pipe = _StackPipe()

    eng.apply_lora(pipe, [a, b], [0.75, 0.5])
    names, weights = pipe.adapters
    assert names == ['lora0', 'lora1']
    assert weights == [0.75, 0.5]
    assert 'good_a' in eng._pool['lora'] and 'good_b' in eng._pool['lora']


def test_apply_lora_single_scale_applies_to_all(tmp_path):
    """强度给单个数值时所有 LoRA 共用。"""
    eng = _make_engine(tmp_path)
    paths = _mk(tmp_path, 'good_a', 'good_b')
    pipe = _StackPipe()

    eng.apply_lora(pipe, paths, 0.6)
    assert pipe.adapters[1] == [0.6, 0.6]


def test_apply_lora_pads_short_scale_list(tmp_path):
    """强度列表不够长时按 0.8 补齐——LoRA 是增强项，不该让整个任务失败。"""
    eng = _make_engine(tmp_path)
    paths = _mk(tmp_path, 'good_a', 'good_b')
    pipe = _StackPipe()

    eng.apply_lora(pipe, paths, [0.4])
    assert pipe.adapters[1] == [0.4, 0.8]


def test_apply_lora_names_the_silent_one_in_a_stack(tmp_path):
    """叠加时逐个校验：空挂的那个要被点名，而不是让另一个把它盖过去。"""
    eng = _make_engine(tmp_path)
    good, bad = _mk(tmp_path, 'good_a', 'bad_b')
    pipe = _StackPipe()

    with pytest.raises(ValueError) as exc:
        eng.apply_lora(pipe, [good, bad], [0.8, 0.8])
    assert '没起作用' in str(exc.value)
    assert 'bad_b' in str(exc.value)
    assert eng._pool['lora'] is None, '失败后不能留脏状态'
    assert pipe.unloaded >= 1


def test_apply_lora_single_still_uses_one_adapter(tmp_path):
    """单个路径是常态，行为必须与改动前一致。"""
    eng = _make_engine(tmp_path)
    (a,) = _mk(tmp_path, 'good_a')
    pipe = _StackPipe()

    eng.apply_lora(pipe, a, 0.8)
    assert pipe.adapters == (['lora0'], [0.8])


def test_probe_prefers_sdxl_when_both_768_and_2048(tmp_path):
    """dims 同时含 768 与 2048 → SDXL，**不能**按 768 判成 SD1.5。

    实测 15 个 SDXL 风格 LoRA（AkihikoStyleXL / BG3EXTurnXL / DiscoLevel …）
    的输入维度集合是 [640,768,1280,2048,2560,3072,5120]——既有 768 又有 2048。
    旧逻辑「两个都有 → 按 768 算 sd15」把它们全判成 SD1.5，挂上去直接报
    「LoRA 与当前底模的模块结构对不上」；改挂 sdxl 后 diff 64~78，全部生效。

    判据：2048 是 SDXL cross-attn 的文本上下文维度，SD1.5 的 UNet 里不存在
    （SD1.5 最大是 ff.net.0.proj 的 2560）。所以只要见到 2048 就是 SDXL。
    """
    p = _fake_st(tmp_path, "xl_style",
                 {"lora_unet_input_blocks_4_1_proj_in.lora_down.weight": (4, 2048),
                  "lora_unet_down_blocks_0_attn2_to_k.lora_down.weight": (4, 768)})
    assert L.probe(p)["base"] == "sdxl"

    # 反向不能误伤：只有 768、没有 2048 的仍是 SD1.5
    p2 = _fake_st(tmp_path, "sd15_only",
                  {"lora_unet_down_blocks_0_attn2_to_k.lora_down.weight": (4, 768),
                   "lora_unet_down_blocks_0_ff_net_0_proj.lora_down.weight": (4, 2560)})
    assert L.probe(p2)["base"] == "sd15"
