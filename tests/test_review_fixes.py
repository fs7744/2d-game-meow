"""评审修复的回归测试（docs/REVIEW-2026-10-28.md 的 P0 项）。

这三个问题都有共同特征：**不报错、只是静默变差**。所以必须有断言钉住，
否则下次重构很容易再退化回去。
"""
from __future__ import annotations

import inspect
import json

import pytest

from workflows import generate as g
from workflows.engine import BASE_MODELS, Engine, LORA_ATTR


class _FakePipe:
    """够 apply_lora 用的最小管线：只记 unet 与已加载的权重。"""

    class _U:
        pass

    def __init__(self) -> None:
        self.unet = _FakePipe._U()
        self.loaded: list[str] = []

    def load_lora_weights(self, path, adapter_name=None):      # noqa: D401
        self.loaded.append(str(path))

    def unload_lora_weights(self):
        self.loaded.clear()

    def set_adapters(self, names, adapter_weights=None):
        pass


@pytest.fixture()
def fresh_engine(tmp_path):
    """独立 models_root，避免和 session 级 eng 共享权重池。"""
    (tmp_path / "checkpoints").mkdir()
    return Engine(tmp_path)


# ---------------- ensure_headroom：判据必须是「空闲」而不是「已用」 ----------------

def test_headroom_keeps_cache_when_plenty_free(fresh_engine, monkeypatch):
    """空闲充足时不该动缓存。

    老实现比的是 `memory_reserved() < 2GB`，任何管线加载后都恒不成立，
    于是**每次调用都清空整个缓存**——这才是 wss/wpc 反复重载 UNet 的原因。
    """
    for k in ("txt2img:sd15", "pose:openpose:1:sd15"):
        fresh_engine._cache[k] = object()

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info",
                        lambda: (20 * 2**30, 24 * 2**30))   # 4GB 被占
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda: 4 * 2**30)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

    fresh_engine.ensure_headroom(need_gb=2.0)
    assert set(fresh_engine._cache) == {"txt2img:sd15", "pose:openpose:1:sd15"}


def test_headroom_evicts_lru_until_enough_free(fresh_engine, monkeypatch):
    """真正不够时才驱逐，且**从最久没用的开始**，腾够就停。"""
    for k in ("a:sd15", "b:sd15", "c:sd15"):
        fresh_engine._cache[k] = object()

    reserved = {"gb": 20.0}                     # 只剩 4GB 空闲

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info",
                        lambda: (24 * 2**30 - int(reserved["gb"] * 2**30), 24 * 2**30))
    monkeypatch.setattr("torch.cuda.memory_reserved",
                        lambda: int(reserved["gb"] * 2**30))

    def _empty():
        reserved["gb"] = max(0.0, reserved["gb"] - 3.0)
    monkeypatch.setattr("torch.cuda.empty_cache", _empty)

    # 需要 6GB 空闲：当前 4GB → 驱逐一条（释放 3GB）= 7GB ≥ 6GB，应当停下
    fresh_engine.ensure_headroom(need_gb=6.0)
    assert list(fresh_engine._cache) == ["b:sd15", "c:sd15"]


def test_headroom_never_evicts_protected(fresh_engine, monkeypatch):
    """`keep` 里的管线绝不能丢——否则刚加载好的会被自己扔掉再重载。"""
    for k in ("txt2img:sd15", "pose:openpose:1:sd15", "animatediff:canny:1:sd15"):
        fresh_engine._cache[k] = object()

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (0, 24 * 2**30))
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda: 24 * 2**30)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

    fresh_engine.ensure_headroom(need_gb=2.0, keep=("pose",))
    assert "pose:openpose:1:sd15" in fresh_engine._cache
    assert "animatediff:canny:1:sd15" not in fresh_engine._cache


def test_headroom_drops_orphan_controlnet(fresh_engine, monkeypatch):
    """管线被驱逐后，它独占的 ControlNet 也要跟着走（否则残留 1.4~2.4GB）。"""
    fresh_engine._cache["pose:openpose:1:sd15"] = object()
    fresh_engine._ctrl["openpose:sd15"] = object()
    fresh_engine._ctrl["canny:sd15"] = object()

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (0, 24 * 2**30))
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda: 24 * 2**30)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

    fresh_engine.ensure_headroom(need_gb=2.0)
    assert fresh_engine._cache == {}
    assert dict(fresh_engine._ctrl) == {}, "没有管线引用时 ControlNet 必须释放"


def test_headroom_keeps_controlnet_still_in_use(fresh_engine, monkeypatch):
    fresh_engine._cache["pose:openpose:1:sd15"] = object()
    fresh_engine._ctrl["openpose:sd15"] = object()

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (0, 24 * 2**30))
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda: 24 * 2**30)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

    fresh_engine.ensure_headroom(need_gb=2.0, keep=("pose",))
    assert "openpose:sd15" in fresh_engine._ctrl


# ---------------- LoRA：状态跟着 UNet 走，不跟着 Engine 走 ----------------

def test_lora_applied_to_every_pipeline(fresh_engine, tmp_path):
    """两条不同管线用同一个 LoRA，**两条都必须真的加载**。

    老实现的 `self._lora` 是池级的：第二条管线命中同一个 key 直接 return，
    但它的 UNet 从来没装过 LoRA —— 于是「W1 用了像素 LoRA，接着跑 W4」
    会静默丢失风格。
    """
    lora = tmp_path / "px.safetensors"
    lora.write_bytes(b"\0" * 16)

    a, b = _FakePipe(), _FakePipe()
    fresh_engine.apply_lora(a, lora, 0.75)
    fresh_engine.apply_lora(b, lora, 0.75)

    assert a.loaded == [str(lora)]
    assert b.loaded == [str(lora)], "第二条管线也必须真的加载"
    assert getattr(a.unet, LORA_ATTR) == getattr(b.unet, LORA_ATTR)


def test_lora_shared_unet_is_not_reloaded(fresh_engine, tmp_path):
    """共享同一个 UNet 的两条管线（pose 复用 txt2img 的权重）不该重复加载。"""
    lora = tmp_path / "px.safetensors"
    lora.write_bytes(b"\0" * 16)

    a, b = _FakePipe(), _FakePipe()
    b.unet = a.unet                    # 共享
    fresh_engine.apply_lora(a, lora, 0.75)
    fresh_engine.apply_lora(b, lora, 0.75)

    assert a.loaded == [str(lora)]
    assert b.loaded == [], "同一个 UNet 已经装过了，不该再装一遍"


def test_lora_scale_change_reloads(fresh_engine, tmp_path):
    lora = tmp_path / "px.safetensors"
    lora.write_bytes(b"\0" * 16)
    p = _FakePipe()
    fresh_engine.apply_lora(p, lora, 0.75)
    fresh_engine.apply_lora(p, lora, 0.4)
    assert len(p.loaded) == 1          # 卸载后重载，最终仍是同一份
    assert getattr(p.unet, LORA_ATTR) == "px.safetensors:0.4"


def test_lora_none_unloads(fresh_engine, tmp_path):
    lora = tmp_path / "px.safetensors"
    lora.write_bytes(b"\0" * 16)
    p = _FakePipe()
    fresh_engine.apply_lora(p, lora, 0.75)
    fresh_engine.apply_lora(p, None)
    assert p.loaded == []
    assert getattr(p.unet, LORA_ATTR, None) is None


# ---------------- 架构判定：底模可以是任意目录名 ----------------

def test_arch_reads_model_index(fresh_engine):
    d = fresh_engine.root / "checkpoints" / "sd15"
    d.mkdir(parents=True)
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), "utf-8")
    assert fresh_engine.arch("sd15") == "sd15"

    x = fresh_engine.root / "checkpoints" / "xl_style"
    x.mkdir(parents=True)
    (x / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), "utf-8")
    fresh_engine._discover_bases()
    assert fresh_engine.arch("xl_style") == "sdxl", "架构要靠 _class_name 判定，不能看目录名"
    assert "xl_style" in fresh_engine.available_models()


def test_arch_unknown_model_rejected(fresh_engine):
    with pytest.raises(ValueError, match="未知底模"):
        fresh_engine.arch("nope")


def test_custom_base_registered(fresh_engine, tmp_path):
    """放一个新目录 = 多一个可选底模（换社区微调不需要改代码）。"""
    d = fresh_engine.root / "checkpoints" / "dreamshaper"
    d.mkdir(parents=True)
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), "utf-8")
    fresh_engine._discover_bases()
    assert "dreamshaper" in BASE_MODELS
    assert fresh_engine._check_model("dreamshaper") == d


# ---------------- WPC / WSS 必须能透传底模 ----------------

def test_wpc_accepts_model():
    """WPC 曾经硬编码 model="sd15"，用户切 SDXL 也不生效且没有告警。"""
    assert "model" in inspect.signature(g.pose_clone).parameters


def test_wss_accepts_model():
    assert "model" in inspect.signature(g.sprite_swap).parameters


def test_animatediff_rejects_sdxl(fresh_engine):
    """AnimateDiff 的 motion module 只能挂 SD1.5：必须显式报错，而不是静默降级。"""
    d = fresh_engine.root / "checkpoints" / "sdxl"
    d.mkdir(parents=True)
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), "utf-8")
    with pytest.raises(ValueError, match="只支持 SD1.5"):
        fresh_engine.animatediff_pipe(controlnet="canny", with_ipadapter=False, model="sdxl")


def test_discovery_skips_inpaint_checkpoints(fresh_engine):
    """inpaint 权重（9 通道 UNet）不能被登记成文生图底模。

    自动发现一开始没排除它，结果 `sd15-inpaint` 出现在前端下拉里，
    选中就在采样时报 `size of tensor a (9) must match b (4)`。
    """
    d = fresh_engine.root / "checkpoints" / "my-inpaint"
    (d / "unet").mkdir(parents=True)
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionInpaintPipeline"}), "utf-8")
    (d / "unet" / "config.json").write_text(json.dumps({"in_channels": 9}), "utf-8")

    fresh_engine._discover_bases()
    assert "my-inpaint" not in fresh_engine.available_models()

    # 4 通道的正常底模仍要被发现
    ok = fresh_engine.root / "checkpoints" / "community_sd15"
    (ok / "unet").mkdir(parents=True)
    (ok / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), "utf-8")
    (ok / "unet" / "config.json").write_text(json.dumps({"in_channels": 4}), "utf-8")
    fresh_engine._discover_bases()
    assert "community_sd15" in fresh_engine.available_models()


def test_discovery_skips_inpaint_checkpoints(fresh_engine):
    """inpaint 权重（9 通道 UNet）不能被登记成文生图底模。

    自动发现一开始没排除它，结果 `sd15-inpaint` 出现在前端下拉里，
    选中就在采样时报 `size of tensor a (9) must match b (4)`。
    """
    d = fresh_engine.root / "checkpoints" / "my-inpaint"
    (d / "unet").mkdir(parents=True)
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionInpaintPipeline"}), "utf-8")
    (d / "unet" / "config.json").write_text(json.dumps({"in_channels": 9}), "utf-8")

    fresh_engine._discover_bases()
    assert "my-inpaint" not in fresh_engine.available_models()

    # 4 通道的正常底模仍要被发现
    ok = fresh_engine.root / "checkpoints" / "community_sd15"
    (ok / "unet").mkdir(parents=True)
    (ok / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}), "utf-8")
    (ok / "unet" / "config.json").write_text(json.dumps({"in_channels": 4}), "utf-8")
    fresh_engine._discover_bases()
    assert "community_sd15" in fresh_engine.available_models()


def test_lora_base_mismatch_reports_cause_not_variant(fresh_engine, tmp_path):
    """SD1.5 的 LoRA 挂到 SDXL 上报的是 size mismatch（640 vs 1280）。

    旧实现把它归因为"LoKr/LoHa 变体不支持"，用户会去换 LoRA——但换多少个都没用，
    真正的原因是拿错了基座。这里钉住诊断文案。
    """
    lora = tmp_path / "px.safetensors"
    lora.write_bytes(b"\0" * 16)

    class BadPipe(_FakePipe):
        def load_lora_weights(self, path, adapter_name=None):
            raise RuntimeError(
                "size mismatch for up_blocks.1.attn2.to_v.lora_B.char.weight: "
                "copying a param with shape torch.Size([1280, 32]) from checkpoint, "
                "the shape in current model is torch.Size([640, 32]).")

    with pytest.raises(ValueError) as exc:
        fresh_engine.apply_lora(BadPipe(), lora, 0.75)
    msg = str(exc.value)
    assert "不匹配" in msg or "另一个架构" in msg
    assert "LoKr" not in msg, "别把基座不匹配说成 LoKr 变体"
    # 失败后不能留下脏状态
    assert fresh_engine._pool["lora"] is None
