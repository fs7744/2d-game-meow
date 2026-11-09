"""画风中立性：**公共提示词不许影响任何画风，画风只由画风库自己那条定义决定**。

回归的是一次实测故障：Q 版（Chibi Anime）出图不像 Q 版，其它画风也被拉平。
根因不是一个，而是一串「丢信息 / 掺风格」：

1. 中文模式把官方三段式画风压成只剩一个中文画风名，Visual description 全丢；
2. 公共段写死「默认日式动画风格」兜底 + 一句泛化的「按这个画风的笔触…」；
3. 官方 race_features（物种特征）中文模式被置空；
4. custom_style 字段从来不进 prompt；
5. PE 请求用 rsplit 切 LoRA 行，把排在它后面的画风段一起砍了；
6. 立绘出图沿用 txt2img 的默认 preset="sprite"，负向词里带 "photo, 3d render"。
"""
from __future__ import annotations

import pytest

from workflows import charactergen as cg
from workflows import presets


BASE = {
    "sex": "female", "age": 22, "hair": "blonde hair", "eyes": "blue eyes",
    "background_color": "green", "framing": "full_body",
    "aesthetics": "masterpiece", "prompt_language": "zh",
}


# ------------------------------------------------------------ 画风定义完整进 prompt

def test_chibi_keeps_its_own_visual_definition():
    """Q 版最要紧的「大头压倒小身体」不能只剩一个名字。"""
    p = cg.character_prompt(BASE, "chibi_anime")
    for word in ("Q 版动漫", "super-deformed", "head dominates",
                 "oversized expressive eyes", "rounded linework"):
        assert word in p, f"Q 版画风定义丢了：{word}"


def test_every_catalog_style_keeps_its_description():
    """画风库 358 条逐条过：只要条目自带描述，就必须原样出现在中文 prompt 里。

    这一条钉死的是「画风由自己影响」——以后谁再加一句「统一按 XX 风格画」
    或把描述截短，都会在这里被拦下。
    """
    checked = 0
    for group in cg.style_catalog().get("groups") or []:
        for item in group.get("styles") or []:
            ref = str(item.get("prompt") or "")
            name, _reference, description = cg._split_style(ref)
            description = description.strip()
            if len(description) < 40:
                continue
            p = cg.structured_prompt(cg.qi2_fields(BASE), ref, "zh")
            assert description[:60] in p, f"画风 {item.get('id')} 的视觉定义被丢了"
            checked += 1
    assert checked > 300, f"只覆盖了 {checked} 条画风，画风库是不是没加载？"


def test_three_line_style_keeps_reference_and_description():
    """标准三段式（40 条）的 Reference 与 Visual description 都要在。"""
    three_line = [
        (_g, _s) for _g in (cg.style_catalog().get("groups") or [])
        for _s in (_g.get("styles") or [])
        if len(str(_s.get("prompt") or "").splitlines()) == 3
        and str(_s.get("prompt") or "").splitlines()[1].startswith("Reference: ")
    ]
    assert three_line, "画风库里应该有标准三段式条目"
    ref = str(three_line[0][1]["prompt"])
    name, reference, description = cg._split_style(ref)
    p = cg.structured_prompt(cg.qi2_fields(BASE), ref, "zh")
    assert reference.strip() and reference.strip() in p, "参考源流被丢了"
    assert description.strip() and description.strip()[:60] in p


# ------------------------------------------------------------ 公共段不许掺风格

def test_no_style_means_no_style_words_at_all():
    """没选画风：公共段一个风格字都不写，绝不替用户决定画风。"""
    p = cg.structured_prompt(cg.qi2_fields(BASE), "", "zh")
    assert "请画一张角色立绘" in p
    for word in ("【画风】", "默认日式动画风格", "按这个画风", "画风：", "Style:"):
        assert word not in p, f"公共段偷偷指定了画风：{word}"
    assert cg.style_block("", "zh") == ""


def test_unknown_style_name_never_falls_back_to_default():
    """译表查不到的自定义画风名：原样用，不兜底成日式动画。"""
    ref = "Style: My Private Style.\nA very particular flat-shaded look with square heads."
    p = cg.structured_prompt(cg.qi2_fields(BASE), ref, "zh")
    assert "My Private Style" in p and "square heads" in p
    assert "默认日式动画风格" not in p
    assert "按这个画风" not in p


def test_style_only_appears_once_as_its_own_block():
    """画风内容只出现在独立画风段里，不混进取景/背景/画质这些公共行。"""
    p = cg.character_prompt(BASE, "chibi_anime")
    public = "\n".join(l for l in p.splitlines()
                       if l.startswith(("请画", "画质要求", "画面里", "取景：", "背景：")))
    assert "super-deformed" not in public and "Chibi" not in public
    assert p.count("【画风】") == 1


# ------------------------------------------------------------ 其它被丢的字段

def test_race_features_survive_chinese_mode():
    """官方物种特征（尖耳朵…）中文模式也要在，别只剩「精灵」一个词。"""
    p = cg.character_prompt({**BASE, "race": "elf"}, "chibi_anime")
    assert "物种特征：" in p and "pointed ears" in p
    assert "种族：精灵" in p


def test_custom_style_reaches_prompt_and_is_not_rewritten():
    p = cg.character_prompt({**BASE, "custom_style": "大头小身，圆润描边"}, "chibi_anime")
    assert "自定义画风要求：大头小身，圆润描边" in p
    assert "custom_style" in cg.QI2_LITERAL_FIELDS, "用户自由文本不能被 PE 扩写掉"


def test_en_template_still_official():
    """英文模式仍走官方三段式，新增字段用空行不占位的规则，老模板不变。"""
    en = cg.character_prompt({**BASE, "prompt_language": "en"}, "chibi_anime")
    for key in ("Draw an image of a character in the following style:",
                "Style reference:", "Style description:",
                "The image contains one character.", "Framing:", "Background:",
                "super-deformed"):
        assert key in en
    assert "默认日式动画风格" not in en


# ------------------------------------------------------------ PE 链路

class _Pipe:
    pass


def test_pe_request_keeps_style_and_drops_only_lora(monkeypatch):
    """PE 改写要看得见画风；只删 LoRA 那一行，别把它后面的行一起砍掉。"""
    seen = {}

    def fake_enhance(pipe, request, **kw):
        seen["request"] = request
        return "一位 Q 版精灵少女，全身入画。"

    monkeypatch.setattr(cg.qwen_text, "enhance", fake_enhance)
    info = {**BASE, "race": "elf", "custom_style": "圆润描边",
            "lora_prompt": "my_lora"}
    r = cg.enhance_character_prompt(_Pipe(), info, "chibi_anime", enhance=True)
    req = seen["request"]
    assert "super-deformed" in req, "PE 请求把画风段砍了"
    assert "自定义画风要求" in req
    assert "my_lora" not in req, "LoRA 触发词不该送进改写器"
    assert r["applied"] == "1"
    assert "super-deformed" in r["prompt"], "PE 之后要补回完整画风定义"
    assert r["prompt"].count("my_lora") == 1, "LoRA 触发词不能重复"


def test_pe_failure_adds_lora_only_once(monkeypatch):
    """改写失败退回原文时，原文已含 LoRA 行，不能再追加一次。"""
    monkeypatch.setattr(cg.qwen_text, "generate_text", lambda *a, **k: "")
    r = cg.enhance_character_prompt(_Pipe(), {**BASE, "lora_prompt": "my_lora"},
                                    "chibi_anime", enhance=True)
    assert r["applied"] == "0"
    assert r["prompt"].count("my_lora") == 1
    assert "super-deformed" in r["prompt"], "退回原文也要有完整画风"


# ------------------------------------------------------------ 出图不挂公共契约

def test_portrait_generate_forces_preset_none(client, monkeypatch):
    """立绘出图必须显式关掉 sprite 契约：客户端手传 preset 也盖不掉。

    `SPRITE_NEG_PAINT` 里写着 `photo, 3d render` —— 选写实 / 3D 画风时
    等于用负向词把自己选的画风压掉；`SPRITE_SUFFIX_PAINT` 的
    `ample empty space` 又跟立绘「占满画面高度」直接打架。
    """
    import app as appmod
    from workflows import characters as ch

    name = "StyleNeutralProbe"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)

    class FakeQ:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

        def qsize(self):
            return len(self.items)

    fake = FakeQ()
    monkeypatch.setattr(appmod, "_q", fake)
    r = client.post(f"/characters/{name}/generate",
                    json={"workflow": "w1", "prompt": "x",
                          "params": {"preset": "sprite", "steps": 25}})
    assert r.status_code == 200, r.text
    assert fake.items, "任务没进队列"
    params = fake.items[0][2]
    assert params["preset"] == "none"
    assert params["steps"] == 25, "别的参数不能被顺手丢掉"
    # 负向词只留角色自己的画质词，不能含 presets 的公共风格压制词
    assert "3d render" not in params.get("negative", "")
    assert "photo" not in params.get("negative", "")


def test_sprite_preset_really_contains_style_suppressors():
    """钉住上游事实：sprite 契约里确实有压制画风的词，所以必须关。"""
    assert "3d render" in presets.SPRITE_NEG_PAINT
    assert "photo" in presets.SPRITE_NEG_PAINT
    assert "ample empty space" in presets.SPRITE_SUFFIX_PAINT
