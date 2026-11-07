"""多角色参考图的主体数 sidecar 与提示词契约。

单主体路径必须**字节不变** —— 这条比「多人时能改对」更重要，因为绝大多数
产线都是单主体，任何意外改写都会静默影响出图质量。
"""
from __future__ import annotations

import base64
import json

import pytest

from workflows import generate as gen
from workflows import posesubjects as ps

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg==")

WPC_INSTR = (
    "The first image is the character reference: keep the same face, hairstyle, "
    "outfit and colors. The second image is the pose condition (a body skeleton "
    "or silhouette). Redraw the same character in exactly that pose, "
    "full body, centered, clean simple background.")


def test_normalize_count_is_conservative():
    assert ps.normalize_count(None) == 1
    assert ps.normalize_count("x") == 1
    assert ps.normalize_count(0) == 1
    assert ps.normalize_count(1) == 1
    assert ps.normalize_count(3) == 3
    assert ps.normalize_count(99) == ps.MAX_SUBJECTS


def test_sidecar_round_trip(tmp_path):
    img = tmp_path / "mannequin.png"
    img.write_bytes(PNG)
    ps.write_subjects(img, 1)
    assert not ps.sidecar_path(img).exists(), "单主体不该留垃圾 sidecar"
    assert ps.subjects_for(img) == 1

    ps.write_subjects(img, 3)
    assert json.loads(ps.sidecar_path(img).read_text("utf-8")) == {"pose_subjects": 3}
    assert ps.subjects_for(img) == 3
    assert ps.subjects_for(str(img)) == 3


def test_missing_or_broken_sidecar_falls_back_to_one(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    assert ps.subjects_for(img) == 1
    ps.sidecar_path(img).write_text("{broken", encoding="utf-8")
    assert ps.subjects_for(img) == 1
    assert ps.subjects_for(None) == 1


def test_single_subject_instruction_is_byte_identical():
    assert ps.adapt_instruction(WPC_INSTR, 1) == WPC_INSTR
    assert ps.adapt_negative(gen.QWEN_FRAME_NEG, 1) == gen.QWEN_FRAME_NEG
    assert ps.adapt_instruction(WPC_INSTR, 0) == WPC_INSTR
    assert ps.adapt_instruction("", 2) == ""


def test_multi_subject_instruction_switches_contract():
    out = ps.adapt_instruction(WPC_INSTR, 2)
    assert "a single character" not in out
    assert "the same character" not in out.lower()
    assert "the two characters" in out
    assert "relative positions" in out
    assert "do not merge or drop any figure" in out
    # 身份描述里的 "the same face" 不能被误改
    assert "keep the same face, hairstyle, outfit and colors" in out


def test_multi_subject_removes_two_characters_from_negative():
    assert "two characters" in gen.QWEN_FRAME_NEG, "前提：w4 负向词里确实有这条"
    out = ps.adapt_negative(gen.QWEN_FRAME_NEG, 2)
    assert "two characters" not in out
    assert "extra head" in out and "missing head" in out, "只删冲突项，别整段丢掉"


def test_solo_instruction_and_frame_rules_become_plural():
    solo = ("This image is a body skeleton showing a pose. Draw a single character "
            "in exactly that pose, full body, centered." + gen.QWEN_FRAME_RULES)
    out = ps.adapt_instruction(solo, 3)
    assert "a single character" not in out
    assert "three characters" in out
    assert "The figures must fill" in out
    assert "The figure must fill" not in out


def test_upload_writes_subjects_sidecar(client, tmp_path, monkeypatch):
    import app as appmod
    from workflows import paths

    monkeypatch.setattr(appmod, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(appmod, "ASSETS", tmp_path)
    r = client.post("/upload?name=mannequin.png&subjects=3", content=PNG,
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pose_subjects"] == 3
    assert ps.subjects_for(body["path"]) == 3
    assert ps.sidecar_path(body["path"]).is_file()


def test_upload_without_subjects_leaves_no_sidecar(client, tmp_path, monkeypatch):
    import app as appmod
    monkeypatch.setattr(appmod, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(appmod, "ASSETS", tmp_path)
    r = client.post("/upload?name=plain.png", content=PNG,
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200
    body = r.json()
    assert body["pose_subjects"] == 1
    assert not ps.sidecar_path(body["path"]).exists()
