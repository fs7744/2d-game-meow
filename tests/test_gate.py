"""公共闸门组件（workflows/gate.py）的单元测试。

这三块是 w4 / wss 共用的，行为一旦变了会同时影响两个工作流，
所以单独钉住：
  · 候选怎么跑（跑满第一轮、没过才补提）
  · 同分取第一个（可复现）
  · 非人形主体不再补提
"""
from __future__ import annotations

from PIL import Image

from workflows import gate


def _img():
    return Image.new("RGBA", (8, 8), (10, 20, 30, 255))


def test_runs_all_candidates_in_first_round():
    """第一轮跑满 candidates 个，哪怕第一个就过闸（w4 的语义：要挑分数最高的）。"""
    seen = []

    def gen(i):
        seen.append(i)
        return _img(), {}

    def score(img, info):
        return {"score": 1.0, "passed": True}

    _img_out, ev, _info, attempts, nh = gate.run_candidates(
        generate=gen, score=score,
        policy=gate.GatePolicy(candidates=3, retries=1, stop_on_first_pass=False))
    assert seen == [0, 1, 2]
    assert attempts == 3
    assert nh is False


def test_retries_only_when_nothing_passed():
    calls = []

    def gen(i):
        calls.append(i)
        return _img(), {}

    def score_fail(img, info):
        return {"score": 0.0, "passed": False}

    gate.run_candidates(generate=gen, score=score_fail,
                        policy=gate.GatePolicy(candidates=1, retries=1))
    assert calls == [0, 1], "没过闸才补提"

    calls.clear()

    def score_pass(img, info):
        return {"score": 1.0, "passed": True}

    gate.run_candidates(generate=gen, score=score_pass,
                        policy=gate.GatePolicy(candidates=1, retries=1))
    assert calls == [0], "过了闸就不补提"


def test_stop_on_first_pass_for_spriteswap_semantics():
    """wss 的语义：过闸就停，不跑满候选。"""
    calls = []

    def gen(i):
        calls.append(i)
        return _img(), {}

    def score(img, info):
        return {"score": 1.0, "passed": True}

    gate.run_candidates(generate=gen, score=score,
                        policy=gate.GatePolicy(candidates=3, retries=0,
                                               stop_on_first_pass=True))
    assert calls == [0]


def test_pick_best_takes_first_on_tie():
    """同分取第一个：候选按 seed 递增生成，第一个就是「标准 seed」的结果。"""
    a, b = _img(), _img()
    img, ev, info, idx = gate.pick_best([
        (a, {"score": 5.0}, {"tag": "first"}),
        (b, {"score": 5.0}, {"tag": "second"}),
    ])
    assert idx == 0 and info["tag"] == "first"
    assert ev["score"] == 5.0


def test_non_humanoid_detection():
    """有前景但检不出关节 → 非人形；画面全空 → 不算（那是生成失败）。"""
    assert gate.is_non_humanoid({"points": 0, "coverage": 0.25}) is True
    assert gate.is_non_humanoid({"points": 0, "coverage": 0.0}) is False
    assert gate.is_non_humanoid({"points": 14, "coverage": 0.25}) is False


def test_non_humanoid_stops_retries():
    """非人形时补提被关掉：只跑第一轮的候选数。"""
    calls = []

    def gen(i):
        calls.append(i)
        return _img(), {}

    def score(img, info):
        return {"points": 0, "coverage": 0.3, "score": 1.0, "passed": False}

    _img_out, _ev, _info, attempts, nh = gate.run_candidates(
        generate=gen, score=score,
        policy=gate.GatePolicy(candidates=1, retries=1))
    assert nh is True
    assert attempts == 1, "非人形不该补提"
    assert calls == [0]
