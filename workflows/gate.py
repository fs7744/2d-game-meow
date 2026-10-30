"""逐帧生成的公共「候选 + 闸门」组件。

w4（姿态序列）与 wss（序列帧替换）做的是同一件事：

    同一帧生成 N 个候选 → 用判据打分 → 挑最好的 → 没过就补提

这段逻辑以前各写一遍，后果是：

  · 补提策略（补几次、seed 怎么错开、候选同分取哪个）有两份实现，
    改一处忘一处；
  · w4 里那个「主体不是人形就别补提」的短路，wss 想用就得复制一遍；
  · 「候选按 seed 递增生成、同分取第一个（保证可复现）」这条约定散落在两处。

抽成组件后，**判据仍然是工作流特有的**（w4 用 DWPose 关节数，
wss 用剪影 IoU + 配色一致度），公共的只有循环、策略与记账。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image


@dataclass(frozen=True)
class GatePolicy:
    """闸门策略：跑几个候选、不过时补提几次、seed 怎么错开。

    `stop_on_first_pass`：第一个候选就过闸时要不要停。
    w4 是 False（跑满候选再挑分数最高的），wss 是 True（过了就停）。
    """
    candidates: int = 1
    retries: int = 1
    seed_step: int = 1000
    stop_on_first_pass: bool = False

    @property
    def attempts_max(self) -> int:
        return max(1, self.candidates) + max(0, self.retries)


def is_non_humanoid(ev: Dict[str, Any]) -> bool:
    """画面里有前景、却一个关节都检不出 → 主体不是人形。

    史莱姆、机甲、道具都是这样：DWPose 检不出关节，闸门永远过不了，
    补提就是纯烧钱（每帧 25~30s 白花）。

    判据是「有前景（coverage > 1%）」+「points = 0」——
    画面全空时**不算**非人形：那是真的生成失败了，该补提。
    """
    return ev.get("points", 0) == 0 and ev.get("coverage", 0.0) > 0.01


def pick_best(cands: List[Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]]
              ) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any], int]:
    """挑分数最高的候选。

    同分取第一个：候选按 seed 递增生成，第一个就是「标准 seed」的结果，
    保证可复现。
    """
    if not cands:
        raise ValueError("没有候选")
    best_i = 0
    best_s = cands[0][1].get("score", 0.0)
    for i, (_img, ev, _info) in enumerate(cands):
        if ev.get("score", 0.0) > best_s + 1e-6:
            best_s = ev["score"]
            best_i = i
    img, ev, info = cands[best_i]
    return img, ev, info, best_i


def run_candidates(
    *,
    generate: Callable[[int], Tuple[Image.Image, Dict[str, Any]]],
    score: Callable[[Image.Image, Dict[str, Any]], Dict[str, Any]],
    policy: GatePolicy,
    on_attempt: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Optional[Image.Image], Dict[str, Any], Dict[str, Any], int, bool]:
    """跑候选、打分、挑最好的。

    `generate(i)`  第 i 个候选 → (图, 附加信息)
    `score(图, 附加信息)` → 带 passed / score 的评价字典
    `on_attempt(评价)` 每次尝试后的回调（收集耗时、抠图后端等）

    返回 (最好的图, 评价, 该候选的附加信息, 尝试次数, 是否判定为非人形)。

    流程与原来一致：
      1. 先跑满 `candidates` 个候选（`stop_on_first_pass=False` 时不提前停）；
      2. 全都没过闸才补提，最多补到 `attempts_max` 次；
      3. **一旦某帧判定为非人形，补提立刻关闭**（没有判据可依据）。
    """
    cands: List[Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]] = []
    attempts = 0
    non_humanoid = False
    passed_any = False

    for i in range(policy.attempts_max):
        if attempts >= policy.candidates:
            # 第一轮跑满还没过闸 → 才有补提的资格
            if passed_any or non_humanoid:
                break
        img, info = generate(i)
        ev = score(img, info)
        attempts += 1
        if on_attempt is not None:
            on_attempt(ev)
        if ev.get("passed"):
            passed_any = True
        if is_non_humanoid(ev):
            non_humanoid = True
        cands.append((img, ev, info))
        if passed_any and policy.stop_on_first_pass:
            break

    if not cands:
        return None, {}, {}, 0, non_humanoid
    img, ev, info, _i = pick_best(cands)
    return img, ev, info, attempts, non_humanoid
