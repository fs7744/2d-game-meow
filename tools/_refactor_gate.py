"""把 w4 与 wss 的「候选 + 闸门」循环换成公共组件 workflows/gate.py。

行为保持不变（逐条对照过原实现）：
  · 候选按 seed 递增生成，第一轮跑满 candidates 个；
  · 全都没过闸才补提，最多补到 candidates+retries 次；
  · 同分取第一个（可复现）；
  · w4 不在第一个候选过闸时提前停（要跑满再挑分数最高的），
    wss 过闸就停；
  · 主体不是人形时不再补提（这条本来就是 w4 独有的，现在成了公共规则）。

改动只是把循环搬进 gate.run_candidates，判据仍由各工作流自己给。
"""
from pathlib import Path

p = Path("workflows/generate.py")
s = p.read_text("utf-8")

# ---------------- 导入 ----------------
old = "from . import framescore"
new = "from . import framescore\nfrom . import gate as gate_mod"
assert old in s
s = s.replace(old, new, 1)

# ---------------- w4：逐帧循环 ----------------
old = '''        best: Optional[tuple[Dict[str, Any], Image.Image, Image.Image]] = None
        spent = 0.0
        used = 0
        for round_seeds in (list(range(candidates)),
                            list(range(candidates, attempts_max))):
            for i in round_seeds:
                # 上限动态：判定为非人形后立刻不再补提
                if used >= (candidates if non_humanoid else attempts_max):
                    break
                dt = _timer()
                sd = seed + i * gate_seed_step
                if use_two_stage:
                    # 阶段 1：只有骨架 + 文字（没人跟它抢姿态）
                    draft = _qwen_call(
                        eng, prompt=solo_instr, negative=neg, ref_image=cond,
                        width=width, height=height, steps=steps, cfg=cfg,
                        seed=sd, model=model, label="w4-draft")
                    # 阶段 2：角色图在前、草图在后 → 换回身份，姿态保住
                    raw = _qwen_call(
                        eng, prompt=reskin_instr, negative=neg,
                        ref_image=[char, draft],
                        width=width, height=height, steps=steps, cfg=cfg,
                        seed=sd, model=model, label="w4-reskin")
                else:
                    raw = _qwen_call(
                        eng, prompt=instr, negative=neg, ref_image=refs,
                        width=width, height=height, steps=steps, cfg=cfg,
                        seed=sd, model=model, label="w4")
                info: Dict[str, Any] = {}
                img, _edge, _subj = _postprocess(
                    raw, eng, matte=matte, matte_backend=matte_backend,
                    denoise=denoise, pixel=pixel, palette=palette,
                    clean_iso=clean_iso, info=info)
                for k in ("backend", "fallback"):
                    if k in info:
                        matte_info[k] = info[k]
                gate_img = info.pop("pre_pixel", None) or img
                ev = quality.evaluate(gate_img.convert("RGBA"), extractor,
                                      prev=prev_gate,
                                      min_points=gate_min_points)
                # **单主体判据**：Qwen 偶尔会把头画丢、把帽子画成浮空的一块。
                # 这种帧 DWPose 照样能检出 16 个关节（躯干还在），闸门放它过去；
                # 但前景连通域数会变成 2+，这才是「缺件」的直接证据。
                if _subj is not None:
                    ev["blobs"] = _subj
                    if _subj > 1 and ev["passed"]:
                        ev = {**ev, "passed": False,
                              "score": ev["score"] - 20.0,
                              "reject": f"前景有 {_subj} 块（头/手与身体分离）"}
                spent += dt()
                used += 1
                # 画面里有东西却检不出关节 → 不是人形，别再为它重提
                if not non_humanoid and ev.get("points", 0) == 0 \\
                        and ev.get("coverage", 0.0) > 0.01:
                    non_humanoid = True
                    print("[warn] 主体不是人形（DWPose 检不出关节）："
                          "关闭补提，后续帧只出 1 次")
                if best is None or ev["score"] > best[0]["score"]:
                    best = (ev, img.convert("RGBA"), gate_img.convert("RGBA"))
                if extractor is not None and ev["passed"] and used >= candidates:
                    break
            if used >= candidates and (extractor is None or best[0]["passed"]):
                break
        ev, frame, prev_gate = best'''

new = '''        spent = 0.0

        def _gen(i: int) -> Tuple[Image.Image, Dict[str, Any]]:
            """跑一个候选：出图 → 后处理 → 交给打分函数。"""
            nonlocal spent
            dt = _timer()
            sd = seed + i * gate_seed_step
            if use_two_stage:
                # 阶段 1：只有骨架 + 文字（没人跟它抢姿态）
                draft = _qwen_call(
                    eng, prompt=solo_instr, negative=neg, ref_image=cond,
                    width=width, height=height, steps=steps, cfg=cfg,
                    seed=sd, model=model, label="w4-draft")
                # 阶段 2：角色图在前、草图在后 → 换回身份，姿态保住
                raw = _qwen_call(
                    eng, prompt=reskin_instr, negative=neg,
                    ref_image=[char, draft],
                    width=width, height=height, steps=steps, cfg=cfg,
                    seed=sd, model=model, label="w4-reskin")
            else:
                raw = _qwen_call(
                    eng, prompt=instr, negative=neg, ref_image=refs,
                    width=width, height=height, steps=steps, cfg=cfg,
                    seed=sd, model=model, label="w4")
            info: Dict[str, Any] = {}
            img, _edge, _subj = _postprocess(
                raw, eng, matte=matte, matte_backend=matte_backend,
                denoise=denoise, pixel=pixel, palette=palette,
                clean_iso=clean_iso, info=info)
            for k in ("backend", "fallback"):
                if k in info:
                    matte_info[k] = info[k]
            gate_img = info.pop("pre_pixel", None) or img
            spent += dt()
            return img.convert("RGBA"), {"gate": gate_img.convert("RGBA"),
                                         "subjects": _subj}

        def _score(img: Image.Image, info: Dict[str, Any]) -> Dict[str, Any]:
            ev = quality.evaluate(info["gate"].convert("RGBA"), extractor,
                                  prev=prev_gate,
                                  min_points=gate_min_points)
            # **单主体判据**：Qwen 偶尔会把头画丢、把帽子画成浮空的一块。
            # 这种帧 DWPose 照样能检出 16 个关节（躯干还在），闸门放它过去；
            # 但前景连通域数会变成 2+，这才是「缺件」的直接证据。
            _subj = info.get("subjects")
            if _subj is not None:
                ev["blobs"] = _subj
                if _subj > 1 and ev["passed"]:
                    ev = {**ev, "passed": False,
                          "score": ev["score"] - 20.0,
                          "reject": f"前景有 {_subj} 块（头/手与身体分离）"}
            return ev

        frame, ev, finfo, used, nh = gate_mod.run_candidates(
            generate=_gen, score=_score,
            policy=gate_mod.GatePolicy(
                candidates=candidates, retries=retries,
                seed_step=gate_seed_step, stop_on_first_pass=False))
        if nh and not non_humanoid:
            non_humanoid = True
            print("[warn] 主体不是人形（DWPose 检不出关节）："
                  "关闭补提，后续帧只出 1 次")
        if frame is None:
            raise RuntimeError("这一帧没有产出任何候选")
        prev_gate = finfo.get("gate")'''

assert old in s, "w4 循环没匹配上"
s = s.replace(old, new, 1)

# ---------------- wss：逐帧循环 ----------------
old = '''        dt = _timer()
        cands: List[Tuple[Image.Image, Dict[str, Any]]] = []
        for k in range(candidates):
            raw = _qwen_call(
                eng, prompt=instr, negative=neg, ref_image=[char, cond],
                width=width, height=height, steps=steps, cfg=cfg,
                seed=seed + k * 1000, model=model, label="wss")
            info: Dict[str, Any] = {}
            cut, _edge, _subj = _postprocess(
                raw, eng, matte=matte, matte_backend=matte_backend,
                denoise=denoise, pixel=0, palette=0, clean_iso=0, info=info)
            for kk in ("backend", "fallback"):
                if kk in info:
                    matte_info[kk] = info[kk]
            ev = framescore.evaluate(cut, frame, char_orig, prev=prev,
                                    area_ref=ref_area)
            cands.append((cut, ev))
            if ev["passed"]:
                break
        best_img, best_ev, best_i = framescore.pick_best(cands)
        frame_scores.append({**best_ev, "attempts": len(cands), "picked": best_i})
        attempts_used += len(cands)'''

new = '''        dt = _timer()

        def _gen(k: int) -> Tuple[Image.Image, Dict[str, Any]]:
            raw = _qwen_call(
                eng, prompt=instr, negative=neg, ref_image=[char, cond],
                width=width, height=height, steps=steps, cfg=cfg,
                seed=seed + k * 1000, model=model, label="wss")
            info: Dict[str, Any] = {}
            cut, _edge, _subj = _postprocess(
                raw, eng, matte=matte, matte_backend=matte_backend,
                denoise=denoise, pixel=0, palette=0, clean_iso=0, info=info)
            for kk in ("backend", "fallback"):
                if kk in info:
                    matte_info[kk] = info[kk]
            return cut, {}

        def _score(cut: Image.Image, _info: Dict[str, Any]) -> Dict[str, Any]:
            return framescore.evaluate(cut, frame, char_orig, prev=prev,
                                       area_ref=ref_area)

        best_img, best_ev, _finfo, used, _nh = gate_mod.run_candidates(
            generate=_gen, score=_score,
            policy=gate_mod.GatePolicy(
                candidates=candidates, retries=0, seed_step=1000,
                stop_on_first_pass=True))
        frame_scores.append({**best_ev, "attempts": used, "picked": 0})
        attempts_used += used'''

assert old in s, "wss 循环没匹配上"
s = s.replace(old, new, 1)

p.write_text(s, "utf-8")
print("ok")
