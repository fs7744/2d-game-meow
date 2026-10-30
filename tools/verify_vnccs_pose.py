"""按 VNCCS PoseStudio 官方工作流跑姿态迁移（不用 ControlNet）。

官方工作流（`MIUProject/VNCCS_PoseStudio_QI2.1` 里的 `VNCCS_PoseStudio_QI2.1.json`）
的节点构成里**没有任何 ControlNet**：

```
LoadImage(角色) ─┐
VNCCS_PoseStudio(骨架) ─┤
                        ├→ TextEncodeQwenImage21 → KSampler(25步, cfg1, euler/simple)
LoraLoaderModelOnly(VNCCS_QI2_PoseStudioV1, 1.0) ─┘
```

固定的指令模板（逐字）：

```
replace the pose of <image 2> with the pose of <image 1>. keep the character of <image 2>
```

- `image 1` = 骨架（VNCCS_PoseStudio 渲染出来的）
- `image 2` = 角色参考图
- 输出尺寸取自**角色图**尺寸（`GetImageSize → EmptyLatentImage`）
- negative 为空字符串，cfg=1

也就是说：**一致性靠专用 LoRA + 双图指令，不靠硬约束**。这条路比接
ControlNet Union 简单得多，而且是作者专门为"保角色"训的。

用法:
    python tools/verify_vnccs_pose.py --ref assets/_cn_none.png \
        --pose assets/_pose_walk_07.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image

from workflows.engine import Engine

#: 官方模板（逐字）
TEMPLATE = ("replace the pose of <image 2> with the pose of <image 1>. "
            "keep the character of <image 2>")

#: 身份/场景补充。官方文档建议"明确提及服装元素"来加强一致性
IDENTITY = (
    "The character is a female warrior with long black hair tied in a low ponytail, "
    "fair skin, wearing tight dark blue leather armor with visible stitching lines and "
    "light blue highlight patches on the shoulders and elbows, a dark cloth belt with "
    "hanging skirt panels, matching blue pants with knee pads, and dark grey tall boots "
    "with silver metal buckles. 2D game character sprite, flat cel shading, "
    "clean black outlines, plain white background, full body, detailed."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="assets/_cn_none.png")
    ap.add_argument("--pose", default="assets/_pose_walk_07.png")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lora", type=float, default=1.0)
    ap.add_argument("--lora-name", default="vnccs_pose",
                    help="models/lora/ 下的文件名（不带 .safetensors）")
    ap.add_argument("--extra", default=IDENTITY, help="附加的身份/场景描述")
    ap.add_argument("--out", default="assets/_vnccs_pose.png")
    ap.add_argument("--canvas", default="pose",
                    choices=["pose", "ref"],
                    help="画布尺寸取骨架（默认，让骨架填满不被缩小）还是角色图（官方做法）")
    args = ap.parse_args()

    eng = Engine("models")
    pipe = eng.txt2img_pipe("qwen-image-2.1")

    print(f"挂 {args.lora_name} (scale={args.lora})", flush=True)
    eng.apply_loras(pipe, [{"name": args.lora_name, "scale": args.lora}])
    print("  生效:", eng.active_loras(), flush=True)

    ref = Image.open(args.ref).convert("RGB")
    pose = Image.open(args.pose).convert("RGB")
    src = pose if args.canvas == "pose" else ref
    # 骨架是 2:3 竖版，取它的比例当画布能让骨架 1:1 填满；取角色图（官方做法）
    # 时骨架只能缩在中间，模型会把它理解成"小幅度动作"。
    scale = 1.5 if args.canvas == "pose" else 1.0
    w = max(256, int(src.size[0] * scale) // 16 * 16)
    h = max(256, int(src.size[1] * scale) // 16 * 16)

    def pad_to(im: Image.Image, W: int, H: int) -> Image.Image:
        """等比缩到框内并居中贴到白底（**不拉伸**）。

        骨架是 512×768（2:3），硬拉到 768×768 会把腿压短，姿态就"跟一半"——
        实测双腿交叉的骨架只被理解成"双腿收窄"。
        """
        r = min(W / im.width, H / im.height)
        nw, nh = max(1, round(im.width * r)), max(1, round(im.height * r))
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(im.resize((nw, nh), Image.LANCZOS),
                     ((W - nw) // 2, (H - nh) // 2))
        return canvas

    print(f"角色图 {args.ref} {ref.size} → 画布 {w}x{h}（按官方：用角色图尺寸）", flush=True)
    print(f"骨架图 {args.pose} {pose.size}（等比贴入，不拉伸）", flush=True)

    prompt = f"{TEMPLATE}\n{args.extra}".strip()
    t = time.time()
    out = pipe(prompt=prompt,
               # **顺序**：image 1 = 骨架，image 2 = 角色（官方就是这样接的）
               image=[pad_to(pose, w, h), pad_to(ref, w, h)],
               width=w, height=h,
               num_inference_steps=args.steps,
               true_cfg_scale=1.0,
               negative_prompt="",
               generator=torch.Generator("cuda").manual_seed(43),
               output_resolution=max(w, h)).images[0]
    dt = time.time() - t
    out.convert("RGB").save(args.out)
    print(f"  生成 {dt:.1f}s → {args.out}", flush=True)

    # 三方对比：角色图 / 骨架 / 结果
    side = Image.new("RGB", (w * 3, h), "white")
    side.paste(pad_to(ref, w, h), (0, 0))
    side.paste(pad_to(pose, w, h), (w, 0))
    side.paste(out.convert("RGB"), (w * 2, 0))
    side.save("assets/_vnccs_compare.png")
    print("  → assets/_vnccs_compare.png（左=角色 中=骨架 右=结果）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
