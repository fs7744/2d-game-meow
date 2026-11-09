"""本地 Whisper 转写（骨骼动画工作台的视位口型同步用）。

单独进程跑，不占主服务进程、也不抢生图显存（固定 CPU）。输出一行 JSON：

    {"ok": true, "text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.4}]}

模型第一次用会从 Hugging Face 下载并缓存到本地；之后离线可用。
可用 MEOW_WHISPER_MODEL 覆盖模型（默认 openai/whisper-base）。
"""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "用法：rig_whisper.py <音频文件>"}))
        return
    path = sys.argv[1]
    model = os.environ.get("MEOW_WHISPER_MODEL", "openai/whisper-base")
    try:
        # 强制 CPU：工作台在生图同一台机器上，转写不该抢显存。
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from transformers import pipeline

        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            chunk_length_s=30,
            return_timestamps="word",
            device=-1,
        )
        out = asr(path)
        words = []
        for c in out.get("chunks") or []:
            ts = c.get("timestamp") or (None, None)
            words.append({
                "word": str(c.get("text") or "").strip(),
                "start": float(ts[0] or 0),
                "end": float(ts[1] or ts[0] or 0),
            })
        print(json.dumps({"ok": True, "text": str(out.get("text") or ""),
                          "words": words}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - 要把真实原因回给前端
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
