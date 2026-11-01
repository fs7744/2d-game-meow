"""统一路径解析：全项目只有这一处定义目录。

**为什么集中**：资产根目录以前在 `app.py` 和 `workflows/generate.py` 里各写了一份
（`ROOT/"assets"`）。于是「把测试产物写到别处」这件事必须改两个地方，漏一个
就继续往真目录里写——pytest 跑一轮，`assets/` 里就多出一堆 `api*` / `w1_*` 的
测试图，把真正要交付的产物淹没。现在只有一个入口，靠环境变量覆盖：

    MEOW_ASSETS_DIR   运行时资产根（产物 / 上传 / 索引）
    MEOW_TEMPLATES_DIR 模板根（跨项目共享的用户资源）

**注意**：`ASSETS` 是**导入时**求值的模块级常量。想在运行时改目录，必须在
import `workflows.paths` 之前设好环境变量——pytest 的 `conftest.py` 正是这么做的
（它在 collection 阶段就设，测试模块的 import 都在其后）。
"""
from __future__ import annotations

import os
from pathlib import Path

#: 仓库根（workflows/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent


def _env_dir(name: str, default: Path) -> Path:
    """环境变量覆盖目录；没设就用默认值。"""
    v = os.environ.get(name)
    return Path(v).resolve() if v else default


#: 运行时资产根：生成产物 / 上传 / index.jsonl / index.db
ASSETS = _env_dir("MEOW_ASSETS_DIR", ROOT / "assets")

#: 各项目的产物目录（项目 id 作为子目录名）
PROJECTS_DIR = ASSETS / "projects"

#: 角色中心（VNCCS 角色库）：`assets/characters/<角色名>/`
#
# **必须在 assets 下**：前端要显示角色封面与 sprite，图片只能走 `/files/`，
# 而那个端点按约定只服务 assets 内的路径（`app._readable_asset_path`）。
# 放别处图就全裂了。
#
# 目录结构照搬 VNCCS `utils.py:405-487`：
#   <角色>/{角色}_config.json + cache/ + Sprites/ Faces/ Sheets/
#   Sprites/<服装>/<表情>/sprite_<表情>_0001.png
CHARACTERS_DIR = ASSETS / "characters"

#: 随项目发行的 **VNCCS 官方素材**（风格库 / 标签库 / 提示词模板），
#: 落在 `web/public/vnccs/character_template/`——前端要直接 fetch 同一份，
#: 后端拼 prompt 也从这里读，**不复制第二份**（复制就会有天对不上）。
VNCCS_TEMPLATE_DIR = ROOT / "web" / "public" / "vnccs" / "character_template"

#: 没有项目归属的上传（旧路径，保留兼容）
UPLOAD_DIR = ASSETS / "uploads"

#: 模型权重根。以前散在各处硬编码 `ROOT / "models"`，路径只有一处定义才改得动。
MODELS_DIR = _env_dir("MEOW_MODELS_DIR", ROOT / "models")
#: Qwen LoRA 目录（**只放 diffusers 原生格式**，见 engine.lora_dir 的注释）
LORA_DIR = MODELS_DIR / "lora"
#: 本地 LLM（gguf）目录。官方 VNCCS 的搜索路径是 `models/llm` 与
#: `models/llm/Qwen3.5-4B` 两层（`nodes/qwen_vl.py:154`），我们固定放在后者，
#: 一层就够 —— 但目录名沿用官方的 `llm`，免得以后照官方代码找不着。
LLM_DIR = MODELS_DIR / "llm" / "Qwen3.5-4B"
#: 人脸检测权重目录。官方 VNCCS 里是 `bbox/face_yolov8m.pt`
#: （`nodes/character_generator.py:845`），ComfyUI 生态的惯例路径。
BBOX_DIR = MODELS_DIR / "bbox"

#: 项目模板根。模板是**跨项目共享**的用户资源（同一套「像素精灵产线」要开好几个
# 项目时复用），不属于任何一个项目的产物，所以不放 assets 下。
TEMPLATES_DIR = _env_dir("MEOW_TEMPLATES_DIR", ROOT / "templates")

#: 用户资料库：**提示词收藏 / 姿势库 / 界面偏好** 在这里落盘。
#
# **为什么不留在 localStorage**：浏览器存储会在「换浏览器 / 清缓存 / 无痕窗口」
# 时凭空消失，而这些都是用户攒了很久的资产；落盘后能备份、能换机器拷走、
# 也不会因为清站点数据全丢。templates 已经是「跨项目共享的用户资源」的路子，
# 库跟它放在一起。
#
# 测试用 MEOW_LIBRARY_DIR 覆盖（conftest 里设），避免跑测试时写真目录。
LIBRARY_DIR = _env_dir("MEOW_LIBRARY_DIR", ROOT / "library")

#: 随项目发行的 **VNCCS 官方姿势库**（191 个预设 json + webp 预览）。
#: 放在静态目录下由浏览器直接取，不用拷进 runtime assets；清单扫描见
#: workflows/poselib.py。
POSE_LIBRARY_DIR = Path(os.environ.get("MEOW_POSE_LIBRARY_DIR")
                        or (ROOT / "web" / "public" / "vnccs-pose" / "PoseLibrary"))

#: 测试产物根：pytest 与 tools/ 下的验证脚本的输出都往这里放，
#: 不再往 `assets/` 里丢（那里只留项目必须的东西）。
TEST_OUT = ROOT / "tests" / "_out"


def ensure_assets() -> Path:
    """确保资产根存在，返回它。"""
    ASSETS.mkdir(parents=True, exist_ok=True)
    return ASSETS


def tool_out(*parts: str) -> Path:
    """验证/实验脚本的输出目录（`tests/_out/` 下）。

    **为什么不写进 `assets/`**：那里的东西是**项目资产**——用户要拿走的图。
    验证脚本产出的对比图、调试图是**开发过程**，混进去以后：
      - 资产清单被几百张 `_xxx_compare.png` 淹没，真正交付的图翻不到；
      - 删项目目录时连带删掉；
      - 新人打开 assets/ 不知道哪些是能用的。

    所以测试与验证产物一律进 `tests/_out/`，`assets/` 只留项目必须的东西。
    """
    d = TEST_OUT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
