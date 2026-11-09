# 项目长期约定（2d-game-meow）

## 总规矩
- 以 `D:\code\gen\others\ComfyUI_VNCCS-main` 为准，不猜。不做 Control Center、原生八方向帧、SeedVR2、Flux/SDXL/FaceDetailer/SAM。
- **只有 Qwen-Image-2.1**，`engine._check_model` 硬拒其它。底模是**全局唯一设置**，节点里没有下拉（`GenParams.model` 只是旧存档兼容，`App.migrate()` 必须清成空串，否则整单失败）。权重是 Qwen **RESEARCH LICENSE**（非商用）。
- 路径只在 `workflows/paths.py`（`MEOW_*_DIR` 可覆盖）：`assets/`=交付、`tests/_out/`=测试、`templates/*.mproj`、`library/`=元数据。图片/精灵帧/权重/节点定义一律不入库。
- 中文化全在 `web/src/labels.ts`（约 1000 条，查不到回落原值），`tests/test_labels.py` 核覆盖率；英文值原样传，只在渲染处显示中文，官方 JSON 一字不改。科普只引开源 upstream，不写商业站。
- 文档：`docs/角色中心-剩余工作与做法.md`、`docs/PLAN-QWEN-ONLY.md`、`docs/角色中心-VNCCS移植评估.md`、`docs/人偶动作导入-glTF-FBX-调研与实现.md`。

## 测试 / 构建（用户明确：别动不动跑全量）
- 全量约 260 例 / 150s，只在合并前跑。前端 `cd web && node smoke.mjs`（**别用 `npm run smoke` / `npm run build`**，自带 vite build 清 dist 触发批量删除守卫）。
- 改前端**必须构建**：`npx tsc -b && npx vite build --outDir dist_new` → `cp -r dist_new/. dist/` → 删 dist_new。截图 `CC_FAKE=1 UI_DIST=web/dist node tools/shot/capture.mjs <tag>` → `tests/_out/ui_review/`。**UI_DIST 相对仓库根**：写成 `dist` 等于目录不存在，所有请求反代到 :8000，后端没起就只有一页 `TypeError: fetch failed`。
- `boot()` 默认拼 `?editor=1`；库首页用例用 `stayInLibrary` + `setupFetch`。`/characters/{name}` 的宽泛 GET 分支会抢请求，专用 mock 必须放它前面。
- 动 `web/public/vnccs-pose/**` 用 `python tools/verify_pose_editor.py` / `tools/verify_anim_import.mjs` / `tools/verify_multi_yaw.mjs`；**别用 `--virtual-time-budget`+`--dump-dom`**。
- **假对象测试会骗人**：fake 必须模仿真实返回类型（`qwen_edit_image` 返 RGBA）；动过 alpha/通道/颜色空间必须真机跑一次。

## 生成 / 抠图 / 像素感
- 可交付 = 单主体 ∧ 无贴边 ∧ 前景 5%~60%。Qwen 768²/24 步：单帧 24~28s、峰值 11~15GB；<16 步糊块。白毛白甲别用白底抠图。
- **默认 `matte=False`，任何「偷偷抠」都是 bug**；查 `generate._postprocess` / `PIPELINE_DEFAULTS` / 前端硬编码 / 产物 meta。不抠必须走原生透明。透明底白边要 `_defringe_white_matte`（`alpha_reused` 时也要）。
- 「像素感」四来源：`SPRITE_SUFFIX` 契约词、alpha 硬切、despeckle、色数量化（**量化不许动 alpha**）。判据 `hard_edge = pixel > 1`；UI 上色数/像素块已删。
- 测速用 `tools/bench_qwen.py`（预热 + 中位数，看 `memory_reserved`）；测后处理用未像素化 `__raw`。批量成本 `t/(1-(1-p)^k)`。
- **角色立绘取景默认全身**：官方 `_character_framing_key` 没写 framing 时是 `cowboy_shot`（半身到大腿），我们改成 `full_body` 兜底（`workflows/characters.py` 默认值也写 `full_body`），进阶里给了「取景」下拉。半身图切精灵帧会缺脚。
- **提示词语言 `character_info.prompt_language`（默认 zh）**：Qwen3-VL 中文原生支持，默认中文、老角色没写也按中文。**结构词/取景/属性值/画质词中文化**（`workflows/textlabels.py` 查 `workflows/labels_zh.json`）。但**画风段和官方数据是例外**：`charactergen.style_block()` 把画风库自己的 Visual description/Reference **英文原文照搬**（中文名 + 英文名 + 原文描述），`race_features` 也照搬 —— 它们是数据不是结构词，硬译/删掉会丢视觉特征（曾导致 Q 版画风不像 Q 版）。
- **画风中立（2026-11-09，别改回去）**：公共提示词**一个风格字都不许写**（旧版「默认日式动画风格」兜底 + 「按这个画风的笔触…」已删），没选画风就完全不提画风；画风只由画风库那条定义决定。立绘出图 `app.character_generate_api` 在 `params.update` 之后**强制 `preset="none"`**（`txt2img` 默认是 `sprite`，负向词有 `photo, 3d render` 会压掉写实/3D 画风）。PE 请求逐行删 LoRA（**别用 rsplit**，会砍掉后面所有行）、成功后补回完整画风段、失败不重复追加 LoRA；`custom_style` 已接通并进 `QI2_LITERAL_FIELDS`。英文模板一字未改。
- **中文译表唯一真源是 `web/src/labels.ts`**，`python tools/build_labels_zh.py` 导出 `workflows/labels_zh.json`（TAG/STYLE/BG）。`tests/test_labels.py::test_backend_labels_json_in_sync` 会拦没重导的情况；同一测试现在还校验 `character_tags.json`（发色/发型/种族/细节那 74 条），以前漏掉导致选择器整排英文。
- **Prompt Enhancement 只在用户点「✨ 润色提示词」按钮时跑**（`/prompt?enhance=true`），打开页面拼 prompt 一律 `enhance=false`；润色结果直接写进界面那段 prompt，出图时 `enhance:false` 交给后端，**别重复跑**。中文模式也照跑 PE：`qwen_text.system_prompt_of(kind,'zh')` 把官方「一律英文」那段换成「一律中文」，其余（八步推理 + JSON 输出格式）一字不改，实测输出整段中文描述。
- **Prompt Enhancement（`workflows/qwen_text.py`）两个坑**：① 官方 t2i system prompt 要求回 `{"rewritten_prompt":...,"wh_ratio":...}`，必须像官方 `_qi2_json_result` 那样解析 JSON，见到 `{` 就判失败等于每次白跑；② `enable_model_cpu_offload()` 的钩子只搬模块不搬输入，PE 期间要**摘钩子**（`remove_hook_from_module`/`add_hook_to_module`）+ 自检每个张量同设备 + 输入按嵌入层设备放，否则报 `wrapper_CUDA__index_select`。实测 fp8 编码器 8.8GB，PE 一次 10~37s。
- **offload 链是易碎状态**：`_all_hooks` 空 / 组件缺 `_hf_hook` → `_execution_device` 退回 cpu，出图报 `Cannot generate a cpu tensor from a generator of type cuda`（**别在 PE 里调 `maybe_free_model_hooks()`**，它内部是 remove+re-add，失败就留个坏状态）。现已双保险：`generate._exec_device()` 让生成器跟随真实执行设备；`Engine._heal_offload()` 每次 `txt2img_pipe()` 检查并重建链。

## 画布 / UI 约定
- 节点「执行中」只看 `job.status` queued/running；**任何异常路径都要退出执行中**（提交 30s 超时、轮询连续 3 次失败、30 分钟看门狗、worker 兜底、404 文案）。
- 「角色图」统一走 `imagePathOf(node)`；`onConnect` 角色图白名单 `['poseclone','spriteswap','posebatch']`，新增节点必须加。
- 侧栏 3 tab（节点/画布/资料库）+ 4 个 `.rail-sec`，只用 `.is-hidden`+CSS 切，别条件卸载；`NODE_ENTRIES` 顺序别重排。
- **设置（底模/抠图/LoRA）只长在顶栏 ⚙ 弹窗**，不许回侧栏（用户提过两次）；侧栏只留 `.rail-settings` 快捷按钮。
- `NodeView='expanded'|'image'|'result'`：出图自动 result，参数区 CSS 隐藏但 DOM 保留。弹窗走 Portal 到 body；灯箱 wheel 要 `{passive:false}`；别用 `opacity` 做低调按钮。
- `--rail-w:252px` 与 `App.tsx`、`store/projects.ts` 的 `x:300,y:96` 同步。`SpritePreview.onMeta` 必须 `useCallback`。
- 外观只改 `styles.css` 末尾「全站样式统一层」，不要再叠层；`:root` 只在顶部一次。像素风第一版被否（贴皮装饰），**别加回来**；画布网格 `GRID_BASE=32` 两边必须同步。
- **column 弹性容器里别用 `flex: 0 0 Npx` 定宽**：主轴是纵向，basis 量的是高度（年龄输入框曾被撑成 76px 高的方块）。宽度只能写 `width` + `flex: none`。
- 角色中心右栏（2026-11-09）：立绘一直累积，`PortraitPane` 是**固定分页**（每页 8 张 + 页码 / 上一下页，数量变了回第 1 页），**不许留「显示全部」入口**（用户明确否掉过一次）；`TaskBar` 与完成提示**排在画廊之前**且 `position: sticky; top: 0`。外观只改 `styles.css` 末尾统一层。前端改完用 `npx vite build --outDir dist` **直接重建 dist**（cp -r 只覆盖不删会堆旧 bundle，浏览器缓存旧 index.html 会加载旧逻辑）；别用 `rm -rf dist_next`（触发批量删除守卫），临时目录用 `mv` 移到 `tests/_out/`。
- 角色中心（2026-11-08）：**画风库是弹窗**（`.cc-style-mask/dialog`，卡片 196px 起，点卡片即选中并关窗，Esc 可关），别再摊回卡片内联。右栏成果区**可拖宽**：`.cc-split` + `--cc-side-w`（默认 420，`localStorage meow.cc.sideW`，双击复位，≤1020px 隐藏）。预览图必须 **flex 居中**：grid 行高 auto 时 `max-height:100%` 解析失败，图会按宽度铺满被裁掉底部。反推属性支持**自选本地图**：`uploadImage()` → `/upload` 的 `path` → `/characters/{n}/analyze` 的 `image`。

## 全局设置 / LoRA（SQLite 权威）
- `metadb.settings` 表 + `GET/PUT /settings`，白名单 `base_model/matte_backend/loras/fast_mode`；启动 `_restore_global_settings()`。`submitMatte` 必须带 `matte_backend`，后端按工作流白名单兜底。
- **勾选 LoRA 只是改设置，绝不加载模型**：`POST /loras/apply` 走 `Engine.set_user_loras()`（只记 `_user_specs` + 失效签名），真正 `load_lora_weights` 推迟到下一次生成的 `txt2img_pipe()`。前端先乐观置 `activeLoras` 再落 `PUT /settings`，失败才 `reloadLoras` 回滚。以前 apply 先 `txt2img_pipe()` → 等 15GB / 权重缺失 400 → 界面「勾了弹回去」。
- `active_loras()`：`_loaded_sig is None` 时**只按规格报**，否则刚取消勾选的那条会从 `_active_loras` 冒出来。
- LoRA 导入两步：`/loras/inspect` 只暂存只读检测，**绝不写 `models/lora/`**；`/loras/commit` 仅 `convert=true` 才调 `tools/convert_lora.py`。ComfyUI 格式直接挂会**静默不生效**。
- 强度滑块每行常显（未挂时 disabled + `.is-off`）。角色中心抠图跟全局方式走（`matte_screen.global_backend()`，2s 缓存）。

## 姿势 / 人偶 / 动作导入
- 「人偶姿势」是官方 PoseStudio **内嵌 iframe**（`studio.html?embed=1`），自写伪 3D 已删。「放大」只能改容器 CSS 为 `fixed`（移动 iframe DOM 会重载 37MB 模型）；`capture()` 前关 skydome；embed class 在 `<head>` 内联加。
- **内嵌是 Utils 0.7.1/0.7.2 基线，别整包升 0.7.3+**（新版 `core.js` 会 import 我们没有的 `normalizeCaptureLens`，iframe 直接崩；比版本用 `git hash-object`）。官方库 191 个 JSON 的 prompt 全是空串。
- 多角色同框：`vnccs-pose/scene.js` 是唯一状态机（`{type:'scene',action}` + `sceneState` 回执），`PoseScenePanel.tsx` 是宿主 UI，状态只在 iframe 一份，`commit=false` 不写 project.json。`web/public` 下 ESM 一律 `.js`（`.mjs` 会被当 text/plain 拒载）。`app.py::PoseSetFrameReq` 必须显式声明字段（Pydantic 吞多余字段）。
- **别用图像识别数主体**（三人图连通域合并成 1）：导出走 `/upload?subjects=N` 写 `.png.json` sidecar，`subjects_for()` 读，老图按 1。
- 动作导入 glb/fbx：本地 three r160 + importmap，FBX 4 个依赖缺一会悄悄回落 CDN。脚底对齐两层（世界空间按 resetPose + `alignFrames()`）。抽帧 `assets/projects/<项目>/poses/`，同名 409。
- 「从图片提姿势」用 MediaPipe `pose_world_landmarks` 拿 3D；`COCO_TO_OP 13→L_ANK` 必须双射。

## 方向帧 / 动作模板 / 试玩台
- 视角模式是一等概念：`topdown-4/topdown-8/side-2`，表在 `store/playtest.ts` 与 `workflows/directions.py`（`playtest.py` re-export），**别写第三份**。官方 VNCCS 没有原生八方向，`YAWS` 是我们转相机。
- 链路：`set.json.direction` → `posegen` 按方向编号 → `spriteatlas` 产 `<组>@<方向>` → `godotexport` ≥2 方向才多动画 `.tres`+`animations.json`（单动画文本有逐字断言）→ 试玩台按朝向选帧。老数据每环退回原行为。
- `entry.js` 的 `animSample` 可选 `directions`：方向 × 帧双层循环，每方向单独 `alignFrames()`；不传时老路径一字不变。`PoseSetFrameReq.direction` 不声明就被吞。
- 动作模板 `workflows/motiontemplates.py` 唯一定义；内置三套是**代码常量不落盘**，用户模板落 `library/motion_templates.json`；`normalize()` 必须幂等。`components/MotionTemplatePanel.tsx` 挂资料库抽屉第三个 tab，用独立 `.mt-*` 类名（别用 `.lib-card`，smoke 按它计数）。
- 试玩台播放与位移**共用一个 rAF**（双时钟会滑步）；`GET /characters/{name}/playtest` 只读扫盘，不新增落盘。

## 资料库 / SQLite / 工作台
- 角色中心（2026-11-09 重构 + 二次修正，**别写回四步工作台**）：四层 `library / stages / stage / pending`。默认全屏库 → 点卡片进**生命阶段列表** → 点阶段进**角色设定**（原定形象内容）→ 另有「待处理」页放原出素材/导出内容（用户说后面再参考，内容一字不动）。UI 上「项目」叫**画布**（内部 `/projects` 不变）。
- **像素化输出尺寸（2026-11-09 修正，别退回去）**：`pixelfy` 输出的是**格子小图**（768×1024 自动只出 48×64），对精灵帧正确，但直接存成立绘会小得没法看。`process_portrait_api` 的 pixel 分支必须①按**长边**解释目标尺寸（短边按原图比例推导，别写死 `[N,N]` 把 3:4 压成 1:1）、②默认 `output=restore` 用 `resize(src.size, NEAREST)` 放大回原尺寸（硬边、不造新颜色、alpha 不半透明），`output=native` 才保留小图。meta 记 `native_size/output_size`。
- 立绘悬浮工具箱（2026-11-09）：hover 出「扣背景 / 完美像素化 / 删除」。两个处理走 `POST /characters/{n}/portraits/process`（`matte.matte_auto` / `pixelfy.refine_batch`，纯确定性不跑模型），弹窗给参数 + 拖动分割线滑动比对，**永远生成新的一份**、不改原图；`portrait_variant_path()` 同目录不覆盖、后缀 `_bg`/`_px`（别用 `_matte.png`，会被 list_portraits 当副本跳过）。删除走 `POST .../portraits/delete` 且必须 `confirm=true`，前端还有确认弹窗。卡片外层是 div（按钮不能嵌套），`.cc-portrait-view` 负责看大图。组件在 `characters/PortraitTools.tsx`。
- 角色中心左边距（2026-11-09）：`cc-main` 是 `padding:0`，顶部栏/面包屑/导航作为它的直接子元素会贴边；统一层末尾把 `.cc-topbar/.cc-crumb/.cc-viewnav/.cc-panel/.cc-next/.chars-err` 左右边距统一到 **24px**（只改统一层），否则左边界参差。
- 角色中心阶段编辑页导航（2026-11-09 再修正）：进入阶段后 `cc-viewnav` 第一项文字必须是**「角色设定」**（高亮、当前位置），列表页才是「生命阶段」；正文不许再有大标题「角色设定」（名字只出现一次）。stage 页该项点击不跳转，返回靠面包屑角色名 / 底部「← 返回生命阶段」。
- 角色中心层级（2026-11-09，用户按截图逐条钉过）：顶部只有**角色名 + 更小描述 + 修改 + 编辑时间**，不许出现「当前服装/画风」行；导航**只有「生命阶段」与「待处理」两项**，「角色设定」属于某个生命阶段、不是同级入口；进入阶段后才出现面包屑 `角色名 / 阶段名`。**列表页没有右栏与拖动条**（`.is-stages` 单列），也不塞待处理卡片/角色档案（前者走导航、后者走「修改」弹窗）；每行以立绘为主（58×82 缩略图最多 5 张），文字用紧凑单行输入。右栏立绘只在阶段编辑/待处理页出现，因为它是角色设定的一部分。
- `workflows/metadb.py`（标准库 sqlite3 / WAL / `library/meta.sqlite3`）是角色与画布元数据权威：tags、创建编辑时间、自定义标签、备注、任务历史（`GET /character-tasks` 已有，**展示 UI 还没做**）、封面。用户明确不必再顾官方目录互通。
- **生命阶段数据**：`workflows/stages.py`；阶段数组/排序/描述在角色 `config.json` 里（数组顺序即展示顺序），`active_stage` 的 info 始终是顶层 `character_info` 的**镜像**——所有出图 API 不用带 stage。`GET/PUT /characters/{n}/stages` 全量提交（新建/改名/排序/删除/激活）；立绘按 `Portraits/s_<id>/` 分目录、默认阶段用老位置；删阶段**不删磁盘文件**；老角色自动补「初始设定」阶段。阶段编辑页的「导入设定」**只复制 info，绝不导入立绘**。所有任务仍走 `runTask` 统一轮询，右栏在 stage/pending 都常驻。

## 收尾状态
- 7 轮（侧栏 / 节点结果视图 / 试玩台 / 多机位抽帧 / 方向闭环 / 动作模板 / 模板编辑 UI）已合并为 `77ee935` 推送 `origin/main`。
- 仍未做：碰撞盒、角色中心「一键按模板补方向帧」、抽帧面板下拉真机验证、双人参考图 GPU 一致性验证（文本契约已接线，单测通过）。
