# 项目长期约定（2d-game-meow）

## 总规矩 / 文档 / 底模
- 一切以 `D:\code\gen\others\ComfyUI_VNCCS-main` 为准，不自己猜。不做 Control Center、8 方向帧、SeedVR2、Flux/MiniMax/Anima/SDXL/FaceDetailer/SAM。
- 文档：`docs/角色中心-剩余工作与做法.md`（待办手册；T1~T5 完成，只剩 T7/T8 暂缓）、`docs/角色中心-VNCCS移植评估.md`、`docs/PLAN-QWEN-ONLY.md` §4/§6、`docs/人偶动作导入-glTF-FBX-调研与实现.md`。
- **只有 Qwen-Image-2.1**（SD 代码已删，权重在 `models/_archive/sd/`），`engine._check_model` 硬拒其它值。别再往 engine/generate 加 StableDiffusion。
  **底模是全局唯一设置，节点里不再有下拉**（`GenParams.model` 只是旧存档兼容字段，
  `App.migrate()` 必须把 sd15/sdxl 清成空串，否则会被提交、整单失败）。
  ⚠️ `models/checkpoints/qwen-image-2.1/LICENSE` 是 Qwen **RESEARCH LICENSE**（非商用），商用前须授权或换底模。
- 路径只有一处定义 `workflows/paths.py`（`MEOW_ASSETS_DIR` / `MEOW_TEMPLATES_DIR` / `MEOW_LIBRARY_DIR` 可覆盖）。
  `assets/`=交付资产，`tests/_out/`=测试产物，模板 `templates/*.mproj`，`library/`=用户攒的元数据。
  **文件 / 图片 / 精灵帧 / 模型权重 / 节点定义一律不入库**。
- 中文化：映射全在 `web/src/labels.ts`（约 1000 条，查不到回落原值），`tests/test_labels.py` 核覆盖率；
  英文标识（`Naked`/`angry 😠`/`Shonen Anime`/`Ballet`、目录名、接口键、prompt、错误码）**值原样传，只在渲染处显示中文**，官方 JSON 一字不改。改完重扫渲染处（`CharactersPage`/`TagPicker`/`LibraryDrawer`/`SpritePreview`/`App`）。
  科普只引开源 upstream（AHEKOT/ComfyUI_VNCCS、AHEKOT/ComfyUI_VNCCS_Utils、MIUProject/VNCCS_PoseStudio_QI2.1），**不写商业托管站**。

## 测试策略（用户明确：不要动不动跑全量）
- 全量约 260 用例 / 150s，只在合并前跑；子集见 README「测试」表。SD 时代测试已删。
- 前端 `cd web && node smoke.mjs`（**别用 `npm run smoke`**，自带 vite build 清 dist 触发批量删除守卫）。
  改完前端**必须先构建**：`npx tsc -b && npx vite build --outDir <新目录>` 再 `cp -r` 覆盖 `dist`；也别用 `npm run build`。
  `boot()` 默认拼 `?editor=1` 深链，库首页用例用 `stayInLibrary` + `setupFetch` 注入假列表。
- 真机截图 `CC_FAKE=1 UI_DIST=<构建目录> node tools/shot/capture.mjs <tag>`（离线假数据，不需后端/GPU），图在 `tests/_out/ui_review/`。
  iframe 内部 jsdom 验不到；动 `web/public/vnccs-pose/**` 用 `python tools/verify_pose_editor.py` 或 `tools/verify_anim_import.mjs`。**别用 `--virtual-time-budget`+`--dump-dom`**。
- **假对象测试会骗人**：fake 必须模仿真实返回类型（`qwen_edit_image` 返 RGBA）；动过 alpha/通道/颜色空间必须真机跑一次。

## 度量 / 生成 / 抠图 / 像素感
- 速度/显存预热 + 多次中位数（`tools/bench_qwen.py`），显存看 `memory_reserved`；测后处理用未像素化 `__raw`。批量成本 `t/(1-(1-p)^k)`。
- 可交付 = 单主体 ∧ 无贴边 ∧ 前景 5%~60%。Qwen 768²/24 步：单帧 24~28s、峰值 11~15GB；<16 步糊块。瓶颈是姿态贴合。白毛/白甲别用白底抠图。
- **默认 `matte=False`，任何「偷偷抠」都是 bug**；排查 `generate._postprocess`、`PIPELINE_DEFAULTS`、前端硬编码、`app.py` 默认值，看产物 meta 的 `matte`。不抠必须走原生透明，否则绿底。
- 透明底白边要 unimultiply（`generate._defringe_white_matte`）；`alpha_reused=True` 时照样 defringe。
- 「像素感」四来源：契约词 `SPRITE_SUFFIX`、alpha 硬切、despeckle、色数量化。**量化不许动 alpha**。
  统一判据 `hard_edge = pixel > 1`；UI 上色数/像素块已删，`custom` 只作旧存档只读回显。

## 异步任务 / 节点画布 / 全局设置
- 节点「执行中」只看 `job.status` queued/running；**任何异常路径都要退出执行中**（提交 30s 超时、轮询连续 3 次失败、30 分钟看门狗、worker 兜底 try、404 明确文案）。
- 「角色图」是上游产物（`image.path`/`job.path`），统一走 `imagePathOf(node)`；`onConnect` 角色图白名单硬编码 `['poseclone','spriteswap','posebatch']`，新增节点必须加。
- 角色迁移批量必须读 `params["engine"]`（默认 wvnccs）；姿势集帧永远是人偶灰人偶（参考模式只认 skeleton/outline）。
- 侧栏 **3 个 tab（节点/画布/资料库）+ 4 个 `.rail-sec`（画布/模板/节点/资料库）**，只用 `.is-hidden`+CSS 切，别条件卸载；
  `NODE_ENTRIES` 顺序别重排（smoke 用 `.rail-add` 索引）。`Ctrl/⌘+K` 聚焦搜索。
  **设置（底模/抠图方式/LoRA）不许放回侧栏**：只长在顶栏 ⚙ 弹窗里（用户明确说「画布里的设置还在」不能再犯），
  侧栏状态条右侧只留一个 `.rail-settings` 快捷按钮去开同一个弹窗。
- `NodeView='expanded'|'image'|'result'`：出图自动 result，参数区 CSS 隐藏 **DOM 保留**；结果条只为有入边节点渲染 `[data-port]`。`?demo=result` 是假结果。
- `--rail-w:252px` 必须与 `App.tsx`、`store/projects.ts` 初始 viewport `x:300,y:96` 同步。`.chars` 是 `position:absolute; inset:0`。
- 弹窗走 **Portal 到 body**；灯箱 wheel 要 `addEventListener(...,{passive:false})`；别用 `opacity` 做低调按钮（近黑底等于隐形）。
- `SpritePreview` 的 `onMeta` 必须 `useCallback` 稳定，否则翻帧弹回第一帧。

## 姿势 / 人偶 / 动作导入
- 「人偶姿势」是官方 VNCCS PoseStudio **内嵌 iframe**（`studio.html?embed=1`），自写 2D 伪 3D 已删，别回写。
  「放大」只能改容器 CSS 为 `position:fixed`（移动 iframe DOM 会重载 37MB 模型）；`capture()` 前要关 `VNCCS_DirectionalSkydome`；embed class 必须在 `<head>` 内联同步加。
- 动作导入 glb/fbx 取帧：本地 three r160 + importmap；FBX 另有 4 个依赖，缺一个会悄悄回落 CDN。不改官方 `mixamo.js`，`anim-skeleton.js` 只登记别名（结构优先、名字兜底）。抽出的帧走 `assets/projects/<项目>/poses/`，同名 409。
  **脚底对齐两层**：世界空间以 resetPose 脚底为基准平移根节点 + 屏幕空间 `alignFrames()`；候选只增不删，舞台「有源帧就渲染」。
- 「从图片提姿势」：COCO_TO_OP `13`→`L_ANK` 必须双射；官方 `convertOpenPoseToPose` 只有 2D，改用 MediaPipe `pose_world_landmarks` 拿 3D，前端 `joints` 带 `reprojected` 归零旋转。
- **内嵌的是 Utils 0.7.1/0.7.2 基线，别整包升级到 0.7.3+**：新版 `core.js` 会
  `import { normalizeCaptureLens } from "./animation.mjs"`，而我们的 `animation.mjs`
  没这个导出且 import 了不存在的 `../shared/custom_select.mjs`（死代码），
  覆盖后 iframe 直接崩。比版本要用 `git hash-object` 比 blob（文件 SHA1 口径不同）。
  我们导出走 `capture(...,{viewport:true})`，官方 0.7.3 的 FOV/Distance 只作用于
  captureCamera 路径，**照搬也影响不到出图**。官方 UI 层 gizmo/Radar/Undo 属第二套
  编辑器，与「iframe 只留画布」冲突；SAM/BiRefNet/weights_only 均无落点
  （0 处 `torch.load`，唯一 .pt 由 ultralytics 加载）。
  ⚠️ 官方库 191 个 JSON 的 `prompt`/`pose_prompts` **全是空串**，别做「保留 prompt」。
- **2026-11-08 已落地（用户明确要支持多角色同框，别再说「只做单主体」）**：
  `vnccs-pose/scene.js` 是唯一的场景状态机（`{type:'scene',action}` + `sceneState` 回执），
  `PoseScenePanel.tsx` 是宿主 UI；状态只在 iframe 一份，`commit=false` 不写 project.json。
  引擎 0.7.2 基线本来就有 `passiveCharacters`，只补了 `setPassiveCharacterPlacement`；
  官方 175 行纯函数 vendored 成 **`character_transform.js`**（`.mjs` 会被 Python
  静态服务当 `text/plain` 而拒载模块 —— web/public 下 ESM 一律用 `.js`）。
  撤销用 `options.captureHistoryContext/onHistoryRestore` 把多角色放置一起记进 history；
  出图前 `hideForCapture()` 藏 gizmo（我们走 viewport:true，官方 layer 过滤不生效）。
  `PoseScene` 落 `project.json.poseScene`；`app.py::PoseSetFrameReq` 必须显式声明
  scene/camera/characters（Pydantic 默认吞多余字段）。
- 收尾已做：`vnccs-pose/pose_guard.js`（动作采样失败还原姿势，含整批抽帧）、
  `matte.release_ai_sessions()`（`app._run` finally 按 job 释放，`MEOW_MATTE_KEEP=1` 常驻）、
  `workflows/posesubjects.py`（多角色参考图 sidecar + 单主体提示词契约改成复数）。
  **别用图像识别数主体**：真机三人图连通域合并成 1、骨架线图细线反而数成多个；
  导出走 `/upload?subjects=N` 写图片旁 `.png.json`，`subjects_for()` 读，老图按 1。
  官方 `_VNCCS_TEMPLATE` 一字未动。w4 传骨架帧目录时 sidecar 不在帧旁会安全降级。
- **仍需 GPU 验证**：双人参考图最终出图是否真保住两个角色（文本契约已接线并单测通过）。

## UI 设计系统（改外观只改末尾统一层）
- `styles.css` 顶部 `:root` 是**唯一**配色/字体来源（只定义一次）。末尾「全站样式统一层」是最终权威层，
  ⚠️ **以后改外观只改这一层，不要再叠第 8 层**。卡片 `var(--r)=4px` / 控件 `var(--r-sm)=3px`，
  边框 `1px solid var(--px-edge)`，阴影 `var(--shadow-hard)=2px 2px 0 rgba(0,0,0,.34)`，
  主视图 tab 绿底 / 其余青激活，文字按钮 30px / 主 CTA 38px，disabled `.42`，小标题统一 `--sans`（中文别用 `--mono`）。
- **像素风第一版被用户否了（「有点诡异」）**：顶栏四色分段条、标题前 2×2 像素角标、主区 8px 满屏点阵、24px 密网格、3px 黑硬投影、每个小按钮都挂投影 —— 贴皮式装饰，**别加回来**。
  收敛版只留功能性几何：48px 疏网格（透明度 .022）、方块滚动条/虚线焦点框/方块步骤号、选中实心阶梯、分格进度条、精灵帧 16px 棋盘格、主按钮按下 1px 位移。
  **判断标准**：像素感来自边界清晰、可点按、像素图按最近邻放大，不来自装饰图形。中文绝不上像素字体。
- **画布网格两边同步**：`Canvas.tsx` 的 `GRID_BASE`(32) 与 `.canvas-grid` 的 `background-size: 32px × 3`。
  以前内联写死 24/120/120 而 CSS 图层顺序是「横线/竖线/点」，渲染出 24×120 长方格（用户明确说难看）。
- CSS 特异性坑：改 `.cc-step-no` 这类被 `.chars-tab.cc-step .cc-step-no` 定义的圆角必须带全前缀。
- 源码卫生：`styles.css` 3553→3440 行，删 7 段整段复制，`:root` 只在顶部一次。三处棋盘格声明、层间同值单行覆盖是刻意保留。
  清理手法：从后往前删 + 首行断言 + 替换次数断言；验证用 Pillow 逐张比对真机截图（18/18 字节级一致）+ smoke 全绿。

## 全局设置 / LoRA 导入（2026-11-07）
- **全局设置以 SQLite 为权威**：`metadb.settings` 表 + `GET/PUT /settings`，
  key 白名单 `base_model` / `matte_backend` / `loras` / `fast_mode`；启动
  `_restore_global_settings()` 灌回 engine。前端 `App` 启动拉一次，改了就 PUT。
- **LoRA 懒恢复**：`Engine.restore_loras()` 只记规格、**不加载管线**（否则启动白占 15GB），
  真正挂载推迟到 `txt2img_pipe()`；`active_loras()` 把待挂载也报成已挂，
  且 `txt2img_pipe` 先剔除文件已不存在的规格（否则每次生成抛 FileNotFoundError）。
- **LoRA 导入必须两步**：`POST /loras/inspect` 只暂存 + 只读 safetensors 头判格式，
  **绝不写 `models/lora/`**；`POST /loras/commit` 只有 `convert=true` 才调
  `tools/convert_lora.py::convert_weights`（服务端不另写一份转换），
  转换完再校验是 diffusers 才原子替换。ComfyUI 格式直接挂会**静默不生效**。
- `submitMatte` 必须带 `matte_backend`（以前漏了，抠图节点的全局方式完全不生效）；
  后端 `_run` 按工作流白名单兜底注入全局值。
- **全局设置只有一个挂载点**：`components/GlobalSettings.tsx` 的
  `LoraSettings` / `GeneralSettings` 只渲染在顶栏 ⚙ 弹窗里（画布与角色中心共用），
  状态在 App 一份。**别再往侧栏塞一份**（用户为这个提过两次）。
- **LoRA 强度滑块每行常显**：未挂载时 disabled + `.is-off` 置灰（默认 0.8），
  不能只在 `activeLoras` 命中时才渲染——用户会以为功能没了。
- 角色中心抠图**跟着全局方式走**：`matte_screen.global_backend()` 读 SQLite
  （2s 缓存）；`auto` 仍是官方幕布键控（`chroma_screen_matte`），
  只有显式选 `u2net`/`color` 才走 `matte.matte_auto(backend)`。

## 全屏库首页 / SQLite 元数据（2026-11-07）
- 角色中心与画布视图都是**两层**：默认全屏库（`CharacterLibrary.tsx` / `CanvasLibrary.tsx`，共用 `.charlib-*`），点卡片才进工作台/编辑器。
  UI 上「项目」一律叫**画布**（内部 scope / API `/projects` 不变）；`?editor=1` 深链直接进编辑器。
- 卡片：3:4 大图 + 图片上 tag（**自定义标签优先**，>5 收 `+N`）、名字/tag 模糊搜索、tag 过滤 OR、名字/创建/最后编辑三档排序；偏好存 `library/prefs.json`；空库中央大号创建按钮。
  `tagLabel()` 只做显示层映射，tag 值原样英文传给后端。
- **SQLite（`workflows/metadb.py`）是角色/画布元数据的权威**：派生 tags、稳定创建/编辑时间、自定义标签、备注、任务历史、封面路径字符串；标准库 `sqlite3` 零安装、单文件、WAL，库在 `library/meta.sqlite3`（.gitignore 忽略）。
  用户明确：**不必再顾忌官方 VNCCS 目录互通**。任务历史已写入并有 `GET /character-tasks`，但**展示 UI 还没做**。
- 角色中心工作台：`.chars.cc` 四步（① 角色档案 ② 定形象 ③ 出素材 ④ 导出），完成标记由磁盘产物推导，
  `userStep` 可回看、任务完成后自动推进，每步只有一个 `.cc-cta`，单步重跑收进 `details.cc-adv`（DOM 常驻）。
  单一 `matte=false`、表情默认 16 步、所有任务走 `runTask` 统一轮询；**不切走用户当前步骤**。
  旧版 5 个并列 tab / 右栏重复按钮 / `browseCostume` 双状态 / 9 条 xxxMsg 已删，别写回。

## 试玩验收台 / 方向维度（2026-11-08 起）

- 视角模式是**一等概念**：`topdown-4 / topdown-8 / side-2`，表在
  `web/src/store/playtest.ts` 与 `workflows/playtest.py` 两处必须一致。
  八方向靠双键向量合成；横板 left/right，right 由 left 镜像派生、带地面与跳跃。
- `GET /characters/{name}/playtest` 只读扫现有 `Sprites/<服装>/<表情>/*.png` +
  `Atlas/atlas.json`，**不新增落盘数据**；缺方向如实报告，默认值标 `source="default"`。
  试玩台挂角色中心步骤 ④ 导出顶部。
- **官方 VNCCS 没有原生八方向**；`PoseStudio.tsx:30` 的 `YAWS` 是我们自己转相机的
  8 个机位（0/45/…/315），角度→方向名映射首次真机出帧要校准一次，别写成官方能力。
  `docs/角色中心-剩余工作与做法.md:86,521` 的「8 方向不做」只针对官方原生支持。
- 播放与位移**共用一个 rAF**，每帧位移 `speed/fps`（对方项目 setInterval + rAF
  双时钟会滑步）；pivot 从 atlas 读并作 `transform-origin`，横板用 `translateY(-100%)`
  把脚底贴到 `.pt-ground` 的 78% 基准线。
- 新增角色接口时：`smoke.mjs` 里 `/characters/{name}` 的宽泛 GET 分支会抢请求，
  专用 mock 分支**必须放在它前面**，否则组件拿到 CharacterDetail、缺字段会崩整棵树。
## 多机位抽帧（2026-11-08 第二轮）

- `entry.js` 的 `animSample` 支持可选 `directions: [{key, yaw}]`：
  **方向循环 × 帧循环**，每方向先 `setYaw` + 双 rAF（等相机矩阵真的更新），
  再逐帧采样，**每个方向单独 `alignFrames()`**（不同 yaw 脚底墨迹行不同，混在一起会互相拉偏），
  帧带 `direction`/`yaw`，结束恢复原机位。**不传 directions 时老路径一字不变**，
  帧里不带 direction —— 现有工具与 730 条冒烟都依赖这条契约，别改坏。
- 宿主 `PoseStudio.tsx` 的抽帧模式默认「当前机位」，可选四/八方向/横板左右，
  方向表复用 `store/playtest.ts`（**只有一张表**）。帧墙角标显示方向。
- 方向元数据链路：`PoseSetFrameReq.direction`（Pydantic 不声明就吞字段）
  → `poseset.EXTRA_FRAME_KEYS` 加 `direction` → `set.json` 每帧 →
  `load_pose_set` 靠 `_frame_extras` 原样回传。`playtest._project_direction_assets()`
  扫 `assets/projects/*/poses/*/set.json` 的 direction，把已抽方向算进就绪判定。
- 真机验证脚本 `tools/verify_multi_yaw.mjs`（直接开 studio.html 顶层，不走宿主 React）。
  实测 4 机位 × 2 帧 → **4/4 种指纹、PNG 长度各异**，证明相机真的转了；
  单方向抽帧仍 3 帧且无 direction。**动过相机/对齐必须真机跑一次**。
- **2026-11-08 方向闭环已打通**：`workflows/directions.py` 是后端**唯一**方向表
  （`DIRECTIONS`/`YAW_MAP`/`direction_from_name`/`frame_filename`），
  `playtest.py` 从它 re-export，前端仍是 `store/playtest.ts` 那一张表（共两张，别写第三份）。
  链路：`set.json.direction` → `posegen.pose_entries()` 读旁边 set.json →
  按方向独立编号落 `sprite_<方向>_NNNN.png`（无 set.json 仍 `sprite_pose_NNNN.png`）→
  `spriteatlas` 额外产出 `<组>@<方向>` 动画（带 yaw/pivot，主组保留全帧）→
  `godotexport.frame_groups()` 靠成片集的 `poseSet` 反查方向，≥2 方向才走
  **多动画 `.tres` + `animations.json`**（单动画文本一字未改，测试有逐字断言）→
  试玩台 `clips.directionFrames` 按朝向选帧、换方向帧索引归零、HUD 标「占位」。
  老角色目录/老成片集在每一环都退回原行为。
- `/pose_dirs` 原来只扫两层，够不到 `assets/projects/<id>/poses/<集>/`（深度 3）；
  现在**定向补扫** `poses/` 与 `pose_frames/` 两类目录，带 `directions` 字段，
  不把全局深度加深（assets 几千文件）。角色中心「姿势素材」下拉终于点得通。
- 验收写回：试玩台「把试玩参数写入导出」→ `CharactersPage.atlasFrameRate`
  → `exportAtlas(name,{frame_rate})` → `atlas.json` → `_read_atlas` 回读，
  fps 只有一个来源（对方项目三套 fps 互不相通的坑没带进来）。
- 新增 `tests/test_direction_pipeline.py`（9 例）钉住整条链；
  后端子集 150 passed / 5 skipped，`smoke 730/730`，真机 34 张 0 错误。
- **2026-11-08 动作模板 registry 已落地**：`workflows/motiontemplates.py`
  是唯一定义，用户模板落 `library/motion_templates.json`（只存元数据），
  **内置三套是代码常量不落盘**（`topdown-walk-4` / `topdown-full-8` /
  `side-platformer-2`），升级不会跟用户改过的文件打架；内置 id 不可覆盖/删。
  字段：`viewMode / directions / yawMap / camera / mirrorFrom /
  actions[{name,clip,frames,fps,loop,oneShot}] / pivot / budget / subjects`。
  `normalize()` 必须幂等（`estimate`/`playback` 会反复调），
  `oneShot` 永远强制 `loop=False`。成本按 `t/(1-(1-p)^k)`，
  **k=1 时是 t/p 不是 t**（要摊失败重跑）；横板 `mirrorFrom` 省一半机位。
  REST：`GET/POST/PUT/DELETE /motion-templates` + `/{id}/estimate`。
- 模板消费链：`PoseStudio` 抽帧面板选模板 → 带出帧数与机位模式 →
  `templateId` 进 `set.json.source` → `godotexport.export_bundle()` 的
  **fps/loop 缺省**时顺着 `render.json.poseSet` 反查模板（显式传参永远优先，
  老调用方不变），`animations.json` 记 `playbackSource/templateId`；
  `playtest` 扫 `source.templateId` 给验收台显示模板名 / fps / 成本估算。
- 新增 `tests/test_motiontemplates.py`（9 例）。后端子集 **159 passed / 5 skipped**。
- **模板编辑 UI 已做**：`components/MotionTemplatePanel.tsx` 挂在资料库抽屉
  **第三个 tab**（`LibraryTab` 加 `'motion'`）。列表区分内置只读（只能「另存为我的」）
  与用户模板（编辑 / 删除需二次确认）；表单只暴露名字 / viewMode / 动作表 /
  进阶预算，`directions`/`camera`/`pivot` 留空由后端推导，**不复制第二套校验**。
  卡片用独立 `.mt-*` 类名，别用 `.lib-card`（smoke 按它计数）。
- `smoke.mjs` 的 `/motion-templates` mock 现在是内存 registry（GET/POST/PUT/DELETE
  全支持），资料库 tab 断言 2→3 并新增 10 条模板 CRUD 断言 → **740/740**。
  真机新增 35/36 两张截图（模板列表 + 编辑表单），0 脚本错误。
- 仍未做：碰撞盒、角色中心「一键按模板补方向帧」、抽帧面板下拉的真机验证
  （要先导入 glb 才出现）、逐帧 Qwen 多角色一致性（需 GPU 真跑）。改动**未提交**。
