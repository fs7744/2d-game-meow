# 项目长期约定（2d-game-meow）

## 文档入口 / 移植总规矩

- `docs/角色中心-剩余工作与做法.md`：待办开工手册（源仓库 file:line、默认值、prompt 原文）。
  T1~T5 已完成；§2 只剩 T7（放大，建议跳过）、T8（图集增强，暂缓）；§9 已核对结论，别当欠账重做。
- `docs/角色中心-VNCCS移植评估.md`（取舍原因）、`docs/PLAN-QWEN-ONLY.md` §4/§6、
  `docs/人偶动作导入-glTF-FBX-调研与实现.md`。
- **一切以 `D:\code\gen\others\ComfyUI_VNCCS-main` 为准，不自己猜。**
  不做：Control Center、8 方向帧、SeedVR2 放大、Flux Klein / MiniMax H3 / Anima / SDXL / FaceDetailer / SAM。

## 底模与许可证

只有 Qwen-Image-2.1（2026-10-30 起 SD 代码已删，`pre-sd-purge` tag 是删除前状态，
权重在 `models/_archive/sd/`）。别再往 engine/generate 加 StableDiffusion。
⚠️ `models/checkpoints/qwen-image-2.1/LICENSE` 是 Qwen **RESEARCH LICENSE**（非商用），
产物出自它，商用前须授权或换底模。

## 测试策略（用户明确要求：不要动不动跑全量）

全量约 260 用例 / 150s，只在合并/发版前跑；子集见 README「测试」表：
engine→test_models+vramopt+stability(110s)｜generate→test_qwen_workflows+denoise+paint_path(47s)｜
presets→test_presets+quality(10s)｜matte/refprep(20s)｜postprocess/pixelfy/sheet(15s)｜
app/assetdb→test_api+review_p2(15s)｜characters/charactergen(9s)｜emotiongen(6s，真 YOLO)｜
clothes/posegen/spriteatlas/library/poselib/labels/projects(各 1~5s)｜
前端→`cd web && node smoke.mjs`（**别用 `npm run smoke`**，它自带 vite build 清 dist
触发批量删除守卫；先 `npx tsc -b && npx vite build --outDir <新目录>` 再 cp 覆盖 dist）｜
tools/docs/README→不跑。SD 时代测试（test_generate/clip/controlnet/lora/train/review_*/fetch）已删除。

- 冒烟跑 `dist/`：**改完前端必须先构建**，否则拿旧产物跑。
- iframe 内部 jsdom 验不到；动了 `web/public/vnccs-pose/**` 用 `python tools/verify_pose_editor.py`
  或 `tools/verify_anim_import.mjs` 真机验（puppeteer）。**别用 `--virtual-time-budget`+`--dump-dom`**
  （虚拟时钟走完等待，解析是真 CPU 活 → 必超时）；断网靠 request interception。
- **假对象测试会骗人**：fake 必须模仿真实返回类型（`qwen_edit_image` 返 RGBA，假抠像没模拟透明像素 RGB=0）。
  动过 alpha/通道/颜色空间的逻辑必须真机跑一次。

## 度量 / 产线判据

- 速度/显存必须预热 + 多次取中位数（单次波动 70%，`tools/bench_qwen.py`）；显存看 `memory_reserved`
  （Qwen cpu offload，加载完 allocated 本就是 0）。测后处理用未像素化 `__raw`（成品命中网格检测虚报 8 倍）。
  批量成本 `t/(1-(1-p)^k)`，t 已是 k 张总时间。
- 可交付 = 单主体 ∧ 无贴边 ∧ 前景占比 5%~60%。Qwen 768²/24 步：单帧 24~28s、峰值 11~15GB；
  低于 16 步出糊块。瓶颈是姿态贴合不是帧间一致性。白毛/白甲别用白底抠图，换 `bg=dark` 或透明底。

## 生成 / 抠图 / 像素感

- **默认 `matte=False`，任何"偷偷抠"都是 bug**。排查四处：`generate._postprocess` 透明分支、
  `pipeline.PIPELINE_DEFAULTS["matte"]`、前端硬编码、`app.py` 默认值；看产物 meta 的 `matte` 字段。
  不抠必须走原生透明（`normalize_background_color` 默认 Transparent），否则出来是绿底。
- 透明底白边：半透明边缘 RGB 是白底预乘，必须 unimultiply（`generate._defringe_white_matte`：
  `C_fg=(C_obs-255*(1-α))/max(α,0.25)`）。`alpha_reused=True` 时照样要 defringe。
- 「像素感」四来源：① 契约词（`SPRITE_SUFFIX`，`pixel<=1` 换 `_PAINT`）；② alpha 硬切（像素档
  `_harden_alpha(240)`）；③ `despeckle`（只像素档）；④ 色数量化——**量化默认不许动 alpha**，
  `alpha.point(a>127?255:0)` 会切出实心白边。
- 统一判据 `hard_edge = pixel > 1`（`test_paint_path.py::test_palette_alone_is_not_pixelation` 钉住）；
  UI 上色数/像素块已删（学业界），`custom` 只作旧存档只读回显，别加回手填。

## 目录 / 存储

- **路径只有一处定义 `workflows/paths.py`**，别在别处写 `ROOT/"assets"`；可用
  `MEOW_ASSETS_DIR` / `MEOW_TEMPLATES_DIR` 覆盖。`assets/`=交付资产，`tests/_out/`=测试产物，
  `tools/` 走 `paths.tool_out()`；模板在 `templates/*.mproj`。
- **浏览器里一个字不留**：项目只在 `assets/projects/<id>/project.json`
  （`workflows/projectstore.py`，临时文件 + replace 原子写），别引入 localStorage 缓存；
  项目 id 与后端 scope 必须同一个。提示词收藏/姿势库/界面偏好走 `workflows/library.py` →
  `library/*.json`，前端用服务端返回值覆盖本地；迁移顺序「先 peek 后删」。
  `library/` 是用户攒的，官方 191 个姿势在 `web/public/vnccs-pose/PoseLibrary/`（静态资源）。

## 异步任务 / 节点画布

- 节点「执行中」只看 `job.status` 是 queued/running。**任何异常路径都要退出执行中**：
  提交 30s 超时、轮询连续 3 次失败、30 分钟看门狗、worker 兜底 try、`/jobs/<id>` 404 明确文案。
- **「角色图」是上游产物**（`image.path`/`job.path`），统一走 `imagePathOf(node)`，不是上游 `charSource`。
- `onConnect` 角色图白名单硬编码 `['poseclone','spriteswap','posebatch']`，新增节点必须加，否则拖线静默忽略。
- 角色迁移批量必须读 `params["engine"]`（默认 wvnccs）：写死 `pose_clone` 会全照抄站姿；
  「生成的和原图一模一样」先查引擎是否被写死。
- 姿势集帧永远是人偶灰人偶：参考模式只认 skeleton/outline，`auto`/`pose` 旧值归一到 skeleton。
- 侧栏 4 tab（节点/项目/资料库/设置）但 **6 个 `.rail-sec` DOM 顺序必须保持**
  项目/模板/节点/资料库/LoRA/设置；切 tab 只用 `.is-hidden`+CSS，别条件卸载（冒烟依赖 DOM 完整）。
  `NODE_ENTRIES` 顺序别重排（smoke 用 `.rail-add` 索引）；`Ctrl/⌘+K` 聚焦搜索、Enter 添加、Esc 清空。
- `NodeView='expanded'|'image'|'result'`：出图自动 result，参数区 CSS 隐藏 **DOM 保留**
  （改条件渲染会让 `.prompt`/`.gen-btn` 消失、连线端口丢、大量冒烟红）。结果条只为有入边节点渲染 `[data-port]`。
  `?demo=result` 是纯前端假结果供截图。

## 姿势 / 人偶 / 动作导入

- 「人偶姿势」是官方 VNCCS PoseStudio **内嵌 iframe**（`studio.html?embed=1`），自写 2D 伪 3D 已删，别回写。
  MakeHuman 网格 + 53 骨；存档 = 官方 `getPose()` JSON（存 18 个 2D 点不精确）。
  「放大」只能改容器 CSS 为 `position:fixed`：**移动 iframe DOM 位置会重载**（37MB 模型重解算）。
  官方 `capture()` 不隐藏方向天穹 `VNCCS_DirectionalSkydome`，导出前必须临时关掉。
  iframe 内不许再有自己的工具条；embed class 必须在 `<head>` 内联同步加（放 module 会闪工具条）。
- 动作导入（glb/fbx 取帧）：3D 是文件自带的，要整套 walk/run 优先走这条。
  解析器是本地 three r160 副本 + `studio.html` importmap；FBX 另有 4 个依赖（fflate/NURBS×2/
  BufferGeometryUtils），缺一个会悄悄回落 CDN，断网才炸。
  不改官方 `mixamo.js`，`anim-skeleton.js` 只登记别名：**结构优先、名字兜底**，认不出要明确报错。
  抽出的帧走 `assets/projects/<项目>/poses/`（`poseset.py`），不进资料库；同名 409、每次存新文件夹。
  **脚底对齐两层**：世界空间以 resetPose 脚底为基准平移根节点（基准只量一次）+ 屏幕空间
  `alignFrames()` 压到同一行（留 5% 底边距）；只对齐脚底、保留身高变化。
  批量出片 `pose_frames/<名字>/`，`candidates[]+selected` **候选只增不删**；面板 ▶ 翻舞台大图（8fps），
  舞台必须「有源帧就渲染」。
- 「从图片提姿势」：已修 COCO_TO_OP `13`→`L_ANK`（必须双射）；已修官方 `convertOpenPoseToPose` 只有 2D
  （侧身照髋线判 −81° 人偶躺倒）→ 用 MediaPipe `pose_world_landmarks` 拿 3D，绕 Y 转正、等比投影，
  前端 `joints` 带 `reprojected` 把 `modelRotation`/`cameraYRotation` 归零（官方 x/y 各自拉满会横向拉伸）。
  肢体交叠仍需人工微调 1~2 关节。

## 中文化 / 文案

`Naked` / `angry 😠` / `Shonen Anime` / `Ballet` 是**数据标识**（目录名、接口键、prompt 内容），
**值传原样英文，只在渲染处显示中文**。映射全在 `web/src/labels.ts`（约 1000 条，查不到回落原值），
`tests/test_labels.py` 逐条核覆盖率；**官方 JSON 一个字不改**。改完重扫渲染处
（`CharactersPage`/`TagPicker`/`LibraryDrawer`/`SpritePreview`/`App`）——漏接比映射表更容易出错。
表情 emoji 别手抄，用 `\p{Extended_Pictographic}` 摘；`neutral` 是我们默认表情，单独断言。
保持英文：提示词示例、官方固定指令、错误码（`MODEL_MISSING`）、文件路径与目录名。
写原理/科普只归开源 upstream（AHEKOT/ComfyUI_VNCCS、AHEKOT/ComfyUI_VNCCS_Utils、
MIUProject/VNCCS_PoseStudio_QI2.1），**不要写 runcomfy.com 这类商业托管站**（用户明确要求）。

## UI 设计系统（2026-11-06 起）

- `web/src/styles.css` 的 `:root` 是**唯一**配色/字体来源；文件末尾「UI 视觉刷新」override 层，
  改组件外观优先在 override 层加，别回头改前 2000 行散落的定位规则。
- 不引用 `Press Start 2P`/`Silkscreen`（从未加载）；中文走系统字体栈，数字/英文走 `--mono`。
  强调色 绿 `#4ade9b` / 青 `#6cc4ef` / 粉 `#ff7a90` / 黄 `#f5c65a`；背景 `--bg`→`--bg-4` 明度分层。
- **`--rail-w:252px` 必须与 `App.tsx`、`store/projects.ts` 初始 viewport `x:300,y:96` 同步**。
  `.chars` 是 `position:absolute; inset:0`（父 `.stage` 不是 flex，`flex:1` 不生效）；
  `.chars-ops` 必须是 `.chars` 直属第三列，别塞回 `.chars-detail`。
- 小部件坑：`.mini` 是 28×28 图标按钮（中文两字必竖排，文字按钮用 `.mini.wide`）；
  **别用 `opacity` 做低调按钮**（近黑画布上等于隐形），要实心底+灰白边框；
  数字输入用 `flex:0 0 52px`，别用 flex-grow；覆盖层弹窗走 **Portal 到 body**（节点卡 overflow:hidden + transform）；
  灯箱缩放 React `onWheel` 是被动监听，必须 `addEventListener('wheel',fn,{passive:false})`，
  拖动松手的 click 要 `stopPropagation` + `justDragged`；`image-rendering` 跟 zoom 走（≥1 最近邻）。
- 真机截图：`node tools/shot/capture.mjs <tag>`（后端需在 8000，`UI_DIST` 指定构建目录，
  图在 `tests/_out/ui_review/`）。加载 `index.html?demo=1` 铺节点，`.rail-item .rail-add` 加节点，
  节点卡片选择器是 `.node.gen`（外层），高级项在 `details.adv`（截图前 `det.open=true`）。
- 构建：`npx tsc -b && npx vite build --outDir <新目录>` 再 `cp -r` 覆盖 `dist`；
  **别用 `npm run build`**（vite 清 dist/vnccs 约 528 文件 >50 触发批量删除守卫）。

## 角色中心：四步工作台（2026-11-06 彻底重构，别写回 tab）

- `.chars.cc` 三栏 = 角色库 / 四步主工作区 / **常驻成果预览**。
  `characters/model.ts` 放常量、`STEPS`、`computeProgress`、`explainError`；
  `characters/Controls.tsx` 放纯展示组件；`CharactersPage.tsx` 只管状态与业务。
- **唯一主线**：① 创建角色 ② 定形象 ③ 出素材 ④ 导出。步骤完成标记由
  `detail.assets` 的帧数与 `atlasInfo.exists` 推导；`userStep` 可手动回看，
  任务完成后 `setUserStep(null)` 自动推进。每步只有一个 `.cc-cta`。
- 旧版已删：5 个并列 `.chars-tab`（现在 `.cc-step` 恰好 4 个）、右栏重复按钮、
  `browseCostume` 双服装状态、9 条 xxxMsg、跨页强跳 `finishOp`。
  **任何动作只留一个入口**，单步重跑收进 `<details class="cc-adv">`（DOM 常驻）。
- 单一 `matte=false`（旧版四处默认值不同）；表情默认 16 步（质量下限）；
  所有任务走 `runTask` 统一轮询（互斥 + 进度 + 错误码话术 + 常驻任务条）。
  **不切走用户当前步骤**，成果靠右栏刷新呈现。
- `SpritePreview` 的 `onMeta` 必须 `useCallback` 稳定：内联函数会让 `loadMeta`
  每次 render 重建，翻到第二帧立刻弹回第一帧。
- `smoke.mjs` 角色用例（46~52）按新 DOM 重写，API 契约全保留；
  mock 的 `PUT/DELETE /costumes` 必须返回 200，否则「新建服装」不生效。
- 真机截图：`CC_FAKE=1 UI_DIST=<构建目录> node tools/shot/capture.mjs <tag>`，
  离线假数据拍 20~24 号四步图，不需要后端 / GPU。
