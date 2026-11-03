# 项目长期约定（2d-game-meow）

## 文档入口

- **`docs/角色中心-剩余工作与做法.md`** —— 角色中心待办开工手册（怎么做），
  每项带源仓库 `file:line`、参数默认值、prompt 原文、落点文件、验证方式。
  T1~T5 已完成；§2 只剩 T7（放大，建议跳过）、T8（图集增强，用户暂缓）。
  §9 是「官方有、我们没做」的核对结论（大部分已不成立，别当欠账重做）。
- `docs/角色中心-VNCCS移植评估.md` —— 功能全表与已拍板的取舍（为什么）。
- `docs/PLAN-QWEN-ONLY.md` §4（九件做不到的事）、§6（删了什么 SD 代码）。
- `docs/人偶动作导入-glTF-FBX-调研与实现.md` —— 动作导入 + 已否决路线。

移植总规矩：**以 `D:\code\gen\others\ComfyUI_VNCCS-main` 为准，不自己猜。**
明确不做：Control Center、8 方向帧（官方没这功能）、SeedVR2 放大、Flux Klein /
MiniMax H3 / Anima / SDXL / FaceDetailer / SAM。

## 底模：只有 Qwen-Image-2.1（2026-10-30 起 SD 代码已删）

`pre-sd-purge` tag 是删除前状态，SD 权重在 `models/_archive/sd/`。
别再往 engine/generate 里加 StableDiffusion —— SD 管线方法已不存在。

⚠️ **许可证**：`models/checkpoints/qwen-image-2.1/LICENSE` 是 Qwen **RESEARCH**
LICENSE（2026-09-20），"FOR NON-COMMERCIAL PURPOSES ONLY"。所有产物都出自它，
商用前必须申请授权或换底模。

## 测试策略（用户明确要求：不要动不动跑全量）

全量约 260 用例 / 150s（含两次 Qwen 加载），只在合并/发版前跑。按改动的子集见
README「测试」表：engine→test_models+vramopt+stability(110s)｜generate→
test_qwen_workflows+denoise+paint_path(47s)｜presets→test_presets+quality(10s)｜
matte/refprep(20s)｜postprocess/pixelfy/sheet(15s)｜app/assetdb→test_api+review_p2(15s)｜
characters/charactergen(9s)｜emotiongen(6s，含真 YOLO)｜clothes(5s)｜posegen(5s)｜
spriteatlas(4s)｜labels.ts→test_labels(1s)｜projectio/paths→test_projects(5s)｜
library/poselib(1s)｜qwen_llm(5s，真推理要 MEOW_SLOW=1)｜前端→`cd web && npm run smoke`｜
tools/docs/README→不跑。

- **冒烟跑 `dist/`**：改完前端必须先 `npm run build`，否则拿旧产物跑（白高兴过一次）。
- **iframe 内部 jsdom 验不到**（PoseStudio）：动了 `web/public/vnccs-pose/**` 用
  `python tools/verify_pose_editor.py` 真机验（无头 Edge + SwiftShader，约 1 分钟）。
- `tests/test_generate.py` / `test_clip` / `test_controlnet` / `test_lora` / `test_train` /
  `test_review_fixes` / `test_review_p1` / `test_fetch` **已删除**（SD 时代模块），
  别在子集表里再写。

## 前端真截图（脱离 GPU）

- 工具在 `tools/shot/`（隔离装 puppeteer-core），驱动系统 Edge
  `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`。
- **`npm run build` 会触发批量删除守卫**（vite 清 dist/vnccs 约 528 文件 >50 被拦）。
  改用 `vite build --outDir dist_shot`（空目录删 0 个），截完即删。
- 加载 `index.html?demo=1` 铺 gen 节点；点侧栏 `.rail-item` 里的 `.rail-add` 加别的节点。
- **节点卡片选择器是 `.node.gen`（外层）**，`.node-kind.poseclone` 只是卡头文字 span。
- 高级项在 `details.adv`，截图前 `det.open=true`。产物落 `docs/img/`。

## 姿势与人偶（2026-11-01 起）

「人偶姿势」节点正文是官方 VNCCS PoseStudio 的**内嵌** iframe（`studio.html?embed=1`）。
自写的 2D 伪 3D 编辑器已删，**别再往回写**。

- MakeHuman 网格 + 53 骨（官方两骨 IK + 极点约束）；朝向靠相机绕 Y 渲染。
- 存档 = 官方 `getPose()` JSON；存 18 个 2D 点还原不精确（深度要重猜）。
- 「放大」只能改容器 CSS 为 `position:fixed`：**移动 iframe 的 DOM 位置会让浏览器
  重载它**（37MB 人偶模型重新解算）。
- 官方 `capture()` **不隐藏方向天穹**（`VNCCS_DirectionalSkydome`，彩虹网格球），
  导出前必须临时关掉。
- iframe 内**不许再有自己的工具条**（朝向/重置/导出）：只在宿主一份，
  `studio.html` 的 `#bar/#hint` 已删，`html.embed` 隐藏规则留作兜底。
  embed class 必须在 `<head>` 内联同步加（放 module 里会先闪一下工具条）。
- `.mini` 是 28×28 图标按钮，中文两字必竖排；文字按钮用 `.mini.wide` 或
  `.pose-editor .mini { width:auto; white-space:nowrap }`。

## 动作导入：glb / fbx 取帧（2026-11-02 起）

「导入动作」读带骨骼动画的 3D 文件，拖滑块选帧或「存入 N 帧」。
**3D 是文件自带的**，没有反推图片的深度歧义 —— 要整套 walk/run 优先走这条。

- 解析器是**本地副本**（`web/public/vnccs-pose/vendor/three/jsm/`，r160），靠
  `studio.html` 的 importmap 把裸名 `three` 指到 vendor。改 three 版本时 jsm 必须同版本。
  FBX 另有 4 个依赖（fflate / NURBS×2 / BufferGeometryUtils），缺一个就"本地失败→
  悄悄回落 CDN"，**断网才炸**。smoke 逐个断言存在。
- **不改官方 `mixamo.js`**：`anim-skeleton.js` 只做别名登记。**结构优先、名字兜底**
  —— 名字只取根锚点，肘腕膝踝沿子链走（Mixamo 的 `LeftLeg` 是小腿，KayKit 的
  `Leg` 是大腿，别指望名字表）。认不出要**明确报错**（缺哪几根 + 文件骨名样本）。
- **抽出来的帧不进资料库**：走 `assets/projects/<项目>/poses/<名字>/`（`poseset.py`），
  同名 409、每次存都是新文件夹。资料库只管跨项目复用的**单个**姿势。
- **脚底对齐要两层**：① 世界空间——以 resetPose 后默认站姿脚底为基准平移**根节点**
  （不是 pelvis 本身），且基准**只量一次**（resetPose 不动 position，每帧量会漂）；
  ② 屏幕空间——透视相机下脚前后移动改变落点，抽完帧再统一把墨迹底边压到同一行
  （`alignFrames()`，留 5% 底边距）。只对齐脚底，**保留身高变化**；腾空帧贴回地面。
- 批量出片（`posebatch.py` + `PoseBatchPanel.tsx`）：`pose_frames/<名字>/`，每帧
  `candidates[] + selected`。**候选只增不删**——旧图必须还能选回来。
- 面板 `▶` 翻的是舞台大图（8fps），舞台必须「有源帧就渲染」（写在 `render` 分支里
  的话，没出图时点播什么都不动）。
- **「角色图」是上游节点的产物**（`image.path`/`job.path`），不是上游节点自己的
  `charSource`。统一走 `imagePathOf(node)`。
- **`onConnect` 的角色图白名单是硬编码的**（`['poseclone','spriteswap','posebatch']`），
  新增带角色图输入的节点必须加，否则拖线**静默忽略**。
- **角色迁移批量必须读 `params["engine"]`**（默认 wvnccs）。曾写死 `pose_clone`，
  结果 12 帧全照抄站姿。**「生成的和原图一模一样」先查引擎是否被写死。**
- **姿势集帧永远是人偶编辑器的灰人偶**：参考模式只认 skeleton / outline；`auto`/`pose`
  是旧存量值，统一归 skeleton（前端归一化 + 后端兜底）。走 auto 会把灰人偶画成
  「黑底彩骨→反相→几乎空白」。
- **真机验证用 puppeteer**（`tools/verify_anim_import.mjs`）。**别用
  `--virtual-time-budget` + `--dump-dom`**：虚拟时钟把等待瞬间走完，而解析是
  真 CPU 活 → 一律超时。断网靠 request interception 拦外部请求。

## 「从图片提姿势」的两层天花板

- 已修：`COCO_TO_OP` 的 `13` 曾写成 `R_ANK`（应 `L_ANK`）→ 左脚踝从没被用上，
  抬腿姿势两条腿叠一点 → 麻花。测试钉了映射必须是双射（纯常量，不跑模型）。
- 已修：官方 `convertOpenPoseToPose` **只有 2D**，靠髋线在画面里的角度反推骨盆倾斜。
  侧身照两髋投影只隔身高 1.5% → 判 −81° → 人偶躺倒。现在用 MediaPipe
  `pose_world_landmarks`（9.4MB，Apache-2.0）拿米制 3D：算躯干 yaw → 绕 Y 转正 →
  等比投影；前端 `joints` 消息带 `reprojected`，把官方的 `modelRotation`/
  `cameraYRotation` **归零**（不归零会拿压扁的髋线再猜一次）。归一化必须**等比**，
  官方 `_projectNamed3DTo2D` 把 x/y 各自拉满会横向拉伸。
- 仍不完美：肢体交叠（手抓脚）3D 精度不够，需人工微调 1~2 关节。

## 目录约定

- **路径只有一处定义**：`workflows/paths.py`。别在别处写 `ROOT/"assets"`。
  可用 `MEOW_ASSETS_DIR` / `MEOW_TEMPLATES_DIR` 覆盖。
- `assets/` = 项目资产（要交付的东西）；`tests/_out/` = 测试与验证产物
  （conftest 在 collection 阶段设 `MEOW_ASSETS_DIR`）；`tools/` 脚本走
  `paths.tool_out()`。`tools/pose_from_image.py` 是用户工具，产物仍放 assets。
- 模板在 `templates/*.mproj`，不在 assets 下（跨项目共享的用户资源）。

## 项目存储与资料库：浏览器里一个字不留

- 项目内容只在服务端磁盘：`assets/projects/<id>/project.json`，写入走
  `workflows/projectstore.py`（临时文件 + replace，原子）。**别再引入 localStorage 缓存**。
- 项目 id 与后端 scope 必须同一个，否则素材和项目定义分家。
- 提示词收藏 / 姿势库 / 界面偏好走 `workflows/library.py` →
  `library/{prompts,poses,prefs}.json`。前端 store 只放纯函数，增删改用**服务端返回值**
  覆盖本地。
- 迁移顺序**先 peek 后删**：`readLegacy()` 只读 → `POST /library/seed`（只补不覆盖）
  → 确认收到才 `clearLegacy()`。
- `library/` 是用户攒的；官方 VNCCS 姿势库（191 个）在
  `web/public/vnccs-pose/PoseLibrary/`，是静态资源，别混。

## 异步任务失败处理

节点「执行中」只看 `job.status` 是 `queued`/`running`。**任何异常路径都要让节点退出
执行中**：提交 30s 超时、轮询连续 3 次失败判 failed、30 分钟看门狗、worker 兜底 try、
`/jobs/<id>` 404 带明确文案。以前 `if (r.status !== 'fulfilled') return n` 会让节点永远卡住。

## 假对象测试会骗人

**单测全绿 ≠ 功能对。** 踩过：`qwen_edit_image` 返 **RGBA** 而所有 `fake_edit` 返
RGB → 贴回时 alpha 混进源图；假 `keyer` 没模拟"透明像素 RGB=0（黑）" → 真抠像把
黑块当前景。规矩：**fake 必须模仿真实返回的类型**；动了 alpha/通道/颜色空间的逻辑
**必须真机跑一次**。

## 度量约定

- 速度/显存结论**必须预热 + 多次取中位数**，单次波动可达 70%（`tools/bench_qwen.py`）。
- 显存看 `memory_reserved`；Qwen 走 cpu offload，加载完 allocated 本来就是 0。
- 测后处理要用**未像素化的 `__raw` 输入**（成品命中网格检测，耗时虚报 8 倍）。
- 批量推理成本：`t / (1-(1-p)^k)`，t 已是 k 张总时间，别再乘 k。

## 产线判据

可交付 = 单主体 ∧ 无贴边 ∧ 前景占比 5%~60%。Qwen 768²/24 步：单帧 24~28s、
峰值 11~15GB；低于 16 步出糊块。逐帧路线短板是**姿态贴合**不是帧间一致性。
白毛/白甲角色别用白底抠图——换 `bg=dark` 或直接勾透明底。

## 抠图必须是用户的选择

**默认 `matte=False`，任何"偷偷抠"都是 bug。** 排查四处：① `generate._postprocess`
的透明分支；② `pipeline.PIPELINE_DEFAULTS["matte"]`；③ 前端硬编码；④ `app.py`
请求模型的默认值。判断有没有抠看产物 meta 的 `matte` 字段。不抠就必须走原生透明
（`normalize_background_color` 默认 Transparent），否则出来是绿底。

## 透明底白边：RGB 是白底预乘的

半透明边缘 RGB = `前景*α + 白*(1-α)`（实测过渡带亮度 237.8 vs 实心 206.4）。
必须 unimultiply（`generate._defringe_white_matte`）：`C_fg=(C_obs-255*(1-α))/max(α,0.25)`。
**坑**：`alpha_trust` 只看 alpha 分布不看 RGB，所以 `alpha_reused=True` 时 RGB 照样
可能是白的 —— 复用 alpha 也要过 defringe。

## 「像素感」四个来源（不只有 pixelate）

① 契约词（`SPRITE_SUFFIX` 带 pixel art 词，`pixel<=1` 时要换 `_PAINT` 版）；
② alpha 硬切（像素档 `_harden_alpha(240)`，插画档只能 `sanitize_alpha`）；
③ 形态学开运算（`despeckle`，只在像素档开）；
④ **色数量化**（白边真凶）：`quantize_palette` 里 `alpha.point(a>127?255:0)` 把
白底预乘的过渡带切成不透明近白 → 实心白边。**量化默认不许动 alpha**。

统一判据 `hard_edge = pixel > 1`（去掉 `or palette > 0`），
`test_paint_path.py::test_palette_alone_is_not_pixelation` 钉住。
UI 层面色数/像素块已删（学业界），`custom` 只作旧存档只读回显，别再把手填加回来。

## 前端小部件踩过的坑

- **`.mini` 是 28×28 图标按钮**（✕/⤢ 单字符用）。文字按钮另起类，否则竖排溢出。
- **别用 `opacity` 做"低调按钮"**：近黑画布上等于隐形。要实心底 + 灰白边框。
- **数字输入框别用 flex-grow 比例分**：会压到 30px，`768` 看不见。用 `flex:0 0 52px`。
- 覆盖层弹窗（提示词放大、看图）**走 Portal 到 body**：节点卡片有 `overflow:hidden`
  + `transform`，就地渲染会被裁且跟着缩放。
- 灯箱缩放：**React 的 onWheel 是被动监听**，`preventDefault` 无效 → 必须
  `addEventListener('wheel', fn, {passive:false})`。拖动松手补的 click 会冒泡到
  "点背景关闭" → 舞台先 `stopPropagation` + `justDragged` 标记。
  `image-rendering` 跟 zoom 走：≥1 最近邻、<1 平滑。

## 界面中文化：只翻译显示层，别改标识

`Naked` / `angry 😠` / `Shonen Anime` / `Ballet` **不是文案，是数据标识**——同时是
磁盘目录名、接口键、写进 prompt 的内容。**值传原样英文，只在渲染处显示中文。**

- 映射表全在 `web/src/labels.ts`（约 1000 条），查不到回落原值。
- **官方 JSON 一个字不改**（`emotions.json` / `character_styles.json` /
  `character_presets_v2.json` / `PoseLibrary/`）。
- `tests/test_labels.py` 逐条核对覆盖率（抓到过小写 `sitting`）。
- **漏接比映射表更容易出错**：改完要重扫渲染处（`CharactersPage` / `TagPicker` /
  `LibraryDrawer` / `SpritePreview` / `App`）。
- 表情 emoji **别手抄**（`😮‍💨` 带 U+FE0F/U+200D），用 `\p{Extended_Pictographic}` 摘。
- `neutral` 是我们这边的默认表情（官方库没有），要单独断言。
- 保持英文的：提示词示例、官方固定指令、错误码（`MODEL_MISSING` 前后端靠它匹配）、
  文件路径与目录名。

## 文档/科普：归开源 upstream

写原理/科普、引用截图时只归开源上游（AHEKOT/ComfyUI_VNCCS、
AHEKOT/ComfyUI_VNCCS_Utils、MIUProject/VNCCS_PoseStudio_QI2.1），
**不要写 runcomfy.com 这类商业托管站**——用户明确要求不替商业公司宣传。
