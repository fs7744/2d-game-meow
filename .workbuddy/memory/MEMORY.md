# 项目长期约定（2d-game-meow）

## 总规矩
- 基准 `D:\code\gen\others\ComfyUI_VNCCS-main`，不猜。不做 Control Center、原生八方向帧、SeedVR2、Flux/SDXL/FaceDetailer/SAM。
- **只有 Qwen-Image-2.1**（`engine._check_model` 硬拒其它）；底模全局唯一、节点无下拉（`GenParams.model` 仅旧存档，`App.migrate()` 清空）。权重非商用。
- 路径只在 `workflows/paths.py`（`MEOW_*_DIR` 覆盖）：`assets/` 交付、`tests/_out/` 测试、`templates/*.mproj`、`library/` 元数据；图片/帧/权重/节点定义不入库。
- 中文化只在 `web/src/labels.ts`（查不到回落原值），`tests/test_labels.py` 核覆盖率；科普只引开源 upstream。

## 测试 / 构建（别动不动跑全量）
- 全量约 260 例 / 150s，只在合并前跑。前端 `cd web && node smoke.mjs`，**别用 `npm run smoke` / `npm run build`**（自带 vite build 清 dist，触发批量删除守卫）。
- 改前端**必须构建**：`npx tsc -b && npx vite build --outDir dist`。别 `cp -r dist_next/. dist/`（堆旧 bundle）；别 `rm -rf dist_next`（946 文件触发守卫），临时目录 `mv` 到 `tests/_out/`。
- **并行两个 `vite build --outDir dist` 会抢 `prepareOutDir` 报 ENOENT 并清空 dist**；别用 `| tail` 的退出码判成败。
- 截图 `CC_FAKE=1 UI_DIST=web/dist node tools/shot/capture.mjs <tag>` → `tests/_out/ui_review/`（**UI_DIST 相对仓库根**）。
- smoke：`boot()` 默认拼 `?editor=1`；库首页用 `stayInLibrary` + `setupFetch`；`/characters/{name}` 宽泛 GET 会抢请求，专用 mock 放它前面；断言输入值读 `.value`。
- 动 `web/public/vnccs-pose/**` 用 `tools/verify_pose_editor.py` / `tools/verify_anim_import.mjs` / `tools/verify_multi_yaw.mjs`。
- **假对象测试会骗人**：fake 必须模仿真实返回类型（`qwen_edit_image` 返 RGBA）；动 alpha/通道/颜色空间要真机跑一次。单测**不能**换掉 `app._q`（worker 里 `_q.get()` 是模块全局查找）。
- 真后端 UI 验证：`MEOW_ASSETS_DIR/MEOW_TEMPLATES_DIR/MEOW_LIBRARY_DIR` 指 `tests/_out`，`.venv/Scripts/python.exe -m uvicorn app:app`（导入仅 2s，重活懒加载）。

## 生成 / 抠图 / 像素感
- 可交付 = 单主体 ∧ 无贴边 ∧ 前景 5%~60%。Qwen 768²/24 步单帧 24~28s、峰值 11~15GB；<16 步糊块；白毛白甲别用白底抠图。
- **默认 `matte=False`，任何「偷偷抠」都是 bug**（查 `generate._postprocess` / `PIPELINE_DEFAULTS` / 前端硬编码 / 产物 meta）。不抠走原生透明；透明底白边要 `_defringe_white_matte`（`alpha_reused` 也要）。
- 「像素感」四来源：`SPRITE_SUFFIX` 契约词、alpha 硬切、despeckle、色数量化（**量化不许动 alpha**）；判据 `hard_edge = pixel > 1`。
- 测速 `tools/bench_qwen.py`（预热 + 中位数 + `memory_reserved`）；后处理用未像素化 `__raw`；批量成本 `t/(1-(1-p)^k)`。
- **立绘取景默认全身**（官方缺 framing 时是 `cowboy_shot`，我们兜底 `full_body`）；半身图切精灵帧缺脚。
- **`prompt_language` 默认 zh**：结构词/取景/属性值/画质词中文化（`workflows/textlabels.py` → `labels_zh.json`）。**画风段与官方数据例外**：`charactergen.style_block()` 的 Visual description/Reference 与 `race_features` **英文原文照搬**（是数据不是结构词，硬译会丢特征，曾导致 Q 版不像 Q 版）。
- **画风中立（别改回去）**：公共提示词**一个风格字都不许写**。立绘 `app.character_generate_api` 在 `params.update` 后**强制 `preset="none"`**（`txt2img` 默认 `sprite`，负向词含 `photo, 3d render`）。PE 请求**逐行删 LoRA**（别 rsplit）、成功后补回画风段、失败不重复追加；`custom_style` 在 `QI2_LITERAL_FIELDS`；英文模板一字未改。
- 译表真源 `web/src/labels.ts` → `python tools/build_labels_zh.py`；`test_backend_labels_json_in_sync` 拦没重导（同测试校验 `character_tags.json` 74 条）。
- **PE 只在点「✨ 润色提示词」时跑**（`/prompt?enhance=true`），拼 prompt 一律 `enhance=false`。坑：① 官方要回 `{"rewritten_prompt":…,"wh_ratio":…}` 必须解析；② `enable_model_cpu_offload()` 钩子只搬模块不搬输入，PE 期间要**摘钩子** + 张量同设备自检，否则 `wrapper_CUDA__index_select`。
- **offload 链易碎**：`_all_hooks` 空 / 组件缺 `_hf_hook` → 退回 cpu，报 `Cannot generate a cpu tensor from a generator of type cuda`（**别在 PE 里调 `maybe_free_model_hooks()`**）；双保险 `generate._exec_device()` + `Engine._heal_offload()`。

## 画布 / UI
- 节点「执行中」只看 `job.status` queued/running；**任何异常路径都要退出执行中**（提交超时、轮询连续 3 次失败、30 分钟看门狗、worker 兜底、404 文案）。
- 「角色图」统一走 `imagePathOf(node)`；`onConnect` 白名单 `['poseclone','spriteswap','posebatch']`，新节点必须加。
- 侧栏 3 tab（节点/画布/资料库）+ 4 个 `.rail-sec`，只用 `.is-hidden`+CSS 切；`NODE_ENTRIES` 顺序别重排；**设置只长在顶栏 ⚙ 弹窗**。
- `NodeView='expanded'|'image'|'result'`：出图自动 result，参数区 CSS 隐藏但 DOM 保留。弹窗走 Portal；灯箱 wheel `{passive:false}`；别用 `opacity` 做低调按钮。
- `--rail-w:252px` 与 `App.tsx`、`store/projects.ts` 的 `x:300,y:96` 同步；`SpritePreview.onMeta` 必须 `useCallback`。
- 外观只改 `styles.css` 末尾「全站样式统一层」；`:root` 只在顶部一次；像素风第一版被否**别加回来**；画布网格 `GRID_BASE=32` 两边同步。
- **column 弹性容器里别用 `flex: 0 0 Npx` 定宽**（主轴纵向）；宽度只能 `width` + `flex: none`。
- `dragId` 是 React state：`dragstart` 后**必须等一次重渲染**再 `drop`。

## 全局设置 / LoRA（SQLite 权威）
- `metadb.settings` + `GET/PUT /settings`；白名单 `base_model/matte_backend/loras/fast_mode`；启动 `_restore_global_settings()`；`submitMatte` 必须带 `matte_backend`。
- **勾选 LoRA 只改设置、绝不加载模型**：`/loras/apply` → `Engine.set_user_loras()`，`load_lora_weights` 推迟到下次 `txt2img_pipe()`；前端先乐观置 `activeLoras` 再 PUT，失败回滚。
- `active_loras()`：`_loaded_sig is None` 时**只按规格报**。导入两步：`/loras/inspect` 只暂存只读检测、**绝不写 `models/lora/`**；`/loras/commit` 仅 `convert=true` 才转换（ComfyUI 格式直挂会**静默不生效**）。
- 抠图会话按任务释放：`matte.release_ai_sessions()` 在 `app._run` finally 一次，`MEOW_MATTE_KEEP=1` 常驻；别每帧重建 ONNX。

## 姿势 / 人偶 / 动作导入
- 「人偶姿势」是官方 PoseStudio **内嵌 iframe**（`studio.html?embed=1`），自写伪 3D 已删；「放大」只能改容器 CSS 为 `fixed`（移动 iframe DOM 会重载 37MB 模型）；`capture()` 前关 skydome。
- **内嵌基线是 Utils 0.7.1/0.7.2，别整包升 0.7.3+**（新版 `core.js` import 我们没有的 `normalizeCaptureLens`）。
- 多角色同框：`vnccs-pose/scene.js` 唯一状态机（`{type:'scene',action}` + `sceneState` 回执），状态只在 iframe 一份；`commit=false` 不写 project.json；`web/public` 下 ESM 一律 `.js`；`app.py::PoseSetFrameReq` 必须显式声明字段（Pydantic 吞多余字段）。
- **别用图像识别数主体**：导出走 `/upload?subjects=N` 写 `.png.json` sidecar，`subjects_for()` 读、老图按 1；多人时提示词改复数并去掉 `two characters` 负向词（`workflows/posesubjects.py`）；官方 `_VNCCS_TEMPLATE` 一字未动。
- 动作导入：本地 three r160 + importmap，FBX 4 依赖缺一回落 CDN；脚底对齐两层（世界 resetPose + `alignFrames()`）；抽帧落 `assets/projects/<项目>/poses/`，同名 409；**失败要回滚**（`pose_guard.js::withPoseRollback` 包 `applyTime`，`animSample` 整批失败从抽帧前快照还原）。
- 「从图片提姿势」用 MediaPipe `pose_world_landmarks`；`COCO_TO_OP 13→L_ANK` 必须双射。

## 方向帧 / 动作模板 / 试玩台
- 视角模式一等概念：`topdown-4/topdown-8/side-2`，表在 `store/playtest.ts` 与 `workflows/directions.py`（`playtest.py` re-export），**别写第三份**。
- 链路：`set.json.direction` → `posegen` → `spriteatlas` 产 `<组>@<方向>` → `godotexport` ≥2 方向才多动画 → 试玩台按朝向选帧；老数据每环退回原行为。
- `entry.js` 的 `animSample` 可选 `directions`：方向 × 帧双层循环，每方向单独 `alignFrames()`；不传时老路径一字不变。
- 动作模板 `workflows/motiontemplates.py` 唯一定义；内置三套是**代码常量不落盘**，用户模板落 `library/motion_templates.json`；`normalize()` 幂等；`MotionTemplatePanel.tsx` 用独立 `.mt-*` 类名。
- 试玩台播放与位移**共用一个 rAF**；`GET /characters/{name}/playtest` 只读扫盘。

## 角色中心（别写回四步工作台）
- 视图 `library / stages / stage / sequence / parts / riganim / pending`。默认全屏库 → 生命阶段列表 → 角色设定；「拆部件」「骨骼动画」「待处理」同级。UI 上「项目」叫**画布**（内部 `/projects`）。
- **层级（用户逐条钉过）**：顶部只有**角色名 + 更小描述 + 修改 + 编辑时间**，不许出现「当前服装/画风」行；改名/改描述/删除/只读档案都在「修改」弹窗。导航**五项：生命阶段 / 序列帧动画 / 拆部件 / 骨骼动画 / 待处理**（序列帧动画排在「角色设定」后、「拆部件」前）；「角色设定」属于某阶段、不是同级入口（进入后第一项文字变「角色设定」并高亮，正文不许再有大标题）；面包屑只在进入后出现。
- **列表页无右栏与拖动条**（`.is-stages` 单列）；每行以立绘为主（58×82、最多 5 张）；`cc-main` padding 0，统一层把 `.cc-topbar/.cc-crumb/.cc-viewnav/.cc-panel/.cc-next/.chars-err` 统一到 **24px**。
- 右栏 `PortraitPane` **固定分页**（每页 8 张 + 页码，数量变了回第 1 页），**不许「显示全部」**；`TaskBar` 排在画廊**之前**且 `sticky top:0`；画风库是弹窗（点卡片选中并关窗）；右栏可拖宽 `.cc-split` + `--cc-side-w`（默认 420，`localStorage meow.cc.sideW`）；预览图 flex 居中。
- **生命阶段**：`workflows/stages.py`；顺序/描述存角色 `config.json`；`active_stage` 的 info 始终是顶层 `character_info` 的**镜像**（出图 API 不用带 stage）；`GET/PUT /characters/{n}/stages` 全量提交；立绘按 `Portraits/s_<id>/` 分目录、默认阶段用老位置；删阶段**不删文件**；老角色自动补「初始设定」；导入设定**只复制 info，不导立绘**。

## 序列帧动画 tab（画布两个节点搬进角色中心）
- 入口 view=`sequence`（`.is-sequence` 单列、不渲染右栏），页面 `components/characters/SequenceAnimPanel.tsx`：把画布「3D 人偶」（`poseedit`）与「参考图批量帧」（`posebatch`）整块搬进来。**两块常驻 DOM**（只 `.is-hidden` 切），37MB iframe 不能卸载。
- 归档走 `workflows/sequence.py`：`assets/characters/<角色>/Sequence[/s_<阶段>]/`（`studio.json` + `mannequin_*.png` + `poses/` + `renders/` + `exports/`）。`poseset._root(character=…, stage=…)` 把根目录从 `assets/projects/<id>/` 切过来，`character/stage` 关键字参数一路透传到 `posebatch / godotexport / motiontemplates`（**别再复制一份实现**）。
- 批量生成用 `generate.register_scope_root()` 把 scope 临时指到阶段目录下 `_tmp/`，`finally` 摘掉并清空；阶段目录不留中间文件。
- 前端 `PoseScope = string | {character, stage}`（老调用方传项目 id 字符串不变）。`PoseBatchPanel` 只在挂载时读一次 `node.params` → 种子必须在挂载前灌好（`seedReady` 门控）。切 tab 后要滚回顶（jsdom 无 `Element.scrollTo`，回退 `scrollTop`）。
- 13 条 `/characters/{name}/sequence*` 路由，`run` 复用 `workflow="wpcbatch"`。

## 骨骼动画编辑器
- 顶栏一级视图三个：画布 / 角色中心 / 骨骼动画（`App.tsx` 的 `view`）。工作台 `components/RigStudioPage.tsx`；角色中心「骨骼动画」tab 保留按阶段归档的产物页，两者共用后端；加 tab 同步 `web/smoke.mjs` 的「顶栏三个一级视图」断言。
- **别再改回独立应用**：编辑器**内嵌**在当前项目的 React 页（`components/characters/rig/`），不是 iframe / 新标签页。`web/src/rig/core/` 是 rigby2d `src/core` 逐字副本，`io/` 是不依赖 UI/Tauri 的纯逻辑，`facade.js` 聚合 + `facade.d.ts` 宽松声明；骨骼/动画/约束必须调 core 真实 API。产物由 `workflows/riganim.py` 按 `RigAnim[/s_<阶段>]/` 归档，发布按 `riganim.EXPECTED_FILES` **七个文件全查**。
- 上游契约：画质档 0/1/2 → low/medium/high（**别再乘 2**），编码循环排空 `encodeQueueSize>8` 且 `color.sort(time)`；`lipsync.readTiming` 只认 Papagayo / Rhubarb(TSV|JSON) / ElevenLabs / 词时间 JSON，返 `{kind,words}` 或 `{kind,cues}`（cues 直接用），SRT 与「单词 起 止」在 UI 层归一化成词时间数组；`retarget()` 自己已写进 `target.animations`（report 字段 `keys/unmatched/skipped`，别 push 两次）；`Pose.apply` 的 rotate/translate 加 setup、scale 乘 setup。
- 场景 / 演员 / 摄影机：演员靠舞台锚骨落位（`actorPlacement`），位置/朝向/缩放改**锚骨 setup**、翻转只改 `actor.flip`；演员图以 `@<id>/<name>` 进共享 Images，读演员工程必须用**独立 Images** 再搬（`openWorkfile` 会 `images.clear()`）；有演员时按舞台 `drawOrder` 逐槽插画，`sceneBounds` 把演员顶点并进取景；`Sel` 有 `{kind:'actor'}`，点选 `scenePose.pickActorIndex`（region 多边形 / mesh 包围盒 / alpha≤0 跳过 / 后加的在上面）；**切文档必须 `actorRef.clear()`**；摄影机 = 舞台里叫 `camera` 的顶层骨，镜头运动 = 给它打关键帧；**深度语义别写反**：`shift = 相机位移 × (1 - focal/(focal+depth))`，深度 0 = 不跟镜头。
- **物理要逐帧推进**：`solvePhysics` 只在 `physicsDelta > 0` 时仿真；`Map` 常驻跨帧复用，每帧把真实 dt 累进 `physicsDelta`，暂停清空；离屏渲染从第 0 帧预热到 startFrame。**按 t 直接采样 + 每帧新建 Pose = 摆动约束全静止**。
- 验证四件套：`tools/verify_rig_{alignment,motion,scene}.mjs`；像素 `cd web && npx vite build --config ../tools/rigshot/vite.config.mjs` 后 `node tools/rig_pixels.mjs`；真指针 `node tools/rig_drag.mjs`（隔离后端 `tests/_out/rig_ui/assets` + 真 Edge）。demo 的 `project.json` 是工作文件格式，别喂 `io.openProject`。像素比对两个坑：背景网格是**屏幕空间**辅助线（先关）、快照要在「已选中」之后拍。

## 立绘工具箱 / 拆部件
- 五动作：扣背景 / 完美像素化 / 换装 / 换表情 / 删除。卡片外层是 div（按钮不能嵌套），`.cc-portrait-view` 看大图；组件 `characters/PortraitTools.tsx`。
- 扣背景 / 像素化走 `POST .../portraits/process`（`matte.matte_auto` / `pixelfy.refine_batch`，**确定性不跑模型**），**永远生成新的一份**；`portrait_variant_path()` 后缀 `_bg`/`_px`（**别用 `_matte.png`**，`list_portraits` 会当副本跳过）。
- **像素化输出尺寸（别退回去）**：`pixelfy` 出的是格子小图（768×1024 → 48×64）。必须①按**长边**解释目标尺寸、②默认 `output=restore` 用 `resize(src.size, NEAREST)` 放大回原尺寸（硬边、不造新色、alpha 不半透明），`output=native` 才留小图。
- 换装 / 换表情走 `POST .../portraits/variant`，**必须用专用链路、别借 w2**：换装 → `wpo`（挂 `Engine.CLOTHES_LORA`）；换表情 → `wpe`（`detect_face + crop_generate + merge_emotion_rgba`，**只改脸、衣服逐像素不变**）。换装六字段 `fields` + 外套 `Dress the character:`，负向 `CLOTHES_NEGATIVE`；默认 25 步 / cfg 3.0。
- **表情多选 / 批量**：`emotions` >1 → `wpebatch`（一个 job、逐张 publish，批量 params **没有 `publish`**）；去重 + 上限 12；单张失败进 `failed` 不连累整批；删除必须 `confirm=true`。
- **拆部件页**（`.is-parts` 单列、无右栏）：`workflows/partsplit.py` —— See-Through 调 pixelforge `tools/see_through_export.py`（独立 venv torch 2.11+cu128 + 17G 权重，`MEOWCUTE_SEETHROUGH_DIR` 覆盖，1024px/25 步 147s / 20 层）、爆炸图 `rigbuild.split_image`（纯 PIL）；产物落 `Parts[/s_<id>]/<时间戳_方法>/`，manifest 记 `stage`；`GET /parts?stage=` 只列该阶段，删除要 `confirm=true`；layers **从底到顶**、画布像素坐标；PSD 由 `tools/parts_to_psd.py` 写，**层名必须 ASCII**（mac_roman）。

## 资料库 / SQLite
- `workflows/metadb.py`（sqlite3 / WAL / `library/meta.sqlite3`）是角色与画布元数据权威：tags、时间、自定义标签、备注、任务历史（`GET /character-tasks` 有后端、**展示 UI 未做**）、封面。

## 收尾状态
- 未做：碰撞盒、角色中心「一键按模板补方向帧」、抽帧面板下拉真机验证、双人参考图 GPU 一致性验证、pixelforge 的 Qwen 爆炸拆解图路线。
