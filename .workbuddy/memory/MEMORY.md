# 项目长期约定（2d-game-meow）

## 角色中心：接手先看这两份文档

- `docs/角色中心-VNCCS移植评估.md` —— 功能全表、模型清单、已拍板的取舍（"为什么"）
- **`docs/角色中心-剩余工作与做法.md` —— 剩余待办的开工手册（"怎么做"）**，
  每项带源仓库 `file:line`、参数默认值、prompt 原文、落点文件、验证方式。
  **T1~T5 全部完成**（表情 / 看图反推 / 一句话造角色 / 换装+服装向导 / 多姿势批量），
  外加一键出全套编排（`workflows/pipeline.py`）与一键建角色（`/characters/from-image`）。
  手册 §3~§7 都标了"已完成"，一节节写了实现依据与取舍，改之前先读那节。
  **§9 是「官方有、我们没做」的逐条核对结论**（2026-11-01）——那份清单大部分已不成立，
  别再当欠账重做。**§2 待办总表里只剩 T7（放大，建议跳过）和 T8（图集增强，用户暂缓）**。

移植总规矩：**以 `D:\code\gen\others\ComfyUI_VNCCS-main` 为准，不自己写自己猜。**
明确不做：Control Center、8 方向帧（官方没有这功能）、SeedVR2 放大（官方支持
`upscaler.mode="off"`）、Flux Klein / MiniMax H3 / Anima / SDXL / FaceDetailer / SAM。

## 底模：只有 Qwen-Image-2.1（2026-10-30 起 SD 代码已删）

`pre-sd-purge` tag 是删除前状态；SD 权重在 `models/_archive/sd/`。
不要再往 engine/generate 里加 StableDiffusion 相关代码——SD 管线方法已经不存在了。
细节见 `docs/PLAN-QWEN-ONLY.md` §6（删了什么）与 §4（做不到的九件事）。

## 测试策略（用户明确要求，2026-10-28）

**不要动不动跑全量测试。** 全量 260 个用例约 150s（含两次 Qwen 管线加载），
只在合并/发版前跑一次。

按改动选子集（详见 README「测试」章节的表）：

| 改动 | 子集 | 约 |
|---|---|---|
| engine.py | test_models test_vramopt test_stability | 110s |
| generate.py | test_qwen_workflows test_denoise test_paint_path | 47s |
| presets.py | test_presets test_quality | 10s |
| matte.py / refprep.py | test_matte test_refprep | 20s |
| postprocess.py / pixelfy.py | test_postprocess test_pixelfy test_sheet | 15s |
| app.py / assetdb.py | test_api test_review_p2 | 15s |
| characters.py / charactergen.py | test_characters test_char_poses | 9s |
| emotiongen.py（表情批量） | test_emotiongen（含一条真 YOLO） | 6s |
| qwen_llm.py（本地 LLM） | test_qwen_llm（真推理要 MEOW_SLOW=1） | 5s |
| clothesgen.py（换装） | test_clothes | 5s |
| posegen.py（多姿势批量） | test_posegen | 5s |
| spriteatlas.py（图集导出） | test_spriteatlas | 4s |
| labels.ts（界面译文） | test_labels | 1s |
| projectio.py / paths.py | test_projects | 5s |
| 前端 web/src | `cd web && npm run smoke`（568 条，纯 mock，不用 GPU） | 60s |
| workflows/library.py / poselib.py | test_library | 1s |
| app.py 加了端点 | test_api（含 library 相关时加 test_library） | 75s |
| tools/ docs/ README | 不跑 | — |

**冒烟跑的是 `dist/`**：改完前端代码必须先 `npm run build`，直接 `node smoke.mjs`
会拿旧产物跑（实测过一次：侧栏名字改了但断言全绿，白高兴）。

**iframe 内部 jsdom 验不到**（PoseStudio 那块）。动了 `web/public/vnccs-pose/**`
或人偶编辑器接线，要用 `python tools/verify_pose_editor.py` 真机验
（无头 Edge + SwiftShader，跑一遍约 1 分钟）。

## 前端真截图（脱离 GPU，2026-11-01 起）

要截真实节点界面（不依赖后端/GPU，纯 DOM），套路：

- 工具放在 **`tools/shot/`**（隔离装 `puppeteer-core`，不污染项目 `package.json`），
  驱动系统 Edge：`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`。
- **`npm run build` 会触发批量删除守卫**（vite 清空 `dist/vnccs` 约 528 文件 >50 上限被拦）。
  改 `vite build --outDir dist_shot`（全新空目录 → emptyDir 删 0 个，不触发守卫），截完即删。
- 加载 `index.html?demo=1` 铺一个 gen 节点；要哪种节点就点侧栏对应 `.rail-item` 里的
  `.rail-add`（"放到画布原点"，实际落可见区中心并错开）。
- **节点卡片选择器是 `.node.gen`（外层），不是 `.node-kind.poseclone`** —— 后者只是卡头里
  的「角色迁移」文字 `<span>`，拿它截只能截到四个字。
- 高级项在 `details.adv`（「更多设置」），截图前把 `det.open=true` 才能看到占高/参考模式/
  底色/透明背景。产物落 `docs/img/`。

`tests/test_generate.py` / `test_clip` / `test_controlnet` / `test_lora` /
`test_train` / `test_review_fixes` / `test_review_p1` / `test_fetch` **已删除**
（SD 时代的模块）。别在子集表里再写它们的名字。

## 姿势编辑：只有官方 3D 人偶（2026-11-01 起）

「骨骼编辑」节点已改名**「人偶姿势」**，正文是官方 VNCCS PoseStudio 的**内嵌** iframe
（`studio.html?embed=1`）。自写的 2D 伪 3D 骨架编辑器已删除，**别再往回写**。

- 摆的是 MakeHuman 网格 + 53 根骨（官方两骨 IK + 极点约束）；朝向是相机机位绕 Y
  渲染，不是 2D 投影假装 3D。
- 产物只有白底灰人偶 PNG（`node.image`）；`node.mannequin` 只为读老工程保留。
- 存档 = 官方 `getPose()` JSON。存 18 个 2D 点做不到精确还原（深度每次都要重新猜）。
- 「放大」只能改容器 CSS 成 `position:fixed`：**移动 iframe 的 DOM 位置会让浏览器
  重载它**，37MB 人偶模型重新解算一遍。
- 官方 `capture()` **不隐藏方向天穹**（`VNCCS_DirectionalSkydome`，彩虹网格球），
  导出前必须临时关掉 —— 否则每张条件图背景都带一层彩色网格。

## 目录约定（2026-10-31 起）

- **路径只有一处定义**：`workflows/paths.py`。别在别的模块写 `ROOT/"assets"`。
  可用 `MEOW_ASSETS_DIR` / `MEOW_TEMPLATES_DIR` 覆盖。
- `assets/` = **项目资产**（用户要交付的东西）：索引、projects/、uploads/、
  可直接当 W4 输入的骨架帧目录。不要往这里丢测试产物。
- `tests/_out/` = **测试与验证产物**：pytest 期间的资产根（conftest 在 collection
  阶段设 `MEOW_ASSETS_DIR`）、验证脚本输出、历史遗留的调试图。
  `tools/` 下的验证脚本输出走 `paths.tool_out()`。
  `tools/pose_from_image.py` 是用户工具，产物仍放 assets（不算测试产物）。
- 模板在 `templates/*.mproj`，**不在 assets 下**——它是跨项目共享的用户资源。

## 项目存储（2026-10-31 起）

- **项目内容只在服务端磁盘**：`assets/projects/<id>/project.json`。
  **浏览器里不留任何项目内容**。
- 写入走 `workflows/projectstore.py`，**原子**（临时文件 + replace）。
- 改前端时别再引入 localStorage 项目缓存——用户明确要求「不存浏览器」。
- 项目 id 与后端 scope 必须是**同一个**：导入时后端按 id 解包素材、
  前端若另起 id，素材和项目定义就分家了。

## 用户资料库：磁盘上，localStorage 一个字不留（2026-11-01 起）

**浏览器存储里不许有任何东西** —— 用户明确要求。提示词收藏 / 姿势库 / 界面偏好
全部走 `workflows/library.py` → `library/{prompts,poses,prefs}.json`
（与 `templates/` 同级，都是跨项目共享的用户资源，备份 = 拷贝目录）。

- 前端 store 只放纯函数（去重键 / 搜索过滤 / `readLegacy*`），
  增删改一律发请求、用**服务端返回值**覆盖本地。
- 迁移顺序**先 peek 后删**：`readLegacy()` 只读不删 → `POST /library/seed`
  （服务端只补不覆盖）→ 确认收到才 `clearLegacy()`。读出来就删的话，
  请求一失败这批收藏就彻底没了。
- `library/` 是**用户攒的**；随项目发行的官方 VNCCS 姿势库（191 个）在
  `web/public/vnccs-pose/PoseLibrary/`，别混在一起 —— 后者是静态资源。

## 异步任务的失败处理

- 节点「执行中」只看 `job.status` 是 `queued`/`running`。
- **任何异常路径都要让节点退出执行中**：提交有 30 秒超时、轮询连续 3 次失败
  判定 failed、30 分钟看门狗。后端 worker 兜底 try、`/jobs/<id>` 404 带明确文案。
- 以前 `if (r.status !== 'fulfilled') return n` 会让节点永远卡住——别写回去。

## 假对象测试会骗人（2026-11-01 review 的教训）

**单测全绿 ≠ 功能对。** 两个真机才暴露的 bug：

- `qwen_edit_image` 返回 **RGBA**，而所有 `fake_edit` 都返回 `Image.new("RGB", ...)`
  → 贴回时 `channels=4`，alpha 被混进源图，透明背景被 255 按羽化权重盖掉。
- 假 `keyer` 直接按颜色判绿，没模拟"透明像素 RGB=0（黑）" → 真实抠像把黑块
  当前景留下，整图失去透明。

规矩：**fake 必须模仿真实返回的类型**（RGBA 就该是 RGBA）；动了 alpha/通道/
颜色空间的逻辑，**必须真机跑一次**再算完。

## 度量约定（本项目踩过坑）

- **任何速度/显存结论必须预热 + 多次取中位数**，单次调用波动可达 70%。
  用 `tools/bench_qwen.py`。
- 显存看 `memory_reserved`（真实占用），`memory_allocated` 会低估。
  **Qwen 走 cpu offload，加载完 allocated 本来就是 0**，别拿它判断"加载没生效"。
- 测后处理要用**未像素化的 __raw 输入**，成品会命中网格检测，耗时虚报 8 倍。
- 批量推理的成本公式：`t / (1-(1-p)^k)`，t 已是 k 张的总时间，不要再乘 k。

## 产线判据

- 可交付率 = 单主体 ∧ 无贴边 ∧ 前景占比 5%~60%（精灵帧要对齐脚底，三条都得满足）。
- Qwen 768²/24 步：单帧 24~28s、峰值 11~15GB；低于 16 步出糊块（PLAN §2.6）。
- 逐帧路线的短板是**姿态贴合**不是帧间一致性（PLAN §2.8）：腿部常被参考图站姿覆盖。
- 白毛/白甲角色别用白底抠图——换 `bg=dark`，或出图时直接勾透明底。

## 抠图必须是用户的选择（2026-11-02）

**默认 `matte=False`，任何"偷偷抠"都是 bug。** 用户原话：「我没选抠图，你为啥要主动扣？」

排查要查四处，少一处就会复现：
1. `generate._postprocess` 的透明分支（曾用 `matte_auto` 兜底全雾 alpha）
2. `pipeline.PIPELINE_DEFAULTS["matte"]`
3. 前端硬编码（曾在「一键出全套」写死 `matte: true`）
4. `app.py` 请求模型的 `matte: bool = True` 默认值

判断"有没有抠"看产物 meta 的 `matte` 字段，不要靠猜。
配套：不抠就必须走原生透明（`charactergen.normalize_background_color` 默认
`Transparent`、`posegen`/`emotiongen` 背景默认空），否则出来是绿底图。

## 透明底的白边：RGB 是白底预乘的（2026-11-02）

模型出的透明 PNG，半透明边缘的 RGB 是 `前景*α + 白色*(1-α)`。实测过渡带亮度
237.8、实心区 206.4，**87% 的过渡像素接近纯白** —— 只保留过渡带（不像素化时
必须保留）就会在深色背景上留一圈白边。

必须做 **unpremultiply**（`generate._defringe_white_matte`）：
`C_fg = (C_obs - 255*(1-α)) / max(α, 0.25)`。

**最坑的一点**：`alpha_trust` 只看 alpha 数值分布，**不看 RGB**。所以
`alpha_reused=True`（alpha 结构可信）时 RGB 照样可能是白的 —— 复用 alpha
时也要过 defringe。真机上出问题的图恰好就是这一支。

## 「像素感」的四个来源：不只有 pixelate（2026-11-02）

用户报「没勾像素化还是有像素感」。四处都会造成，后三处更隐蔽：

1. **契约词**：`SPRITE_SUFFIX` 带 `16-bit pixel art, hard outline, flat colors`，
   负向词压 `anti-aliasing`。`pixel<=1` 时用 `presets.build(..., pixelate=False)`
   → 换成 `SPRITE_SUFFIX_PAINT` / `SPRITE_NEG_PAINT`。
2. **alpha 硬切**：`_postprocess` 里 `_harden_alpha(240)` 把过渡带砍成 0/255。
   像素档要硬边，插画档只能 `sanitize_alpha`（<16 清零、>240 拉满、中间留着）。
3. **形态学开运算**：`despeckle(denoise_iters>0)` 削掉 <3px 突起（发丝/挂饰）。
   只在像素档开。
4. **色数量化**（2026-11-02 补，**白边的真凶**）：`quantize_palette` 里那句
   `alpha.point(a > 127 ? 255 : 0)`。白底预乘的过渡带（RGB 220~240、α 中段）
   被硬切成"不透明近白" → 深色背景上就是实心白边，发梢也被削成硬色块。
   实测一张真产物：5708 个半透明像素全消失，其中 1556 个变成不透明（亮度 223）。
   **量化默认不许动 alpha**；硬边由 `pixel > 1` 决定（`binary_alpha=True`）。

判据统一为 **`hard_edge = pixel > 1`**（2026-11-02 起去掉 `or palette > 0`）：
只填色数不再构成像素化 —— 契约词、量化、开运算、硬切全部只看 pixel。
`tests/test_paint_path.py::test_palette_alone_is_not_pixelation` 钉住这条。
**别再用"看数值有没有 pixel"判断是否像素化了。**

**UI 层面色数/像素块已经删掉**（用户要求学业界：Midjourney/即梦那类产品
不让人填色数和像素块）。画风由「产出风格」一档决定，`custom` 只作旧存档
只读回显（disable）。别再把手填入口加回来。

## 前端小部件踩过的坑（2026-11-02）

- **`.mini` 是 28×28 图标按钮**（✕ / ⤢ 这种单字符用）。文字按钮另起类，
  否则两个字会竖排换行并溢出边框。
- **别用 `opacity` 做"低调按钮"**：连图标带边框整体半透明，近黑画布上等于隐形。
  要实心底 + 灰白边框，hover 换强调色。
- **数字输入框别用 flex-grow 比例分**：会被压到 30px，`768` 直接看不见。
  用 `flex: 0 0 52px` 固定，剩余空间给旁边的下拉。
- 覆盖层弹窗（提示词放大、看图）**走 Portal 到 body**：节点卡片有
  `overflow:hidden` + `transform`，就地渲染会被裁掉且跟着画布缩放。

## 界面中文化：只翻译显示层，别改标识（2026-11-01 起）

界面上的 `Naked` / `angry 😠` / `Shonen Anime` / `Ballet` **不是文案，是数据标识**——
同时是磁盘目录名、后端接口键、以及要写进 prompt 的内容。**改一个就全盘失效**。

规矩：**值一律传原样英文，只在渲染处显示中文。**

- 映射表全在 `web/src/labels.ts`（约 1000 条）。查不到就回落到原值
  （用户自建的服装名/姿势名本来就该原样显示）。
- **官方 JSON 一个字不改**（`emotions.json` / `character_styles.json` /
  `character_presets_v2.json` / `PoseLibrary/`）—— 改了就不好升级官方版本。
- `tests/test_labels.py` 逐条核对覆盖率：官方加了新条目而我们忘译，测试直接红。
  已抓到过 `sitting`（小写变体）这种漏。
- **漏接比映射表更容易出错**：映射做对了，渲染处漏接照样露英文。
  改完必须**重新扫一遍渲染处**（`CharactersPage` / `TagPicker` / `LibraryDrawer` /
  `SpritePreview` / `App`），不能只验映射表。
- 表情 emoji **别手抄**（`😮‍💨` 带 U+FE0F 和 U+200D），从原 `key` 末尾用
  `\p{Extended_Pictographic}` 正则摘。
- `neutral` 是**我们这边的默认表情**（官方库从 angry 开始，没有这一条），
  官方对齐测试测不到，得单独断言。
- 要保持英文的：提示词示例（`blonde hair, twin tails`）、官方固定指令
  （`Dress character: White underwear`）、错误码（`MODEL_MISSING` ——
  前端和测试靠它做字符串匹配，翻译会断契约）、文件路径与目录名。

## 文档/科普：归开源 upstream，别宣传商业站点（2026-11-01）

写原理/科普文档、引用截图时：**只归到开源上游**（AHEKOT/ComfyUI_VNCCS、
AHEKOT/ComfyUI_VNCCS_Utils、MIUProject/VNCCS_PoseStudio_QI2.1），
**不要写 runcomfy.com 这类商业托管/二次发布站点**——用户明确要求不替商业公司宣传。
讲解文案可直接用上游官方描述（中文/英文皆可），截图用上游官方仓库或用户贴的图，
不要链商业页面。
