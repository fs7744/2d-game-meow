# 2D Game Asset Pipeline（Qwen-Image-2.1 本地版）· 完整手记与变更记录

> **这份文件是项目的完整技术手记**（原名 `README.md`）。
>
> - 想先知道"这是什么、怎么跑起来" → 看 [README.md](README.md)
> - 想知道"为什么这么做、实测数据是多少" → 就是本文
>
> **代码注释里写「见 README「底模」」之类，指的都是本文的章节**——
> 改名时没有去改那些注释（几百处，改了反而容易断链）。

单卡 RTX 4090 上的本地 2D 游戏资产生成工作台。**当前运行态是 Qwen-Image-2.1 单底模**：
文生图 / 图生图（指令改图）/ 角色迁移（双图条件）/ 原生透明底 / 像素化 / 精灵图打包。
代码化推理（diffusers，不依赖 ComfyUI）+ FastAPI 服务。

> **SD 时代已归档**：SD1.5 / SDXL / SDXL 动漫 / Refiner 底模、ControlNet、IP-Adapter、
> AnimateDiff、27 个 SD LoRA 全部移到 `models/_archive/sd/`（**没有删**）。
> 完整基线备份在 git tag `sd`。恢复方法见「底模」章节。
> SD 代码路径仍在仓库里，所以回滚只是搬回权重 + 改一个常量。

设计基线来自资料库文档 `RTX4090_2D_Game_Pipeline` v1.1；本仓库最初是全栈 SD1.5，
现在运行态换成 Qwen——两代的实测结论与取舍见文末「版本选型」与「Qwen 能做什么」。

---

## 2026-11-09 · 立绘悬浮工具箱（扣背景 / 完美像素化 / 删除）

用户要求：立绘图上 hover 出工具箱；两个处理都在弹窗里给参数，处理完有
**滑动比对**看差异；处理结果是**新的一份**而不是改原图；删除要确认防手抖。

### 后端

新增两个端点（都走 `characters.portraits_dir()` 做 realpath 校验，挡 `..` 与软链）：

- `POST /characters/{name}/portraits/process`：`op=matte` 走
  `matte.matte_auto`（方式 / 容差 / 去白边 / 补洞；「自动」跟随全局抠图设置），
  `op=pixel` 走 `pixelfy.refine_batch`（目标尺寸 / 色数 / 网格吸附 / 每格取色 /
  量化前增强 / 透明背景 / 描边重建）。两个算法都是**纯确定性后处理，不跑模型**。
  结果返回 `{before, after, path, name, meta}`，前后 url 供弹窗比对。
- `POST /characters/{name}/portraits/delete`：必须带 `confirm=true`，后端再验一次。

`characters.portrait_variant_path()` 生成**同目录、不覆盖**的新文件，名字带
`_bg` / `_px` 后缀（刻意不用 `_matte.png` 结尾——`list_portraits` 会把它当
publish 时生成的副本跳过）。文件名带 4 位随机码，同一秒连续派生两份也不会撞。
新文件落在原图所在目录，所以自动继承同一个生命阶段 / 服装 / 表情。

### 前端

- `PortraitPane` 的卡片外层由 `<button>` 改成 `<div>`（按钮不能嵌套），
  里面 `.cc-portrait-view` 负责看大图，`.cc-portrait-tools` 在 hover / 键盘
  聚焦时浮出三个按钮。
- 新组件 `characters/PortraitTools.tsx`：`PortraitProcessDialog`（参数表单 +
  处理并比对 + `CompareSlider` 拖动分割线 + 改参数重做 / 放弃这份 / 完成）与
  `PortraitDeleteDialog`（写明不能撤销，取消与确认分开）。弹窗走 Portal 到 body。
- 「放弃这份」和「改参数重做」会自动删掉刚生成的预览版本，不留垃圾文件。
- `CharactersPage` 传 `onChanged`，处理/删除后只刷画廊，不重启整页数据流。

### 验证

`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 813/813**
（新增用例 65：工具箱三项 / 参数弹窗 / 滑动比对拖动 / op 请求体 / 删除取消不发请求 /
确认才带 `confirm:true`）、新增 `tests/test_portrait_tools.py` **7 例**（原图字节不变、
新文件不覆盖、抠图角透明主体保留、像素化 16×16、越界路径 400、删除需确认）、
后端回归 **112 passed / 5 skipped**。截图工具同步到新导航并新增 hover 截图
`23e-portrait-tools`。

## 2026-11-09 · 角色中心：修掉顶部贴边

用户反馈「左边太靠边了，不好看」。根因：`.chars.cc .chars-detail.cc-main` 是
`padding: 0`，而顶部栏 / 面包屑 / 导航都是它的**直接子元素**，所以这三行完全贴着
容器左边；下方 `.chars-body.cc-panel` 反而有 18px 左右边距，形成一条参差的左边界。

统一层末尾把 `.cc-topbar` / `.cc-crumb` / `.cc-viewnav` / `.cc-panel` /
`.cc-next` / `.chars-err` 的左右边距全部对齐到 **24px**（border-bottom 仍在
最外圈，保持通栏分隔线）。只改统一层，不动中间的历史定义。

验证：`tsc -b` / `vite build` ✓、`check-hints` 34/34、
**`node smoke.mjs` 792/792**（新增静态断言钉住 24px 对齐），
截图 `tests/_out/ui_review/pad24/23-cc-workspace.png` 里返回按钮、面包屑、
导航按钮与「生命阶段」标题落在同一条左边界上。

## 2026-11-09 · 角色中心二次修正：头部层级 / 导航 / 阶段列表以立绘为主

用户对照两张界面截图逐条指出的问题，全部按「层级」而不是「藏起来」来修：

1. **顶部只有角色名 + 更小的描述**，最右是「修改」与编辑时间。删掉
   `角色库 / X / 生命阶段` 那串面包屑式的返回栏和「当前服装：… · 画风：…」
   那一行——服装与画风属于角色设定，不该出现在角色级头部。「修改」弹窗承接
   改名 / 改描述 / 删除角色 / 只读档案（时间、标签、备注、素材数）。
2. **导航降为两项**：只有「生命阶段」与「待处理 · 出素材 / 导出」。
   「角色设定」是某个生命阶段的一部分，不再是同级入口；进入阶段后
   「生命阶段」仍高亮，说明还在它下面。
3. **进入阶段后才有面包屑**：`角色名 / 阶段名`，下面才是角色设定的内容；
   右栏立绘是角色设定的一部分，此时才出现。
4. **生命阶段列表页只放阶段本身**：删掉塞在列表里的「待处理：出素材 / 导出」
   卡片和「角色档案」折叠区（前者走导航，后者走「修改」弹窗）；
   **整个右栏与拖动条都不渲染**，容器加 `.is-stages` 走单列 grid。
   理由是每行已经在显示该阶段的立绘，再放一个全局立绘面板反而分不清归属。
5. **一行以立绘为主**：缩略图放大到 58×82、最多 5 张横向铺开（没有立绘给占位
   框），标题 / 描述改成紧凑的单行无框输入，操作按钮竖排；文字不再用大 textarea
   占掉半行。

配套：`tools/shot/capture.mjs` 的假数据补了 `/stages` 并把 `/portraits` 改成
`includes`（前端现在带 `?stage=`），否则截图会落到 404 分支。

验证：`tsc -b` / `vite build` ✓、`check-hints` 34/34、
**`node smoke.mjs` 791/791**（用例 46 重写为头部 / 两项导航 / 无右栏 / 面包屑
断言；`ccStep(2)` 改为点行内「进入设定」而不是点导航按钮）。

## 2026-11-09 · 角色中心重构：生命阶段（一个角色多套设定）

用户明确四点：① 创建角色只要名字 + 可选描述，**不要参考图**；② 进角色后是
**生命阶段列表页**，每行 title / 描述 / 几张立绘设定图，可新建、拖拽排序、点击
进入编辑；③ 阶段编辑页 = 原「定形象」内容 + 立绘，改名「**角色设定**」，并加
「从已有阶段导入设定」快捷方式（**不导入立绘**）；④ 原「出素材 / 导出」内容
一字不动，整体挪到单独的「**待处理 · 供参考**」页，后面再定。
用户还说「现有的测试数据都不管」，所以不迁就旧角色数据与旧用例。

### 数据模型（`workflows/stages.py`）

阶段列表、顺序、角色描述放在角色自己的 `config.json` 里，**数组顺序即展示顺序**，
不额外建表：

```json
{ "description": "...", "active_stage": "default",
  "stages": [{"id": "default", "title": "初始设定", "description": "", "info": {...}}],
  "character_info": {...} }
```

- `character_info` 始终是**当前激活阶段 info 的镜像**：现有全部出图 API
  （`/prompt`、`/generate`、姿势/表情/图集）都不用带 stage 参数，切换阶段就等于
  切换「当前设定」。`update_character_info` 也同步写回阶段，避免下次 normalize 吞掉改动。
- 立绘按阶段分目录：`Portraits/<服装>/<表情>/` 是默认阶段的老位置（已有立绘原地不动），
  其它阶段落 `Portraits/s_<id>/...`；`list_portraits(name, stage_id)` 过滤，默认阶段
  遍历时跳过 `s_*`。删阶段只从列表移除，**不删磁盘文件**。
- 老角色第一次读 stages 自动补一个「初始设定」阶段（info = 原 character_info）。
- 新端点：`GET/PUT /characters/{name}/stages`（PUT 是全量数组：新建 / 改名 / 描述 /
  排序 / 删除 / 激活都走它，服务端 sanitize 限 50 条、去重 id）；`POST /characters`
  收 `description`；`/portraits?stage=` 按阶段取；`/generate` 的 publish 带
  `stage`，worker 按阶段目录收编立绘。

### 前端四层

`CharactersPage` 的 `view` 从 `library/workspace` 改成
`library / stages / stage / pending`，四步 `StepBar`、`activeStep`、`computeProgress`
整套删除，换成 `.cc-viewnav` 三个入口：

- **stages**：`.cc-stage-row` 一行一阶段，可编辑 title / 描述（blur 保存）、拖拽
  ⠿ 排序（`dropStage` 整包 PUT）、显示该阶段最多 4 张立绘缩略图与张数、
  「进入设定 →」、删除；「＋ 新建阶段」弹表单。角色档案（改名/删除/标签/备注）
  收进 `<details>` 折叠区。
- **stage**：标题「角色设定」，原定形象的画风/属性/prompt/生成立绘内容一字不动；
  新增「从其它阶段导入设定」下拉 + 导入按钮，明确标注「只复制角色设定，不导入立绘」。
  右栏 `PortraitPane` 只显示当前阶段的立绘（分页 8 张）。
- **pending**：原第 ③ ④ 步 JSX 原样搬进来，顶部一条「待处理 · 供参考」说明。
  右栏在 stage / pending 都常驻，任务条不会因为换页失联。

`CharacterLibrary` 新建弹窗删掉参考图上传，只留名字 + 描述 textarea；
`model.ts` 的 `STEPS[2]` 标签、App 标题、各处「定形象 / 四步 / 第 ② 步」文案
全部改成新架构的说法。阶段样式只加在 `styles.css` 末尾统一层。

### 验证

`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 784/784**
（用例 46 重写为三层导航与阶段列表断言；新增用例 64：3 阶段列表 → 新建第 4 个 →
拖拽把新阶段挪到首位 → 进入角色设定 → 导入其它阶段 info 且立绘张数/缩略图不变 →
待处理页仍有出素材与导出）；新增 `tests/test_stages.py` **10 例**；
后端 `test_stages + test_characters + test_api + test_style_neutral +
test_qwen_workflows + test_labels` **153 passed / 5 skipped**。

> 拖拽回归踩过一个坑：`dragId` 是 React state，`dragstart` 后必须等一次重渲染再
> `drop`，否则闭包里读到的还是 null（冒烟里为此加了 120ms 等待）。

## 2026-11-09 · 画风中立：公共提示词不碰画风 + 右栏画廊限量 / 进度条常驻

用户反馈「Q 版画风生成效果不对，其他画风也不能被影响，公用的提示词应该不影响
任何画风，画风由自己影响」。查下来不是一个 bug，而是一串「丢信息 / 掺风格」，
顺手把中途追问「还有别的被丢了吗」也做了完整审计。

### 1) 被丢的 / 被污染的东西（全部定位并修掉）

| # | 问题 | 后果 |
|---|---|---|
| 1 | 中文模式 `structured_prompt` 把官方三段式画风压成**只剩一个中文画风名** | 画风库 358 条的 Visual description / Reference 全丢。Q 版 `chibi_anime` 最要紧的「大头压倒小身体、圆润线条、夸张大眼」没了，出图当然不像 Q 版 |
| 2 | 公共段写死 `画风：… or 默认日式动画风格` +「按这个画风的笔触、轮廓、配色」 | 没选画风 / 译表查不到时**强加**一个日式动画风格，所有画风都被公共词干扰 |
| 3 | `qi2_fields` 把官方 `race_features` 在 zh 下置空 | 用户只填「精灵」一个词时，尖耳朵等物种视觉特征真的没了 |
| 4 | `custom_style` 全仓库只有默认值一处引用 | 存档里写了自定义画风，从不进 prompt |
| 5 | PE 请求用 `rsplit("\nLoRA 触发词：")[0]` | 会砍掉 LoRA 行**之后所有内容**；画风段排在后面时改写过程看不到画风定义 |
| 6 | PE 失败退回原文后又追加一次 LoRA | 触发词重复，会把 LoRA 权重拉偏 |
| 7 | 立绘出图沿用 `gen.txt2img` 默认 `preset="sprite"` | 追加 `flat plain background / ample empty space`（跟立绘「占满画面高度」打架），且 `SPRITE_NEG_PAINT` 里有 **`photo, 3d render`** —— 选写实 / 3D 画风等于用负向词把自己压掉 |

### 2) 修法：分层，画风段只引用画风库自己的定义

- 新增 `charactergen.style_block(ref, lang)`：中文画风名 + 英文原名 +
  **画风库自己的 Visual description / Reference 原文**，独立成段、放在属性行之前
  （跟官方「风格在前」一致）；没选画风返回空串。
- **公共段一个风格字都不写**：删掉「默认日式动画风格」兜底和那句泛化风格指令，
  只留「请画一张角色立绘」+ 画质 / 构图 / 背景 / 属性。
- **取舍**：画风描述保留官方英文原文 —— 它是画风库的**数据**，不是官方英文结构词
  （结构标签中文化即可）。Qwen3-VL 中英双语原生支持，硬译 358 条反而失真；
  信息完整优先于「整段纯中文」。
- PE 请求改成**逐行删 LoRA**（不再 rsplit 切尾部）；PE 成功后照官方再补一次
  完整画风段；PE 失败不重复追加 LoRA。
- 恢复 `race_features`（独立「物种特征」行，原文照搬）、接通 `custom_style`
  （加入 `QI2_LITERAL_FIELDS`，不被 PE 扩写掉）。**英文模板一字未改**。
- `app.character_generate_api` 在 `params.update()` **之后**强制 `preset="none"`：
  客户端手传 `preset=sprite` 也盖不掉，公共构图词与 `photo/3d render` 负向词
  彻底不进角色立绘。画布那条 `/generate` 不受影响（精灵帧仍可用 sprite 契约）。

### 3) 右栏 UI

- 立绘是**一直累积**的（`publish_portrait` 不覆盖），一次铺几十上百张 `<img>`
  首屏布局/解码全白做。`PortraitPane` 默认只铺最近 8 张，给「显示全部 N 张 /
  收起」按钮，头部仍显示总数。
- `TaskBar`（含完成提示）从画廊**之后**移到画廊**之前**，CSS 在统一层加
  `position: sticky; top: 0`，立绘再多、滚动到哪儿都能看见生成进度。
- 顺手删掉 `styles.css` 里重复了一整遍的画廊样式块。

### 验证

`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 756/756**
（新增用例 63：注入 23 张立绘 → 默认 8 张 → 展开 23 张 → 收起 8 张，
并钉死任务条顺序与 sticky）；新增 `tests/test_style_neutral.py` **13 例**
（逐条遍历 358 条画风断言视觉定义都在、公共段无风格词、PE 不砍画风、
LoRA 不重复、`preset=none`）；后端 `test_style_neutral + test_characters +
test_qwen_workflows + test_labels + test_api` **143 passed / 5 skipped**；
真机截图 `tests/_out/ui_review/style_neutral/` 右栏布局正常。

> ⚠️ 仍需 GPU 真跑：Q 版与其它画风的**视觉**改善幅度只能实拍对比确认；
> 这里钉死的是「喂给模型的信息完整且不被公共词干扰」。

### 4) 同日修正：立绘画廊改成**真分页**（不是「默认 8 张 + 显示全部」）

用户看了上一版后指出「还是一下全展示了，要分页」——原来那版默认只铺 8 张，
但「显示全部 N 张」一点开又把所有 `<img>` 铺回 DOM，等于留了个一次铺满的入口。

- `PortraitPane` 改为 `PORTRAIT_PAGE_SIZE = 8` 的**固定分页**：`slice(page*8, (page+1)*8)`，
  任何时刻只渲染一页；给「‹ / 页码（页数多时首尾 + 当前页附近，中间省略号）/ ›」
  与「第 x/y 页 · 共 N 张」。**删掉「显示全部」按钮**，没有一次铺满的路径。
- 立绘数量变化时 `useEffect` 回到第 1 页（列表按时间倒序，新出的立绘要看得见）；
  页数变少时 `safePage` 夹住越界页码；首/末页按钮 disabled。
- 分页样式只加在 `styles.css` 末尾统一层（`.cc-portrait-pager` / `.cc-page-btn`）。
- 顺带发现 `dist/assets` 堆了 44 个历史 bundle（其中包含旧的「显示全部」版），
  `cp -r dist_next/. dist/` 只覆盖不删除，浏览器若缓存旧 `index.html` 就可能
  继续加载旧逻辑。改成 `vite build --outDir dist` 让 vite 自己清空重建，
  现在只剩 1 个 js + 1 个 css，`index.html` 明确指向分页版。

冒烟用例 63 重写为分页回归：23 张 → 第 1 页 8 张（最新那张起）→ 下一页 8 张
（内容不同）→ 第 3 页 7 张 → 上一页回第 2 页；并断言**不存在**「显示全部」入口、
首末页按钮禁用、页码高亮、任务条顺序与 sticky。
`node smoke.mjs` **765/765**、`check-hints` 34/34、`tsc -b` ✓。

## 2026-11-07 · 设置彻底移出画布侧栏 + LoRA 强度每行常显

上一轮我把设置提到顶栏，但**为了保住「6 个 `.rail-sec`」的 DOM 契约，又在侧栏
留了一份**。用户的反馈很直接：「怎么画布里面的设置还在？」—— 这就是同时留两个
入口的代价：看起来像没搬。这次彻底搬走，并把因此暴露的两个问题一起修掉。

### 设置只剩一个家：顶栏 ⚙ 弹窗

- 侧栏删掉「LoRA」「设置」两个 Section，以及「设置」这个 tab（4 tab → 3 tab：
  节点 / 画布 / 资料库）。`.rail-sec` 从 6 段变 4 段：
  画布 / 模板 / 节点 / 资料库。
- 侧栏状态条右侧保留一个 `⚙` 小按钮（`.rail-settings`）——**它只是快捷方式**，
  点开的是同一个弹窗，不复制任何内容。
- 侧栏的 props 一口气删掉 12 个（loras / activeLoras / fastMode / baseModel /
  matteBackend / skipped / dir / count / onRefresh …），只留 `onOpenSettings`。
  少一份 props 就少一处可能不一致的状态。
- 冒烟同步把「六个菜单分组」改成「四个」、并新增三条反向断言：
  **侧栏里不许再出现 `.matte-backend-select` / `.lora-import-btn`**，
  以及点状态条 `⚙` 能打开弹窗。

### LoRA 强度：每一行都显示

以前 `{act && <label className="lora-scale">…}` —— 只有**勾上之后**才出现强度条。
没挂任何 LoRA 时（也就是刚装好的默认状态）界面上一条强度都没有，用户会以为
这个功能被删了。现在：

- 每行都渲染强度滑块，默认值 0.8；未挂载时 `disabled` + `.is-off` 置灰
  （opacity .42），`title` 写「先勾上左边的框，强度才会生效」。
- 勾上之后原地变成可拖，仍是 0.1~1.5 / 步进 0.05，并写回全局设置。

### 顺手修掉的一个真 bug

弹窗里的 `LoraSettings` 一开始**没把 `skipped / count / dir / onRefresh` 传下去**
（这几个 prop 只加给了当时还在的侧栏那份），表现为：头部显示「扫到 2 个」、
待转换分组空、刷新按钮点了没反应。是冒烟用例 60 抓出来的，已补齐。

### 验证

`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 695/695**
（用例 10 改四个分组 + 侧栏无设置断言；39/40/56/57/58/59/60 统一改成先开顶栏
⚙ 弹窗；39 新增「每个 LoRA 都有自己的强度滑块 / 未挂载置灰」）。
真机截图 `sx_ui`：侧栏只剩三个页签、状态条有 ⚙，弹窗里每行都有强度条。

## 2026-11-08 · 收尾：失败回滚、会话释放、多角色提示词契约

把上一轮如实留下的三个尾巴全部处理掉。

### 1) 动作导入失败不再清空用户摆好的姿势（官方 0.7.1 同类修复）
`applyTimeInner()` 在任何校验**之前**就 `viewer.resetPose()`；关键点算不出、
骨架映射失败或抛异常时，用户摆了半天的姿势会被清成站姿，界面只报一句
「这一帧套不上人偶」。

- 新增零依赖模块 `vnccs-pose/pose_guard.js`：`snapshotPose` / `restorePose` /
  `withPoseRollback`，返回 false 或抛异常都还原（含 modelRotation / camera /
  IK effector / 极点），并把关节标记重新露出来。
- `applyTime` 包一层；`animSample` 整批失败时还要从**抽帧前**的快照还原
  （单帧回滚挡不住前 N-1 帧已经改过的状态）。
- `loadAnimation` 失败时保留上一次成功载入的 state（它全程不动 viewer 骨骼）。
- `tools/verify_pose_rollback.mjs`：假 viewer 真单元测 6/6（成功不还原、
  false 还原、抛异常还原后原样抛出、getPose 挂了不二次崩、无快照 no-op）。

### 2) 抠图会话按任务释放（官方 0.7.5「Keep Models Loaded」的等价物）
`matte._AI_SESSIONS` 以前永久常驻、无释放入口，跟 Qwen 加载管线要的
17GB headroom 抢显存。`qwen_llm.py` 早就有「默认用完卸、可 keep」的范式。

- `workflows/matte.py` 加 `clear_ai_sessions()`（能 dispose 就 dispose + gc）、
  `release_ai_sessions()`、`MEOW_MATTE_KEEP=1` 常驻开关、`ai_session_count()`。
- `app._run` 的 finally 调一次 `matte.release_ai_sessions()`：**一个 job 内
  跨上百帧复用一次会话，job 结束才放**，不是每帧重建 ONNX。
  释放失败绝不影响已经成功的生成任务。

### 3) 多角色参考图与「单主体」提示词契约
上一轮加了多角色同框，但下游姿态链路写死 `a single character` /
`Redraw the same character`，更糟的是 `QWEN_FRAME_NEG` 把
**`two characters` 列为负向词** —— 双人参考图接上去，模型会被明确要求只画一个。

- 主体数走**确定性元数据**，不做图像识别：真机三人图因轻微接触连通域会合并
  成一块（数成 1），而 w4 常见的 OpenPose 骨架线图又全是断开细线（会数成多个）。
  导出人偶图时前端把 `scene.characters.length` 一起发给 `/upload?subjects=N`，
  服务端在图片旁写 `.json` sidecar（文件在磁盘、不进数据库）；生成时按参考图
  路径读。老图没有 sidecar 一律按 1 处理。
- 新增 `workflows/posesubjects.py`：sidecar 读写、`adapt_instruction()`（单主体
  → 复数、补「保持相对位置、不要合并或丢人」）、`adapt_negative()`（去掉
  `two characters` 负向词）。**count==1 时原样返回，老产线提示词一个字节不变**。
- 接进 `wpc`（pose_clone）与 `w4`（poseswap）两条文本指令链路，产物 meta 加
  `pose_subjects`。**官方 `_VNCCS_TEMPLATE` 一个字没动**（配套 LoRA 的固定指令）。
- ⚠️ 如实说明：文本契约已接线并有单测，但「双人参考图最终出图是否真的保住两个
  角色」需要 GPU 真跑一轮对比才能定；w4 若传的是骨架帧目录而非人偶图，
  sidecar 不在帧旁，会安全降级成单主体契约。

### 验证
`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 730/730**
（新增 62：三角色导出 → `/upload?subjects=3` → 静态契约断言）、
后端 `test_posesubjects + test_matte + test_api + test_pose_scene_contract +
test_poseset` **83 passed / 6 skipped**、`verify_pose_rollback.mjs` 6/6、
真机 `verify_pose_scene.py` 6/6、`verify_pose_editor.py` 通过、
`verify_anim_import.mjs` 全部通过（5 个骨架用例 + 断网本地解析器）。

## 2026-11-08 · 多角色同框 + 整角色变换 + 撤销/构图/阶段进度

用户明确说「我们也要支持多角色同框」，于是把上一轮按「单主体」前提否掉的
整角色 Move/Rotate、World/Local、Move & Rotate All、每 tab 独立位置、
精确 XYZ / Radar 全部做了，连同撤销重做、姿势构图还原、分阶段进度一起落地。

### 关键判断：引擎不用换，死代码不用激活

盘点后发现**内嵌的 0.7.2 基线引擎本来就有多角色能力**：
`passiveCharacters` Map、`upsertPassiveCharacterFromActive()`、
`setPassiveCharacterState()`，`capture()` 连 passive mesh 一起渲染。
官方 0.7.4 多出来的只是「状态 + UI」，纯算法是 175 行零 UI 依赖的
`character_transform.mjs`。`TransformControls.js` 也早已在 vendor 里。
所以：

- 只给 `core.js` 补一个 12 行的 `setPassiveCharacterPlacement()`（原样对应官方）；
- 把官方纯函数模块原样 vendored（**扩展名改 `.js`**：Python 静态服务把
  `.mjs` 猜成 `text/plain`，浏览器会拒载模块，实测会让
  `tools/verify_pose_editor.py` 报「脚本加载失败 entry.js」）；
- 其余全写在自写宿主里，官方那套 `animation.mjs` / `characters.mjs` /
  `video.mjs`（约 4000 行死代码、绑死自己的 DOM）一行没碰。

### 新增 `web/public/vnccs-pose/scene.js`（iframe 状态机）
统一 `{type:'scene', action}` 协议：load / select / add / remove / mode /
space / linkAll / tab / addTab / transform / camera / undo / redo，回执
`sceneState{scene, pose, yaw, history, commit}`（commit=false 是拖动中间态，
宿主只刷界面不写 project.json）。

- 每个姿势 tab 有独立的 `placements[tab][角色 id]`；切换 tab 从当前复制一份，
  不会让用户以为刚摆的位置丢了。
- **新人偶自动错开**（横向 5 单位/个）并**自动取景**：真机导出图里第一次
  加角色时新人偶压在第一个身上（默认 0,0,0），第二次又因单主体机位把
  第三个裁掉半身。`fitScene()` 按所有角色横向跨度把相机拉远居中，
  只在角色集合/切换时调用，拖 gizmo 时不动相机。
- 联动移动用官方 `translateCharacterPosition`（到边界时整组一起停），
  联动旋转用 `rotateCharacterAboutPivot` 绕各自骨盆补偿位置。
- **撤销栈复用引擎预留的扩展点**：`options.captureHistoryContext()` 把每个
  角色的放置与姿势一起写进 history，`onHistoryRestore()` 再取回 ——
  一次 Ctrl+Z 会同时退骨骼和整角色位移，不是只退一半。
- 出图前 `hideForCapture()` 把 gizmo 手柄藏掉（我们用 `viewport:true`，
  官方那套 editor-layer 过滤不生效）。

### 宿主 UI（`PoseScenePanel.tsx`）
角色 chip（选 / 删 / 加，最多 12）、摆姿势/移动/旋转三态、世界/本地轴、
联动开关、姿势位 tab（最多 16）、XYZ + 缩放 + 三轴旋转共 7 个数值框、
可拖的 X/Y Radar、撤销/重做按钮 + Ctrl/⌘+Z / Shift+Z / Ctrl+Y。
数值框失焦或回车才提交，避免每敲一个字符写一次存档和撤销栈。
状态**只在 iframe 里一份**，宿主只发消息、收回执。

### 数据契约
- `poselib.py` 清单带 `cameraParams`（zoom 117 个非 1、范围 0.5~4.17；
  offset_x/y 各 70 个非零）和精简的 `characters[]`。
  ⚠️ 官方库 191 个 JSON 的 `prompt`/`pose_prompts` **全是空串**，不做「保留 prompt」。
- `LibraryDrawer` → `PosePreset` → `App.applyPose` 把构图参数与多角色合成
  `PoseScene`，落进 `project.json` 的 `poseScene`，并用官方
  `snapToCaptureCamera(zoom, offset…)` 还原构图（不再靠 tags 猜）。
- 节点重开时 `restoreCamera` 还原自己摆的视角；pose 没带 camera 时不会
  被官方默认 `(0,0.5,4)` 把人偶缩成一点。
- `poseset.py` 帧表原样透传 `scene/camera/characters`，`app.py` 的
  `PoseSetFrameReq` 补上三个可选字段（Pydantic 默认吞多余字段，不补就全丢）。
- 分阶段进度：后端 `pipeline.py` 早就发 `step`，只是 `ApiJob.progress` 少
  这个类型 → 补上后 TaskBar 显示「阶段 2/3 · 姿势」。

### 验证
`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 721/721**
（新增用例 61 覆盖 2 角色 chip / 三态 / World-Local / 联动 / 姿势位 /
7 个数值框 / Radar / 失焦提交 X=4.5 / 撤销快捷键 / 静态产物契约）。
**真实浏览器** `tools/verify_pose_scene.py`（无头 Edge + SwiftShader）：
载入 2 角色 → X=4.5 → 撤销退回 0 → 加第 3 个角色 → 导出白底 PNG，**5/5 通过**；
基线 `tools/verify_pose_editor.py` 恢复通过（768² 白底灰人偶）。
后端 `test_pose_scene_contract + test_poseset` **18 passed**、
`test_api` **35 passed**、新增契约单测覆盖姿势库字段、姿势集 extras 往返、
Pydantic 不吞字段。

## 2026-11-07 · LoRA 列表改成「扫描目录」：刷新 + 待转换 + 就地转换

### 起因：截图骗人 + 静默过滤

用户看到界面上只有 2 个 LoRA，实际 `models/lora/` 里有 **11 个**。查下来两个原因：

1. **上一轮的真机截图用的是离线假数据**（`CC_FAKE=1` 的 mock 只写了 2 条），
   不是程序只认出 2 个 —— 这个锅得截图背；
2. 但程序里确实埋着同类问题：`available_loras()` 对格式不对的文件**直接
   `continue` 掉**，用户把 ComfyUI 格式的权重拷进 `models/lora/` 后，
   界面上根本看不到，只会觉得「怎么少了一大半」。静默过滤比报错难查。

### 改法

后端：

- `Engine.scan_loras()`：**列出目录里所有 `.safetensors`**，带
  `format / need_convert / convertible / reason`；读不出头的也不吞，报成
  `unknown` 并附原因。
- `available_loras()` 收窄成「scan 里 format == diffusers 的那些」，
  语义更准确（可直接挂载），行为对挂载/恢复不变。
- `GET /loras` 返回 `{items, active, skipped, count, dir}` ——
  `skipped` 是格式不对的，`count/dir` 让界面能显示「扫到 11 个 · 目录 models/lora」。
- 新增 `POST /loras/convert`（`lorastore.convert_in_place`）：把已经在目录里的
  ComfyUI 文件**就地**转成 diffusers，原名不变；**转换前先把原始字节备份到
  `library/lora_backup/`**（不在 models/lora 下，不会被当成另一个 LoRA 扫出来），
  失败时原文件一个字节都没动。
- 不提供「扫描入库」这种中间层：目录就是权威，扫一次就是一次真实读取。

前端（`GlobalSettings` 的 LoRA 段）：

- 头部一行：`扫到 N 个 · 可挂 M · 待转换 K` + `↻ 刷新`，下面一行显示
  `目录：models/lora/`（title 里是绝对路径）。拷完文件点一下就行，不用重启。
- **待转换分组**：格式不对的文件逐条列出（名字 / 体积 / ComfyUI 格式），
  能转的给「转换」按钮，不是本底模的明确写「无法转换」并附原因。
- 转换成功后自动刷新，条目从「待转换」移到「可挂」列表；提示里说明已备份。

### 验证

`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 687/687**
（新增 60：头部数量与目录 → 待转换 2 条 → 点刷新重新 GET → 点转换打到
`POST /loras/convert` → 自动刷新后可挂从 2 变 3、待转换只剩 1）、
`pytest test_lora_import` **17 passed**（新增 4 条：扫描不隐藏、就地转换保名字+备份、
已是 diffusers 不重复转、其它架构拒绝且不动原文件）。
截图假数据也改成 8 个可挂 + 2 个待转换（`count: 10`），不再用两条骗自己。

## 2026-11-07 · 全局设置提升为全站入口（角色中心也能改、也生效）

上一轮把底模 / 抠图方式 / LoRA 收到了 SQLite，但它们的**界面**还长在画布侧栏里：
一进角色中心就再也改不到，看起来像「画布的设置」。这轮把界面提到全站。

### 一个实现，两个挂载点

新增 `web/src/components/GlobalSettings.tsx`，导出 `LoraSettings` /
`GeneralSettings` / `GlobalSettingsDialog`：

- **画布侧栏**：仍然是原来的「LoRA」「设置」两个 `.rail-sec`（6 段 DOM 顺序不变），
  只是内容改成引用共享组件；
- **顶栏 ⚙ 设置**：新增按钮 + Portal 弹窗，画布和角色中心都能开，两段内容完整。

关键是**只有一份状态**（App 的 `loras/activeLoras/matteBackend/fastMode`）：
以前同一个动作在两个地方各写一遍，迟早不一致；现在两边都只是这份状态的两个视图。
冒烟用例 59 专门验这个：在角色中心把抠图方式改成 u2net，切回画布进编辑器，
侧栏读到的必须是同一个值。

### 抠图方式真的对角色中心生效了

以前角色中心的「额外抠一次背景」固定走 `matte_screen.chroma_screen_matte`
（官方幕布键控），跟全局设置毫无关系 —— 上一轮我在 UI 上写了「不受影响」，
这轮把它接通：

- `matte_screen.global_backend()` 从 SQLite 读全局方式（2 秒小缓存：批量抠
  上百帧时不该每帧开一次库连接）；
- `auto` **保持原来的官方幕布键控**（角色出图背景是已知纯色幕，这套就是为它调的，
  换成语义分割反而是降级）；
- 只有用户明确选了 `u2net` / `color`，才切到 `matte.matte_auto(backend)`，
  与画布节点走同一条路；模型目录用 `MODELS_DIR/matte`（与 `engine.matte_dir` 一致）。
- 角色中心那两个勾选项后面直接显示当前方式（`方式：自动` / `语义分割` / `纯色键控`），
  点不了但一眼看得到，改去顶栏 ⚙。

LoRA 本来就是引擎级全局状态（`apply_loras` 的 user 槽），角色中心的生成同样带着它；
之前缺的只是「在角色中心改」的入口，现在有了。

### 验证

`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 676/676**
（新增 59：角色中心无侧栏 → 顶栏 ⚙ 可开 → 弹窗含通用+LoRA → 改完写回 /settings
→ 切回画布侧栏是同一个值）、`pytest test_matte_screen` **10 passed**
（新增 3 条：读设置、未知值回落 auto、`auto` 不走 matte_auto 而显式选择才走）、
`pytest test_characters + test_matte + test_api` 全绿。
真机截图新增 `27-gs-from-characters`（角色中心里打开）与
`28-settings-global`（画布侧栏同一份）/ `29-lora-convert`。

## 2026-11-07 · 底模收归全局 + 全局设置落 SQLite + LoRA 导入（格式不符先问再转）

### 节点里不再选底模

后端早就是 Qwen-only（`engine.DEFAULT_MODEL` + `_check_model` 硬拒其它值），
但前端还留着整套「每个节点一个底模下拉」：`GenParams.model`、`WORKFLOW_OF`、
`shortModel`、`usableModels` 与 `setNodeModel` 的推荐参数联动，一共 120 行只为
渲染一个**永远只有一项**的下拉。这轮全部删掉：

- 节点卡头不再有 `.node-model`；参数区第一个控件就是提示词，视觉上更直接。
- 提交时底模统一用全局 `model`（`app._run` 再兜一层），节点里无处可改。
- **旧存档的坑**：以前节点可能钉着 `sd15` / `sdxl`，删了下拉之后这个值仍会被
  当 `params.model` 提交，而 `_check_model` 只认 Qwen → **整单直接失败**。
  `migrate()` 现在把非 Qwen 的旧值一律清成空串，弹窗里也验过（用例 22b）。
- `/models` 的冒烟 mock 从「sd15/sdxl/sdxl-anime」改成 Qwen-only：以前底模联动
  用例跑的是与生产**相反**的数据，属于自欺欺人的绿。

### 全局设置进 SQLite（`metadb.settings`）

抠图方式 / 生效的 LoRA / 快速模式以前只在 React `useState` 里，重开一次全丢。
现在服务端是权威：

- `metadb` 新增 `settings` 表（key/value + JSON），白名单只收
  `base_model` / `matte_backend` / `loras` / `fast_mode`；未知抠图后端归一成 `auto`。
- `GET/PUT /settings`；启动时 `_restore_global_settings()` 把设置灌回 engine。
- **LoRA 懒恢复**：`Engine.restore_loras()` 只记规格，不为了恢复去加载管线
  （那会在用户还没点「生成」时白占十几 GB 显存）；真正挂载推迟到
  `txt2img_pipe()` 第一次调用。`active_loras()` 会把「待挂载」也报成已挂，
  否则界面刷新后显示「没挂任何 LoRA」，实际下一次生成却带着它们。
- 补齐一处真 bug：`api.submitMatte()` 从来没带 `matte_backend`，所以侧栏
  选的「抠图方式」对抠图节点（wbg）**完全不生效**，恒走后端默认 `auto`。
  后端 `_run` 也补了一层：接受 `matte_backend` 的工作流没带这个键时，用全局值。

### LoRA 导入：先检测、再问、才转换（`workflows/lorastore.py`）

ComfyUI 生态发的 Qwen LoRA 大半是 `diffusion_model.` + `lora_A/B`，而
`pipe.load_lora_weights()` 只认 `transformer.` + `lora.down/up`。直接挂上去
diffusers 只打一句 "unexpected keys found" 就继续 —— **看起来导入成功，实际
一点效果没有**，比报错难查得多。所以做成两步：

1. `POST /loras/inspect`：原始字节流式落进 `library/lora_staging/`（2GB 上限），
   只读 safetensors 头，返回 `format / need_convert / convertible / reason /
   sample_keys`。**这一步绝不写 `models/lora/`**。
2. `POST /loras/commit`：`convert=true` 才调 `tools/convert_lora.py` 的
   `convert_weights()`（同一套键名映射与 img_mlp gate_up 拆分，服务端不另写一份），
   转换结果再校验一次必须是 diffusers 格式，然后原子替换进 `models/lora/`。
   未确认转换 → 明确 400；不是本底模的 LoRA（`unknown`）→ 报错并附头几个键名。
3. `DELETE /loras/{name}` 删文件 + 从全局设置里摘掉挂载记录（不加载管线）。

前端：侧栏 LoRA 段加「＋ 导入 LoRA（.safetensors）」、每条一个删除按钮与内联
确认；格式不符弹 **Portal 到 body** 的确认框，说清「ComfyUI → diffusers」、
张量数、体积、转换后文件名与键名示例，用户点「确认转换并导入」才落盘。

### 验证

`tsc -b` / `vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 668/668**
（重写用例 10 / 22 / 22b / 23，新增 56 全局设置、57 转换确认、58 直接导入与删除）、
`pytest tests/test_models.py tests/test_review_round2.py tests/test_lora_import.py`
**30 passed**（新增 13 条：格式判定、未确认拒绝、确认后转换、非本底模拒绝、
文件名清洗、设置白名单、懒恢复剔除缺失文件）。真机离线截图新增
`27-settings-global` / `28-lora-convert`。

当时还留着一个缺口：角色中心的「额外抠一次纯色幕布」固定走
`matte_screen.chroma_screen_matte`，**不受全局抠图方式影响** ——
下一节「全局设置提升为全站入口」已把它接通（`auto` 仍用这套幕布键控，
显式选 `u2net`/`color` 才切到 `matte.matte_auto`）。

## 2026-11-07 · 全屏库首页 + SQLite 元数据 + 「项目」改名「画布」

用户连续提了几条，一起落地：

1. 角色中心不该在左边留一条窄角色库，一进来整个页面就该是角色库列表；
   每张卡展示角色图片、名字、tag（tag 多就贴在图片上）；名字 / tag 模糊搜索、
   选 tag 过滤、名字 / 创建时间 / 最后编辑时间排序。
2. 一个角色都没有时，页面中央放一个**大号创建按钮**。
3. 「很多信息没存下来」→ 加本地小型数据库，无需单独安装，有更好的用更好的；
   后来明确：**文件、模型等不进数据库**；也**不必再顾忌官方 VNCCS 目录互通**。
4. 画布侧同样处理，但以「项目」为角度；「项目」不好听，UI 上改名**画布**。

### 数据库选型：SQLite（Python 标准库）

- `sqlite3` 在标准库里，**零安装、单文件、无服务进程**；开 WAL + 事务，
  本机单用户 / 后台 worker 多线程读写够稳。
- DuckDB 要额外 pip 且面向 OLAP 分析；TinyDB 只是 JSON 换皮（无事务 / 索引 / 并发），
  等于回到现状；libSQL / Turso 需要额外依赖或同步服务。
  对「几十到几百条元数据 + 点查排序」这个场景都是过度工程。

**分工（重要）**：SQLite 是**角色 / 画布元数据的权威存储**，磁盘只放文件与生成用的
属性 config。库里存：派生标签、稳定创建 / 编辑时间、自定义标签、备注、任务历史摘要，
外加封面**路径字符串**。图片、精灵帧、模型权重、节点定义一律留在磁盘，绝不入库。
自定义标签 / 备注这类磁盘上没有、也不该塞进 prompt config 的信息，终于有地方放了。
按用户要求，不再为兼容外部 VNCCS 目录格式牺牲数据模型。

`workflows/metadb.py`（新）：`characters` / `canvases` / `tasks` 三表 + WAL + UPSERT 同步；
库文件在 `library/meta.sqlite3`（已被 .gitignore 忽略，测试用 `MEOW_LIBRARY_DIR` 覆盖）。
`created_at` 首次落库后固定（文件系统 ctime 跨平台不可靠），`updated_at` 取磁盘 mtime
与库里记录的较大值 —— 刷新页面不会把时间推到「现在」。

### 角色中心：库 → 工作台两层

- 新增 `CharacterLibrary.tsx`：全屏网格，卡片 3:4 大图 + 底部渐变上的 tag chips
  （自定义标签优先，超过 5 个收成 `+N`）；名字 / tag 模糊搜索、tag 过滤（OR）、
  名字 / 创建时间 / 最后编辑时间三档排序 + 升降序；排序 / 过滤偏好写回
  `library/prefs.json`（复用现有 `getLibrary` / `savePrefsApi`，不另起存储）。
- 卡片上可编辑自定义标签与备注（`PATCH /characters/{name}/meta`）、删除角色。
- 点卡片才进四步工作台；工作台**删掉左侧窄角色库**，只剩主工作区 + 常驻成果，
  顶部加「← 角色库」。第 ① 步从创建表单改成**角色档案**（名字 / 创建 / 编辑时间 /
  服装 / 帧数 / 标签 / 备注 + 改名 / 删除），创建只发生在库里，不再有重复入口。
- 空库时中央一个大按钮「创建第一个角色」，不用去角落找「新建」。

### 画布侧：同样两层 + 改名

- 新增 `CanvasLibrary.tsx`，复用同一套 `.charlib-*` 样式与交互契约：全屏画布网格、
  大图（项目目录里最新的产物图，只记路径）、图片上节点类型 tag、搜索 / 过滤 / 排序、
  空库大按钮「创建第一张画布」、自定义标签与备注。
- 单位仍是磁盘上的 `assets/projects/<id>/project.json`，**后端 scope / API 路径不变**
  （改名会牵动几十处调用），但界面上「项目」一律叫「画布」：侧栏分组、新建 / 导入 /
  删除、模板说明、状态条、默认名、提示文案。
- `?editor=1` 深链直接打开编辑器（分享 / 恢复上次画布 / 冒烟用）；点顶栏「画布」只在
  从角色中心切过来时回库，已在画布视图时保持当前层。
- 编辑器顶部浮动小条「← 画布库」（`.stage` 里 Sidebar / Canvas 都是 absolute 定位，
  不能插普通块挤掉锚点）。

### 后端改动

- `workflows/characters.py`：`character_summary` 增加派生 `tags` 与文件系统时间戳；
  `_character_tags` 从 `character_info` 拆逗号字段（值仍原样英文，中文只在渲染处映射）。
- `app.py`：`GET /characters` / `GET /projects` 先同步 SQLite 再合并元数据；
  新增 `PATCH /characters/{name}/meta`、`PATCH /projects/{scope}/meta`、
  `GET /character-tasks`；`_run` 的 `finally` 记任务历史摘要（状态 / 耗时，不记文件路径）。
- `web/src/api.ts`：`CharacterSummary` / `ServerProjectMeta` 扩展字段 + 两个 meta 写接口。
- `tools/shot/capture.mjs`：`CC_FAKE=1` 离线假数据扩到多角色 / 多画布，
  截图流程改成「画布库 → 编辑器 → 角色库 → 工作台四步」。

### 验证

`tsc -b` ✓、`vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 642/642**
（新增用例 53/54/55 覆盖卡片 tag、名字/tag 搜索、tag 过滤、三种排序、
meta 写入、空库大按钮、画布库进编辑器与返回）、角色后端子集 1 passed、
`metadb` 同步真实数据（2 张画布 / 1 个角色）✓、真机截图 20 张。

## 2026-11-07 · CSS 源码卫生债清理

上一轮 review 里留了一条没动的债：`styles.css` 有整段复制粘贴的重复定义、
注释与实现互相打架的补丁、以及 `:root` 被定义两次。这轮清掉，**目标视觉零变化**。

### 删掉的 7 段（3553 → 3440 行，共 -113 行）

| 段 | 行数 | 说明 |
|---|---|---|
| 抽帧挑帧整段（`.pose-frames` ~ `.anim-order`） | 47 | 连注释一起 100% 复制了两遍 |
| 「舞台」段（`.pose-batch-stage`） | 18 | 同上，含 2 行注释 |
| 「导出行」段（`.pose-batch-export` / `-gif`） | 13 | 同上 |
| `.pzoom` 补丁式第二份 | 13 | 两份定义、两份注释互相否定 |
| `.pose-batch-btn` 第二份 | 10 | 见下 |
| `.chars-wizard-input` 第二份 | 7 | 含注释完全相同 |
| 末尾二次 `:root` | 10 | 见下 |

用脚本按行号**从后往前**删，每段首行做内容断言；文本替换逐条断言出现次数。
第一次脚本把最后一条替换打在了已被删掉的注释上，断言失败且**没有写盘**，修掉后重跑。
清理后复查：≥8 行的整段重复为 0。

### 顺手修掉两处「注释与实现打架」

- `.pose-batch-btn`：第一份注释明确写着「底色必须用 `--bg-3`，用 `--bg` 会比卡片
  还暗、看起来像透明的洞（实测被报过）」，但后一份复制时写成了 `background: var(--bg)`
  并且生效。删掉后一份，恢复 `--bg-3`。**这是本次唯一有意的视觉变化，且是修 bug**；
  18 张截图场景里没有 pose-batch 面板，所以像素对比没有覆盖到它。
- `.pzoom`：基础定义 + 后面的「不要用 opacity 做低调」补丁覆盖，两份注释互相否定。
  合并成一份，直接写补丁后的最终值（`rgba(0,0,0,.86)` / 灰白边 / hover `.16`），
  最终计算样式完全不变。
- `:root` 原本定义两次（顶部 8px/5px/柔光，末尾覆盖成 4px/3px/阶梯），把最终值
  直接写回顶部，并移入 `--px-edge`，全站只剩一处定义，引用值不变。

### 刻意保留的两类重复（不是漏删）

1. **三处棋盘格声明**（`.lightbox-img` / `.node-result-img` / `.sprite-stage`）：
   是三个不同元素的相同声明，分属 3 个 override 层，抽共享规则要改动 3 处规则体、
   收益 14 行而风险不小，保留并在注释里各自说明。
2. **各 override 层之间的同值单行覆盖**（`.mini:hover`、`.gpu.ok`、
   `input::placeholder` 等约 20 处）：是各层的有意兜底，删掉基础层那份会让该层
   失去覆盖能力；真正的统一已由末尾「全站样式统一层」负责，不再动前面。

### 验证

`tsc -b` ✓、`vite build`（同时验证 CSS 可解析）✓、`check-hints` 34/34、
**`node smoke.mjs` 604/604**。
关键证据：把清理前后两次真机截图逐张用 Pillow 比对，
**18/18 张 PNG 字节级完全一致** —— 纯源码清理没有引入任何视觉回归。

## 2026-11-07 · 全站 UI 一致性 review 与整改

用户反馈「很多地方样式还是不统一，画布那个长方格子特别难看」。做了一次
全量只读审计（styles.css 3400+ 行 / 6 层 override / 全部组件），结论分两块。

### 根因 1：画布网格是**真 bug**，不是审美问题

`Canvas.tsx` 用内联 `gridStyle` 写死三层 `backgroundSize`：
`24*z / 120*z / 120*z`（沿用早期「点 / 横线 / 竖线」的图层顺序）。
但 CSS 后来把图层顺序改成了「横线 / 竖线 / 点」，尺寸没跟着改 ——
于是横线每 24px、竖线每 120px，铺出一片 **24×120 的长方格**；
`backgroundPosition` 还只给了 2 层值，第 3 层回落复用第 1 层。
CSS 里写的 48px 正方格被内联样式完全覆盖，从未生效。

**修法**：`Canvas.tsx` 的 `GRID_BASE = 32`，只给一个 `backgroundSize` 值，
所有图层复用它；CSS `.canvas-grid` 同步改成 `32px × 3`。
平移 / 缩放时格子始终是正方形。两边必须同步改，只改一边就会再长出长方格。

### 根因 2：6 层 override 叠加，同类控件各写各的

审计出的具体数值（都是同一屏内并存）：
- 圆角：节点卡 **10px** / 卡头 9px / 弹窗 12px / 一堆 999px 胶囊 /
  角色中心 3~4px 方块 —— 五套
- 阴影：侧栏 28px 柔光 / 弹窗 60px 柔光 / 顶栏 14px 柔光 /
  节点与按钮 2px 阶梯 —— 两套体系
- 按钮高度 **22/24/26/28/30/32/34/36/38/40px 十种**；
  disabled 透明度 0.3/0.35/0.4/0.42/0.45/0.5/0.55 七种；
  hover 主色绿 / 青 / 白 / 无 / brightness 五套
- 激活态四套：绿底、青边、绿边、青框配绿影
- 边框透明度 0.10 / 0.16 / 0.22 三套；绿底深字 `#08160f`/`#07160e`/`#07131b` 三种
- 绕过 `:root` 的旧纯黑底 `#0d0d10`/`rgba(12,12,16,.9)` 等 20+ 处；
  `var(--purple, #5b57eb)` 等 fallback 与变量真值不符；`--pixel` 是从未被引用的死变量
- 特异性不够导致「列清单式覆盖」漏网：`.chars.cc .chars-card`(0,3,0)
  压过 `.chars-card`(0,1,0)，`.lib-tab`/`.lib-src-btn`/`.train-panel` 整类漏改

### 整改

1. 修网格真 bug（见根因 1）。
2. `styles.css` 末尾加**「全站样式统一层（最终权威层）」**：只做对齐，
   不动布局 / 定位 / 类名 / DOM。统一到两档圆角（卡片 4px / 控件 3px）、
   `1px solid var(--px-edge)` 边框、`var(--shadow-hard)` 阶梯投影、
   主视图 tab 绿底 / 其余 tab 青激活、文字按钮 hover 青 / 图标按钮 hover 绿、
   文字按钮 30px / 主 CTA 38px、disabled 一律 `.42`、图片容器统一 `--bg-1`、
   小标题统一 `--sans`。刻意用虚线表达「空 / 新增」的元素保留语义。
3. 基础层清理：删死变量 `--pixel`、统一 `--chip = var(--sans)`、
   修 7 处 `var(--purple, #5b57eb)` / 4 处 `var(--yellow, #ffd34d)` /
   `var(--cyan-2, ...)` / `var(--line, ...)` 的错误 fallback、
   绿底深字三种统一成 `#08160f`。
4. 中文小标题不再用 `--mono`（Consolas 无中文字形，低分辨率会糊成方块）；
   原生 multiple 下拉行高放宽。

**约定写进 CSS 注释：以后改外观只改这一层，不要再叠第 8 层。**

### 验证

`tsc -b` ✓、`vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 604/604**、
真机 18 张截图（画布 / 侧栏四页 / 节点三态 / 角色中心四步与进阶展开）。
截图确认：网格是整齐的 32px 正方格，节点卡 / 弹窗 / tab / chip 圆角一致，
按钮与阴影不再各写各的，中文清晰无错位。

## 2026-11-06 · 全站加回像素风味（只改几何，不动字体）

仓库最初就是「近黑底 + 1px 硬边框 + 硬投影 + 四色点缀」的像素风，视觉刷新为了
可读性把它磨圆了。这层把**几何感**加回来，配色、字体、类名、DOM、数据流一个不改，
仍是 `styles.css` 末尾的 override 层。

### 做法

- 重定义设计令牌：`--r/--r-sm` 8px/5px → **2px 方块角**，
  `--shadow-hard/--shadow-soft` 柔和扩散 → **3px/2px 阶梯硬投影**，
  新增 `--px-edge`（更实的 1px 硬边）。前面 3000 行里几十处 `var(--r)` /
  `var(--shadow-hard)` 一次性统一，不用逐个选择器补。
- 顶栏底部一条 **24px 四色像素分段条**（绿/青/粉/黄，硬边无渐变）；
  画布网格改成 **24px 方格 + 格心像素点**；角色中心主区加 **8px 淡点阵**；
  精灵帧透明底用标准 **16px 棋盘格**。
- **2×2 像素角标**：卡片标题与步骤大标题前的 `::before` 用 6px 方块 +
  两级 `box-shadow` 画 L 形点阵，纯 CSS 不引图片；空状态卡右上角同理。
- 交互手感：按钮 `:active` 整体右下位移 2px 并收掉投影；选中态用实心阶梯
  代替发光；进度条切成 8px 一格；滚动条 thumb、焦点框（虚线）、步骤号、
  任务状态点全部方块化；数字/英文（`--mono`）加 1px 硬阴影做 8-bit 立体感。

### 明确不做的事

**不给中文上像素字体**：仓库里从来没有可用的中文点阵字体，离线环境也加载不到
外链字体；强行靠 `image-rendering` 或字距只会把中文糊成一团。中文正文一字未动，
像素感全部来自几何、纹理、阴影和位移。文件顶部那条「logo 是 512px 下采样到
26px，别用 `image-rendering: pixelated`」的注释也继续保留，只对精灵帧这类
**放大显示**的像素图开最近邻。

### 验证

`tsc -b` ✓、`vite build` ✓、`check-hints` 34/34、**`node smoke.mjs` 604/604**、
真机截图 18 张（画布 / 侧栏四页 / 节点三态 / 角色中心四步与进阶展开），
确认中文清晰、无错位、无遮挡。

踩到一个 CSS 特异性坑：`.cc-step-no` 前面是 `.chars-tab.cc-step .cc-step-no`
（0,3,0），像素层里只写 `.cc-step-no`（0,1,0）压不过那个 `border-radius: 50%`，
必须带全前缀才能把步骤号真正变成方块。

### 2026-11-07 修正：装饰过度，收敛

用户看完真机截图反馈「有点诡异」。回看确认问题不在「像素」而在**贴皮**：
顶栏四色分段条在 1600px 宽下重复十几组、每个标题前都挂一个 L 形小方块、
主区铺满 8px 噪点，再叠 24px 密网格 + 3px 黑硬投影 + 每个小按钮都有的
投影，和低饱和深色界面打架，又花又脏。

**撤掉**：彩色分段条、标题像素角标、空状态角标、主区 8px 点阵、
数字/英文 1px 硬阴影文字、中文标题硬阴影、所有小图标按钮的黑投影、
`▪` 折叠符号、窄屏点阵兜底规则。
**收敛**：圆角 2px → 4px/3px（8px→2px 太生硬），硬投影 3px 黑块 →
2px `rgba(0,0,0,.34)`，硬边 `.20` → `.16`，画布网格 24px → 48px 且透明度
降到 `.022`，按下位移 2px → 1px 且只给主按钮。
**保留**（功能性、不抢眼）：方块滚动条 / 虚线焦点框 / 方块步骤号 /
选中实心阶梯 / 分格进度条 / 精灵帧 16px 棋盘格 / 像素图最近邻放大。

判断标准写进 CSS 注释了：**像素感来自边界清晰、可点按、像素图按最近邻放大，
不来自贴上去的装饰图形**。以后别把彩条、角标、满屏点阵加回来。

## 2026-11-06 · 角色中心彻底重构：从「5 个并列 tab」到「四步工作台」

用户反馈：「非常难用，用户根本不知道怎么用」。核对代码后确认不是缺功能，是信息架构
补丁堆叠出来的：5 个并列 tab（属性/出图/服装/表情库/素材）+ 右栏 4 组重复按钮 +
中栏 3 张欢迎卡 + 3 套互相竞争的编号引导，同一操作有 2~3 个入口（「一键出全套」
两处、「导出图集」两处、「换白内衣」三处），读完所有说明仍不知道第一步点哪里。

### 业界同类产品的共识（交互层，不涉及模型）

参考了主流 AI 游戏素材 / 角色生成产品的通行做法：角色是**带持久 ID 的对象**而非
一段提示词；用户面对**一条唯一的线性主线**；**成果预览常驻**在视野里；
**结构化选择题**优先于自由输入框、专业参数渐进披露；把**耗时、前置条件和能力边界**
写在按钮旁边而不是报错里。这些都不需要改后端，纯粹是前端信息架构。

### 重构后的骨架

- **四步主线**：① 创建角色 → ② 定形象 → ③ 出素材 → ④ 导出。完成标记由磁盘上
  真实存在的立绘 / 帧 / 图集推导（`computeProgress`），不靠用户记点到哪；
  默认停在第一个未完成步骤，完成后自动推进，也可手动回看任意一步。
- **常驻成果区**（右栏）：立绘与精灵帧翻看器一直可见，附服装/表情切换、帧缩略图、
  素材统计与统一任务状态条。这是「结果优先」，不再是填完表单才知道有没有出图。
- **每步只有一个主 CTA**：②「生成第一张立绘」、③「一键出全套」、④「导出图集」；
  换装 / 表情 / 姿势的单步重跑收进「进阶」折叠区，不再和主线抢注意力。
- **统一状态**：删掉 `browseCostume`/`costume` 双服装状态（服装页选 A、素材页跑 B）、
  统一 `matte=false`（旧版出图/表情/流水线/后端四处默认值不同）、
  9 条 `xxxMsg` 灰小字 → 一条常驻 `TaskState`（互斥、进度、耗时、错误码话术）。
  所有任务共用一个 `runTask` 轮询器，不再 7 份循环、7 种超时文案。
- **前置条件显性化**：没有立绘时第 ③ 步直接给挡路卡 + 一键回 ②，不再「点了才 400」；
  「新建服装」改内联输入并立即落盘（旧版只改本地 state，切角色就消失）；
  改名/删除换成内联 UI，不再弹 `window.prompt/confirm`。
- **文案**：去掉「官方 QI2 / latent 步长 / cfg / V1/ / config.json / assets/characters」
  这类实现细节；属性字段改成点选标签（值仍原样英文传给后端），手填只在进阶里；
  表情默认从 25 步改成 16 步（实测质量下限，旧默认恰好是页面自己写着「约 27 分钟」
  的最慢档）；一键出全套按帧数给出耗时预估。

### 文件与验证

- 新增 `web/src/components/characters/{model.ts,Controls.tsx}`（常量/纯函数/展示组件），
  `CharactersPage.tsx` 从 1900 行单文件重写为状态 + 业务 + 四步面板；
  `styles.css` 末尾追加 `.cc-*` override 层（沿用 `:root` 变量）。后端 API 一个没改。
- `web/smoke.mjs` 角色用例同步重写：旧用例把「5 个 tab / 8 字段 / 5 个下拉 /
  3 张缩略图」等 DOM 数量写死，全部换成新架构断言，API 契约（建档、prompt、
  PE、生成、反推、向导、标签、服装、姿势、流水线、内衣、表情、图集、翻帧）一条没丢。
  mock 补上 `PUT/DELETE /costumes`（旧 mock 落 404，导致「新建服装」请求发了却不生效）。
- `tools/shot/capture.mjs` 加 `CC_FAKE=1` 离线模式：页面内假接口 + 本地图标，
  不接后端 / 不占 GPU，拍四步工作台与进阶展开态。
- **验证数字**：`tsc -b` 通过、`vite build` 通过、`check-hints` 34/34、
  `smoke.mjs` **604/604 通过**、真机截图 13 张基础场景 + 5 张四步场景，
  确认成果图常驻、主 CTA 可用、进阶展开 157 张表情卡不撑坏布局。
- 踩到一个真 bug：`SpritePreview` 的 `onMeta` 必须用 `useCallback` 稳定引用，
  内联箭头函数会让 `loadMeta` 每次 render 重建、把刚翻到的第二帧弹回第一帧。

## 2026-11-02 · 批量出图和角色图一模一样 —— 引擎写死了 wpc

用户报：12 帧结果跟原始角色图一模一样（站姿没变）。

### 根因：批量无视节点的引擎设置

「角色迁移」节点的 `pcEngine` 默认是 **VNCCS**，但 `posebatch.run_batch` 写死
调 `generate.pose_clone`（**wpc**）。所以界面上选的是 VNCCS、实际跑的是 wpc。

用用户那套真实素材做 A/B（同一张站立角色图 + 同一组灰人偶帧、同 seed）：

| 引擎 | 帧 1 vs 帧 6 平均差（/255） | 结果 |
|---|---|---|
| wpc（auto） | 0.08 | 12 帧全照抄角色图站姿 |
| wpc（skeleton） | 0.27 | 还是照抄 |
| **VNCCS** | **28.4** | 真正跟住了跑步/踢腿 |

wpc 会把「站立角色图」当主参考、灰人偶条件图当陪衬；VNCCS 的图序反着来
（骨架在前、角色在后 + 专用 LoRA），所以跟得住。

### 修法

- `posebatch.run_batch` 按 `params["engine"]`（wvnccs 默认 / wpc）走，两条引擎
  各自显式传参；VNCCS 步数 `max(20, steps)`、`transparent = bg=="transparent"`；
  `job["result"]` 里记 `engine` / `ref_mode` 方便排查。
- `/pose_renders/{name}/run` 加 `engine` 字段（默认 wvnccs）；面板透传
  `node.params.pcEngine`。
- **参考模式归一化**：姿势集帧永远是人偶编辑器的灰人偶，`auto`/`pose` 是旧
  存量值，统一归到 `skeleton`（面板在源头归一化，后端再兜底一次）。否则灰人偶
  会走 DWPose 提取 → 条件图被画成「黑底彩色骨架 → 反相 → 几乎空白的细线」。
- 节点「更多设置」加**引擎下拉**（VNCCS 推荐 / 旧版 wpc）；选 VNCCS 时「参考」
  下拉禁用（它不走 ref_mode）。

### 验证

- **真跑**：`posebatch.run_batch`（真实引擎，2 帧）→ `engine=wvnccs`、
  帧间差异 **29.4**、候选正常落盘；生成的跑步帧肉眼复核过（前腿抬起、摆臂）。
- 单测 `test_run_batch_defaults_to_vnccs`（默认不退回 wpc、transparent 与
  steps>=20 都传到 vnccs）；`test_run_batch_appends_candidates` 固定走 wpc 分支。
- `verify_pose_batch` 补：**点「全部生成」抓请求体**，断言 `engine=wvnccs`、
  `ref_mode=skeleton`（前端契约，不靠后端兜底）。
- smoke 635、verify_frame_pick 26、verify_anim_import 5、pytest 53 全绿。

## 2026-11-02 · 批量点了不出图（漏传姿势集名）+ 主按钮与透明底

### 1) 「全部生成」建了一堆空格子，但一张图都没有 —— **后端漏传 pose_set**

`/pose_renders/{name}/run` 组 params 时只放了 `project` / `render`，**没放
`pose_set`**；而 `posebatch.run_batch` 第一件事就是拿它去 `poses/<名字>/set.json`
找源帧，缺了直接抛「缺成片集名或姿势集名」。

坑在于**失败发生在 worker 里**：`render.json` 已经建好了（所以界面上一排
空格子），接口也早返回了 `job_id`，而面板只显示 `job.result.failed`（整批抛错
时它是 `undefined`）→ 用户看到的就是「点了没反应」。

- 端点从 `detail["poseSet"]` 补上 `params["pose_set"]`，取不到就 400 明说；
- 面板补一条 `job.status === 'failed'` 的红字（整批失败必须看得见）；
- `tests/test_poseset.py` 补 `run_batch` 的两条：缺 pose_set 必须抛、
  给了之后候选要**追加**进 `render.json` 且自动选中；
- `tests/test_api.py` 补端点级断言：直接查 `_jobs[job_id]["params"]["pose_set"]`
  （队列 `put` 换成空操作，不惊动 GPU worker）。

### 2) 「全部生成」改成绿色大按钮

之前用 `.pose-batch-btn`，底色是 `--bg`(#0a0a0a) —— 比卡片底 `--bg-2`(#121216)
还暗，看着像"透明的洞"，也不像主操作。改成和「迁移角色」同一套 `gen-btn`：
整行、32px、品牌绿、带文字帧数（`全部生成（12 帧）`）。

次级按钮（导出 GIF / Godot / 重生成）底色从 `--bg` 提到 `--bg-3`，至少比卡片亮一档。

### 3) 底色加「透明底」

用户要的是**底色下拉里的透明选项**（不是抠图）。三处配套：

| 层 | 改动 |
|---|---|
| `types.ts` | `BgTone = 'white' \| 'dark' \| 'transparent'` |
| 节点「更多设置 → 底色」 | 角色迁移 / 姿态批量都加「透明底（模型直接出 alpha）」 |
| `PoseBatchPanel` | `bg` 原样透传（旧代码 `=== 'dark' ? 'dark' : 'white'` 会把透明悄悄换回白底） |
| `generate._qwen_pose_clone` | `bg="transparent"` → `_qwen_call(transparent=True)`（提示词加 alpha 尾句）+ `_postprocess(transparent=True)`；meta 里记 `transparent` |
| `api.submitVnccsPose` | 底色选透明 ≡ 勾透明，两条引擎语义一致 |

> 透明底 ≠ 抠图：透明底是让模型**直接画 alpha**，背景干不干净由模型决定；
> 要稳定透明得另勾「抠背景」。这条写进了 select 的 tooltip。

## 2026-11-02 · 批量点了不出图（漏传姿势集名）+ 主按钮与透明底

### 1) 「全部生成」建了一堆空格子，但一张图都没有 —— **后端漏传 pose_set**

`/pose_renders/{name}/run` 组 params 时只放了 `project` / `render`，**没放
`pose_set`**；而 `posebatch.run_batch` 第一件事就是拿它去 `poses/<名字>/set.json`
找源帧，缺了直接抛「缺成片集名或姿势集名」。

坑在于**失败发生在 worker 里**：`render.json` 已经建好了（所以界面上一排
空格子），接口也早返回了 `job_id`，而面板只显示 `job.result.failed`（整批抛错
时它是 `undefined`）→ 用户看到的就是「点了没反应」。

- 端点从 `detail["poseSet"]` 补上 `params["pose_set"]`，取不到就 400 明说；
- 面板补一条 `job.status === 'failed'` 的红字（整批失败必须看得见）；
- `tests/test_poseset.py` 补 `run_batch` 的两条：缺 pose_set 必须抛、
  给了之后候选要**追加**进 `render.json` 且自动选中；
- `tests/test_api.py` 补端点级断言：直接查 `_jobs[job_id]["params"]["pose_set"]`
  （队列 `put` 换成空操作，不惊动 GPU worker）。

### 2) 「全部生成」改成绿色大按钮

之前用 `.pose-batch-btn`，底色是 `--bg`(#0a0a0a) —— 比卡片底 `--bg-2`(#121216)
还暗，看着像"透明的洞"，也不像主操作。改成和「迁移角色」同一套 `gen-btn`：
整行、32px、品牌绿、带文字帧数（`全部生成（12 帧）`）。

次级按钮（导出 GIF / Godot / 重生成）底色从 `--bg` 提到 `--bg-3`，至少比卡片亮一档。

### 3) 底色加「透明底」

用户要的是**底色下拉里的透明选项**（不是抠图）。三处配套：

| 层 | 改动 |
|---|---|
| `types.ts` | `BgTone = 'white' \| 'dark' \| 'transparent'` |
| 节点「更多设置 → 底色」 | 角色迁移 / 姿态批量都加「透明底（模型直接出 alpha）」 |
| `PoseBatchPanel` | `bg` 原样透传（旧代码 `=== 'dark' ? 'dark' : 'white'` 会把透明悄悄换回白底） |
| `generate._qwen_pose_clone` | `bg="transparent"` → `_qwen_call(transparent=True)`（提示词加 alpha 尾句）+ `_postprocess(transparent=True)`；meta 里记 `transparent` |
| `api.submitVnccsPose` | 底色选透明 ≡ 勾透明，两条引擎语义一致 |

> 透明底 ≠ 抠图：透明底是让模型**直接画 alpha**，背景干不干净由模型决定；
> 要稳定透明得另勾「抠背景」。这条写进了 select 的 tooltip。

## 2026-11-02 · 修「姿态批量」两个真 bug：连线接不上、播放没内容

用户报的：**下拉能选角色图，手动拖线不行**；**有播放按钮但没有播放的内容**。

### 1) 拖线接不上（两处，第二处才是"选了也说不接"的根因）

- **`onConnect` 的角色图白名单漏了 `posebatch`**：`['poseclone','spriteswap']`
  里没它，拖过去被**静默忽略**（下拉走的是 `charSource` 字段，不经过白名单，
  所以"能选、不能连"）。已补，并在注释里写明"新增节点类型必须加这里"。
- **`charPathsOf` 传错了对象**：它读的是「传入节点的 `charSource`」，
  而「角色图」是**上游节点的产物**（`image.path` / `job.path`）。
  把上游 `image`/`gen` 节点整个丢进去，取到的是它自己的 `charSource` —— 永远空。
  于是**接好了也一直报"先接一张角色图"**，`全部生成` 一直是禁用的。
  抽出 `imagePathOf(node)`（`image` 取 `image.path`，生成类取 `job.path||raw`），
  两个地方共用。

### 2) 播放没有内容

大图区（对照视图）原本写在 `renderName && render` 分支里 —— **只有出过图才渲染**。
所以抽完帧点 `▶` 时，屏幕上唯一的变化是 38px 缩略格的高亮，等于"什么都没发生"。

拆出独立**舞台** `<div class="pose-batch-stage">`：

- 有源帧就渲染（**没出图也渲染**，单栏就是原始姿势），播放才有内容；
- 出图后自动变两栏（左原始 / 右结果），**两栏同一条时间轴**一起翻；
- 播放条：`▶ 播 / ■ 停` + `播放中 3/12`（静止时 `第 3 / 12 帧`）+ `8fps`；
- `对照原始` 勾选框挪到播放条（`margin-left:auto` 贴右）；
- **点缩略格 / 换姿势集 / 开始出图都走 `stopPlay()`**（清定时器 + 置 false）。
  只 `setPlaying(false)` 不 clearInterval 的话，定时器还在改 `playIndex`，
  再点播会从"半路"开始 —— 这类半截状态最容易让人觉得功能没写完。

顺手：结果缩略网格从 `auto-fill` 换 `auto-fit`（4 帧时 auto-fill 留一排空格子，
把缩略图挤到 44px）。

验证：`tools/verify_pose_batch.mjs` 扩到 **34 条**，新增的几条都是"真拖、真播"：

| 断言 | 怎么验的 |
|---|---|
| 拖一条角色图连线 | **真的鼠标拖**（pointerdown → elementFromPoint → pointerup），图上 `path.link.char` 从 1 条变 2 条 |
| 接上后「全部生成」可点 | 拖之前 disabled=true，拖完 disabled=false（这条才是路径解析的正面证据） |
| 没出图也有舞台 | `.pose-batch-stage` 存在、单栏、`第 1 / 4 帧` |
| 播放时大图真的在换 | 播放前/后的 `img[src]` 不同 + 标注走到「第 3 帧」（mock 每帧给**不同 URL**，否则永远绿） |
| 停下回到静止态 | 标签从「播放中 3/4」回「第 1 / 4 帧」，按钮回 `▶ 播` |

## 2026-11-02 · 「姿态批量」独立成节点 + 对照 / GIF / Godot 导出

**单张和批量拆成两个节点**。上一版把批量塞在「角色迁移」的折叠区里，问题是：

- 两种模式混在一张卡片里，用户永远要先想"我现在是在批还是单张"；
- 还会出现"选了姿势集却没接角色图"的半截状态 —— 而且报错文案把锅甩给了
  角色图，**角色图明明接了**。真因是 `charSource` 只存 `nodeId`，路径得按
  节点现算；面板却去读 `charSource.path`（永远空）。现在由 App 把
  `charPathsOf(id)` 透传下来，不再靠猜。

新节点 `posebatch`「姿态批量」，三件事：

1. **先看清**：选完姿势集立刻铺出**源帧缩略条**，`▶` 连着播 —— 先确认这
   是不是你要的那套动作，再决定要不要花一小时出图。
2. **能对比**：结果区 **左原始 / 右结果**并排（`.pose-batch-pair`），一屏内
   就能看出哪一帧没换好；逐帧「重生成」追加候选，旧的永远留着。
3. **能交付**：
   - **GIF**（原始帧 / 成片各一个）：`workflows/poseset.export_gif`，PIL 合成，
     写进集目录的 `preview.gif`，给团队审核用。成片只取**当前选中的候选**。
   - **Godot 4 工程包**（`workflows/godotexport.py`，照 pixelforge 的
     `godot2d.py` 做的）：逐帧 PNG + `SpriteFrames(.tres)` +
     `AnimatedSprite2D(.tscn)` + 中文 README + zip。zip 条目带 `<folder>/` 前缀，
     解到工程根目录 `res://` 路径恰好成立；写完自检一遍（每个 `res://` 引用都
     能在包内找到），不通过就在结果里报出来。

### 两个格式细节（踩了才知道）

- **GIF 只有 10ms 精度**：要 125ms，文件里是 120ms（按厘秒存）。所以 `export_gif`
  先自己对齐到 10 的倍数再返回实际值，不然界面写 125、文件里 120 对不上账。
- **Pillow 会合并连续相同帧**：四张一样的图写出来是单帧 GIF（优化，不是 bug）。
  测试里必须用**内容不同**的帧，否则断言"应该 4 帧"会红。

验证：`tests/test_poseset.py` +2（GIF 合成 / 跳过未选中的帧）、
`tests/test_godotexport.py` 新增 5 条（load_steps / ext_resource 数 / autoplay /
zip 前缀 / 名称清洗）、`test_api` 补两个导出端点；
新 `tools/verify_pose_batch.mjs` 在真浏览器里点界面：源帧 4 格 → 对照两栏 →
点第 3 帧跟着换 → 两个导出都出下载链接（13 条断言）。
smoke 633 条；截图 `tests/_out/pose_batch/pose_batch_node.png`
（从截图里抓到「帧率」输入框 30px 装不下两位数，已改 40px）。

## 2026-11-02 · 抽帧改成「先抽出来挑」（照 pixelforge 的拆解帧）

上一版是"填帧数 → 点一下 → 直接存盘"，用户全程看不见抽出来了什么。
照 `D://code//pixelforge` 的「拆解帧 · 挑帧」节点重做交互：

**抽帧不再落盘**：点「抽帧看看」先把整条动作抽成一摞缩略图摆在帧墙上，
点格子勾选，最后点「存为姿势集（N）」才写盘。抽错了重新抽，不污染项目目录。

**两把筛子**（只在显示层生效）：

| 筛子 | 用途 |
|---|---|
| 跳帧步长 每 N 帧取 1 | 匀速动作一眼压掉 3/4 |
| 近似度过滤 < N% 当作重复 | 起步/定格/收势那种"连着几帧没动"压成 1 张 |

**筛选只改「显示哪些」，勾选是显式状态** —— 拖滑杆不会把你已经挑好的帧
悄悄换掉，点「应用筛选」才采纳。这是 pixelforge 那边踩过坑的地方，照抄。

另外：工具条（应用筛选 / 反选 / 清空 / 显示全部）、统计（已选 N/共 M、
间距中位、首尾差、近重复对数）、双击某帧全屏放大、序号按**帧序**标。

### 指纹这块踩了三个坑（都是真机才暴露）

1. **alpha 合成把整帧压成纯白**：`viewer.capture` 出的人偶图 alpha 整张是 0
   （内容在 RGB 里）。按"透明合成到白底"算，8 帧指纹一模一样 —— UI 上表现为
   「间距中位 0.0% / 近重复 7 对」。改成只看 RGB。
2. **白底稀释差异**：16×16 整幅灰度平均差，两个完全不同的走路姿势只差
   **0.2%**，滑杆等于没用。改成先按主体包围盒（32×32 上扫非白，正方形裁、
   保持长宽比）再缩。
3. **还是不够**：人偶是白底上的一根细灰人形，灰度平均仍被留白稀释
   （**0.2~0.4%**）。最终换成**剪影二进制指纹**，差异口径改成
   `改动格数 / 两帧墨迹的并集` —— 现在相邻帧 2~6%（Idle）、10~30%（走跑），
   滑杆范围 0~40% 才有意义。

代价：指纹只认"形状变没变"，同一姿势换了打光/颜色会判成一样。人偶是纯灰模型，
不存在这个问题；真接入彩色素材时要重新看这条。

验证：新 `tools/verify_frame_pick.mjs` —— 真浏览器里**像用户一样点界面**：
抽 8 帧 → 帧墙 8 格（默认全选）→ 步长调 2（显示 4 格、勾选仍 8）→ 显示全部
→ 点格子取消 → 清空后保存按钮禁用 → 应用筛选回到 4。18 条断言全绿，
截图存 `tests/_out/anim_import/frame_pick.png`。

## 2026-11-02 · 动作 → 一整套成片（播放 / 项目姿势集 / 批量和候选）

把"从一条动画抽帧"这条链从"存进资料库就完"补成了完整产线：

**人偶编辑器**

- **播放**：`animPlay` 一条消息，按 **wall clock** 推进 mixer（不是 rAF——
  一帧要跑关键点提取 + 全骨 IK，rAF 会被拖成 15fps 且越播越偏）。播放中
  不往宿主推 `pose`（那是"拖动松手"的存档语义，25 次/秒会把项目文件写穿），
  只推 `animTick` 让滑块跟着走；停下时恢复到滑块那一帧。
- **抽帧数量由用户填**：以前写死 2~24，一条慢动作 30 帧抽不全。现在 1~240
  （上限跟后端 `poseset.MAX_FRAMES` 对齐）。
- **不再存进跨项目的资料库**：一整套动作（N 张 768² 人偶图）进
  `library/poses.json` 会把库文件撑爆，而且它属于这次采样、这个项目。
  改为 `assets/projects/<项目>/poses/<用户取的名字>/`（`set.json` + `f_001.png`…），
  **同名一律 409**（不覆盖），每次存都是新文件夹。

**角色迁移**

- 节点里新增「批量：用姿势集出整套」：选一个姿势集 → 取输出文件夹名 →
  逐帧生成到 `assets/projects/<项目>/pose_frames/<名字>/`。
- **候选只增不删**：每帧存 `candidates[]` + `selected`。不满意就"重生成"
  （追加一张并自动选中），**旧图原地留着**，点一下就能选回去——用户很可能
  看完新的又觉得上一版更好。
- 预览：大图 + 缩略网格（一屏看完整套）+ 每帧候选条；`▶` 按 8fps 翻"当前选中
  的那张"当动作预览。
- 同名文件夹：当成"接着往里加候选"；但如果是**另一套姿势集**的成片，
  直接报错让用户改名（帧表对不上会张冠李戴）。

**存储与接口**（`workflows/poseset.py` + `workflows/posebatch.py`）

- 姿势集：`GET/POST /projects/{id}/pose_sets`；
  成片集：`GET/POST /projects/{id}/pose_renders`、
  `.../{name}/select`、`.../{name}/run`（走同一个单卡队列，前端挂 `job_id` 就有进度）。
- 名字清洗复用 `characters.ensure_safe_name`（挡 `/`、`..`、Windows 保留名），
  再叠 `safe_join_under` 兜底。
- 创建时间戳**严格递增**：同一毫秒连存两个集会让"新的在前"退化成目录遍历顺序。
- 顺手删掉了上一轮加的 `/library/poses/batch`（`add_poses`）——那正是这轮要
  换掉的做法，留着两套"存 N 帧"只会让人分不清该用哪个。

**验证**：`tests/test_poseset.py` 8 条（同名拒绝 / 候选只增不删 / 名字越界 /
坏帧不留半套 / 候选上限）；`test_api` 补端点级 409 与 4xx；
`tools/verify_anim_import.mjs` 新增播放断言（1.2s 播放 → 14 个 `animTick`、
t 单调到 1.21s、`animPlaying` on/off 各一次）；smoke 615 条。
面板排版用真实 CSS 单独截图复核过（`tests/_out/pose_batch_panel.png`）——
`.mini` 是 28px 图标按钮，「全部生成」四个字会被挤成竖排，必须另起
`.pose-batch-btn` + `nowrap`。

---

## 2026-11-01 · review 修掉 4 个 bug（两个只有真机能暴露）

这一轮不是加功能，是把上一轮的东西**真机跑一遍**之后修的。两个最要命的 bug
都是"假对象测试"掩盖掉的 —— 单测全绿，真机全错。

### 1. 贴回把 patch 的 alpha 也混进去了（**真机暴露**）

`qwen_edit_image` 返回的是 **RGBA**（`generate.py:455` 的 `convert("RGBA")`），
而 `paste_crop_direct` 按 patch 的通道数决定混几个通道
（官方 `patch[:, :, :, :result.shape[-1]]`）—— 于是 alpha 跟着一起混，
源图的透明背景被 patch 的 255 按羽化权重盖掉。

真机数据：900² 源图 52.5% 透明 → 贴回后背景 alpha 变成 **61** → 再走 refine
就彻底不透明。用户看到的是"表情生成把透明底吃掉了"。

官方没这个问题是因为它的 patch 恒为 3 通道（`_safe_image_batch` 的输出）。
我们必须在贴回前 `convert("RGB")`。

**为什么单测没抓到**：所有 fake_edit 都返回 `Image.new("RGB", ...)`。
现在有一条专门返回 RGBA 的回归测试。

### 2. 抠像前没把透明区合成到幕布色（**真机暴露**）

透明像素的 RGB 是 **0（黑）**。直接送抠像的话，抠像只认幕布色，**黑块被整块
当成前景留下来** —— 一张 52.5% 透明的源图，生成完变成 100% 不透明（一片黑）。

官方 `_prepare_emotion_detailer_input:3651-3666` 是先 `rgb*a + screen*(1-a)`
合成过的。`chroma_key_patch` 现在也这么做。

**为什么单测没抓到**：假 keyer 直接按颜色判绿，没有模拟"透明变黑"这一步。

### 3. 图集会把 `V1/V2` 归档当成表情打包

`staged_image_batch` 重新生成时把旧图搬进 `Sprites/<服装>/<表情>/V1/`。
`collect_frames` 扫子目录当表情 → 图集混进上一版的帧，动画组还多出
`Naked/neutral/V1` 这种名字（引擎按组播动画时会闪一帧旧图）。

### 4. 中断残留的 `.sprites-*` 会让图集导出**直接崩**

原子发布用 `tempfile.mkdtemp(prefix=".sprites-")` 建临时目录，进程被杀会残留。
名字以点开头 → `ensure_safe_name` 拒绝（点不在白名单）→ 抛 ValueError。
用户只看到一句"导出失败"，完全猜不到原因。

`list_costumes` 也有同样的问题（会把 `.sprites-*` 当服装返回）。

### 顺带补的 UX 缺口

图集导出完**前端只有一行文字路径**，既看不到图也拿不到文件。现在有：
缩略图（棋盘底看得出透明区域）、新标签打开原图的入口、`atlas.json` 的入口。

另外图集过大（> 48 帧 或 > 4096px）时会在 notes 里提示按服装分批 ——
多数引擎的纹理上限是 4096，超了上传直接失败。

### 验证

- 真机：900² 透明底源图 → 表情生成 43.5s，**透明占比 52.5% → 52.5%**，
  0% 被破坏（修之前是 52.5% → 0%）
- 后端 183 passed / 5 skip；前端 smoke 571/571

---

## 2026-11-01 · 表情贴边重抠补完 + 内衣底图 + 精灵图集导出

三件事，前两件是照官方补的，第三件是我们自己的工程环节。

### 1. 表情的 alpha 精修（官方第二段）

官方表情链路是**两段**：贴回整图（第一段）+ `bg_remove`（第二段）。之前只做了
第一段，表现是"生成图里本该透明的地方贴着一块幕布色的脏斑"——因为贴回只混
RGB、不动 alpha。

补的三个函数（都在 `workflows/emotiongen.py`）：

| 这里 | 官方 |
|---|---|
| `detailer_region_mask` | `_expanded_detailer_region:3683-3706` |
| `merge_emotion_rgba` | `_merge_emotion_rgba:3707-3748` |
| `refine_alpha_edges` | `_run_emotion_bg_remove:3750-3901`（单帧版） |

三个要点：

- **`detailer_mask` 就是裁切框矩形**（`:1692-1698`，QI2 分支没有 FaceDetailer）。
  "贴边"指框膨胀 8px 后碰到原图 `alpha < 0.999` 的地方 —— 碰上才重抠那一小块。
- **混合必须在预乘空间**。半透明发梢（alpha 0.25）混进不透明新图时，直接对
  straight RGB 做 lerp 会让那块偏亮；预乘后按 `lerp(rgb*a)` 再 un-premultiply
  才是物理正确的。
- **羽化取 `max(mask, avg_pool(mask))`** 而不是直接替换成平滑结果：单纯平均会把
  蒙版边缘（值 1）拉到 0.5 以下，重抠不到边，接缝照旧。

`run_batch` 顺带改成按源图分两条路：透明底 → 局部重抠；不透明纯色幕布 → 整图抠。
以前一律整图抠，透明源图会被再抠一遍，容易啃掉半透明发梢。

### 2. 内衣底图（官方 `remove_clothes`）

`character_generator.py:878-900` + `_run_remove_clothes:2282-2340`。

**它是一套具体服装（白色内衣），不是"去掉衣服"**：官方实现就是一条换装指令
`Dress character: White underwear` 加 Clothes LoRA，产物给后续每套服装当统一起点
（这样新衣服不会带着上一套的袖子残留）。

跟换装的区别只有三处：prompt 一句话（**没有背景段**）、采样 **20 步 / cfg 1.0**
（换装是 25 / 3.0）、输入要先**压到白底**（QI2 encode 不吃 alpha；不压的话
模型看到"角色周围一圈黑边"，经常照着画一圈黑轮廓）。

`POST /characters/<名>/clothes/underwear`，产物落 `Sprites/<服装>/neutral/`，
默认服装名 `Underwear`。实测 39.3s。

**`instruction` 那个字段不用管**：`DEFAULT_WIDGET_DATA` 里给了
`GENERATOR_QWEN_INSTRUCTION`，但 QI2 分支的 `_qi2_encode` 只吃 prompt 和
target_size（`:1430-1435`），那句长指令根本没进模型。

### 3. 精灵图集导出（**不是官方功能**）

`workflows/spriteatlas.py`。官方只管把帧出到 `Sprites/<服装>/<表情>/`，
引擎要的是"一张大图 + 坐标表"——中间这步没人做的话，每个接这套资产的人都要
自己写一遍打包脚本，而且十有八九会把脚底对齐搞错。

**对齐规则**（最容易错的地方）：

```
统一画布 = 所有帧里最大的 (w, h)
每帧在画布内：水平居中 + 底边对齐
```

底边对齐是因为精灵帧要站在地面上——脚底是唯一在所有姿势里都该固定的位置。
按左上角拼的话，高一点的帧会让角色在动画里"跳"一下。

另外三条经验：

- **帧与帧之间必须留缝**（默认 2px）。贴边的话引擎线性过滤会把邻居的像素吸进来，
  表现为边缘一圈杂色。
- **绝不 resize**：尺寸不同的帧是"放进统一画布"（透明底 + 偏移），不是拉伸——
  这一套管线出来的图是像素化的，重采样直接毁掉网格。
- 输出是 TexturePacker JSON Hash（Aseprite 能直接导入），另加一层 `animations`：
  按「服装/表情」分好组，引擎侧不用自己归类。

**与引擎对接**：`meta.canvas` 是建 sprite 用的尺寸（**别按 `frame.w/h` 猜**），
`meta.pivot = (0.5, 1.0)` 是底边中点——地面站位就是帧的底边。
单帧动画会在 `notes` 里标出来（引擎照 loop 播会"卡住不动"）。

`POST /characters/<名>/atlas` 导出、`GET` 查统计。实测 2 帧 → 1540×768 / 85KB。

### 顺带修的一个测试卫生问题

`test_api_underwear_enqueues` 一开始没 mock `remove_clothes`，worker 线程真的去
加载 Qwen 管线跑了一遍——单跑没事，跑全量时把单卡队列占了好几分钟，
`test_api` 里几条互不相干的断言跟着失败（看起来像功能坏了）。
现在测试里显式 `monkeypatch.setattr(clothesgen, "remove_clothes", ...)` 并等任务收尾。
全量相关测试从 85s 降到 25s。

---

## 2026-11-01 · 角色中心 T5：多姿势批量生成（pose_generation）

官方 `character_generator.py:2108-2223` 的 QI2 分支，落在 `workflows/posegen.py`
+ `generate.qwen_pose_image`。角色中心的 T1~T5 到这里**全部做完**。

### 三处不能改的

| 项 | 官方 | 为什么 |
|---|---|---|
| 指令 | `Replace the pose of <image 2> with the pose of <image 1>. Keep the character of <image 2>. {prompt}` | `image 1` 是**姿势**、`image 2` 是**角色**；顺序反了就是"把角色的姿势套到骨架上"，出来是一张骨架 |
| 种子 | 整批共用一个（`pose_sampler.inherit_pipe=True`） | 每张随机会让同一套动作的衣服/发型/画风互相飘 |
| 采样 | `steps 20 / cfg 1.0 / denoise 1.0`（`:825-833`） | 20 步是这套 LoRA 的训练点；cfg=1 是单次前向 |

另外 official 是 `+ prompt` **直接接在同一行**（不是换行）——跟画布那个「角色迁移」
节点用的 `_VNCCS_TEMPLATE`（换行拼）不是同一条，两条产线各照自己的原文走。

### 发布是**整批替换**，不是逐张追加

`staged_image_batch`（旧帧进 `V1`），这叫"一次出一整批"。而表情那条是**逐张追加**
（`publish_image`）。搞混的后果是把先出的帧挤进 V1 —— 这个坑项目里踩过一次，
所以两条路的函数分开、注释里互相指明。

### 还要往 prompt 追一句背景

官方 `_prompt_with_solid_background:1827-1843`：非透明底时追
`Change background to solid {bg} color`，透明底追 `Transparent background with
alpha channel.`。**必须要**——姿势图与人偶图都带白底，不指定背景出来就是白底，
抠像没法定阈值，也跟其它帧拼不到一起。（真机第一版就漏了这句，出的是白底图。）

### 还要往 prompt 追一句背景

官方 `_prompt_with_solid_background:1827-1843`：非透明底时追
`Change background to solid {bg} color`，透明底追 `Transparent background with
alpha channel.`。**必须要**——姿势图与人偶图都带白底，不指定背景出来就是白底，
抠像没法定阈值，也跟其它帧拼不到一起。（真机第一版就漏了这句，出的是白底图。）

### 串行而不是 batch

QI2 是逐张 encode→sample→decode（`:2205`），每张的 latent 尺寸可能都不一样，
batch 只会把显存顶爆。

### 接口与界面

`POST /characters/<名>/poses/generate`（`poses` 或 `pose_dir` 二选一）。
素材页底部加「多姿势批量」：选骨架帧目录 → （可选）一句追加描述 → 生成，
带进度条（一批几十张，没进度只能干等）。

### 顺带修的一个测试卫生问题

`test_create_conflict_and_bad_name` 不清残留：上一次跑到一半中断就会留下 `Dup`
目录，下次在**第一句**就撞"角色已存在"，看起来像功能坏了。现在用例自己先清理。

---

## 2026-11-01 · 角色中心 T4：换装（Clothes Designer）+ T6 服装向导

官方 `clothes_designer.py:413-448`（prompt）+ `:493-790`（QI2 出图分支），
服装向导 `:885-994`（原计划的 T6，跟 T4 一起做掉了）。

### 换装**不是"只存描述"**

这是最容易理解错的地方：它是把五个字段拼成一条图生图指令，拿**裸装底图**
重画一张**换装后的底图**，存成 `Sprites/<服装>/neutral/sprite_pose_0001.png`。
之后所有姿势/表情都从这张底图出发。所以衣服是"画进图里"的。

三个必须逐字照抄的部分：

```
Dress the character:
{top} {bottom} {head} {shoes} {face}        ← 顺序是官方写死的，不是表单顺序
flat uniform pure green background, exact RGB (0, 255, 0), hex #00FF00, no gradient
```

背景段把 **RGB 与 hex 都写进 prompt** 不是啰嗦：只说 "green" 时模型会画渐变、
墨绿甚至带紫的绿，抠像 `tolerance` 一放宽就把衣服一起吃掉了。蓝底还要额外
在负向词里压住紫色/渐变。

负向词 `(naked, nude, nipple, penis, vagina:2.0)` 是**带权重的**（`:2.0`），
不是普通关键词——换装时"别把人画成裸的"优先级必须压过"画衣服"。

### Clothes Core LoRA

`VNCCS_QI2_ClothesCoreV2.6.safetensors`，**强度 1.0、clip 强度 0、没有触发词**
（`:286,301,711-716`）——它让模型"会换衣服"，不是某个角色/风格。挂了它就
独占 LoRA 槽位（`fast_lora=False`），所以这条路径不叠快速模式的 Lightning
（官方换装也是 25 步全跑）。

### 采样与底图

- `steps 25 / cfg 3.0`（`:555-556`）、`target_size 1024`（`:163`）。
- 底图 = `Naked` 最新一张，没有就退 `Original`（`:479`）。**不能用目标服装
  自己的图**：新服装本来就是空的，而换装的定义是"从裸装改成这套"。
- "最新"用 `(mtime, path)` 判，并且**跳过 `V1/V2` 归档目录**（`:203-205`）——
  不跳过的话重生成一次，"最新底图"就指到上一版，换装会照旧图改。

### 实测（RTX 4090）

灰人偶 + "白色护士服、薄荷绿镶边、护士帽、白鞋" → **37.8s**（768²/8 步/cfg 3）。
图上确实穿上了这套衣服、背景是纯绿、姿态与体型不变。

### 接口

| 端点 | 作用 |
|---|---|
| `POST /characters/<名>/clothes/design` | 出换装底图（走单卡队列，挂 Clothes LoRA） |
| `POST /characters/<名>/clothes/wizard` | 一句话 → 五字段（不落盘） |

前端「服装」页：一句话拆服装 → 字段灌进表单 → 保存 → 「出换装底图」。
**出图前会先把表单存进 config**：换装读的是 config 里的服装信息，不先存
就等于拿上一次的内容去画。

### 顺带修的一个真 bug

**LoRA 文件名带点会让 peft 崩**：adapter 名沿用了文件名，而
`torch.nn.Module.add_module` 禁止名字含 `.`，于是
`VNCCS_QI2_ClothesCoreV2.6` 直接抛 `KeyError: module name can't contain "."`。
现在 adapter 名统一走 `_adapter_name()` 清洗（文件名不变，用户认的是文件名）。

---

## 2026-11-01 · 角色中心 T3：AI 向导（一句话造角色）

官方 `character_creator_v2.py:1740-1869`，纯文本入口（**不需要 mmproj**），
跟 T2 共用 `workflows/qwen_llm.py` 的加载/超时/解析。采样
`max_tokens=900 / temperature=0.3`。

### 真正值钱的不是 prompt，是**后端后处理**

官方 `:588-604` 那层纠错比 prompt 更管用——模型的默认审美会把三个字段填错，
每个都有专门的规则：

| 字段 | 模型爱怎么错 | 官方的纠法 |
|---|---|---|
| `race` | 写成 `afro_student` / `black student` / `asian girl` | 只留 `RACE_OPTION_TAGS` 或预设库里的物种；全空且原话有人类线索才回落 `human` |
| `skin_color` | 几乎总填 `fair/pale skin` | **用正则从用户原话反推并覆盖**；原话没提肤色就直接置空（不许猜默认值） |
| `body` | 只写个 `body` / `unknown` | 剔占位词；没写 build 就从原话补；**女性没提胸围强制加 `small_breasts`** |

`small_breasts` 那条不是审美：QI2 的结构化 prompt 里有一行 `Breast size:`，
空着模型会自己乱猜，官方宁可给个保守默认值。
（顺带注意：官方在"body 已经带 build 词"时会**提前返回**，那时不补胸围。）

### 注入标签白名单

user prompt 里用 `json.dumps` 塞进 7 组标签（race / skin_color / hair / eyes /
face / body / additional_details）与 20 条 `skin_color` 可选值。来源是
`character_presets_v2.json`（前端也读同一份，不复制第二份）。

不注入的话模型会自己编 `silver_hair_ponytail_v2` 这种没人认识的 tag，
而 QI2 的风格 LoRA 只认官方那套写法。

### 中文本地化（**官方没有，我们补的**）

官方那三组正则只认英文。中文描述（"一个黑人女学生"）会命中"没有肤色线索"
分支，把模型读对的 `dark skin` **清空**，`race` 也变空串——而中文是我们这边
的主流输入。补了一组中文关键词，**优先级与官方完全一致**（黑皮 → 棕皮 →
古铜 → 橄榄 → 苍白 → 白皙），只加词不改判断顺序。代码里 `_CN_*` 那段有说明。

### 实测

"一个苍白的精灵女剑士，银发红瞳，身材纤细，脸上有疤" → **40.8s**：

```
race elf / skin_color pale skin / hair silver hair, shoulder-length hair…
eyes red eyes, luminous irises / face …, scar across one cheek
body slim build, small_breasts / sex female / age 25
```

`small_breasts` 是后处理补的，其余都是模型读出来的——说明规则和模型配合得对。

### 接口

`POST /characters/wizard`（body: `{description, timeout}`）。跟看图反推一样
**不落盘**，前端把字段摆成表格，用户点「用这些属性」才写进 config。

---

## 2026-11-01 · 角色中心 T2：看图反推属性（本地 4B 模型）

官方 Cloner 的旁路（`character_cloner.py:219-412`）搬进 `workflows/qwen_llm.py`。

**它是旁路，不是依赖**：出图链路一次都不碰它。模型没装 / fork 装错 / 推理卡住，
都必须退成"手填"，而不是把界面卡死。

### 链路

图片 → 压到背景色上（透明图不压会变黑，模型从"黑边"读出不存在的特征）
→ 缩到 512 → JPEG q85 → base64 data URL 塞进 `image_url` 内容块
→ Qwen3.5-4B（mmproj 多模态，`n_ctx=4096`）→ `max_tokens=1024, temperature=0.2`
→ `parse_wizard_json` 三级降级 → 白名单过滤 → 11 个字段。

### 跟官方的两处必要差异

| 项 | 官方 | 我们 | 为什么 |
|---|---|---|---|
| 超时 | **没有**（`preview_runtime.py:25-52` 只有串行锁） | 默认 300s，超时回 `TIMEOUT` | 阻塞调用卡住时官方只能重启进程，完全查不出原因 |
| 显存 | 模型常驻 | **用完就卸**（`MEOW_LLM_KEEP=1` 可留驻） | 4.5GB 常驻跟 Qwen 图像管线抢 24GB，切回去出图就可能 OOM。代价是每次多 3~5s 加载，而推理本身 20~40s |

### 解析为什么要有三级降级

模型输出**经常不是合法 JSON**：包一层 json 代码块、包成数组、写尾逗号、掺一段自然语言。
官方 `qwen_vl.py:89-118` 的办法是：

1. `json_repair.loads`（修坏 JSON；官方 requirements 里有，我们也加进 `requirements.txt`）
2. 结果是 list → 取 `[0]`
3. 剥代码块 → 正则捞 `{...}`
4. 还不行 → `PARSE_ERROR`，**把模型原话放进 `raw`** 返回（不然用户只看到"失败了"）

### 白名单是防注入

官方 `:374-388` 只留 11 个 key。模型偶尔会多吐 `positive_prompt` / `config` 这类键，
直接塞进角色 config 就能改到别处去。测试里钉了一条（多吐的键必须被丢掉）。

### 实测

人偶图（768²）：加载 4s + 推理 19s ≈ **23s**，输出 `skin_color: gray skin` /
`aesthetics: 3d render, silhouette` 这种看图说话的结果——**质量归真图看，单测不管**。

### 接口

`POST /characters/<名>/analyze`（`image` 可空 = 取这个角色最新一张精灵帧）。
**只返回字段，不写 config**——属性错了会一路影响后面所有出图（prompt 由它拼），
必须让用户确认。错误码照官方：`DEPENDENCY_MISSING` / `MODEL_MISSING` /
`MODEL_INVALID` / `MMPROJ_MISSING` → 503，`TIMEOUT` → 504，
`PARSE_ERROR` / `INVALID_RESPONSE` → 502，前端按码给不同话术。

---

## 2026-11-01 · 角色中心 T1：表情批量生成（只改脸）

官方 Emotion Studio 的 QI2 分支（`character_generator.py:1591-1708` 单张 +
`emotion_generator_v2.py:675-761` 编排）整套搬进 `workflows/emotiongen.py`。

**它跟"再出一张整图"的区别**：表情是把**脸框那块裁出来放大重画**，再按羽化蒙版
贴回原图，所以脖子、衣服、动作逐像素不变。一套姿势帧换 5 个表情，动作不会漂。

### 管线（每一段都有官方行号）

```
YOLO 找脸框（外扩 2×50px） → 裁 → 正方形补边到 32 的倍数（replicate）
  → 按总像素档位放大（2048 → 2MP，边长向下取 32 倍数）
  → Qwen 指令改图（denoise 1.0，只给表情 prompt）
  → 缩回对齐尺寸 → 切掉 pad → 羽化蒙版贴回原图（alpha 不动）
  → Sprites/<服装>/<表情>/sprite_<表情>_<源帧序号>.png
  → Faces/<服装>/<表情>/face_<表情>_<源帧序号>.png（生成的那块脸原图）
```

### 六个纯几何函数（零模型，逐行翻译）

| 这里 | 官方 | 备注 |
|---|---|---|
| `pad_to_square_multiple` | `:1571` | **replicate 补边**，不是补黑/灰 |
| `crop_feather_mask` | `:1516` | 边缘斜坡：`min(到上下边, 到左右边)/feather` |
| `paste_mask` | `_blur_detail_mask:1449` | 斜坡再高斯模糊，半径 = feather |
| `paste_crop_direct` | `:1483` | **只混合到 patch 的通道数** —— patch 是 RGB 时不动 alpha |
| `cut_padding` | `:1671` | 贴回前把 pad 反向切掉 |
| `crop_target_size` | `:1636` + `ImageScaleToTotalPixels` | `target_size` 是**总像素档位**不是边长 |

### 几个容易抄错的地方

1. **`target_size` 不是边长**：`megapixels = target_size/1024`，2048 就是 2.0 MP
   （1024² → 1440²）。边长**向下**取 32 的倍数，向上取会白算 token。
2. **dilation 官方用了两次**：传进 Impact 的 `detect()`（内部已扩一次）后又手动
   扩一次，净效果 `2 × dilation`（默认 100px）。这是照抄不是笔误——脸框要带上
   头发和脖子，模型才知道这张脸长在什么身上。
3. **QI2 分支只用 template**：不拼 `face_details`、种族、发型、服装（`:4142`）。
   拼了反而会把脸画成另一个角色。
4. **负向词要多一串水/汗**（`emotion_generator_v2.py:669`）：不然"难过"会顺手
   画一串眼泪，而精灵帧不要那个图层。
5. **产物序号跟源帧对齐**：`sprite_pose_0003.png` → `sprite_angry_0003.png`
   （官方 `save_index`，`:4541`）。不是"下一个序号"——那样两张对不上，翻图就错位。

### 两处刻意的取舍（都有理由，不是猜的）

| 项 | 官方 | 我们 | 为什么 |
|---|---|---|---|
| 推理边长 | 2048 → 1440² | **封顶 1024** | 24GB 单卡 1440² 单张 60s+ 且显存压线；脸框本身只有 200~400px，1024² 已远超源信息量。要按官方原值跑传 `crop_max_side=2048` |
| 人脸检测设备 | ComfyUI 里跟模型同卡 | **CPU** | Qwen 的 transformer 已经占着卡，YOLO 再挤进去会让出图阶段显存碎片化；CPU 上一张 1024 的图约 100ms |

### 实测（RTX 4090，人偶 768² 源图，脸框 256²）

- 1024² / 8 步 / cfg 3.0：**42.7s 一张**（cfg>1 是真 CFG，每步两遍 transformer）
- 25 步（官方默认）约 **110s 一张**：一批 3 帧 × 5 表情 ≈ 27 分钟。
  所以界面上**必须有进度条**，`/jobs` 的 `progress` 每张更新一次。
- 贴回后除脸以外逐像素不变，接缝不可见（羽化带里是渐变的中间值，不是 0/255 硬边）。

### 接口

| 端点 | 作用 |
|---|---|
| `POST /characters/<名>/emotions/detect` | 列源帧 + 检出脸框（**不跑生成模型**） |
| `POST /characters/<名>/emotions/generate` | 提交批量任务（走同一个单卡队列，返回 `job_id`，`/jobs` 带 `progress`） |

先检脸框再提交是官方 Emotion Studio 的顺序：没有源帧或一张脸都没检出时要立刻
说清楚，别等提交完 3 分钟才告知"没源帧"。

---

## 2026-11-01 · 3D 人偶取代骨架编辑 + 资料库落盘

### 1. 「骨骼编辑」节点 → 「人偶姿势」节点：2D 骨架编辑器整个删掉

以前是**两套并存**：节点正文是自写的 2D 伪 3D 骨架编辑器，旁边再挂一个
「编辑 3D 人偶」按钮（弹窗打开官方 PoseStudio）。用户的一句话点破了：
「都有 3D 人偶了，还用骨架图编辑人偶干嘛」。

现在只有官方那套，而且是**正文内嵌**（不是弹窗）：

| 项 | 以前 | 现在 |
|---|---|---|
| 编辑对象 | 18 个 2D 关节（x/y + 手填深度 z） | MakeHuman 网格 + 53 根骨，官方两骨 IK + 极点约束 |
| 换朝向 | 绕竖轴正交投影「假装 3D」 | 相机机位绕 Y 渲染，0/90/180 是真的正面/侧面/背面 |
| 产物 | 两种：OpenPose-18 线骨架 + 3D 人偶图 | **只有白底灰 3D 人偶** |
| 存档 | 18 个 2D 点（重开要重新猜深度） | 官方 `getPose()` JSON，弧度级还原 |

删掉的：`web/src/components/SkeletonEditor.tsx`（含自写的 `renderSkeletonPNG`、
7 个 2D 预设、方位角滑块）、`MannequinHost.tsx`（弹窗壳）。

新增：`web/src/components/PoseStudio.tsx`（内嵌壳）。
iframe 用 `studio.html?embed=1`（只留画布，控制条由宿主提供）；「放大」是
把容器改成 `position:fixed` —— **不移动 iframe 节点**，一动浏览器就重载，
37MB 的人偶模型要重新解算。

### 2. 官方 VNCCS 姿势库（191 个）进了项目

`web/public/vnccs-pose/PoseLibrary/`（7.2MB：195 个 json + 186 张 webp 预览）。
json 是 PoseStudio 原生格式，`setPose()` 直接吃，不用转换。

- `GET /pose_library` 给清单（**只给路径**，7MB 本体不塞进 JSON）；
- 清单带 `_library.tags`，所以能按 `front view` / `side view` / `rear view`
  筛，也能顺手给导出机位一个合理默认值（侧面 90°、背面 180°）；
- 分类大小写去重过（官方库里 `Kneeling` 和 `kneeling` 是两批人写的）。

### 3. 提示词收藏 / 姿势库 / 界面偏好：全部落盘，localStorage 一个字不留

用户要求：`localStorage` 不存任何东西。以前提示词收藏和蒙版颜色都在浏览器里，
清一次站点数据就全没了。

- 落点 `library/`（`prompts.json` / `poses.json` / `prefs.json`），
  与 `templates/` 同级——都是跨项目共享的用户资源，备份就是整个目录拷走；
- 写入原子（临时文件 + `os.replace`），坏文件一律当空库读回；
- 迁移顺序是**先 peek 后删**：`readLegacy()` 只读不删 → `POST /library/seed`
  （服务端只补不覆盖，两个浏览器的存档都能留）→ 确认收到才 `clearLegacy()`。
  反过来的话，请求一失败这批收藏就彻底没了。

### 4. 资料库浏览面板（`LibraryDrawer`）

收藏是会持续变多的东西，侧栏那点高度十条之后就只能瞎翻。所以给了一个统一的
浏览面板：两个 tab（提示词 / 姿势）、搜索（提示词连负向词一起搜，姿势连标签一起搜）、
卡片网格带预览缩略图、Esc 关闭。

姿势页分两个来源：**VNCCS 官方库**（191，带分类筛选）与**我存的**（自己攒的，
跨项目复用）。官方那条可以一键「收藏」进自己那栏。

### 5. 实测/踩坑

- **导出图背景有一层彩色网格**（占画面 4%）。根因：官方 `capture()` 会隐藏
  gridHelper / skeletonHelper / 关节标记，但**没有**隐藏方向天穹
  （`VNCCS_DirectionalSkydome`，那个彩虹经纬球）。修法是 capture 期间临时藏掉。
  实测：导出 PNG 从 132KB → 39KB，彩色像素 4.2% → 0.00%。
- 真实浏览器（无头 Edge + SwiftShader）在**节点卡片那么小的 280px** 下验过：
  嵌入模式画布正常铺满、套官方姿势、导出 768² 白底灰人偶，左上角 (255,255,255)、
  人偶完整不裁切。
- 前端冒烟 **415/415**；新增后端 `tests/test_library.py` 11 条。

---

## 2026-11-01 · 角色中心续：精灵帧翻看器 + 纯色幕抠像

### 1. 素材页换成官方那套翻看器（`SpritePreview.tsx`）

以前素材页是「把所有图一次性塞进列表」。精灵帧一多（几十上百张）首屏就卡死，
官方的做法是**两步**：先问数量，再按帧号单张取。

移植自 `web/vnccs_common.js:739` `createSpritePreviewNavigator`，几个关键点都是
踩出来的经验：

| 官方的做法 | 为什么 |
|---|---|
| 先 `poses/meta` 拿 count，再 `poses?index=N` 单张取 | 一次性塞 N 条 URL 进 JSON，帧一多就卡住 |
| requestId 竞态保护 | 没有它，快速换服装会「晚到的旧请求覆盖新图」 |
| 相邻帧预取 `index±1` | 翻页手感全靠它 |
| `index % count` 取模 | 翻到最后再翻一下绕回第一张，不会走到「第 0 张之前」 |
| `count > 1` 才显示导航条 | 只有一帧时给按钮 = 点了没反应还没人解释 |

枚举规则照抄 `utils.py:936-981`：6 个候选根目录（指定服装 → Naked → Original、
各自先 Neutral 再根目录）**有优先级、第一个非空就停**。所以不给服装名时退到
最基础的裸装帧，不会把所有服装的图混着翻。

**官方目录是首字母大写**（`Sprites/Original/Neutral/`），我们归档用小写
`neutral` —— 两种都列进候选，否则把别人的角色库拷进来一张都翻不出来。

### 2. 纯色幕抠像（上一节已提交 b63862b）

官方 `nodes/chroma_screen_matte.py` 整文件搬运，纯 torch 零模型。接在角色出图的
收编步骤（官方 bg_remove 阶段）：另存 `_matte.png`、**原图保留**、抠失败只记
`matte_error` 不判任务失败。

**白底/灰底不自动抠**：幕布色是按边框取样的，白毛白甲会被一起判成背景。

---

## 2026-11-01 · 再续：官方属性标签选择器 + 精灵帧翻看器

### 属性标签选择器（characterTags.ts / TagPicker.tsx）

手填"blonde hair / blonde_hair / Blonde"三种写法混着用，出来的图会飘。官方的
解法是给每个属性字段配一组标签，点选后写进 tag 里的标准写法。移植自
`web/character_presets.mjs`（全文 60 行，两个纯函数）。

**用哪份数据是个坑**：官方有两套标签库 —— `character_presets_v2.json`
（creator_v2，15 个分类，覆盖全部字段）给 QI2；`character_tags.json`
（legacy，6 个分类只有 hair/eyes/face）给 SDXL 老 UI。官方是 `mode === 'qi2'`
时强制 creator_v2（`:4000`），我们既然只留 QI2 就用前者。

照搬的两个细节：写入的是 tag 的**空格形式**、已选项按 **tag/label/synonyms
三种别名**匹配且**保留用户自定义文本**（否则手写的描述会被"选一次标签"清掉）。

后端 `race_features` 本来就吃这套格式（按逗号拆 + 处理权重），所以**多选做混血**
（官方 UI 上的 "Combine species for hybrids"）直接能用。字段入口是 8 个
（FIELD_CATEGORIES 7 个 + `breast_size` 是官方调用处的特例）。

---

## 2026-11-01 · 权重下载脚本 `tools/fetch_vnccs_weights.py`

四个可选权重（AI 向导的 gguf、表情的人脸框、二次元风格与换装一致性两个 LoRA）
装齐要 4.5GB，之前只写在文档里等人手动下。现在一条命令搞定，并且**核对过一遍**：

**真实体积**（走镜像查的 files_metadata，跟之前记的估算不一样）：

| 权重 | 体积 |
|---|---|
| Qwen3.5-4B-Q8_0.gguf | 4.17 GB |
| mmproj-F16.gguf | ~1 GB |
| VNCCS_QI2_AnimeOverhaulV2 | **152 MB**（之前估 300） |
| VNCCS_QI2_ClothesCoreV2.6 | **152 MB** |
| face_yolov8m.pt | 49.6 MB |

核对时抓出三个坑：

1. **mmproj 的文件名跟官方代码对不上**。官方 `qwen_vl.py:68` 写
   `mmproj-Qwen3.5-4B-F16.gguf`，仓库里**没这个文件**（实际 `mmproj-F16.gguf`），
   照抄会 404。官方自己也是先找短名（`_find_qwen_vl_mmproj`，:171-187）。
2. **LoRA 下完必须转格式**：官方是 ComfyUI 格式，我们的引擎只认 diffusers 原生，
   不转挂上去会**静默不生效**（什么都不报，只是"出图跟没挂一样"）。
   脚本自动跑 `convert_lora.py`。
3. 两个 LoRA 真实文件名**带版本号**（AnimeOverhaulV2 / ClothesCoreV2.6）。

`face_yolov8m.pt` 官方只给了路径（`bbox/face_yolov8m.pt`）没给来源，仓库是
用户提供的（Tenofas/ComfyUI），已联网核对存在（49.6 MB）并真跑下载验证通过。

脚本行为：已存在且校验通过就跳过（不重复下 4GB）；校验照官方做（GGUF magic +
>1MB，`.pt` 看 zip magic 但**不 torch.load**，没装 ultralytics 时会炸）；
`--dry-run` 真的不写磁盘；`--endpoint` 支持国内镜像（这里直连 huggingface.co
是 502，镜像通）。

llama-cpp-python 必须装 **JamePeng fork**（上游没有 Qwen35ChatHandler）。官方
是装按 Python/CUDA/平台拼好的预编译 wheel（`install.py:21-47`），不是
`pip install llama-cpp-python`。脚本会打印该装的 wheel URL，**不自动装**——
装 wheel 有换掉 torch 的风险。

顺带：`ROOT / "models"` 这种硬编码收进 `paths.MODELS_DIR`，新增
`LORA_DIR` / `LLM_DIR` / `BBOX_DIR`（路径只有一处定义）。

验证：`tests/test_fetch_weights.py` 12 条（离线，不联网）；真跑下载过
face_yolov8m.pt（49.6MB → 校验 → 落盘 → 清掉空壳目录）。

---

## 2026-11-01 · 五个权重装齐，顺手修好项目里挂不上的 4 个 LoRA

### 1. 权重全部到位（实测）

| 权重 | 体积 | 验证 |
|---|---|---|
| Qwen3.5-4B-Q8_0.gguf | 4.27 GB | 落 `models/llm/Qwen3.5-4B/` |
| mmproj-F16.gguf | 641 MB | 同上 |
| VNCCS_QI2_AnimeOverhaulV2 | 152 MB | 转格式后引擎能列出 |
| VNCCS_QI2_ClothesCoreV2.6 | 152 MB | 同上 |
| face_yolov8m.pt | 49.6 MB | 人偶图上检出人脸 conf 0.751 |

**下载速度差 10 倍**：直连 huggingface.co 502（代理拦），hf-mirror 通但只有
1.2 MB/s；走 127.0.0.1:7897 代理 **11.7 MB/s**。脚本加了 `--proxy`。
（中途用镜像下了 577MB/4.2GB 才测出这个差距，切代理后 6 分钟下完。）

### 2. 运行时依赖

- **llama-cpp-python 0.3.46（JamePeng fork wheel）**：装完实测
  `Qwen35ChatHandler` 能 import；**torch 没被动**（装前装后都 2.6.0+cu124）。
  ⚠️ 别加 `--no-deps` —— 那样会缺 `diskcache`，handler import 不起来
  （我就是这么踩的，补装 diskcache 才好）。
- **ultralytics 8.4.196**：`YOLO('face_yolov8m.pt')` 类别 `{0:'face'}`，
  实测检出人脸。依赖清单里写清了 llama-cpp-python 不能装官方版及原因。

### 3. 项目里 4 个 LoRA 一直挂不上（静默）

`available_loras()` 只列出 7 个，但目录里有 11 个文件 —— 4 个是 ComfyUI 格式，
挂上去**一个层都匹配不到，界面什么都不报**，只是"出图跟没挂一样"。

其中 `nsfw-posture-v3` / `qwen21-nsfw-image-edit` 是 ComfyUI 的**第三种变体**：
没有 `diffusion_model.` 前缀、键名带 LoRA 名（`.lora_A.default.weight`）。
`convert_lora.py` 原先只认带前缀的两种，遇到它一个键都转不出来。
已扩展 `_strip_prefix`（认无前缀）与 A/B 判定（正则兼容 `.default`）。

现在 11 个 LoRA 全部可用。四种格式都能转：

| 格式 | 键样例 |
|---|---|
| diffusers 原生 | `transformer...lora.down.weight` |
| ComfyUI | `diffusion_model...lora_A.weight` |
| ComfyUI 变体 | `transformer_blocks...lora_A.default.weight` |

---

## 现状

**底模只有 Qwen-Image-2.1**——SD 的代码与权重都已移除（见「底模」章节）。

| 模块 | 状态 |
|---|---|
| 引擎层（懒加载 / 显存守卫 / fp8 量化缓存 / cpu offload） | 已完成，实测通过 |
| W1 文生图 | ✅ 768²/16 步 ≈ 10s，峰值 11.4GB |
| W2 图生图（指令改图） | ✅ 源图当参考、提示词当指令；raw 原图一律落盘 |
| W3 局部重绘 | ⚠️ **近似实现**：整图指令改图 + 蒙版贴回（选区外逐点不变） |
| W5 局部 AI 重绘 | ⚠️ **近似实现**（同一机制，多一步色板吸附） |
| W4 姿态序列 | ⚠️ **近似实现**：骨架当参考 + 指令软引导；可选「两阶段」把姿态偏差 102px → 77px |
| W4 质量闸门 | ✅ 关节数 / 前景连通域 / 贴边三道判据；**非人形主体自动跳过补提**（每帧省 25~30s） |
| 画布「角色迁移」wpc | ✅ 双图条件（角色图 + 骨架/轮廓）；ref_mode 支持非真人素材 |
| 画布「序列帧替换」wss | ✅ 逐帧重绘 / 确定性换皮两模式，共享色板 |
| 画布「像素转换」wpx / 「转像素」topixel | ✅ 不跑模型，纯后处理（网格对齐 + 色板量化 + 轮廓描边） |
| 抠图 / 去白边 | ✅ 色键 + 语义分割 + 抠空自动回退 + **已有 alpha 直接复用** |
| 原生透明底（RGBA） | ✅ Qwen 的 VAE 第 4 通道（提示词冲突会自动清理） |
| **速度** | ✅ 单帧 **10.1s**（关真 CFG + 16 步，见 `docs/PLAN-QWEN-ONLY.md` §2.10） |
| 显存 | ✅ 峰值 **11.4GB**（transformer 与文本编码器各一份 fp8 缓存） |
| FastAPI + 资产索引 | ✅ 单卡串行队列、参数按签名过滤、产物落 projects/<scope>/ |
| 游戏动效导出 | ✅ GIF/APNG + Godot 4 tres/tscn |
| **人偶姿势节点** | ✅ 官方 PoseStudio 的 3D 人偶（内嵌），拖关节摆姿势、8 个导出机位、从照片提姿势、姿势库存取 |
| 视频（Wan2.2） | ⚠️ 有工具脚本，**没接画布路由** |
| LoRA / LoRA 训练 / ControlNet / AnimateDiff / Refiner / hires | ❌ 已随 SD 一起删除 |

## 实测数据

本机 RTX 4090 24GB / 32GB 内存，Qwen-Image-2.1（transformer 与文本编码器各一份
fp8 缓存），**单次前向（cfg=1.0）**，每次跑前预热、多次取中位数。

| 用例 | 耗时 | 峰值显存 |
|---|---|---|
| **文生图 768² / 16 步（默认档）** | **10.1s** | 11.4GB |
| 文生图 768² / 24 步 | 12.7s | 11.4GB |
| 文生图 768² / 24 步 + 真 CFG（cfg=4） | 21.1s | 11.4GB |
| 文生图 1024² / 16 步（立绘档） | 14.9s | ~13GB |
| 文生图 1024² / 24 步（立绘档） | 19.8s | ~13GB |
| 文生图 640² / 24 步 | 10.9s | 11.4GB |
| 文生图 512² / 24 步（**不推荐**，见下） | 8.9s | 11.4GB |
| 逐帧 w4（768²/16 步，含抠图与后处理） | 12.9s/帧 | 11.4GB |
| 逐帧 w4 两阶段（草图 + 换皮） | 25.8s/帧 | 11.4GB |
| 冷启动加载（两块 fp8 缓存） | ~20s | — |

**一帧的时间花在哪**（`tools/bench_qwen_profile.py`，768²/24 步中位数）：

| 段 | 耗时 | 占比 |
|---|---|---|
| **DiT 前向** | **18.84s** | **86%** |
| 文本编码 | 1.79s | 8% |
| VAE 解码 | 1.12s | 5% |
| 权重搬运（cpu offload） | 0.12s | 0.5% |

三个由此得出的结论：

1. **真 CFG 是最大的浪费**：`true_cfg_scale > 1` 且给了负向词时，管线每个 step
   跑**两遍** transformer（48 次 vs 24 次）→ 21.1s vs 13.6s。
   画质对照（6 个提示词）没掉，所以默认关掉（cfg=1.0，界面上的 CFG 仍可调）。
2. **步数下限是 16**（不是旧文档里的 24）：旧结论是在真 CFG 开着时测的，
   已经作废。**白/浅色主体**（白猫、白甲、幽灵）建议 20~24 步。
3. **不要低于 640²**：512² 上 DWPose 经常检不出关节（白猫骑士 0 个），
   质量闸门会误判。

完整拆账与实验脚本见 `docs/PLAN-QWEN-ONLY.md` §2.6 与 §2.10。

## 姿态提取（W4 的入口）

```
图片 / GIF / 视频 / 精灵图
        ↓  YOLOX-L（onnx, 640×640，person 类）
   人体检测框 → 扩成正方形 ×1.25 → 缩到 288×384
        ↓  RTMPose（dw-ll_ucoco_384.onnx, SimCC 头）
   17 个 COCO 关键点 → 补 neck → 重排为 OpenPose-18 → 按官方配色渲染
        ↓
   ControlNet OpenPose 条件图
```

```bash
# 单图 / 精灵图（按网格切）/ 视频（每 3 帧取一帧）
python tools/pose_from_image.py sprite.png          -o assets/pose
python tools/pose_from_image.py sheet.png --grid 8x1 -o assets/pose
python tools/pose_from_image.py walk.mp4  --step 3   -o assets/pose
```

两个实现细节值得记：

1. **不用 controlnet_aux 的 DwposeDetector**：它依赖 mmpose + mmdet + mmcv（Windows 编译极痛），
   而且要 .pth 不是 onnx。这里直接用 DWPose 官方导出的两个 onnx 自己串链路。
2. **检测器可换**：`--detector auto|yolox|torchvision|none`。`auto` 优先 YOLOX，
   权重不在时退到 torchvision FasterRCNN（权重来自 download.pytorch.org，国内可达），
   两者都没有则整图回归——实测整图模式对人形精灵也能出 17/18 个关节，只是比例偏差更大。

实测（CPU，512² 输入）：

| 模式 | 单帧耗时 | 可见关节 |
|---|---|---|
| 整图回归（无检测） | 26ms | 17/18 |
| YOLOX 检测 + 裁剪 | 515ms | 17/18 |

检测的价值在**构图不规整**的输入（人物偏在一角、多人、画面留白多）；精灵图本来就占满画面时，
整图模式又快又够用。

## 画质：为什么差，以及怎么调

用户连续反馈「生成质量差」，实测下来是三层叠加的原因，按影响从大到小：

### 1. 参数档位不对（已修）

**每个底模的推荐档位由后端给**（`engine.RECOMMEND`，经 `/models` 的 `recommend` 下发），
前端拿到就填，不自己拍。Qwen-Image-2.1 的推荐值是
**768² / 24 步 / cfg 4.0 / 不像素化（pixel=0, palette=0）**。

两条实测依据：

- **768² 而不是 1024²**：Qwen 是逐帧路线，1024² 的图 token 是 768² 的 1.78 倍——
  实测 768² 一帧 24~28s，1024² 约 45s；而精灵成品最后要缩到 32~128px，
  1024² 那点细节在缩图上根本看不见。**只有立绘档才用 1024²**（不缩图，笔触要留住）。
- **步数 16 起步**：旧结论「低于 16 步出糊块」是在真 CFG 开着时测的，已作废。
  现在 16 步（默认）已经稳定；**白/浅色主体**（白猫、白甲、幽灵）用 20~24 步，
  低步数下它们对比度不够、细节发白（详见 `docs/PLAN-QWEN-ONLY.md` §2.6）。

### 2. 底模能力上限（Qwen 能画复合概念了，短板换成了姿势）

同一句 "cat knight with katana"：

- **SD1.5**（已归档）：画得出「猫 + 红袍」，但**画不出刀、画不出铠甲**——拟人角色
  拿武器这种复合概念超出它的能力，肢体与躯干也会糊成一团。
- **Qwen-Image-2.1**：刀、铠甲、披风、面部表情都能到位，而且**直接就是像素风**
  （所以默认 `pixel=0`，再过一遍像素化反而把它的色阶优势抹平）。

**Qwen 现在的短板不在单张画质，而在逐帧的姿态贴合**：之前只能靠
「骨架图当参考 + 指令」软引导，实测腿部常被角色参考图的站姿覆盖（平均偏差
108px/768²，见 PLAN §2.8）。动作幅度要大，现阶段得接受「换一个姿势」而不是
「精确复制这个姿势」。

**ControlNet Union 已接通**（2026-10-30）：`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`

它是**双流**（VACE 风格）：control block 是完整 Transformer block 的副本，
每步都重跑、吃 timestep 调制，所以 skip 随采样步变化（实测 U 型
0.352→0.273→0.393）。按「算一次复用」做会退化成常量偏置——8 步勉强能用、
24 步必崩，出图直接是骨架的彩色线段。

实测（**竖版 768×1024 / 40 步 / scale 1.0 / 几百字详述 prompt**）：完整角色，
有脸有发饰，姿态与骨架逐点对应。峰值显存 18.7GB（24G 卡够用，不需要 fp8 化）。

**这个分支对配置极敏感，三条要一起对**：
1. **画布必须匹配内容**——2D 角色是竖构图就用竖版（768×1024），
   用 768² 方形时模型会把人物撑满、把头顶出画面（表现为"有身体没头"）
2. **prompt 要详述**（头发颜色、发饰、服装、姿态、背景、光线），
   只写 "a 2D game character sprite" 会让模型直接照抄骨架画个线条人
3. 强度 scale 1.0 是官方值，竖版下完全正常

四个坑：条件 latent 要做 mean/std 归一化、control block 要**继承**而不是组合
（否则 key 多 `block.` 前缀）、`init_stream` 要加回 base hidden、
mask 通道填 0 而非 1（填 1 等于"保留空的 masked-image"，出全透明图）。
**有 control 时必须 `use_kv_cache=False`**。
细节见 `docs/PLAN-QWEN-ONLY.md` §2.15，验证脚本 `tools/verify_cn_portrait.py`。


**VNCCS PoseStudio：不用 ControlNet 的姿势迁移（2026-10-31）**

社区那个号称"完美保持角色一致性"的工作流，实际做法是**专用 LoRA + 双图指令**，
全流程没有一个 ControlNet 节点。指令是固定的一句：

```
replace the pose of <image 2> with the pose of <image 1>. keep the character of <image 2>
```

（`image 1` = 骨架、`image 2` = 角色图；25 步、cfg 1、LoRA 权重 1.0。）
LoRA 是 `MIUProject/VNCCS_PoseStudio_QI2.1`（152MB，ComfyUI 格式，
用 `tools/convert_lora.py` 转成 diffusers）。实测对比：

| | ControlNet Union | **VNCCS LoRA** |
|---|---|---|
| 画风 | ❌ 会变 | ✅ 与参考图一致 |
| 头部 | ❌ 方形画布下丢失 | ✅ 完整清晰 |
| 肢体 | ⚠️ 偶发多腿 | ✅ 正常 |
| 服装/发型 | ⚠️ 部分丢失 | ✅ 全保住 |
| 姿态精度 | ✅ 精确 | ⚠️ 能跟 |

**要保角色一致性走 VNCCS；只要结构、不在乎外观才用 ControlNet Union。**
验证脚本 `tools/verify_vnccs_pose.py`，细节见 `docs/PLAN-QWEN-ONLY.md` §2.16b。

**已接入产品**：
- **角色迁移节点**：「引擎」下拉默认 VNCCS（可切回旧版）+「额外参考图」列表，
  同一角色的正面/侧面/细节图一起喂（最多 5 张附加，后端上限 6 张）。
  载荷字段 `char_images`（数组，第一张同时写进 `char_image` 兼容旧客户端）
- **序列帧替换节点**：`engine=wvnccs` 逐帧挂 LoRA，**帧间角色不漂**（4 帧实测
  同一张脸同一套衣服）。原「引擎」下拉里的 AnimateDiff 与「显存档位」是 SD 时代
  的死选项，已换成 VNCCS / 旧版
- **姿态序列节点**（w4）：同样有引擎下拉，默认 VNCCS。VNCCS 需要角色图，
  没选时节点上会给出提示；两阶段模式在 VNCCS 下自动关闭
- **换装 / 换脸**：也可以给多张参考图（`outfits` / `faces`），指令会自动改成
  「这些是同一件服装 / 同一个人」，否则模型会拼出好几套衣服
- **透明底可用**：实测主体占比 16.6%、半透明 2.1%、背景 83% 全透
  （`tools/verify_vnccs_alpha.py`）。之前"VNCCS 不能配透明底"是错的，
  那是 ControlNet 路线的结论连坐过来的
- **朝向必须写进指令**（2026-10-31 实测）。骨架图分不清角度：白底骨架 90° 不加
  方位词会被画成**正面**，加一句 `side view, facing right` 就是标准右侧面。
  也试过 VNCCS 那种"3D 人偶渲染带朝向"，0°/90° 都被还原成正面——那个 LoRA 只在
  他们自己的 PoseStudio 渲染上训过，认不出我们画的人偶。所以节点上多了「朝向」
  下拉（自动=跟骨骼编辑器的方位角 / 正面 / 8 个方向），提交时拼成
  `viewing direction: …`。同时修掉一个真 bug：角色迁移这条**漏了骨架转白底**，
  黑底骨架会被读成"背对镜头"，出来是背影（另两条 VNCCS 路线早就转了）。
- **整套移植官方 PoseStudio（2026-11-01）**。上面那条"人偶渲染传不了朝向"的结论
  其实是「我们自绘的人偶不在 LoRA 分布里」——换成**官方那套渲染器**（MakeHuman
  网格 + Phong 材质 + 相机机位）后，朝向直接烘焙进图：0°/90°/180°/270° =
  正面/侧面/背面/侧面，全部正确，**不用再写任何方位词**（写了反而压过姿势）。
  - 官方源码整套入库 `web/public/vnccs-pose/`（core.js 8938 行 + morph_runtime +
    openpose.js + vendor/three），MakeHuman 网格（CC0）与皮肤贴图一并入库；
    `entry.mjs` 只是接线（约 100 行），**没有一行自己的 IK 或渲染**
  - 骨骼编辑节点新增「**编辑 3D 人偶**」——打开官方的**编辑会话**（弹窗），
    交互与官方 UniCanvas 的 pose layer 一致：在 3D 视口里**直接拖橙色关节点**
    摆姿势（官方 raycast + IK + 极点约束），选朝向，点「导出人偶图」。
    产物存 `node.mannequin`（与线框骨架 `node.image` 并存）；
    角色迁移的「动作参考」**优先用人偶图**，没有才退回线框骨架
  - **修「导出超时」**：本机 Python 的 `mimetypes` 不认识 `.mjs`（给 text/plain），
    浏览器对 module script 强制 MIME 校验 → 官方渲染器的模块**一个都没加载**，
    页面永远停在「加载中…」，父页只能空等到超时。两处都修了：
    入口与 morph_runtime 改名 `.js`（.js 的 MIME 本来就对，不依赖重启后端）；
    app.py 里再注册一次 `mimetypes.add_type('text/javascript', '.mjs')`
  - 宿主页面加**早期错误捕获**（inline script）：模块没加载时立刻上报，
    不再让父页空等；等待上限从 60 秒放宽到 180 秒（37MB 网格首次要十几秒）
  - **后端零改动**：`_vnccs_skeleton` 只把「低饱和+暗」的像素换成白底，
    人偶图本就是白底灰人偶，判据 `mx<70` 不会命中
  - 验证：真实浏览器（headless Edge + swiftshader）里 postMessage 往返拿到
    169KB 的 768² PNG；主页面 iframe 零 console 错误；前端 smoke 383/383
  - 详见 `docs/VNCCS-REAL.md` 第 6 节


### 全项目 review（2026-10-31 上午）

对整个项目做了一轮全面审查（静态 + 实测 + 回归测试），修掉 21 个问题。
最严重的三个：

1. **扩图的 Outpaint LoRA 永久留在共享管线里** → 跑一次扩图之后，
   所有后续生成都带着它出图（而 meta 写着 `lora: None`）
2. **工作流临时挂的 LoRA 会顶掉用户挂的** → 用户在界面挂的 LoRA 会被
   快速模式静默卸载。现在改成 `user` / `temp` 双槽位合并加载
3. **路径穿越** → `startswith` 是字符串前缀比较，`assets2/` 能通过，
   `DELETE /assets` 能删到 assets 目录之外

另外把 meta 里"说的和实际做的不一样"的地方都改准了（步数、快速模式是否生效、
负向词、抠图、尺寸），产线判据不再算错。

配套：`tests/test_review_round2.py`（12 条回归测试）、
`tools/verify_review_quality.py`（六条工作流的质量实测 + LoRA 泄漏检查）。

### 骨骼编辑器：人形 / 线框 / 叠加三种视图

拖关节摆姿势时，如果只有线条骨架，很难看出"这个姿势像不像人、朝向对不对"。
编辑器现在多一层**有体积的人形体**（躯干是面、四肢是带锥度的柱、头是椭圆，
按深度排序遮挡，转到侧面身体会自然变窄；背面没有五官、有发际线）。

**导出格式一个字没改**——仍然是 OpenPose-18 黑底线框，Qwen / ControlNet 认的
就是那个格式。人形体只是编辑时的参考显示。

顺带把预设姿势的比例按真人 8 头身重映射过（旧版是"大头短腿"，
头占身高 22%、腿只占 34%）。这不只影响观感：导出的骨架比例失真，
模型照着画出来的人也是 Q 版。

效果自查（不用启动前端）：

```bash
cd web && npx vite build --ssr src/components/SkeletonEditor.tsx --outDir ssr-tmp
node skeleton-preview.mjs 行走,跑步,张手 0,45,90,180
```

### 3. 像素画的散落杂色（已做清洗，但未根除）

开了像素化（`pixel>0`）并量化到 32 色后，角色周围会冒出一粒粒红 / 青 / 橙的点。
排查时有个反直觉的发现：**它们不是独立小块**——实测 1024² 图上 25 个前景块，
最小的也有 2368px，压根没有孤立小块。所以面积过滤和形态学开运算都拿它们
没办法（那两种手段只动 alpha，而这里要动的是**颜色**）。

做了两层处理（`clean_iso` 与 `strip_fringe`，产线默认各 1 次）：

- **`strip_fringe`**：把「紧贴背景、颜色被背景混过」的一圈剥掉。
  原先只有色键后端走这一步，语义分割后端没走 —— 这才是白点的主要来源，
  补上之后残留从 30~40 个降到 15~20 个。
- **`clean_isolated`**：4 邻域判定的孤立点，主体内部的改成邻居色，
  飘在外面且 5×5 邻域只有自己的直接删。**5×5 那层保护不能省**：像素画的
  1px 斜向轮廓四邻同样全透明，只看 4 邻域会把整条对角线吃掉（实测踩过）。

残留的十来个点生成模型自身的噪声，继续加码会开始伤细节，就此打住。

### 4. 产出风格要选对（前端一档搞定）

画布「图生成」节点上只有一个**产出风格**下拉，一档同时定「提示词契约 +
像素块 + 色数」三件事：

| 风格 | 契约 | 像素块 | 色数 | 等效有效像素 |
|---|---|---|---|---|
| **像素精灵·Illustrious** | anime | 8 | 32 | 128px（1024²） |
| 像素精灵（16-bit） | sprite | 6 | 32 | 128px |
| 像素精灵·高清 | sprite | 4 | 48 | 192px |
| 像素·8-bit 粗颗粒 | pixel | 8 | 16 | 96px |
| 动漫立绘（不像素化） | anime | 0 | 0 | 原图 |
| 写实插画（不像素化） | portrait | 0 | 0 | 原图 |
| 水墨国风（不像素化） | wuxia | 0 | 0 | 原图 |
| 自定义参数 | — | 手填 | 手填 | — |

**「像素精灵·Illustrious」是唯一带装备的档位**：选它会把底模切到
`sdxl-anime`、挂上 Pixelizer LoRA、尺寸设 1024²、像素块 8（= 128 有效像素）。
实测可交付 89%、硬边 0.407（产线真精灵帧 0.447，不挂 LoRA 只有 0.198）。

手动配这四项很容易漏——尤其 LoRA 的触发词是 30 个字符的单 token
（`PixelArt_IL_One_Token_style_M_01`），漏一个字符就完全不生效且不报错。
所以触发词由**后端按 `loras.json` 的 trigger 字段自动前置**，界面上不用填。

**为什么不再让用户填像素块和色数**：这两个参数强耦合（768² 配 6px 块 =
128 有效像素），换算要用户在脑子里做，而填错**不会报错**、只是出图"看着
不对"——实测有人把 64 当成成品尺寸填进去，512/64 = 8px 有效像素，细节全没。
产线上真正该选的是「要什么成品」，不是「块多大」。

`自定义参数` 才暴露原始输入（老用户/特殊需求）。

**风格档位会老实回退**：如果单独改了色数、或者切了 LoRA（LoRA 会带自己的
契约），下拉会显示「自定义参数」而不是骗人说还在预设档上。

- 像素 LoRA 必须配 `pixel` 或 `sprite` 契约，配错了会出彩色色块怪物（见下节）。
- 武侠 LoRA 必须配 `wuxia` 契约（压印章与水墨背景），且抠图要用 u2net。

## 底模：只剩 Qwen-Image-2.1（SD 代码与权重都已移除）

### SD 时代的东西在哪

权重只是**移到** `models/_archive/sd/`（28G checkpoints + 12G controlnet +
3.1G IP-Adapter + 1.7G AnimateDiff + 2.2G LoRA），但**代码已经删了**
（提交 `1831ade`，engine 1493→570 行、generate 3028→1777 行）。

```
models/_archive/sd/
├── checkpoints/   sd15, sd15-inpaint, sdxl, sdxl-anime, sdxl-refiner
├── controlnet/    SD1.5/SDXL 的 openpose / canny / tile
├── ipadapter/     ViT-H 编码器 + SD1.5/SDXL 两套 adapter
├── animatediff/   SD1.5 的 motion module
└── lora/          27 个 SD LoRA 权重 + loras.json 备份
```

**保留没动的**：`qwen-image-2.1`、`matte/`（抠图模型，Qwen 也用）、
`pose_keypoint/`（DWPose，骨骼提取 + 质量闸门用）。

### 真要用回 SD：只能整段回退代码

```bash
git checkout pre-sd-purge    # SD 代码 + 归档前的运行态（推荐，能直接跑）
# 或者只回退那两刀：
git revert e92775e    # 先恢复前端（LoRA / 训练面板）
git revert 1831ade    # 再恢复后端（SD 管线 + 路由 + 工具）
# 然后把权重搬回去：
mv models/_archive/sd/checkpoints/sd15 models/checkpoints/
mv models/_archive/sd/controlnet models/ && mv models/_archive/sd/ipadapter models/
mv models/_archive/sd/animatediff models/
```

`pre-sd-purge` 是删除前的最后一个状态（tag 已推远端）。**注意**：回退会把
Qwen 后期的修复一起带走（蒙版贴回、尺度归一、fp8 编码器、底色参数都在删除之后
才定型），所以能整段回退、别指望 cherry-pick。

`engine.DEFAULT_MODEL` 仍是唯一的默认底模来源，`/models` 的 `current` 字段带出来；
前端初始选它。**不要在前端写死底模名**——SD 时代写死的「有 SDXL 就用 SDXL」
在权重归档后会挑一个根本没装的底模，这是当年踩过的坑。

### Qwen-Image-2.1 能做什么、不能做什么

| 工作流 | Qwen | 说明 |
|---|---|---|
| 文生图 / 图生图 | ✅ | 图生图是**指令改图**（没有 strength 语义） |
| 角色迁移 | ✅ | **双图条件**：角色图 + 骨架图进同一个 vision context；上身跟随好，**腿部常被参考图的站姿覆盖**（见 §2.8 实测） |
| 抠图 / 像素转换 / 转像素 | ✅ | 原生 RGBA（alpha 66~68%），开透明底会自动跳过抠图 |
| 局部重绘 / 局部 AI 重绘 | ✅ **真 inpaint** | 蒙版内换噪声 + 每步把蒙版外拉回原 latent（ComfyUI 的 no-offset 做法）。模型**看得见蒙版**，指令准确执行 |
| 姿态序列 / 序列帧替换 | ⚠️ **近似** | 仍是「骨架/轮廓当参考图 + 指令」软引导。**两者都能走 VNCCS 引擎**（`engine=wvnccs`）：帧间角色不再漂，但姿态幅度仍不如硬约束 |
| LoRA | ✅ | **Qwen 自己的 LoRA 能挂**（fp8 量化后也行，实测只慢 1.1s）。但必须是 diffusers 原生格式，ComfyUI 格式要先转 |
| **扩图** | ✅ | 中性灰 padding + Outpaint LoRA + 原图贴回；无接缝、无灰块 |
| **换装 / 换脸** | ✅ | 两图 + `<image1>/<image2>` 结构化指令，不需要专用权重。**支持多张参考图**（`outfits` / `faces`），指令会自动声明"这是同一件/同一人" |
| **姿态迁移（VNCCS）** | ✅ **推荐** | 角色迁移 / 姿态序列 / 序列帧替换的默认引擎：专用 LoRA + 双图指令，画风/发型/服装一致，头不丢、不多腿。25 步，画布按骨架比例自动定，透明底可用。**骨架图要先转白底**（黑底会让模型画成背面） |
| **ControlNet Union** | ✅ **能硬压结构** | 8 种条件 + 蒙版 inpaint，7.55GB，峰值 18.7GB。双流注入；**竖版画布 + 详述 prompt** 下出完整角色，姿态逐点对应 |
| Refiner / AnimateDiff | ❌ | 主干与文本编码器都不同，一点都不通用 |

⚠️ 的两个在前端节点上有橙色「近似」标记（`/models` 的 `caps.approx`）。
十个工作流的实现方式、实测数据、以及**明确做不到的事**见
[`docs/PLAN-QWEN-ONLY.md`](docs/PLAN-QWEN-ONLY.md)。

**局部重绘是怎么做成的**（2026-10-30 实测推翻旧结论）：
diffusers 的 `QwenImage21Pipeline` **没有** `mask` 参数，但这不等于模型做不到——
ComfyUI 的 `Qwen Image 2.1 Inpainting | No-Offset Edits` 工作流用的
**就是标准 2.1 权重**（没有专用 inpaint 权重），靠两件事实现：

1. 初始 latent 里，蒙版内换成噪声
2. 采样循环里**每一步**把蒙版外拉回原 latent
   （ComfyUI `comfy/samplers.py:641-642`，**采样器级**机制，与权重无关）

第 2 步在 diffusers 里用管线现成的 `latents=` 入参 + `callback_on_step_end`
钩子复刻，核心是 flow matching 的 `x_t = (1-σ)·x0 + σ·ε`。
实测（768²/20 步）：蒙版内指令准确执行、蒙版外最大像素差 19~46（视觉无差，
差异来自 VAE 重建本身）、~15s。旧的「整图改图 + 蒙版贴回」保留为对照
（`strategy="whole"`，逐点为 0 但模型看不见蒙版）。
实现见 `workflows/qwen_mask.py`，坑见该文件头部注释。

**LoRA 的格式坑**：ComfyUI 格式（`diffusion_model.` + `lora_A/B`）与 diffusers
原生格式（`transformer.` + `lora.down/up`）键名不同，**ComfyUI 格式挂上去会
静默不生效**（比报错更坏）。转换：`python tools/convert_lora.py in.safetensors out.safetensors`。
挂载/卸载在侧栏「LoRA」分组里（全局状态，对所有生成生效）。

**扩图怎么做的**：源图贴到更大画布、空白填**中性灰 #808080**（灰色本身就是
"这里要补内容"的信号，填别的颜色会污染新区域色调），挂 Outpaint LoRA，
提示词以它的触发句开头。原图区域最后软边贴回——扩图的语义是"扩展"，
原图不该变。实测 768² 右扩 50% → 1152×768，14s，右侧延续了森林/地面/雾气，
无灰块、无可见接缝。注意：**画布尺寸必须对齐 32**（不是 16），
否则管线会静默 resize，meta 里记的尺寸就是假的。

**换装 / 换脸**：两者本质相同，都是**两图 + 结构化指令**，不需要专用权重。
关键是指令里写清楚每张图管什么，并用官方的多图引用语法 `<image1>/<image2>`
（比"第一张图"这种自然语言更稳）。实测换装后服装特征来自图 2、人还是图 1；
换脸后脸像图 2、姿势与场景是图 1。

**快速模式**：侧栏「LoRA」分组顶部的开关，挂 Lightning 少步蒸馏 LoRA +
步数压到 8。实测 24 步 17.3s → 8 步 9.3s（-46%），画质几乎不掉。
逐帧路线（姿态序列 / 序列帧替换）建议开着——单帧 37s 降到 9.3s，8 帧 ≈ 75s。

### SD1.5 / SDXL 时代（权重已归档，代码已删）

这一段原来有四小节：SD1.5/SDXL 底模、SDXL 版 LoRA 实测、`sdxl-anime`（Illustrious）、
SDXL 版 ControlNet。**全部失效**——权重在 `models/_archive/sd/`，代码在提交 `1831ade`
删掉了，它们引用的 `tools/fetch_*.py` 也一起删了。别再照旧命令去下权重。

当时实测里仍然通用的三条（接别的底模时照样成立）：

1. **尺寸不是越大越好。** SDXL@1024 反而比 @768 差：角色占画面比例更大、更容易贴边，
   可交付率 92% → 75%，每张还慢一倍。前端填多大的依据应该是「成品要缩到多少 px」。
2. **姿态强度与身份强度是两条独立旋钮。** 调错一个的表现是「姿势对了但不像这个角色」
   或反过来。Qwen 没有这两条旋钮（双图条件由模型自己分配），但这条诊断思路仍适用：
   先看是姿态跑偏还是身份跑偏，再决定改指令还是换参考图。
3. **`clip_skip` 只对 SD1.5 系动漫模型成立**，SDXL 有两个文本编码器，设了会静默失效。
   这个字段现在整个删了（Qwen 的文本编码器是 Qwen3-VL，没有跳层语义）。

要看完整历史：`git show pre-sd-purge:README.md`。

### Qwen-Image-2.1：唯一的底模（原生透明底 / 指令改图）

```bash
# 43GB：transformer 14G（另有 6.7G fp8 缓存）+ text_encoder 22G + vae 1.3G
# 放 models/checkpoints/qwen-image-2.1/，需要 diffusers≥0.41 + transformers≥5.21
# 文本编码器再生成一份 fp8 缓存（8.8G，一次性）：python tools/make_qwen_fp8.py
```

它不是"更大的 SD"：**7B 单流 DiT + Qwen3-VL 视觉语言编码器**，一个权重同时干
文生图 / 指令改图 / 原生 RGBA 三件事。接入后它出现在「底模」下拉里，
`arch` 是 `qwen21`，与 SD 系列**不共用任何组件**（LoRA、ControlNet、Refiner 全部不通用）。

| 项 | 实测值（RTX 4090 24G，fp8 双模块） |
|---|---|
| 文生图 768² / 8 步 | 10.45s，峰值 **11.41GB** |
| 指令改图 768² / 8 步 | 11.92s，峰值 **11.82GB** |
| 产线口径（768² / 24 步） | 约 28~30s/帧，峰值 15.4GB |
| 原生透明底 | alpha 占 66~68%，边缘无脏毛边 |
| 像素风 | **直接出像样的像素画**，不需要像素 LoRA |

**显存/内存峰值主要在文本编码器上**：`text_encoder` 是 Qwen3-VL（8.77B，bf16
16.3GB），量化成 fp8 后峰值从 16.59GB 降到 11.41GB（-31%）、速度快 25%、
内存少 7.5GB、加载快 44%。同 seed 出图平均差 **1.10/255**（结构一致）。
不满意就 `MEOW_QWEN_TEXT_FP8=0` 退回 bf16。细节见 `docs/PLAN-QWEN-ONLY.md` §2.1。

**几个必须知道的坑**（都是实测踩出来的）：

1. **`true_cfg_scale` 要配负向词才有用**。管线的判据是
   `do_true_cfg = true_cfg_scale > 1 and has_neg_prompt`，不传 `negative_prompt`
   时 CFG **静默失效**（cfg=1 与 cfg=4 输出逐字节相同）。代码里恒定传 `QWEN_NEG`。
2. **透明底要双段式声明**。VAE 有第 4 通道，但模型不会自己想起来用——
   提示词首尾必须各加一句官方原话：
   ```
   This is an RGBA image with transparency. <正文> The image has alpha channel and the background is transparent.
   ```
   改写说法会掉回不透明。而且**不能同时用 SD 的像素契约后缀**——
   `SPRITE_SUFFIX` 里的 "flat plain background" 和"背景透明"直接矛盾，
   实测会画出白底。透明场景走短后缀（只留构图约束）。
3. **尺寸必须是 16 的倍数**，否则 diffusers 只 warn 一句就静默改尺寸。
4. **本机 32G 内存装不下全量 bf16**（13.3G + 16.3G 一次性读会 swap）。
   两块都走预量化 fp8 缓存：transformer 6.7G、text_encoder 8.8G
   （`python tools/make_qwen_fp8.py` 生成一次），仍靠 `enable_model_cpu_offload`
   在编码后把文本编码器搬出显存。见「Qwen 的显存/内存」一节。
5. **图生图没有 `strength`**。SD 是"加噪强度"，Qwen 是"照指令改"——
   同一张源图，SD 调 strength 是渐变，Qwen 下改多少全看提示词怎么写。

`transparent` 开关同时出现在「图生成」和「转像素」节点上，只有 Qwen 底模可勾
（`/models` 的 `caps.transparent`）。开它之后「抠图」会自动跳过——
重复处理反而把 alpha 弄脏，白猫白底这类场景分割模型会把脸一起切掉。

**透明底现在有三道保险**（都是踩过坑加的，细节见 PLAN §2.7）：

1. 提示词里的「白底/纯色底」措辞会被自动清掉。留着它等于给模型一个和透明
   声明直接冲突的正面要求——实测 alpha 会变成一层雾（77.5% 的像素停在半透明）。
2. `matte_auto` 先看输入有没有可信 alpha（`matte.alpha_trust`：真透明区足够多、
   半透明中间值 < 12%）。可信就**原样复用**，不再拿分割模型重猜一遍。
3. 真要重抠时，`flatten_white` 把透明区合成到白底再送分割（以前直接
   `convert("RGB")`，透明区留着 VAE 噪声色），结果再做一次 `fill_holes`
   把主体内部被误判的洞补回实心（额头、胸口那种）。

### 角色迁移在 Qwen 上：双图条件（角色图 + 骨架图）

用户反馈「SD1.5 / SDXL 的角色迁移效果都不行」，实测后换了条路：**不靠
ControlNet，靠 Qwen 的多图条件**。

SD 那条路是两个**互相拉扯**的约束——IP-Adapter 抢身份、OpenPose ControlNet 抢
姿态，同一张角色图 + 同一张动作图，出来的经常是「身份丢了」或者「姿势变形」。
Qwen 的多图条件是**同一个 vision context**：两张图一起进模型，由它自己分配
「谁是身份、谁是姿势」。

流程（`generate._qwen_pose_clone`）：

1. 动作参考图 → DWPose 骨架图（和 SD 那条路同一个 `refcond.build_condition`，
   真人照片走骨架、像素/非人形素材走剪影，auto 自动降级）
2. 角色图 → `refprep.prepare_reference`（透明区填白、裁主体、像素图最近邻缩放）
3. `image=[角色图, 骨架图]` + 一段固定英文指令（"第一张是角色参考，保持脸/发型/
   服装/配色；第二张是姿态条件，按它的姿势重画同一个角色"）
4. 用户填的角色描述接在指令后面（可选，只补语义）

实测（RTX 4090 24G，768² / 28 步 / cfg 4）：

| 路线 | 耗时 | 峰值 | 结果 |
|---|---|---|---|
| **Qwen-2.1 双图条件** | 72s | 19.3GB | 上身跟随好、主色继承；**腿部会退回参考图的站姿**（2026-10-30 重测，见 `docs/PLAN-QWEN-ONLY.md` §2.8） |
| SD1.5 ControlNet+IPA | 7s | 5.3GB | 同一输入，姿态基本没跟上 |

代价是慢（SD1.5 快 10 倍），换来的是姿态正确率。角色迁移节点切到 Qwen 底模时
会自动换成 768²/28 步/cfg 4，并**隐藏「姿态」「身份」两根滑块**——它们是
ControlNet / IP-Adapter 专属的，Qwen 路径没有强度旋钮。

限制：Qwen 不支持 LoRA（不同主干与文本编码器），也不吃 SD 的提示词契约后缀；
序列帧替换（wss）仍只支持 SD 系（逐帧生成 + 帧间对齐，Qwen 的速度不现实）。

### 骨骼编辑节点：拖关节 + 换视觉方向

上面那节解决了「模型能读懂骨架」，这节解决「骨架从哪来」——以前只能从真人
照片里提，现在可以**自己摆**。

侧栏「节点 → 骨骼编辑」新建一个节点，正文里是一个可拖的 OpenPose-18 骨架：

- **拖圆点**改关节位置（18 个 COCO 关节点，热区放大到 4.5 单位，缩放后也点得中）
- **8 个方向按钮**：正面 / 右前 / 右侧 / 右后 / 背面 / 左后 / 左侧 / 左前，
  另有 0~359° 滑块可任意角度
- **深度滑块**：选中关节后调它前后的 z，用来做「手往前伸」「肩往后转」
- **7 个姿势预设**：站立 / 张手 / 行走 / 跑步 / 跳跃 / 拳击 / 举手
- **导出骨架图**：前端按 `tools/make_pose.render` 的 18 色 19 边画法生成 PNG
  （黑底、彩色肢体 + 关节圆点），上传后挂在节点上，直接连到角色迁移节点的
  「动作参考」输入

**换视觉方向怎么实现的：伪 3D 投影。** 每个关节存 `(x, y, z)`，z 是前后深度，
渲染时绕竖轴旋转再正交投影：

```
rx = 0.5 + (x - 0.5) * cos(a) + z * sin(a)
ry = y
```

只有 2D 坐标的话「侧面」只能靠左右镜像硬凑，肩膀一转就叠成一个点；有 z 之后
0° 正面 / 90° 侧面 / 180° 背面自然过渡。拖动时把屏幕横向位移按
`(cos a, 0, sin a)` 分解回 x/z——侧面视角下横向拖动自动变成「改深度」。

**两个必须配套的改动**：

1. **新增 `skeleton` 参考模式**（`refcond.MODES`）。骨架图上没有真人，
   走 `auto` 会因 FasterRCNN 检不到人体而**静默降级成剪影**，把画好的骨架
   毁成一张黑图。所以骨骼节点连过来时前端自动把 `ref_mode` 切成 `skeleton`，
   后端原样 resize 使用，不再跑人体检测。
2. **方位角补成方向词**。骨架的朝向信息 Qwen 有时读不准（实测侧面骨架
   仍可能出正面人），而方位角在编辑器里是确定值——提交时自动追加
   `viewing direction: side view, facing right` 这类短语，不改用户写的提示词。

实测（768² / 8 步，同一角色图 + 三个方向的骨架）：**正面 / 侧面 / 背面都
正确生成了**——背面能看到后脑与背部，侧面身体明显收窄；侧面那张双臂在
投影上重叠（符合真实侧面），所以预设里给左右手腕加了前后深度差，让侧面看
一前一后。单张 20~46s（同进程连续生成会降到 20s 左右）。

### 从图片自动摆姿势

编辑器底部的「📷 从图片」：选一张图 → 上传 → 后端 `POST /pose/extract` 跑
本机 DWPose（FasterRCNN 检人 + RTMPose 关键点）→ 返回 18 个归一化关节点
（`normalized_points`，人物已居中占满画面，正好是编辑器的 0~1 坐标）→
前端补上深度 z，变成可旋转的骨架。姿势不用手动摆了，方向还是点上方按钮换。

实测：一张真人站姿照 → **18/18 关节全部识别**，旋转到 0°/90°/180° 后骨架
都还像人形，没有叠成一条线。

**四个必须知道的限制**（都是单张 2D 图的固有信息缺失，不是 bug）：

1. **只对真人 / 写实人像有效**。检测器和 RTMPose 是在 COCO 自然照片上训的，
   像素画、插画、版画一个框都检不出（和角色迁移 `pose` 模式同一个限制）。
   这类素材请用上面的姿势预设手动摆，或走角色迁移的「轮廓」参考模式。
2. **2D 图没有深度**。z 是按「右侧在前、左侧在后」给的保守启发式
   （手腕/脚踝再加大前后差），不是真实三维重建——换方向后四肢的前后关系
   是假设的，可以用深度滑块改。目标是可辨识（别叠成一条线），不是物理精确。
3. **正面 / 背面在 2D 骨架上是对称的**，自动分不出来。后端只在「肩宽很窄」
   时给出侧面猜测（仅用于提示），导入一律按正面，方向交给用户点按钮——
   若原图本身就是侧面照，直接当正面再旋转会有偏差，原图尽量用正面。
4. 路径只能读 `assets/` 内的图片（新写了只读版 `_readable_asset_path`，
   不复用删除接口那个带额外规则的 `_safe_asset_path`），路径穿越返回 400。

> 偶发慢：连续生成时观察过一次第二张 252s（正常 24s），8 步连续三次复测
> 46/24/20s 均正常，判断是本机内存换页偶发，不是管线 bug。

### 调研：MultiAngle-T8 能不能用于角色迁移 / 序列帧替换

结论：**姿态迁移部分不能直接用，只有 AnyAngle LoRA 值得单独拿来做多视角立绘。**

[Comfyui-Qwen-Image-2.1-MultiAngle-T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-MultiAngle-T8)
是个 ComfyUI 节点（不是 diffusers 库），核心是「三维相机重打光」：用 TripoSplat
把原图重建/用 MakeHuman 人偶摆姿势，在真实三维相机里取景，导出 POSE/深度/Canny
引导图，再喂给 Qwen Image 2.1 生成新机位图。

不能用的原因（按重要性）：

1. **它自己没有 ControlNet 端口**。作者自己写明"底模把结构图作为**图像参考**理解，
   没有独立的 ControlNet 控制端口；遵循程度取决于底模与提示词，不能保证
   像专用控制模型一样严格"。我们现在有 Union 的 Pose 条件可以做硬一点的控制
   （虽然仍是软约束级别，见 §2.15 实测），比它的"纯参考图"要强。
2. **实测精度不够产线**。它自己公布的主矩阵：32 张里只有 **7 张基本符合、
   2 张部分符合**，九人场景会增人/丢动作。序列帧要求"帧数和动作逐一对应"，
   这个合格率直接判死。
3. **它是 ComfyUI 生态**（依赖 ComfyUI 0.36 原生节点 + 浏览器 WebGL），
   我们是 diffusers 进程内推理，移植等于重写。

值得单独拿的只有一个：**AnyAngle LoRA**（`QI2.1_AnyAngle.safetensors`，
出自 `lilylilith/QI_2.1_AnyAngle`）。它是给 Qwen Image 2.1 做**相机方位控制**的
LoRA（配合 3D 重建 + Blender 重打光出的参考图使用），对我们的"同一角色出
正面/侧面/背面三视图"有用——这条路不需要 3D 重建，直接 Qwen + LoRA + 角度提示词
即可，可以之后单独接一个"多视角立绘"节点。优先级不高：现有 W4 姿态序列 +
IP-Adapter 已经能覆盖精灵帧的主要需求。

> 下载受阻：本机走代理访问 huggingface.co / hf-mirror.com 持续 502
> （`Tunnel connection failed`），ModelScope 直连可用但没有这两个 LoRA 的镜像。
> 需要时先解决网络，或从别处拿到文件手动放进 `models/lora/`。

### 调研：3D 骨骼 / 姿势编辑器（Fisher-Pose、VNCCS PoseStudio）

结论：**模型侧我们已经用 Qwen 双图条件拿到了「骨架 → 姿势」的能力**；
这两个方案多出来的是**交互层**（3D 人偶拖拽摆姿势），不是更强的模型控制。

| 方案 | 核心 | 我们的判断 |
|---|---|---|
| ComfyUI-Fisher-Pose | 内置 3D 人偶 + 209 根 OpenPose 骨骼，拖拽出精确骨骼图 | ComfyUI 生态（浏览器 WebGL 3D），我们是 diffusers 进程内推理，移植等于重写 |
| VNCCS PoseStudio（Qwen Image 2.1） | 3D 姿势编辑器 + 专用 LoRA（`MIUProject/VNCCS_PoseStudio_QI2.1`），骨架图当参考图喂 Qwen | LoRA 是**在 VNCCS 自己的骨架格式上训的**，我们出的是 DWPose/OpenPose 格式，格式不同则 LoRA 遵循度未知；且下载同样受阻 |

两者共同的关键点：**姿态靠「骨架图」传递**，不是靠模型内部参数。Qwen-Image-2.1
本身就能读懂 OpenPose 骨架图（但「读懂」≠「照做」，实测腿部常被参考图站姿覆盖，见 PLAN §2.8），所以：

- **不需要**为了读骨架而额外训/下 LoRA（VNCCS 那条）
- **不需要**为了摆姿势而引入 3D 重建（AnyAngle 那条）
- 真要提升，方向是**交互**：做一个能拖关节的 2D 骨骼编辑器，或接 PoseStudio
  导出的骨架图作为额外输入。这属于产品体验升级，不影响产出质量

如果一定要"硬约束"，本机 pixelforge 里有 `qwen-image-20b` + `qwen-image-controlnet-union`
（ControlNet Union 自带 pose 条件）。但 union 是给 **20B** 训的
（`joint_attention_dim=3584`），Qwen-Image-2.1 的文本编码器是 **4096**，两者
结构对不上——**2.1 用不了这个 union**。接 20B 是另一条工程线（显存/加载都要重做）。

## 挑帧：批量候选 + 质量闸门（W4）

逐帧出图的老问题不是「画得丑」，而是**偶尔会出废帧**：没画出手、腿糊成一坨、
脚被画布切掉。这类帧人眼一秒能挑出来，但自动化管线以前只能照单全收，打包出来的
走路循环里突然抽一帧——不逐帧看根本发现不了。

`poseswap` 现在默认开闸门（`gate=True`）：

```
每帧 -> 生成 candidates 个候选（种子 seed, seed+1000, ...）
     -> 每个候选打一次分（workflows/quality.py）
     -> 取分数最高的一个
     -> 若最高分那个没过闸门，再补提最多 gate_retries 次
     -> 仍不过就留最好的那个，并在 meta 里记 passed=false
```

**分数量什么**（全部从产物图上直接量，不看生成过程）：

| 信号 | 含义 | 权重 |
|---|---|---|
| `points` | 姿态模型在**生成图**上重新检出的人形关节数（0~18）。手断了、画成两个半身，点数立刻掉 | +10/点 |
| `coverage` | 抠图后前景占比。偏大=半身/糊成一坨，偏小=人太小（像素化后没细节） | +5 落在区间内 |
| `edges` | 贴边（被画布切断）。精灵要对齐脚底，脚被裁掉则对齐必跳 | −8/条边 |
| `drift` | 与上一帧的 alpha 质心偏移（按对角线归一化）。帧间跳太远播放会抖 | −20×偏移 |

闸门判据是 `points >= gate_min_points（默认 8）` 且无贴边。**没有姿态模型时判定为
「无法判定」而不是「不合格」**，闸门自动关闭并告警——否则会拿不到判据还无限重提。

```python
g.poseswap(eng, pose_images=poses, prompt="...", matte=True,
           candidates=2, gate=True, gate_min_points=8, gate_retries=2)
# meta 里能查到：attempts（总生成次数）、gate_passed（过闸门帧数）、
#               frame_scores（每帧的 points/coverage/edge/drift/score/attempts）
```

代价：闸门每帧多几十毫秒（姿态模型跑 CPU，4090 上单帧约 10ms 量级），比重新出
一帧便宜两个数量级。`candidates=1, gate_retries=0` 就退化成原来的行为。

**非人形主体**（史莱姆、机甲、道具）：DWPose 一个关节都检不出来，闸门永远
过不了。这时补提就是纯烧钱（每帧 25~30s），而且 meta 里 `gate_passed=0`
看起来像生成失败。现在一旦判定为「不是人形」就**自动关掉补提**，
并在 `meta.gate_non_humanoid` / `meta.gate_note` 里写明原因。

实测（Qwen，768²/16 步，`candidates=1, gate_retries=1, matte=True`）：
单帧约 12.9s、峰值 11.4GB。人形帧 18/18 关节、无贴边；
史莱姆这类非人形帧只出 1 次（不再白补提一次）。

## 画质：一轮代码 review 改了什么

不是「调参玄学」，每条都有对照实验（`python tools/ab_quality.py`，同 seed 只改一个变量）。

**① 正向后缀被静默丢弃（真 bug）**
`presets.build` 原来只对**负向词**做 77 token 截断，正向是 `用户描述 + 后缀` 直接丢给
CLIP。用户主体描述一写长，被丢掉的恰好是后缀（`flat plain background` / `hard outline`），
于是背景变脏、边缘变糊——这就是「提示词越长越难看」的真实原因。现在先给后缀留预算，
超长的部分截主体。

**② 没有逗号的长文本会让负向词变成空串（真 bug）**
`fit` 按逗号分段累加，第一段就超预算时 `break` 得只剩空列表，返回 `""`——
用户粘贴一整串空格分隔的负向词，等于什么都没写。现在退化成按词截断。

> **2026-10-31 更新：①② 那套 77 token 预算机制已整体删除。**
> 底模换成 Qwen-Image 后，tokenizer 的 `model_max_length` 是 262144、
> diffusers 的 `encode_prompt()` 只做 padding **不做 truncation**——模型侧没有
> 长度上限。所以 `presets.build()` 现在只做拼接，**一个字都不截**
> （`fit` / `count` / `rough_tokens` / `CHUNK_BUDGET` 一并删了）。
> 上面两段留作历史：它们说明"替用户做减法"能造成多大伤害——一次是砍掉了后缀，
> 一次是靠高估的估算器把 35 token 当成 113 来砍。

**③ 像素化降采样：LANCZOS → BOX**
LANCZOS 有负瓣，色块交界处会振铃出「过冲色」（黑线旁边冒一圈更亮的边），这些假色
再被量化，就成了多出来的脏色阶。实测同一张图：LANCZOS 用满 32 色且边缘发毛，
BOX 只用到 30 色，格子单色率 1.0（有测试钉住）。

**④ clip_skip 自实现（上游参数在本机会崩）**
diffusers 0.41 的 `clip_skip=` 分支写死 `text_encoder.text_model.final_layer_norm`，
本机 transformers 5.21 把 `CLIPTextModel` 摊平了（没有 `text_model` 这层）——
一传就 `AttributeError`。`workflows/clip.py` 自己算 embeds 再喂管道：
`clip_skip=1` 与管道默认**逐元素一致**（有测试），`2` 跳过最后一层（与默认的
余弦相似度 0.79，是实打实另一个向量）。节点「更多设置 → 跳过层」可选，
LoRA 注册表也能声明推荐值。

视觉上（`docs/img/clip_skip_ab.jpg`）：`cs1` 偏光泽渐变（高档 3D 感，但渐变多、
量化后杂色多），`cs2` 偏平涂 + 硬描边（更适合转像素）。代价是 `cs2` 下多主体
略多（2/3 vs 3/3），所以**默认仍是 1**，需要时手动开。

**⑤ 触发词实测有害（改数据，不是改代码）**
`px_illustrious` 注册的触发词是 `pixelart`，UI 上有「+ 填入」按钮。三个种子实测：

| 用法 | 主体数 | 单主体率 |
|---|---|---|
| 不加触发词 | 1 / 1 / 1 | **3/3** |
| 加 `pixelart` | 5 / 3 / 2 | **0/3** |

加了触发词反而把画面变成一整个「史莱姆图案墙」。注册表已把触发词清空并写明原因。
同一个 LoRA 一定会画彩色/纹理底，所以注册表新增 `matte: true`——选中它时节点自动
勾上抠图（以前默认不勾，出来就是蓝底图，没法直接用）。

同一 seed 的成品对照（上排旧用法：加触发词、不抠图；下排新默认）：

![旧用法 vs 新默认](docs/img/quality_before_after.jpg)

左列 seed 7、右列 seed 8。旧用法整幅变成「史莱姆图案墙」，新默认是单只、透明底、
像素块干净的成品。`meta.subjects` 也印证：旧用法数不出主体（没抠图），新默认 1 个。

**⑥ 多主体写进产物 meta**
SD1.5 画小体型角色（史莱姆尤甚）爱复制一份，负向词压不住，于是不猜、直接量：
抠图后 `foreground_blobs` 数面积 ≥400px 的前景块，写进 `meta.subjects`；节点里
`subjects > 1` 直接黄字提示换种子。这是**下界**（u2net 只分割显著主体，一整墙
史莱姆可能只剩 1 块），所以 1 不代表干净，但 >1 基本确实脏。

**⑦ 抠图后端：量过了，默认不动，补了防呆**
原本怀疑「像素画该用色键，因为 u2net 边缘软」。四个样本量下来两个结论：
alpha **都是二值**（中间 alpha 占比 0，边缘硬度假设不成立）；色键在像素 LoRA 的
彩色底上几乎抠不动（前景 0.90~0.98），在白袍角色上会把袍子抠碎（3 块）。所以
`auto` 继续走 u2net。真正该修的是另一件事：u2net 会**静默抠空**（草稿档只剩
0.44% 前景 → 交出一张空精灵），现在按前景占比做合理性检查并回退色键，
`matte_used` / `matte_fallback` 记进 meta。

**Qwen 之后又补了一件事：先看输入有没有 alpha。** 以前 `auto` 永远重抠一遍，
而 Qwen 出图本来就带 VAE 的第 4 通道——重抠纯属自找麻烦（白毛角色的额头、
脸颊会被判成背景咬掉）。现在 `matte.alpha_trust` 三个数一量（真空透明区
≥10%、半透明中间值 ≤12%、前景 1%~95%），可信就直接复用，`meta.matte_used="alpha"`
且带 `alpha_reused`；alpha 是雾状（提示词冲突那类）才真的重抠。
显式选 `u2net` / `color` 后端则跳过这套判断——那是用户明确要求重抠。

### 白毛/浅色主体 + 白底：抠图这件事本身不可靠（已知限制）

实测同一张白猫骑士（白毛、白甲、白底）：直接送 RGB 给 isnet → 头顶被切角、
胸口挖洞；先合成白底再送 → 头顶与胸口救回来了，**两条腿又被吃**（腿也是白的）。
前景占比 20.9% vs 15.6%，两种喂法各丢不同部位。

结论不是「调参」，而是**别在这类素材上走抠图**：Qwen 出图时直接勾「透明底」，
alpha 由模型自己给（干净时前景 31%、全透明 67.8%、半透明中间值只有 2.9%）。
抠图是 SD 时代的补丁——那时候模型不会输出 alpha，只能事后猜。

## LoRA / 训练：已随 SD 一起删除

**这一整块没了**（提交 `1831ade`）。原因不是"懒得维护"，而是 Qwen 根本不吃：

- LoRA 是挂在 UNet/DiT 注意力投影层上的增量权重，键名要与主干严格对齐。
  Qwen 的主干是 `QwenImage21Transformer2DModel`（单流 DiT），SD1.5/SDXL 的
  27 个精选 LoRA 一个都挂不上。
- 训练那边同理：`workflows/train.py` 训的是 SD 的 UNet LoRA，产出对 Qwen 无用。

一起删掉的：`workflows/lora.py` / `clip.py` / `train.py`、`/loras` 与 `/train`
五条路由、前端的 LoRA 下拉与「训练 LoRA」面板、`models/lora/` 的注册逻辑。

**现在的风格控制手段**：
1. 提示词契约（`presets.py`，产出风格档位就是它的预设组合）
2. 参考图 —— 图生图是指令改图，双图条件（角色 + 骨架）能锁身份
3. 原生像素风 —— Qwen 直接就能画，不需要像素 LoRA（这是当年上 LoRA 的主因）

历史记录还在：`git show pre-sd-purge:README.md` 能看到完整的 LoRA 调研结论
（哪些能用、哪些静默空挂、触发词为什么必须自动前置）。那些结论对 SD 仍然有效，
只是这个工作台不再跑 SD 了。

## WebUI（画布 · 文生图 / 图生图）

React 18 + Vite + TS，**零 UI 库、零画布库**（与 pixelforge 同思路），构建产物 231KB / gzip 76KB。

**节点可拖宽**（右边缘拖拽，260~620px）：参数横排时 300px 会挤成两列、标签被截断。
宽度存在节点上（刷新后保持），连线端点也跟着实际宽度走。

**每个参数都有 tooltip**：hover 参数名显示三段——做什么 / 影响什么 / 推荐填多少。
说明集中维护在 `GenNodeCard.tsx` 的 `PARAM_HINTS` 里（同一参数在多个节点复用，
散着写必然分叉）。推荐值尽量写本项目实测过的数字，不写"视情况而定"。
带虚下划线的参数名表示有说明——只加 `title` 不加视觉提示，用户不会去 hover。

```bash
npm run check:hints     # 检查每个 hint('X') 都有说明（已挂进 npm run smoke）
```

这个检查是必需的，不是洁癖：`hint()` 查不到键时界面上只是**没有说明**，
不报错、不崩溃、任何测试都不会失败。「像素块」就这样漏了一整轮
（表里 37 条、代码引用 38 个），最后是用户发现的。所以现在两处兜底：
`hint()` 对未知键返回「未定义说明：X」而不是空串，外加这道静态检查。

**一条命令启动**（自动体检 → 按需构建前端 → 起服务 → 开浏览器）：

```bash
./start.sh                  # Git Bash / macOS / Linux
start.bat                   # Windows 双击或命令行
```

| 参数 | 作用 |
|---|---|
| `--port 8010` | 换端口 |
| `--rebuild` / `--no-build` | 强制重建前端 / 跳过构建 |
| `--check` | 只体检（Python、依赖、CUDA、权重、前端、端口），不启动 |
| `--dev` | 只起 API，前端另开 `cd web && npm run dev`（:5173 代理） |

体检项与实际输出：

```
✓ Python：.venv/Scripts/python.exe（Python 3.13.14）
✓ 依赖齐全（torch / diffusers / transformers / fastapi / uvicorn）
✓ CUDA 可用（NVIDIA GeForce RTX 4090）
✓ Qwen-Image-2.1 权重就绪（9 个 safetensors）
✓ 前端产物是最新的
```

手动方式：

```bash
cd web && npm ci && npm run build     # 构建
cd .. && python app.py                # FastAPI 同时托管 dist 与 API → http://127.0.0.1:8000
cd web && npm run dev                 # 或开发模式（:5173，代理到 8000）
npm run smoke                         # 无浏览器渲染冒烟（jsdom，384 项断言）
```

> `start.bat` 刻意保持**纯 ASCII**：cmd.exe 按 OEM 代码页解析 .bat，中文注释会切断命令行
> （实测把 `setlocal` 切成 `ld`）。中文提示只留在 `start.sh`。

**布局**（参考了 InvokeAI 的「队列可见/参数分层」、SD.Next 的「折叠压制参数面」、pixelforge 的视觉与画布机制）：

```
┌ 顶栏：品牌 · GPU · 缩放 [− 100% + 1:1 适应] ┐
├───────────────────────────────────────────┤
│ ┌节点列表┐   ┌─ 图生成卡 ────┐  ┌─ 图生图卡 ─────┐
│ │图生成 +│   │ 提示词 / 尺寸   │  │ 源图 ▾ / 重绘   │
│ │图生图 +│   │ 更多设置 ▸     │  │ 提示词 / 更多 ▸ │
│ └────────┘   │ [生成] 预览     │  │ [生成] 预览     │
│              │ ↳ 图生图        │  └────────────────┘
│              └────────────────┘                   │
└───────────────────────────────────────────────────┘
```

**设计决策**

| 项 | 做法 | 理由 |
|---|---|---|
| 节点 | 两种：**图生成**（w1 文生图）与 **图生图**（w2），参数与结果同卡 | 生成与结果是一件事，拆两张卡会让每次生成多一次连线 |
| 图生图的源 | 卡片里选「画布上已出过图的节点」，或在生成节点下方点 **↳ 图生图** 自动接上 | 源图必须落到磁盘绝对路径（后端直接 `Image.open`），所以走节点引用而不是上传 |
| 源图用哪张 | 优先用上游的 **raw**（后处理之前的原图），次选成品 | 拿已经像素化到 128 有效像素的成品回炉，模型只会把色块重新糊成插画 |
| 双视图 | 出图后可切「只看图」，卡片只剩画面 + 种子徽标 | 画布上做对比时，参数是噪音 |
| 新建节点 | 节点列表**拖拽到落点**，或点 **+** 放到原点 (0,0) | 拖拽处落点用 `(client - viewport)/zoom` 换算成世界坐标 |
| 复制/删除 | 卡头按钮 + `Ctrl/Cmd+D` 复制、`Delete` 删除 | 输入框内不拦截按键 |
| 画布 | 自研 viewport(overflow hidden) + world(translate/scale) + 视差点阵网格 | 不上 react-flow：它会把节点模型绑死在库里；InvokeAI 的 README 已现身说法 dockview 之痛 |
| 持久化 | localStorage 防抖 400ms 保存 nodes + viewport | 刷新不丢布局 |
| 画布平移 | **拖空白处**（或空格 / 中键）；滚轮以光标为锚点缩放 | 节点多了全靠它 |
| 左边栏 | 菜单式层级：**项目** / 节点 / 提示词收藏 / 设置，各自可折叠，整条可收起成图标条 | 原来两张浮层卡片会互相挤压、占满画布左上角 |
| 项目 | 每个项目一套独立节点+视口，存 localStorage 的 `meow.projects.v2` | 不同关卡/不同素材线各画各的，互不干扰；老存档自动迁移 |
| 训练入口 | 顶栏「训练 LoRA」浮层面板，有进度条/loss/显存 | 不用切命令行 |
| **提示词语言** | **中英文都行**；输入含中日韩字符时提示「中文没问题，构图契约由后端自动补」 | Qwen 的文本编码器是 Qwen3-VL，官方示例本身就是**几百字的中文详述 prompt**（见 PLAN §5 ③）；SD 时代那条「CLIP 只认英文、中文被拆成字节级乱 token（'背剑侠客' → `è ĥ Į å ī`）」已随 SD 管线作废 |
| **提示词收藏** | 提示词框右上角 ☆ 收藏，左栏「提示词收藏」列出，点条目填入选中节点 | 常用提示词不必反复输入；按文本去重、可删除、localStorage 持久化 |

### 项目：每个项目是一套独立画布

左栏最上面是「项目」分组。一个项目 = 一套自己的节点 + 自己的视口 + 自己的连线，
切项目就是整套换掉，**互不干涉**（A 项目的节点不会漏到 B 项目）。

- **新建**：分组里的「+ 新建项目」，得到一个空画布，顶栏副标题会显示当前项目名。
- **切换**：点项目名即切换；当前项目高亮，条目上写着「N 个节点 · 当前」。
- **改名**：双击项目名，变成输入框，Enter 确认、Esc 取消。
- **删除**：点项目右边的 ✕ → 弹确认框：
  - 「仅从列表移除」：只从列表里去掉，项目内容留在 localStorage；
  - 「真删（含文件）」：把这个项目里**所有节点的产物**（PNG / sidecar / raw）
    一起从磁盘删掉，不可恢复。
- **最后一个项目不能删光**：删光了会自动补一个空的「我的项目」。

**存储：项目内容只在服务端磁盘上，浏览器里什么都不存。**

```
assets/projects/<项目id>/
├── project.json             项目定义（节点 + 视口 + 元信息）
├── w1_..._0000.png          产物
├── uploads/                 本项目拖进来的本地图
└── masks/                   局部重绘的蒙版
```

- **为什么落盘**：项目原先只存在浏览器 `localStorage` 里，实际问题很具体——
  容量就 5MB 左右，一个大项目直接写不进去（`saveStore` 只能默默吞掉异常，
  用户毫无察觉，改动全丢）；换浏览器 / 换机器 / 清缓存，项目就没了。
- **写入是原子的**：先写临时文件再 `os.replace`。半截的 JSON 会让项目打不开，
  而自动保存是高频触发的，不能冒这个险。
- **浏览器零存储**：`localStorage` 里只剩提示词收藏这类界面偏好，
  **没有任何项目内容**。
- **老存档迁移**：旧版本时代的 `meow.canvas.v1` 会在启动时读一次、传到服务端、
  然后**从浏览器删掉**——迁移完不留副本。

**接口**：

| 端点 | 作用 |
|---|---|
| `POST /projects/allocate` | **分配一个保证没用过的项目 id** |
| `GET /projects` | 项目列表（摘要，不带节点）+ 当前打开哪个 |
| `GET /projects/<id>` | 读完整定义（节点 + 视口） |
| `PUT /projects/<id>` | 保存定义（原子写） |
| `DELETE /projects/<id>?files=1` | 删项目（files=1 连产物目录一起删） |
| `PUT /workspace` | 记「当前打开哪个项目」 |

前端启动时拉服务端列表 → 逐个拉完整定义 → 侧栏与画布就位；
改动后防抖 400ms 写回服务端。

### id 冲突怎么办

项目 id **由服务端分配**（`POST /projects/allocate`，随机后缀 + 存在性检查）。
不用「时间戳 + 序号」本地生成，是因为**两个标签页在同一毫秒建项目会拿到
完全相同的 id**（两边序号都从 0 开始，实测 `pmdr5q6ns0`）——后保存的直接
覆盖先保存的，用户毫无察觉地丢了项目。

三类冲突分别这样处理：

| 冲突 | 处理 | 结果 |
|---|---|---|
| **新建时 id 已被占用** | `PUT` 带 `create=true` → **409** | 不覆盖，调用方换 id 重来 |
| **两个窗口编辑同一个项目** | `PUT` 带 `expect_updated`（我读到的版本号）→ **409** | 提示「已被其他窗口修改」，两边内容都保住 |
| **导入到已存在的 id** | 解包前检查目标目录 → **报错** | 不写进去（否则两个项目的素材混在一起，删项目时连带删掉别人的图） |

前端收到 409 会弹提示（「保存冲突：…」），**不静默失败**——
静默失败等于用户以为存上了，其实没存。

> **版本号与时间戳是两回事**：`expect_updated` 是「我读到的那个版本」
> （乐观锁的比对基准），`updated` 是「本次保存的新时间戳」（写进磁盘，
> 作为下一个版本号）。搞混的话本地一改就跟磁盘对不上，每次保存都误报冲突。
> 另注意 `flushCurrent` **不能**动 `updated`——它是版本号，不是修改时间。

**产物也隔离**：每个项目的文件都在自己的目录里：

```
assets/
├── index.jsonl                  全局索引（每条带 scope 字段）
├── projects/
│   ├── p1a2b3/                  项目 A 的产物
│   │   ├── w1_..._0000.png      成品
│   │   ├── w1_..._0000.png.json sidecar
│   │   ├── w1_..._0000_raw.png  后处理之前的原图
│   │   └── uploads/             本项目拖进来的本地图
│   └── p4d5e6/                  项目 B 的产物
└── uploads/                     历史遗留（项目功能之前上传的）
```

- 生成时前端带上 `scope`（项目 id），后端把产物写进 `assets/projects/<scope>/`。
- **跨项目引用会被后端拒绝**：拿 A 项目的图去 B 项目做图生图，直接报
  「源图不属于当前项目」。同项目内引用不受影响。
- 上传（拖进来的本地图）也进本项目目录：`/upload?project=<id>`。
- 资产清单可按项目过滤：`GET /assets?project=<id>`。
- **真删项目 = 删整个目录**（`DELETE /assets` 传 `projects/<id>`），产物、上传、
  帧序列一起清掉，索引里的条目同步摘掉。
- 产物在子目录里之后，`/files/<basename>` 取不到了——后端在结果里直接给
  `url`（相对 assets 的路径），前端用它显示。

> 历史文件（项目功能之前生成的，还在 `assets/` 根目录）仍可访问，也能当图生图的源，
> 它们属于「公共区」；但它们不会被任何项目的新产物覆盖或混进去。

### 项目导出 / 导入（.mproj 单文件）

项目本来只活在浏览器的 localStorage 里——换台机器、换个浏览器就没了。
现在项目能导出成一个 `.mproj` 文件（zip），**连它引用的素材一起打包**：

- **导出**：项目行右边的 `↧` 按钮 → 下载 `<项目名>.mproj`。
- **导入**：左栏「↑ 导入项目」→ 选一个 `.mproj` → 得到一个**新项目**。

**这个文件是自包含的**：节点里引用的图片、蒙版、骨架帧目录都会被收进去。
换台机器导入，素材照样在——不用再手动把图一张张搬过去。

导入的落点是 `assets/projects/<新项目id>/imported/`，**绝不覆盖已有项目**：

| 引用形态 | 例子 | 导入后 |
|---|---|---|
| 单个文件 | `params.maskPath` | 同名文件 |
| 目录 | `psDir` 指向骨架帧目录 | **同构的目录**（内部结构保留） |
| 通配 | `assets/_pose_*.png` | 折成目录（后端接受目录，更稳） |

路径与 `/files/...` 链接在导入时**全部重算**；引用的文件当时不在盘上时，
该路径会被清空（留着一个指向不存在文件的绝对路径，只会让用户点了生成才报错）。

**怎么找到素材**：不去猜「节点里哪些字段是路径」，而是在整棵节点 JSON 上递归找
**看起来像路径且真实存在**的字符串。所以以后加新节点类型，这块不用改。

### 预设模板：一套产线，反复开新项目

同一个项目结构往往要开好几次（同一套「像素精灵产线」跑不同角色）。
左栏「模板」分组就是干这个的：

- **存模板**：项目行右边的 `★` 按钮 → 当前项目存成一个模板。
- **从模板新建**：点模板 → 开一个**新项目**，节点、参数、素材全都在。
- **删模板**：模板右边的 ✕。

模板存在 `templates/<id>.mproj`（**不在 `assets/` 下**——它是跨项目共享的
用户资源，不属于任何一个项目的产物）。同名的模板不覆盖，自动加时间后缀。

### assets/ 里放什么，tests/_out/ 里放什么

这个项目有两类完全不同的文件，以前混在一个目录里：

```
assets/                       项目资产（要交付的东西）
├── index.jsonl               全局索引
├── index.db                  索引的 SQLite 读缓存
├── projects/<项目id>/        每个项目的产物 / 上传 / 蒙版
├── uploads/                  历史遗留的上传
└── pose_walk/ pose_pixel/    可直接当 W4 输入的骨架帧目录

tests/_out/                   测试与验证产物（开发过程，不入库）
├── assets/                   pytest 期间的资产根（见下）
├── templates/                pytest 期间的模板根（见下）
├── ui/                       沙盒实例的资产根（tools/serve_sandbox.py）
├── legacy_assets/            历史上散在 assets/ 根下的调试图
├── legacy_projects/          历史上各项目目录里的测试产物
├── quarantine_templates/     历史上写进真 templates/ 的测试模板
└── _*/                       各个验证脚本 / 实验脚本的输出
```

**为什么要分开**：`assets/` 里的东西是用户要拿走的。混进几百张
`_xxx_compare.png` 之后，资产清单被淹没、真正交付的图翻不到，删项目时还会
连带删掉。新人打开 `assets/` 也不知道哪些是能用的。

**pytest 期间资产根和模板根都改到 `tests/_out/` 下**，不是仓库的 `assets/` /
`templates/`：`tests/conftest.py` 在 collection 阶段就设好 `MEOW_ASSETS_DIR` 与
`MEOW_TEMPLATES_DIR`，`workflows/paths.py` 据此解析路径。这样跑一轮测试，
仓库的 `assets/` 与 `templates/` 一点不脏。

**模板根漏过一次，代价是用户模板列表里多出 5 个垃圾文件**：早先 conftest 只隔离了
资产根，`save_template()` 就直接往真 `templates/` 写。`API_34266.mproj` /
`t_io_d4cdd081.mproj` 这种名字带随机后缀，用户根本分不清哪些是自己存的（实盘
检查过：那几个文件全是 `test_projects.py` 的残留，没有一个是用户建的）。

现在 `conftest.py` 有一条 **autouse 的看守夹具**：开场先断言两个目录都在
`tests/_out` 下，否则**一条测试都不许跑**。理由是隔离完全依赖「conftest 早于所有
模块 import」这个时序，一旦有人把它打破（新增上层 conftest、把 import 提到环境变量
赋值前），隔离就静默失效——测试照样全绿，产物全落进用户目录，没人会发现。

**界面验证必须用沙盒实例**：

```bash
python tools/serve_sandbox.py          # 资产根/模板根指向 tests/_out/ui/
```

**为什么不能直接 `uvicorn app:app` 做截图验证**：前端首屏逻辑是「一个项目都没有
就新建一个」，于是无痕的 headless 浏览器每连一次，就在用户的项目列表里多一个空
项目，还会把 `workspace.json` 的 `current` 指过去——用户下次打开发现画布被切走了。
实测在 20:58 / 21:05 各撞过一次（其中一个空目录残留到现在才清）。

**为什么必须走脚本而不是命令行前缀**：`paths.ASSETS` 是 **import 时求值**的常量，
`uvicorn app:app` 会在导入 `app` 时连锁导入 `workflows`，环境变量必须在那之前设好。
脚本能保证顺序；手写前缀容易漏设置其中一个（比如只设 ASSETS 没设 TEMPLATES）。

路径解析只有一处定义（`workflows/paths.py`），可用环境变量覆盖：

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `MEOW_ASSETS_DIR` | 运行时资产根 | `assets/` |
| `MEOW_TEMPLATES_DIR` | 模板根 | `templates/` |

`tools/` 下的验证脚本（`verify_*.py` / `exp_*.py` / `bench_*.py`）输出统一走
`paths.tool_out()`，一律落在 `tests/_out/`。`tools/pose_from_image.py` 是
**给用户用的工具**（抽骨架帧给 W4 当输入），它的输出仍然由 `-o` 指定，
默认 `assets/pose/`——那是用户要用的素材，不算测试产物。

### 抠图节点：去背景 / 换背景

左栏「节点」组多了一个**抠图**（紫色标题）。它**不用模型**——纯图像处理，
所以不占显存、瞬间出结果。

**源图两种来法**（都支持）：

1. **连线**：从任意节点（生成节点 / 图片节点 / 另一个抠图节点）的 ⤢ 拉一条线过来；
2. **直接加图**：把本机图片**拖到这个节点上**，它就成为源（不用先建图片节点）。

**三种背景方式**：

| 方式 | 结果 |
|---|---|
| 透明（去背景） | 产物是带 alpha 的 PNG，背景全没了 |
| 纯色 | 前景叠到指定颜色上（带取色器） |
| 换背景图 | 前景叠到**另一张图**上（背景图从画布上的图片节点里选，cover 裁切不拉伸） |

指定了背景时会**额外**存一份 `<name>_cut.png`（透明底前景），回头想换别的背景
直接拿它再走一次就行。

抠图节点的产物也是一张图，可以：点开全屏预览 / 下载 / 当别的节点的源（再接一个
图生图继续改）/ 继续连线给下一个抠图节点。

后端对应 `POST /generate` 的 `workflow: "wbg"`：

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "workflow": "wbg",
  "params": {"init_image": "D:/.../assets/projects/p1/xxx.png", "bg_color": "#204080"}
}'
```

### 局部重绘节点（w3）

左栏「节点」组里的**局部重绘**（粉色标题）：源图 + 笔刷选区 + 提示词，
只重画涂出来的区域，**选区外像素逐点不变**（后端羽化蒙版向内收缩后贴回，
这条承诺有测试钉着）。

**源图**：连线接过来，或把本机图片直接拖到节点上。
**蒙版**：点「✎ 绘制蒙版」打开全屏笔刷，涂出要重绘的区域（白色 = 重绘区）。

笔刷是照着 pixelforge 的 MaskEditor 移植的，那边已经踩过一轮坑，几个关键经验：

| 做法 | 为什么 |
|---|---|
| **三层画布**：图 / 蒙版 / 覆盖 | 覆盖层是半透明红（给眼睛看），擦除时用 `destination-out` 只擦自己，下面的图不动 |
| **撤销用「操作列表 + 全量重放」**，不存 ImageData 快照 | 1024² 的快照是 4MB/步，撤 20 步就是 80MB；操作列表每步几十字节，重放一张图几毫秒 |
| **笔刷一律硬边** | 柔和边界交给后端的羽化（那边按像素算、可复现）；画布上做软笔刷只会得到不稳定的半透明，「选区外不变」也没法验证 |
| **覆盖层用不透明色 + CSS opacity** | 直接画 `rgba(0.4)` 的话，同一笔拖动时相邻线段会互相叠加（0.4→0.64→0.78…），看起来像「有的地方没涂上」 |
| 蒙版颜色/透明度记 localStorage | 换过一次颜色的人不希望每次重开又要再挑一遍 |
| **H 键临时隐藏蒙版** | 用原图核对「到底涂到哪了」 |

操作：画笔 / 橡皮 / 矩形框选，笔刷大小滑杆，滚轮缩放、空格拖动平移，
`Ctrl+Z` 撤销、`Ctrl+Shift+Z` 重做，「✕ 清空」重来。

蒙版存在**项目目录的 `masks/` 下、同名覆盖**（每个节点一份）——所以反复涂抹
不会攒出一堆垃圾文件。节点里只记路径与缩略图 URL。

后端对应 `POST /generate` 的 `workflow: "w3"`（`image` + `mask` 两个路径），
蒙版另有一个 `POST /mask` 端点负责存 PNG。

### 局部 AI 重绘节点（w5）

和上面的「局部重绘（w3）」并列的另一种做法：**裁剪「选区 + 上下文」→ 交给通用模型
重画 → 贴回原图**。左栏「节点」组里是「局部AI重绘」（绿色标题）。

**和 w3 的区别**（选哪个看场景）：

| | w3 局部重绘 | **w5 局部 AI 重绘** |
|---|---|---|
| 模型 | SD1.5 **专用 inpaint 权重** | **任意底模**（含 SDXL、像素 LoRA） |
| 分辨率 | 512 封顶 | 选区放大到 512~1024 再画 |
| 能挂 LoRA | ❌ | ✅ |
| 原理 | 管线原生 mask | 裁剪 → 图生图 → 贴回 |

所以：**像素素材改局部**必须用 w5（w3 的 inpaint 权重画不出像素风），
**大幅改写整块内容**用 w3 更省事。

**操作**：源图（连线或拖图）+ 笔刷蒙版（与 w3 共用同一个编辑器）+ 提示词 + 重绘幅度。

「更多设置」里的四项：

| 参数 | 默认 | 作用 |
|---|---|---|
| 上下文 | 0.35 | 选区外带多少上下文（按选区尺寸的比例）。越大模型看到的参照越多，也越容易改到选区外 |
| 外扩 | 8 px | 蒙版先外扩一圈，给模型「往哪长」的余地 |
| 羽化 | 4 px | 贴回时的边缘过渡。0 = 硬边（像素素材常用） |
| 接缝色彩校正 | 开 | 只在**边界带**做色彩校正。关掉的话模型新颜色可能和周围对不上 |
| 色板吸附 | 关 | 把重绘区里「接近源色」的杂色吸回源图色板（像素素材建议开） |

**「选区外的像素一个都不变」是逐像素可验证的硬承诺**（有测试钉着）：贴回的羽化
只向内——羽化权重 = 模糊(蒙版) × 二值蒙版，区域外的权重恒为 0。

**重绘幅度（strength）的实测含义**（像素素材，25 步，512² 源图）：

| strength | 效果 |
|---|---|
| 0.80 | **选区几乎没变**——模型保留原样，等于没重画 |
| 0.90 | 改动开始明显（默认值） |
| 0.95 | 改动最明显，但硬边会留矩形拼接痕 |
| 1.00 | 整块重画，风格漂移最大 |

所以默认给 **0.90**。另外实测：硬边（羽化 0）在重绘幅度大时会看到明显的矩形拼接
痕，**羽化 8px + 边界色彩校正**能把接缝收掉（`docs/img/ai_repaint_ab.jpg` 的
hard / soft 两组对照）。

![局部 AI 重绘对照](docs/img/ai_repaint_ab.jpg)

上面这组对照还暴露了一个坑：**透明底素材送模型时必须把透明区填白**，不能直接
`convert("RGB")`——那样透明区会变黑，模型会把背景也当成内容重画，出来的选区
周围带一圈脏边。

两个踩过的坑（都写进注释了）：

- **送模型前必须转 RGB**。RGBA 图进 VAE 会变成 4 通道，UNet 首层按 4 通道卷积，
  激活尺寸直接爆炸——实测 512² 报 `Tried to allocate 36.00 GiB` 当场 OOM。
- **约束条款由后端拼成英文**。不是为了"模型只认英文"——Qwen3-VL 支持中文，
  用户正文写中文完全没问题。英文的作用是和 `presets` 的英文后缀拼在一起时
  语言统一，模型更容易分清"哪句是产线契约、哪句是用户描述"。
  SD 时代"整句约束 100+ token 必然在 77 处被截断"的结论也作废了：
  现在 `presets.build()` **不截断**（模型侧也没有 truncation）。

后端对应 `POST /generate` 的 `workflow: "w5"`：

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "workflow": "w5",
  "params": {"image": "D:/.../assets/projects/p1/src.png",
             "mask": "D:/.../assets/projects/p1/masks/mask_x.png",
             "prompt": "give the knight a golden helmet",
             "context": 0.35, "expand": 8, "feather": 4, "palette_snap": true}
}'
```

### 姿态序列节点（w4）

左栏「节点」组里的「姿态序列」。输入不是单张图，而是一组**骨架帧**：

```bash
# 从图片 / 视频 / 精灵图提骨架，产物是一个目录
python tools/pose_from_image.py walk.mp4 --step 3 -o assets/pose_walk
```

节点里的目录下拉会列出 `assets` 下所有含 ≥2 张图的目录（后端 `/pose_dirs`，20ms 扫完），
也可以手填目录或通配（`assets/pose_walk/*.png`）。后端用**自然序**展开
（`frame_2` 排在 `frame_10` 前）——字典序会让 10 帧以上的序列乱序，而且不报错。

它和「序列帧替换」的区别：wss 是拿现成的精灵图**换角色**，w4 是从骨架**从零生成**，
所以能吃候选 + 闸门这套产线机制。

结果区会显示 `候选 N/帧 · 补提 X 次 · 闸门 M/K`——候选数不等于设定值就说明被显存降级了。

**「更多设置」里的三个开关**：

| 开关 | 作用 | 实测 |
|---|---|---|
| **底色**：白底 / 深色底 | 白毛、白甲角色用白底，抠图会把脸和腿一起判成背景 | 深色底不丢头顶与腿 |
| **角色图**（正文里） | 不选就只靠提示词描述角色，帧间身份会漂 | 给了之后帧间身份更一致，但它也会「抢姿态」 |
| **两阶段** | 先用骨架出姿态草图，再用角色图换皮 | 姿态偏差 **102px → 77px**，代价是每帧多一次生成（33s → 50s），服装细节会漂 |

两阶段的原理与数据见 `docs/PLAN-QWEN-ONLY.md` §2.9。

**非人形主体**（史莱姆、机甲、道具）：DWPose 检不出关节，闸门永远过不了。
现在一旦判定为「不是人形」就自动关掉补提（每帧省 25~30s），产物 meta 里
`gate_non_humanoid=true`、`gate_note` 写明原因——不再显示成「闸门 0/8 像失败」。

### 局部 AI 重绘的几何预览

节点上有一张小预览图（源图 + 三层色带），拖「外扩 / 羽化」滑块时实时更新：

| 颜色 | 含义 |
|---|---|
| 红 | 你涂的选区 |
| 橙 | 蒙版外扩的环（给模型留"往哪长"的余地） |
| 黄 | 贴回的羽化过渡带 |

为什么值得做：这两个参数改完只能点一次生成才知道效果，一轮十几秒；而它们的
后果很具体——**外扩太大会吃掉邻格**（像素素材直接废），羽化太大接缝会糊。
这些都能本地算出来，不需要模型。

实现上有个坑：`blur + 阈值化` 根本不能当膨胀用。实测 blur 会把边缘灰度拉低，
阈值化后反而**缩小**（r=16 时边界退了 3px）。所以按后端的
`repaint.dilate`（PIL `MaxFilter`，方形核）重写成可分离的方形膨胀，
行/列扫描 O(w·h)，与半径无关，拖滑块不卡。**算法与后端逐像素一致**
（r=1/2/3/5/8 全部对上）。

它只画几何范围，不预测画面内容——重绘幅度改变的是"模型画成什么样"，
那部分必须生成才知道，界面上也写明了。

### 角色迁移节点（wpc）

给一张**动作参考图** + 一张**角色图**，生成角色做出参考图那个动作的图。
左栏「节点」组里是「角色迁移」（青色标题）。

![角色迁移](docs/img/pose_clone.jpg)

上面这组对照：动作参考图（真人双臂张开）→ 提取骨架 → 角色图（蓝色史莱姆）
→ 结果是**史莱姆做出了双臂张开的站立姿态**，外观（蓝色、圆润、眼睛）保留。

三件套各管一件事：

| 输入 | 通路 | 作用 |
|---|---|---|
| 动作参考图 | 骨架或轮廓 → **ControlNet** | 定姿态（硬约束） |
| 角色图 | **IP-Adapter** | 定身份/外观（软约束） |
| 提示词 / LoRA | — | 定画风与细节 |

**关键实现：骨架必须归一化。** ControlNet OpenPose 是空间敏感的——条件图里人物
站在哪、占多大，生成的人就站在哪、占多大。参考图里的人物可能缩在角落，直接用
原始骨架会画出「角落小人」。所以先按检测到的人体框把关键点归一化（人物居中、
占画面 `占高` 比例），再渲染骨架图。

**「占高」参数**（默认 0.8）就是人物在画布里占的高度比例。

三个强度参数：

- **姿态**（ControlNet，0.2~1.5，默认 1.0）：越大越贴参考动作
- **身份**（IP-Adapter，0~1.5，默认 0.8）：越大越像角色图
- **占高**（0.3~1，默认 0.8）：人物在画布里占多高

### 参考模式：不止真人，游戏素材也能用

**参考模式**参数决定「动作参考图」怎么变成 ControlNet 条件图：

| 模式 | 做法 | 适用 | 实测 |
|---|---|---|---|
| **自动**（默认） | 先试骨架，**认不出人体就自动降级到轮廓** | 通用 | 真人走骨架、游戏素材走轮廓 |
| 骨架 | 人体检测 → RTMPose → OpenPose-18 | **只对真人有效** | 真人照 18/18 关节全中 |
| 轮廓 | 抠主体 → 剪影 → 自制软边缘 | **任意素材**：像素画、怪物、机甲、非人形生物 | 轮廓 IoU **0.899** |

为什么必须有轮廓这条路：姿态模型**只认得出真人**。实测像素画上人体检测器
**一个框都检不出**（放大、LANCZOS 抗锯齿、加描边、重模糊全都试过，仍然 0 框），
因为 FasterRCNN 是在 COCO 自然照片上训练的。游戏素材想用骨架路线是走不通的。

轮廓路线保留的是**剪影形状**（含手持道具），所以能把像素角色、怪物当动作参考。
代价是轮廓里的道具也会被抄走（参考图里的剑会出现在结果里）。

> **一个关键的坑：条件图不能只给边缘。**
> canny 条件图天然是「白线画在黑底」，模型看到满屏黑底加一个闭合轮廓，最省力的
> 解释就是「整屏都是画布」——于是它把整个画面填成了纹理。实测最终成品里主体
> **只剩 3 个像素**（参考帧是 677 个），完全不可用。
>
> 解法是给**自制软边缘**：保留参考图自身的灰度结构（让模型知道「这个闭合形状
> 内部是有内容的实体」），再把边缘加重成硬边。同一张图、同一套参数下对比：
>
> | 条件图 | 成品主体像素（参考 677） |
> |---|---|
> | 只给边缘 | **3** ← 不可用 |
> | 白底 + 中灰主体 | 628 |
> | 白底 + 灰主体 + 内部结构边 | 590 |
> | **自制软边缘** | **709**（轮廓 IoU 0.899） |

后端对应 `POST /generate` 的 `workflow: "wpc"`：

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "workflow": "wpc",
  "params": {"pose_image": "D:/.../pose.jpg", "char_image": "D:/.../hero.png",
             "prompt": "a blue slime monster with big eyes",
             "cn_scale": 1.0, "ipa_scale": 0.8, "fill": 0.8,
             "ref_mode": "auto"}
}'
```

> 顺带修了一个一直存在的 bug：随包的 `yolox_l.onnx` 输出的是**未解码**的
> head 结果（前 4 列是网格坐标 + log 尺寸），且 anchor 布局与官方 YOLOX 的
> anchor-major（25200 条）不一致——8400 恰好等于格子数，说明每格只输出一个
> anchor。按官方 decode 解出来的框全是垃圾（实测 `(-1,1,1,4)`），人体裁剪
> 完全错位，姿态回归置信度全线偏低。**它没有可靠的解码方式**，所以默认检测器
> 改用 torchvision 的 FasterRCNN（直接输出像素坐标，实测同一张图 **18/18
> 关节全中**）。yolox 后端仍可显式指定，但结果不可信。
> 这个 bug 一直影响 W4 的质量闸门——闸门一直在用错误的人体框打分。

### 序列帧替换节点（wss）：游戏动画序列帧 → 换角色

给一张**游戏动画精灵图** + 一张**角色图**，把精灵图里的角色换成目标角色，
**动作保持不变**，输出仍然是同样网格的精灵图。
左栏「节点」组里是「序列帧替换」（紫色标题）。

这解决的是产线上最常见的场景：手上有一段现成的游戏动画（走跑攻击的序列帧），
想把它里面的角色换成自己的角色，但动作必须一模一样。

**两种方式**：

| 方式 | 做什么 | 跑模型 | 帧间漂移 | 适用 |
|---|---|---|---|---|
| **AI 重绘** | 轮廓条件（动作）+ IP-Adapter（身份） | 是 | 有（会按脚底锚点对齐） | **换体型**：把骑士换成史莱姆 |
| **换皮** | 明度重映射到角色色板，alpha 原样保留 | **否** | **零漂移** | **换配色**：同体型换色，毫秒级 |

**只有一个引擎**：逐帧生成。

SD 时代的「AnimateDiff（一次生成整段，跨帧时序注意力）」已随 SD 一起删除——
它依赖 SD1.5 的 motion module。Qwen 没有等价物，逐帧路线就是现在这条
（每帧 25~30s，见 `docs/PLAN-QWEN-ONLY.md` §2.8）。

**残缺帧补提**还在：逐帧量出不合格的帧（剪影 IoU < 0.40 或主体面积塌陷），
用同样的条件重跑一次顶上，好的帧保留。补提帧的身份可能跟不上，用**整段共享
色板重新上色**修回来。

### 像素转换节点（wpx）：任意图 → 像素完美

左栏「节点」组里的**像素转换**（黄色标题）。**不跑模型**，纯确定性处理，所以
瞬间出结果、不占显存。

它和 W1 里那个「像素化」不是一回事：

| | W1 的像素化（pixel 参数） | **像素转换（wpx）** |
|---|---|---|
| 原理 | BOX 降采样 + 最近邻放大 + 中位切分量化 | **网格检测 → 吸附 → 每格单色** |
| 网格 | 不对齐（AI 图的像素块常在 37.3px 这种非整格上，边界被切在中间混成脏斑） | 对齐 |
| 每格一色率 | 不保证 | **100%**（有测试钉着） |
| 输出 | 原尺寸放大版 | **原生像素尺寸**（128×128 这种） |

流程：**FFT 检测网格 → Sobel 吸附到最强梯度峰 → 每格中位色采样 → 共享调色板
量化 → alpha 二值化 → 孤立噪点清理**。

![像素转换对照](docs/img/pixelfy_ab.jpg)

上面这组对照（第 1 张是源图 768²，第 2~5 张是各种参数）：`auto_32`（自动检测
网格 + 32 色）像素感最强，`nosnap`（关掉网格检测、退化成普通降采样）明显更糊、
边缘有毛刺——这就是网格对齐的价值。

几个关键设计：

- **输出只存原生像素尺寸**（如 128×128），**不预放大 8 倍**。放大交给渲染端
  （前端 `image-rendering: pixelated`），这样以后想换 4 倍 / 16 倍不用重算。
- **每格取色默认用中位色**（median）。实测重建误差 2.4、色数 550；多数表决
  （k-means k=2）误差 5.0、色数 1378；直接取格中心最差（误差 11.9）。
- **共享色板**：一批图先各自网格对齐，再汇总提取**一块**色板统一映射。
  逐张各自取色板会显著偏色（第一张的主色占满色位，后面图的主体只剩几档）。
- **色板外颜色数必须为 0**——这是验收指标，写进 meta 随时可查。
- 网格尺寸离谱时（远大于边长 / ≤ 0）**立刻返回**，不会死循环。
- **网格吸附到标准尺寸**（默认 16 的整数倍）：AI 出的「像素风」本来就没有
  严格网格（实测检测出 157×153 这种），这种尺寸的精灵图没法用。吸附后
  输出一定是 16 / 32 / 64 / 128 这种能用的大小。`吸附=0` 可关掉。

参数：边长（0 = 自动检测）、色数（0 = 不量化）、取色方式、网格吸附、网格检测
开关、纯色背景转透明。

后端对应 `POST /generate` 的 `workflow: "wpx"`：

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "workflow": "wpx",
  "params": {"init_image": "D:/.../assets/projects/p1/src.png",
             "colors": 32, "sample_method": "median"}
}'
```

### 转像素节点（topixel）：任意图 → 像素画

左栏「节点」组里的**转像素**。**这是「像素生成质量差」的正解**——
不要在提示词里让模型直接画像素。

**不只动漫人物**：照片、插画、3D 渲染、手绘图都能转——像素化那一半是通用的
（网格对齐 + 色板量化 + 轮廓描边），跟前面出的是什么图无关。
只有「没给源图、需要用描述先生成」时才会走动漫底模。

| | 让 SD 直接画像素 | **动漫大图 → 像素化** |
|---|---|---|
| 构图 | 常常裁到腿（实测「全身」提示词也出半身） | **全身完整** |
| 颜色 | 连续渐变，不像色块 | **色块干净** |
| 描边 | 没有结构线 | **1px 轮廓，色数受控** |
| 输出尺寸 | 768² 但有效像素只有 128 | **原生 128²/192²** |

根因是**两种能力互相打架**：像素风要求极简色块和硬边，而人物结构（脸、手、
持握武器）需要足够分辨率才能画对。让模型一次做两件事，它会在两者间妥协。
拆开做则各取所长——动漫底模负责「画对」，像素化负责「压成像素」。

**两种用法**：

1. **只写描述**：`1girl, red cape, blue armor, full body`
   → 先出 1024² 大图 → 转成 192² 像素画。一步到位。
2. **接一张现成的图**（任何图都行）：直接转，不跑模型、秒回。

参数：边长（默认 192，装得下「全身 + 武器」）、色数（默认 32）、
对比度增强（默认 1.0）、轮廓描边（默认开）、抠背景（默认开；转风景/照片时关掉）。

后端 `POST /generate` 的 `workflow: "topixel"`（旧名 `wap` 仍可用）：

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "workflow": "topixel",
  "params": {"prompt": "1girl, red cape, blue armor, holding sword, full body",
             "target": [192, 192], "colors": 32}
}'
```

#### 像素化的三步质量增强（都是实测出来的）

这三步是这次修「像素太差」的核心，缺一个质量就掉：

**1. 色板提取必须排除透明像素**（真 bug）

抠图后的图透明区 RGB 是**纯黑**（`convert("RGB")` 把 alpha=0 变成 0,0,0），
而角色图前景常常只占 **16%**。不排除的话色板被「背景黑 + 抠图边缘白」吃掉
大半色位——实测一张 1024² 动漫立绘转像素后**整张只剩 12 色、几乎全白**，
主体真正的银发蓝甲一个都没进色板。修复后色板从 12 色恢复到 32 色。

顺带修了序列帧替换里的同一个问题：那里把 RGBA `convert("RGB")` 再送进
`refine_batch`，透明区（常占 70%+）同样污染色板，导致「色板只有 2 色」。

**2. 量化前拉对比度**（业界第一步）

色板受限后，原本靠细微渐变区分的东西会糊成一片。实测同一张图转 192²/32 色：

    不预增强   红披风偏暗红、金发偏土黄、整体发灰
    强度 1.0   鲜红 + 亮黄，色块边界站得住

业界（hermes pixel-art、hypeart 的 photo-to-pixel 综述）都把这步列在第一位，
且**色板越小越要强**。

**3. 轮廓描边**（四阶段流水线的第 3 步）

量化后相邻色块只是「颜色不同」，没有「结构线」，缩小看就是一坨色斑
（业界叫 melt）。加上外轮廓的 1px 描边后，盔甲和披风才立得起来。

两个坑：

- **描边色必须从色板里选**。第一版直接乘 0.7 压暗，色数从 32 涨到 45~49——
  像素画的硬约束是「颜色数 ≤ 色板」，凭空多出中间色等于破功。现在改成
  「找色板内更暗的一档」，色数严格不变（实测 32 → 24~29）。
- **只描外轮廓，不描内部结构边**。对**所有**色差边界描边会把脸毁掉——
  实测同一张角色图，全边界描边后脸部五官糊成一块灰；只描轮廓时脸保持干净、
  盔甲与披风边界照样立起来。像素精灵的立体感主要来自外轮廓。

### 全屏预览与下载

图在卡片里只是缩略图，看不清像素细节。三种方式进全屏（灯箱）：

- **点缩略图**（参数视图下方的预览图）
- **点卡头的 ⤢ 按钮**
- 图片节点的本地图直接点图

灯箱是全屏遮罩，显示原图（`image-rendering: pixelated`，像素画放大不糊），顶部有文件名和
**「⭳ 下载」**按钮。三种关闭方式：点背景、按 `Esc`、点「✕ 关闭」。

下载走 `GET /files/<相对路径>?download=1`——后端会带上
`Content-Disposition: attachment; filename="xxx.png"`，浏览器直接存盘而不是内联打开。
不带 `download=1` 时是内联显示（预览用）。

### 删除：确认框 + 真删

删节点不再是一按就消失。点 ✕ 或按 `Delete` 会弹确认框，三个出口：

| 出口 | 行为 |
|---|---|
| 取消 | 什么都不做 |
| 仅移除节点 | 只从画布拿掉，**磁盘文件留着**（后悔了还能在 assets 里找回） |
| 真删（含文件） | 调 `DELETE /assets`，把产物、sidecar、raw 一起删掉，**不可恢复** |

几个细节：

- **被引用时禁掉真删**：如果这些文件还被别的节点当源图，弹窗会说明「还有 N 个节点
  把它当源图」，并把真删按钮置灰——否则会默默把下游图生图节点变成「没有源的节点」。
- **删除会断开连线**：删掉的节点如果被别的节点连着，那些节点的源会被清空（不悬空）。
- **索引同步清理**：真删会把 `assets/index.jsonl` 里对应的条目摘掉，`/assets` 不会
  再列出已经不存在的文件。
- **路径穿越被拒**：后端只允许删 `assets/` 目录内的文件，`../` 一律跳过（返回
  `skipped`，前端能看到「哪几条没删掉」）。

### 图片节点、拖本地图、连线

- **把本机图片拖进画布** → 变成一个「图片」节点（黄色标题）。图片会先 POST 到
  `/upload` 存进 `assets/uploads/`，画布存的是磁盘绝对路径 + 展示 URL 两份：
  前者给图生图当 `init_image`，后者给 `<img>` 显示。
- **拖到图生图节点上**（而不是空白处）→ 直接成为那个节点的源，不用先建节点。
- **图片节点可以自由改变展示大小**：右下角有拖拽手柄（按原图比例缩放），
  也可以直接改「宽」输入框（60~900px）。高度按原图比例自动算。
- **连线**：每个节点卡头右侧有一个青色连接点，从这里拖一条线到图生图节点上，
  就把它设成了源（画布上会画一条贝塞尔曲线）。图生图节点也可以在卡片里用下拉换源。
  只有图生图节点能接源——文生图没有「底图」这回事。

### 图生图节点（w2）

用途：拿一张已有图**改内容不改构图**——换配色、加装饰、把草稿修成成品、把半身补成全身。

- 建节点：左栏「图生图 +」，或在任意已出图的节点下方点 **↳ 图生图**（自动接上源图、
  自动带过去提示词，落在源节点右边 340px）。
- 卡片里可以直接换源（下拉列出画布上所有出过图的节点），**重绘幅度** 0.15~0.90：
  0.3 只动细节，0.55 默认（明显改内容又不丢构图），0.85 基本重画。
- 尺寸随源图，卡片里没有宽高输入（后端也不接受，传了会被 diffusers 忽略）。
- 后处理与 W1 完全同一套（抠图 / 像素化 / 量化 / 去孤立点），所以图生图出来的也是
  成品精灵，不是「半成品再手工处理」。
- **没有源图时生成按钮是禁用的**，不会提交一个必然失败的任务。
- 产物同样多存一份 `<name>_raw.png`，可以继续往下接第二个图生图节点（链式重绘）。

实测（同 seed，左=文生图，右=以左的 raw 为源的图生图 strength=0.45）：构图、轮廓、
像素块完全保持，只在细节上重画——这正是图生图该有的样子（要大改就把幅度调到 0.7+）：

![文生图 → 图生图](docs/img/img2img_chain.jpg)

## 抠图 / 动效导出

```python
r = poseswap(eng, pose_images=poses, prompt="blue robot",
             pixel=4, palette=16, matte=True,          # 出透明底
             export="assets/export/robot", anim_name="walk", fps=10)  # 出 Godot 工程包
```

导出产物（`export_godot`）：图集 PNG、`SpriteFrames.tres`（每帧一个 AtlasTexture）、
`AnimatedSprite2D.tscn`（`texture_filter = 1` 最近邻、`autoplay` 已开、
`offset.y = -cell_h/2` 脚底对齐）、引擎无关 JSON、GIF/APNG 预览、zip 包。
全是确定性文本生成，同输入逐字节可复现。

抠图两个后端（`matte_backend="auto"` 默认有分割模型就走第二个）：

| 后端 | 耗时 | 适用 | 实测短板 |
|---|---|---|---|
| `color`（连通域色键） | **0.02s/帧** | 平坦纯色背景的精灵 | 白袍/浅色角色贴白底会被误抠（实测把脸抠没） |
| `u2net`（isnet-general-use, 170MB） | 1.6s/帧 | 复杂背景、水墨装饰、浅色主体 | 慢 80 倍 |

同一张武侠图对比：`color` 把背后的圆形水墨装饰整个留下、底部黑块也在；
`u2net` 只留人，脸、剑、衣摆全在。**武侠 / 立绘类一律走 u2net。**

像素画也一样——这条是量过的（`python tools/ab_matte.py`，768² 四个样本）：

| 样本 | `color` 前景占比 / 连通块 | `u2net` 前景占比 / 连通块 | IoU |
|---|---|---|---|
| 像素 LoRA 史莱姆（彩色纹理底） | 0.98 / 1（几乎没抠） | **0.12 / 1**（只留主体） | 0.13 |
| 像素 LoRA 白猫骑士（灰底） | 0.91 / 1（脸被当背景抠没） | **0.41 / 1** | 0.33 |
| 武侠白袍（白底） | 0.35 / **3**（袍子被抠碎） | **0.42 / 1** | 0.63 |
| W4 姿态帧（灰底） | 0.195 / 1 | 0.191 / 1 | **0.97** |

![抠图后端对照](docs/img/matte_ab.jpg)

每行三联：原图 / `color` / `u2net`。第二行最能说明问题——`color` 把白猫的**脸和身体
当背景抠掉**（只剩红披风），`u2net` 完整保留。

两个后端的 alpha **都是二值**（中间 alpha 占比 0）——所以「u2net 边缘软、不适合
像素画」的假设不成立，`auto` 继续默认 u2net 是对的。

但 u2net 会**整张抠空**：草稿档（256²/8 步）实测只剩 0.44% 前景，下游贴边检测 /
主体计数 / 像素化全拿到空数据，最后交出一张空精灵且不报错。现在 `matte_auto`
会检查前景占比，落在 1%~95% 之外就回退色键并取更合理的那份，同时把
`matte_used` / `matte_fallback` 写进产物 meta。

抠图三个要点：

1. 只抠**与画面边界连通**的同色区域——否则白猫肚子贴着白底会被一起吃掉。
2. 去白边要**迭代**：混色边通常 1–3px 厚，单次膨胀只剥掉最外一圈（实测残留 184 像素 → 迭代 3 次后 0）。
3. alpha 一律二值。像素素材要硬边，半透明会引入新的脏边。

## W5 视频驱动（Wan2.2 而非 MimicMotion）

选 Wan2.2 TI2V-5B 而不是 MimicMotion，理由很实际：Wan 已在盘上（pixelforge 的 37GB），
MimicMotion 要另下 ~16GB（代理 480KB/s ≈ 9 小时）。MimicMotion 的优势是「真人动作驱动」，
等有明确需求再单独接。

```bash
python tools/video.py t2v "pixel art cat knight walking, flat background" -o assets/clip
python tools/video.py i2v assets/cat.png "the same character walking" -o assets/clip
```

实测（4090 24GB，bf16 + enable_model_cpu_offload）：

| 项 | 数值 |
|---|---|
| 加载 | 23–30s |
| 25 帧 384² / 10 步 | **26.8–30.6s（1.22s/帧）** |
| 峰值显存 | **10.7GB** |

比预期省：pixelforge 记的 19GB 是 fp8+offload；bf16 + 仅 offload 实测 10.7GB 就够。
注意视频底座要和 SD1.5 分开跑（先 `eng.free()`），否则两套权重抢显存。

**一个坦诚的限制**：人体姿态模型对**非人形角色**（四足/怪物）几乎无效——实测把生成的
四足生物视频喂进去，5 帧里只有 3 帧能提够关节。W5 用在人形角色上才顺。

## 测试

```bash
python -m pytest tests -q                          # 275 个用例，约 150s（含两次 Qwen 管线加载）
MEOW_SLOW=1 pytest tests -m slow                  # 真跑 Qwen 的端到端组（约 3 分钟）
```

`-m slow` 那组会**真的加载权重出图**：十个工作流各出一张（w1/w2/w3/w5/w4/wpc/topixel…），
加上 API 层的队列往返，用来确认「改完出图链路没坏」。默认不跑，因为它每张 10s 起、
整组 1 分多钟，日常改代码时全量测试要保持在 2 分钟内。

**按改动选测试，不要每次都跑全量**（全量 2.5 分钟，只在合并/发版前跑一次）：

| 改了什么 | 跑这些 | 约 |
|---|---|---|
| `workflows/engine.py` | test_models test_vramopt test_stability | 110s |
| `workflows/generate.py` | test_qwen_workflows test_gate test_denoise | 45s |
| `workflows/presets.py` | test_presets test_quality | 10s |
| `workflows/matte.py` / `refprep.py` | test_matte test_refprep | 20s |
| `workflows/postprocess.py` / `pixelfy.py` | test_postprocess test_pixelfy test_sheet | 15s |
| `app.py` / `workflows/assetdb.py` | test_api test_review_p2 | 15s |
| `workflows/projectio.py` / `paths.py` | test_projects | 5s |
| `workflows/projectstore.py`（项目落盘） | test_project_store | 3s |
| `workflows/emotiongen.py`（表情批量） | test_emotiongen（含一条真 YOLO） | 6s |
| `workflows/qwen_llm.py`（本地 LLM：反推 + 向导） | test_qwen_llm（真推理那条要 `MEOW_SLOW=1`） | 5s |
| `workflows/clothesgen.py`（换装） | test_clothes | 5s |
| `workflows/posegen.py`（多姿势） | test_posegen | 5s |
| `workflows/spriteatlas.py`（图集导出） | test_spriteatlas | 4s |
| `workflows/characters.py` / `charactergen.py` | test_characters test_char_poses | 9s |
| `tools/*` `docs/*` `README` | 不跑（工具靠自身输出验证） | — |
| 合并 / 发版前 | 全量 | 150s |

**测试产物不进 `assets/`**：`tests/conftest.py` 在 collection 阶段设
`MEOW_ASSETS_DIR=tests/_out/assets`，pytest 期间的一切生成物（产物 PNG、索引、
上传、蒙版）都落在 `tests/_out/` 下。仓库的 `assets/` 只放项目资产，
跑一轮测试它一点不脏。详见「assets/ 里放什么，tests/_out/ 里放什么」。

纯 CPU 的子集（presets / postprocess / matte / repaint / assetdb / qwen_workflows）
都在 1 分钟内。**只有真正加载 Qwen 权重的三条**（test_models 的 /models、
test_stability 的 free 与泄漏、test_matte 的缓存素材）需要 GPU。

| 文件 | 覆盖 | 用例数 |
|---|---|---|
| `tests/test_qwen_workflows.py` | **十个工作流的能力表与「近似」标记**、**局部重绘选区外逐点不变**、**output_resolution 真的对齐**、**逐帧尺度归一**、**透明底 alpha 不被白底吃掉**、**Qwen 路径不许碰已删的 SD 管线**、**非人形主体不补提**、**两阶段每帧调两次模型** | 37 |
| `tests/test_gate.py` | 公共闸门组件（候选怎么跑、同分取第一个、非人形不补提） | 6 |
| `tests/test_e2e_qwen.py` | **十个工作流各真跑一次模型**（`-m slow`，约 80s） | 10 |
| `tests/test_spriteatlas.py` | **底边对齐 + 水平居中**、尺寸不同的帧不拉伸、货架打包换行与留缝、**TexturePacker 结构**（frames/meta/animations）、帧名与动画分组、单帧动画给 note、原子写不留 .tmp、API 导出 + 统计 | 15 |
| `tests/test_posegen.py` | **指令逐字**（图序 / 同行接 prompt）、**整批共用种子**、**整批替换（旧帧进 V1）**、每张一句/一句话通吃、单张失败不丢整批、姿势来源（路径 /files / 目录展开 / 去重）、接口 enqueue + progress | 14 |
| `tests/test_clothes.py` | **换装 prompt 逐字**（指令句 / 字段顺序 / 背景 RGB+hex / 权重负向词）、clone 模式、蓝底负向词、**底图优先级 Naked→Original**、**跳过 V1 归档**、出图落盘命名、进度、**接口只返回不落盘** | 21 |
| `tests/test_qwen_llm.py` | **解析三级降级**（纯 JSON / 代码块 / 数组 / 带解说 / 尾逗号 / 废话）、**属性收敛**（list→逗号、sex/age clamp）、**白名单挡注入**、**透明图压背景色**、**送模型的图缩到 512**、**GGUF magic 校验**、**装错 fork 报 DEPENDENCY_MISSING**、**推理超时**、**错误码 → HTTP 状态**、**不写 config** | 30 |
| `tests/test_emotiongen.py` | **pad 到 32 倍数用 replicate**、**羽化蒙版是斜坡+模糊（不是硬边）**、**贴回只动框内、alpha 不被贴出方块**、**推理尺寸按官方总像素公式**、**prompt 模板逐字**、**dilation 用了两次**、**产物序号跟源帧对齐**、**没源帧要说清原因**、**单张失败不丢整批**、**队列里的 progress** | 34 |
| `tests/test_matte.py` | 背景估计、连通域不误抠内部同色区、去白边、alpha 二值、**u2net 抠空自动回退**、**已有 alpha 复用（IoU 1.000）**、**雾状 alpha 仍重抠**、**补内部洞但不动通到边界的缺口** | 18 |
| `tests/test_api.py` | 各工作流队列往返、**未知 workflow 报错**、蒙版端点、上传端点、真删（连带 sidecar/raw、路径穿越）、资产按项目过滤、**参数按签名过滤并记 params_ignored** | 33 |
| `tests/test_projects.py` | **项目归档往返**（.mproj 打包/解包）、**目录与通配引用还原成同构目录**、**路径与 URL 一起重算**、缺失素材清空而非报错、**Windows 反斜杠相对路径**、模板存/列/实例化/删 | 14 |
| `tests/test_project_store.py` | **项目落盘存取往返**、**原子写（不留半截 JSON / 临时文件）**、**路径穿越被清洗**、坏文件不让列表 500、删项目（保留/连产物）、**HTTP 增删改查**、**任务 404 带明确文案**、**id 分配唯一**、**新建撞 id 报 409 不覆盖**、**乐观锁检出并发修改**、**导入到已存在 id 被拒** | 27 |
| `tests/test_models.py` | **只剩一个底模时接口自洽**：available / arch / recommend / caps.approx / refiner=false、未知底模报错 | 5 |
| `tests/test_postprocess.py` | 锚点、逐帧对齐、打包几何、色板上限、alpha 二值、像素块单色性、前景块计数 | 11 |
| `tests/test_pixelfy.py` | 网格检测、每格一色、共享色板、噪点清理、网格吸附 | 20 |
| `tests/test_sheet.py` | 精灵图网格自动检测、切分、前景掩膜 | 16 |
| `tests/test_pose.py` | 检测器切换、18 点完整性、骨架渲染、双后端一致性、单例 | 12 |
| `tests/test_refcond.py` | 轮廓条件、auto 降级、强制骨架报错、换皮 | 18 |
| `tests/test_refprep.py` | 参考图预处理、帧级评分 | 19 |
| `tests/test_quality.py` | 前景占比/质心/漂移/贴边、无模型时「无法判定」 | 11 |
| `tests/test_io_paths.py` | 视频抽帧、GIF、切片 CLI、非 8 倍数尺寸、环境自检、**测试目录不越界** | 12 |
| `tests/test_presets.py` | 契约拼接与优先级、**长提示词原样送达（一个字都不截）**、后缀保持短 | 8 |
| `tests/test_repaint.py` | 选区外逐点不变、色彩校正、色板吸附、RGBA 进 VAE 会炸 | 16 |
| `tests/test_stability.py` | 连续出图显存不涨、**free() 清空缓存**、跨进程同 seed（`MEOW_SLOW=1` 手动跑） | 5 |
| `tests/test_vramopt.py` | W4 候选峰值估算（768² 单帧/加一个候选、24G 卡只够 1 个、未知架构按 Qwen 算） | 4 |
| `tests/test_denoise.py` | `strip_fringe` 剥白边、`clean_isolated`、斜向轮廓不被吃 | 9 |
| `tests/test_anim.py` | GIF/APNG 帧数、Godot tres/tscn 导出 | 9 |
| `tests/test_video.py` | numpy/PIL 帧兼容、mp4 落地、Wan 真实生成（`RUN_WAN=1`） | 6 |
| `tests/test_review_p2.py` | 抠图后端选择、网格对齐、资产索引增量、**删条目不能误伤同名近似文件** | 10 |
| `tests/test_review_round2.py` | **LoRA 双槽位（临时挂载不顶掉用户的）**、**路径穿越（`assets2/` 不算 `assets/`）**、**meta 真实性（记的步数是实际执行的）**、重复字典键 | 12 |

合计 **351 条**（`pytest --collect-only` 实测；`-m slow` 的 `test_e2e_qwen`
10 条也在内，跑起来才真加载模型）。改测试后顺手核一遍这张表——
数字对不上说明有文件被整段删掉或重复定义覆盖了。

**SD 时代的 8 个测试模块已一起删除**（`test_clip` / `test_controlnet` /
`test_generate` / `test_lora` / `test_train` / `test_review_fixes` /
`test_review_p1` / `test_fetch`）：它们测的全是 LoRA 注入、ControlNet 条件、
clip_skip 换算、AnimateDiff 时序这些已经不存在的机制。

已知限制：骨架质量决定成败——实测 8 帧序列里有 2 帧只提取到 8/18 个关节（手臂与身体重叠），
这两帧生成时会自由发挥，出现朝向漂移。产线上应当在提取后按「可见关节数」筛帧；
W4 侧对应的兜底是上面那套**质量闸门**（生成后再检一遍，点数不够就补提）。

测试过程中修掉的两个真 bug：**inpaint 不传 width/height 时 diffusers 默认输出 512×512**（与源图不符，
贴回时 `images do not match`）；**composite 的羽化是双向扩散**（会把选区外像素一起吃掉），
改为先腐蚀再模糊。

## 目录

```
2d-game-meow/
├── app.py                  FastAPI 服务入口
├── workflows/
│   ├── paths.py            统一路径解析（资产根 / 模板根，可用环境变量覆盖）
│   ├── projectio.py        项目归档 .mproj：打包 / 解包 / 模板
│   ├── projectstore.py     项目落盘存储（assets/projects/<id>/project.json）
│   ├── engine.py           pipeline 懒加载、显存守卫、LoRA 热切换
│   ├── generate.py         w1/w2/w3/w4 四个工作流
│   ├── quality.py          帧质量评分（关节数 / 占比 / 贴边 / 漂移）——挑帧与闸门
│   ├── repaint.py          局部 AI 重绘的图像运算（裁剪 / 羽化 / 接缝校正 / 色板吸附）
│   ├── perfect_pixel.py    完美像素网格对齐（FFT 检测 / Sobel 吸附 / 每格单色采样）
│   ├── pixelfy.py          像素精修（共享调色板 / alpha 二值 / 噪点清理）
│   ├── clip.py             自实现 clip skip（上游参数在 transformers 5.x 下会崩）
│   └── postprocess.py      像素化、调色板量化、锚点对齐、精灵图打包
├── tools/
│   ├── fetch_weights.py    权重下载（走 hf-mirror）
│   ├── smoke.py            逐阶段测速/测显存
│   ├── make_pose.py        COCO-18 骨架图生成 / 渲染
│   ├── pose_from_image.py  从图片 / 视频 / 精灵图批量提骨骼
│   ├── video.py            文生视频 / 图生视频 CLI
│   ├── fetch_ranged.py     按 Range 分块下载（抗限流断流）
│   ├── bench.py            稳态基准
│   ├── ab_repaint.py       局部 AI 重绘 A/B（重绘幅度 / 羽化 / 色板吸附，同源单变量）
│   ├── ab_pixelfy.py       像素转换 A/B（网格检测 / 目标尺寸 / 色数，同源单变量）
│   ├── ab_quality.py       画质 A/B（采样档 / 后处理档 / 背景干净度，同 seed 单变量）
│   ├── ab_matte.py         抠图后端 A/B（色键 vs u2net：占比/白边/连通块/IoU）
│   ├── ab_base.py          基座 A/B（多权重 × 多 prompt × 多 seed：多主体/贴边/背景脏/块单色/关节）
│   ├── bench_sprite.py     精灵管线单位成本（预热+中位数；每"可用帧"耗时与显存峰值）
│   └── check_env.py        环境自检
├── models/                 权重（checkpoints / controlnet / ipadapter / loras / ...）
├── templates/              项目模板（*.mproj，跨项目共享的用户资源）
├── assets/                 项目资产：产物 + sidecar(.json) + index.jsonl
└── tests/_out/             测试与验证产物（不入库，仓库的 assets/ 保持干净）
```

## 环境（本机实测）

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24GB（驱动 621.01 / CUDA UMD 13.5） |
| CPU / RAM | i9-13900KF 24C32T / **31.8GB** |
| OS | Windows 11 26300 |
| Python | 3.13.14，torch 2.6.0+cu124（`cuda.is_available()=True`） |
| 注意力 | torch SDPA（Windows + py3.13 无 xformers 轮子） |
| 模型源 | `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 直连不可达） |

安装：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 任务失败时，节点不会卡在「执行中」

生成是异步的（提交 → 排队 → 出图 → 轮询）。中间任何一环出问题，
节点都**必须能自己退出「执行中」状态**，否则按钮一直禁用、进度条一直转，
用户只能刷新页面（刷新后节点还在，还是卡着）。

三道兜底：

| 情况 | 处理 |
|---|---|
| **提交请求挂住** | `fetch` 加 30 秒超时（`AbortController`）。没有超时的话，服务端不响应时请求会挂到天荒地老，节点永远「排队中…」 |
| **轮询查询失败** | 连续 3 次失败（约 2 秒）就判定失败，节点退出「执行中」并显示原因 |
| **任务消失 / 卡死** | 服务端返回 404（任务被淘汰或服务重启）→ 判定失败；超过 30 分钟没结束 → 判定超时 |

后端侧也做了兜底：worker 线程的整个执行过程包在 `try` 里，任何漏网的异常
都把任务标成 `failed`（以前任务会永远停在 `queued`，前端那边节点就一直排队）。

> 「执行中」判定只看 `job.status` 是 `queued` 还是 `running`，
> 失败（`failed`）与完成（`done`）都会让按钮恢复可点。

## 快速开始

```bash
# 1) 拉权重（魔搭，约 12.6GB：底座 4.0 + inpainting 2.2 + IP-Adapter 2.4 + ControlNet 4.0）
python tools/fetch_modelscope.py            # 全部
python tools/fetch_modelscope.py sd15       # 只拉某一段
python tools/fetch_modelscope.py --list     # 先看清单

# 2) 环境自检 + 分阶段实测（写 assets/_bench.json）
python tools/check_env.py
python tools/make_pose.py          # 生成 COCO-18 骨架帧（无 DWPose 时用）
python tools/smoke.py w1
python tools/smoke.py w3
python tools/smoke.py w4 --pose-file assets/_pose_walk_00.png --no-ipa

# 3) 起服务
python app.py            # http://127.0.0.1:8000/docs
```

提交任务：

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "workflow": "w1",
  "params": {"prompt": "pixel art cat knight, red cape, katana, front view",
             "width": 512, "height": 768, "steps": 25, "pixel": 4, "palette": 16}
}'
```

上传一张本地图（画布「拖本地图进画布」走的就是它；请求体是**原始字节**，不是 multipart）：

```bash
curl -X POST "http://127.0.0.1:8000/upload?name=hero.png" \
     --data-binary @hero.png -H "Content-Type: image/png"
# -> {"path": "D:/.../assets/uploads/ab12cd34ef_hero.png",
#     "url": "/files/uploads/ab12cd34ef_hero.png", "w": 512, "h": 512, "format": "PNG"}
```

真删（连 sidecar 与 raw 一起删，并把索引里的条目摘掉）：

```bash
curl -X DELETE http://127.0.0.1:8000/assets -H "Content-Type: application/json" \
     -d '{"paths": ["D:/.../assets/w1_20261028_090000_0000.png"]}'
# -> {"deleted": [...], "skipped": [...], "not_found": [...], "count": 3}
```

`assets/` 之外的路径会被跳过（`skipped`），`index.jsonl` 不可删。

拿到 `path` 就能当图生图的源：`{"workflow": "w2", "params": {"init_image": "<path>", ...}}`。
校验：体积 ≤20MB、真能解码（坏文件 400）、格式白名单（PNG/JPEG/WEBP/BMP/GIF）、
磁盘名只用 ASCII（中文名要 URL 编码，会坑掉某些代理）。

## 设计约束（都是踩过的坑）

### 前端（这两条是用户实测报上来的）

1. **卡头按钮上不能启动拖拽**。`setPointerCapture` 会把后续 pointerup 重定向到卡头，
   浏览器按「pointerdown 与 pointerup 的最近公共祖先」派发 click —— 于是 click 落到卡头，
   「切回参数视图」永远点不动。修法：`e.target.closest('button, input, textarea, ...')`
   命中就直接 return，不抓手。
2. **画布平移不能靠 `data-bg` 判断**。`.canvas-grid` 是 `pointer-events:none`，点空白时
   事件目标是 `.canvas-viewport` 自己（没有 `data-bg`），判据永远不成立 → 画布完全拖不动。
   改成反向排除：不在节点/面板/控件上就算可拖。
3. **`.block` / `.fold-body select` 的样式漏了**，深色底上 LoRA 下拉几乎看不见。
   现在 `.fold-body` 里的 select 与 number 统一有边框、`color-scheme: dark`、深色 option。


1. **像素感由后处理保证，不靠提示词**。模型只会画出「看起来像像素」的糊图；
   必须走「提示词契约 → 网格吸附 → 共享调色板」三道关。
2. **逐帧生成必须做锚点回拉**。模型每帧会自己重新构图，角色在画面里上下漂移；
   按首帧脚底锚点做**整体整数平移**拉回，只平移不缩放（缩放会抹掉下蹲/抬腿的真实大小变化）。
3. **跨帧一致性靠固定 seed + 固定 LoRA + 固定文本**，不是靠提示词描述得详细。
4. **局部重绘只改选区**：羽化蒙版**向内**收缩后贴回，选区外像素逐点不变。
5. **单卡串行**：一个 worker 线程 + FIFO 队列，不做并发，避免两套 pipeline 抢显存。

## 版本选型：为什么全栈 SD1.5

| 能力 | SD1.5 | SDXL |
|---|---|---|
| 文生图 / 图生图 | 可行 | 画质与 prompt 遵循更好 |
| 局部重绘 | **更优**（专用 inpaint 权重，专为 512 训练） | inpaint 专模弱于自身基座 |
| 姿态 → 精灵帧 | **更优**（lllyasviel OpenPose v1.1，业界默认件） | 社区二次训练，边缘姿态更飘 |
| 视频驱动 | **唯一选项**（MimicMotion 底座即 SD1.5） | 无 |
| 权重体积 / 单步耗时 | 2GB / 基准 | 7GB / 慢 3–4 倍 |
| 授权 | CreativeML OpenRAIL-M | Community License（年营收 >$1M 需付费） |

关键判断：**游戏精灵的交付分辨率是 512–768，不是 1024**。SDXL 出 1024 再缩到 128 只会把细节噪声一起缩进去。
SD1.5 在 512 上「一次到位 + 16 色量化」更贴产线，且被视频模型绑死（MimicMotion 无 SDXL 版）。

---

## 评审修复（2026-10-28）

完整评审见 `docs/REVIEW-2026-10-28.md`。这一轮修的是「不报错、只是静默变差」的那一类问题，
以及把省下来的显存换回吞吐。**每条都有回归测试**（`tests/test_review_fixes.py` /
`test_review_p1.py` / `test_review_p2.py`）。

### 修掉的确定性缺陷

| 问题 | 后果 | 现在 |
|---|---|---|
| `ensure_headroom` 判据写成「已用量 < 2GB」 | 每次 wss/wpc 都把**整个缓存清空**（cache 形同虚设），ControlNet 还不跟着释放 | 改为按**空闲显存**判、LRU 驱逐、`keep` 保护正在用的管线、孤儿 ControlNet 一起回收 |
| LoRA 状态记在 Engine（池级） | 「W1 用了像素 LoRA，接着跑 W4」→ 第二条管线**静默没加载** | 状态挂在 UNet 上；共享 UNet 时天然一致，不重复加载 |
| WPC / WSS 硬编码 `model="sd15"` | 切 SDXL 无效且无提示 | 参数透传；AnimateDiff + SDXL 显式报错 |
| WSS 默认路径把 preset 拼两遍 | 白吃 15~20 token 预算 | 只 build 一次 |
| 底模写死在代码里 | 换社区微调要改两个文件 | `models/checkpoints/<任意名>/` 自动登记，架构按 `_class_name` 判定 |

### 质量

| 改动 | 依据 |
|---|---|
| 提示词预算改用**真实 BPE 计数**（`presets.count`），并在超 77 时**分段编码**（`clip.encode_long`，77×2） | 实测 `WUXIA_NEG` 真实 **35 token**，而旧的 `rough_tokens` 估成 **113** → `fit()` 一直按高估值提前砍词。注释里"写到 113 尾部全被丢弃"的结论是错的，真问题是**没超也照砍** |
| 质量闸门改用**像素化之前**的图评分（`_postprocess` 暴露 `pre_pixel`） | 128 有效像素、32 色的产物上姿态模型本来就检不出几个点，好帧会被判废并触发无谓补提 |
| 主链路像素化接**网格对齐**（`pixelate(grid_align=True)`） | 只对"已经是像素画"的输入生效（≤512 色才检测；命中要求格数与请求值 ±30%）。**连续调 AI 图没有可检测网格，0.5ms 内回退 BOX**——原先"主链路也走网格对齐"的预期不成立，实测 768² 的 W1 产物与 6px 噪声图都检测不出 |
| 少步蒸馏（`distill="lcm"`）+ 高清修复（`hires=1.5`） | 蒸馏把 30 步压到 8 步，省下的预算换候选数；hires 是「512 出结构 → 放大补细节」，避开 SD1.5 ControlNet 的 512 训练分辨率问题。**两者都必须 A/B**：蒸馏会削弱 LoRA 风格保真。注意 hires **不省显存也不省时间**（见下） |
| 抠图优先用 BiRefNet（放了权重就自动切）、`alpha_mode="soft"` | 细长结构（剑/飘带）与半透明边；像素素材仍默认 binary |

### 显存 / 吞吐（实测 RTX 4090）

| 项 | 前 | 后 |
|---|---|---|
| txt2img + pose 同时驻留 | 7.36GB（两份 UNet） | **3.91GB**（共享 UNet） |
| 候选批量推理（512²，8 步） | 串行 2 张 1.23s | 一次 batch **0.70s（1.75×）**；n=4 为 1.40× |
| 连续工作流 | 每次清空缓存后重载 | 缓存真正生效；AnimateDiff 前主动清场（`drop_others`） |

批量推理**不是逐像素一致**（不同 batch 走不同 conv 内核，差 1~4/255），但同配置重跑完全一致，
跨进程同 seed 可复现的测试仍然通过。

### 服务层

- `/assets` 改走 SQLite 增量索引（`workflows/assetdb.py`）：只解析 jsonl 新增的字节，查询取一页；
  索引库不可用时自动退回全量解析。
- `_jobs` / `_trains` 加 LRU 上限（200 / 50，跳过运行中的任务）。
- 训练与生成共用一把 GPU 锁——「不进生成队列」不等于「可以并发占卡」。

### 高清修复（hires）：已随 SD 一起删除

`hires` / `hires_strength` / `hires_steps` 这套「先出结构、再低幅度放大重画」
是 SD 时代的参数，**代码与界面都已删除**（提交 `1831ade` / `e92775e`）。

Qwen 没有对应的两段式：它的图 token 数随边长走，1024² 是 768² 的 1.78 倍图
token（实测 45s vs 25s），所以「要更清晰」的做法是**直接出 1024²**，
而不是「768² 放大重画」。立绘档位已经默认这么做了（见 `genStyles.ts`）。

### SD1.5 能画出参考图那种全身像素角色吗？—— 分三项看，别混着说

拿一张"全身、纯色背景、带明暗层次的像素角色"当目标，实测（sd15 base v1.5，
人形 prompt knight / wuxia / mech × 2 seed）：

| 目标 | 现状 | 能不能靠参数救 |
|---|---|---|
| **单主体** | 768² 下多主体率 17% | 能：512² 可压到 8%，但见下 |
| **纯色背景** | 已基本达标（浅灰纯底、无杂物） | 不需要：脏背景由抠图兜住 |
| **全身不裁** | **贴边率 67%**（头被切 / 脚缺失） | **不能**：见下 |
| **像素观感** | 由后处理兜住（块单色率恒为 1.000） | 不需要：模型不用画像素 |

#### 被数据否掉的两个"看起来对"的办法

1. **"回到 512 原生分辨率"**。它确实把多主体率从 17% 压到 8%（SD1.5 在自己
   的训练分辨率上不会画成场景），但**人形 prompt 的贴边率变成 100%**——
   模型在更小的画布上把人放得更大。我一度只看了单张样本（恰好是完整的），
   以为找到了解法；扩大到 6 张就被推翻。**单张样本不能当结论。**
2. **"加构图词 / feet visible"**。`small figure in frame, ample empty space`
   两次测出 50% 与 67%（基线 67%），`feet visible, standing` 是 67% 且多主体
   升到 33%——**都在噪声内，没有证明有效**。（词已留在契约里，因为无害。）

#### 加了一个真能用的像素 LoRA：`pixel_sprites_32`

魔搭 `mushenL/2D-Pixel-Toolkit-2D` 的 Sprites_32 分支（36MB / rank 16，SD1.5），
已下载并登记。**必须带触发词 `pixelart`**，选中后点提示词旁的「+ 填入」：

| 配置 | 可交付率 | 多主体 | 贴边 |
|---|---|---|---|
| 不挂 LoRA | 17% | 17% | 67% |
| 挂它但**不加触发词** | 17% | 17% | 67% ← 完全没反应 |
| 挂它 + `pixelart` | **50%** | **0%** | 50% |

#### 「像素块」是**色块边长**，不是成品尺寸 —— 界面现在会显示有效像素

这个字段极易被理解成「我想要的成品像素尺寸」（填 64 = 想要 64px 的像素画），
但它的语义是「每个色块多少像素」，所以 `512 + 像素块 64` 得到的是 **8×8**——
画面当然只剩一团色。这类设置**不报错**，只是出图看着不对，而人会把锅甩给模型。

节点里现在直接算出并展示 `有效像素 = 边长 / 像素块`（如 768/6 → `128px`），
色块过疏（<64px）或色数过少（<8）就告警并写明该怎么填；`max` 挡不住手输，
`onChange` 里也夹一道。参考图那种密度约 **128px 有效像素 / 32 色**。

（人形 prompt knight/wuxia/mech × 2 seed，各 6 张。**样本小，幅度仅供参考**，
但"不加触发词就完全不起作用"这条是干净的对照。）

> 这条**修正了项目里一条旧结论**：此前记着"像素 LoRA 一律不加触发词"，
> 来源是 px_illustrious（它加触发词会让多主体率上升）。对这个 LoRA 正好相反。
> 所以 trigger 该不该填，必须每个 LoRA 单独测，不能一刀切。

#### 结论：别再调 SD1.5 了，用 SDXL@768

同一句提示词（knight in silver armor…）、同一套后处理，实测对比：

| 底模 | 全身 | 单角色 | 背景 | 可交付率(768²) |
|---|---|---|---|---|
| SD1.5 base | 头/脚常被切 | 多主体 17% | 需要抠图 | 33% |
| **SDXL 1.0** | **完整** | **1 个** | **干净** | **92%** |

SD1.5 上我先后试过降分辨率、加留白词、feet visible、换像素 LoRA——
只有加对 LoRA 触发词那次把可交付率从 17% 提到 50%，裁切始终下不来。
**继续调参没有意义**，这是 base v1.5 的构图能力边界。

所以：**默认底模改成 SDXL**（有 SDXL 权重时自动选中）。显存不够
（SDXL 出图约 11GB、W4 约 15GB）就在侧栏切回 SD1.5，那里也写了代价。

推荐配方（SDXL）：**768×768 / 24 步 / CFG 6.5 / 契约 pixel / 像素块 6（→128px）
/ 色数 64 / 勾抠图**。注意 SD1.5 的 LoRA（含 pixel_sprites_32）**不能**用于 SDXL，
像素感靠后处理兜住就够了。

像素化改成**块内中位数取色**（不是 BOX 平均）：平均会把背景色按比例混进边缘块，
轮廓外挂一圈灰边、看着像抗锯齿；中位数的边缘更硬、色块更纯。对比图见
`assets/_sample_cmp.png`（左=平均，右=中位数）。

色数 64 是实测选的：同一张 SDXL 产物按 16/32/64 量化后对比（见
`assets/_pal_cmp.png`），64 色最干净——披风与甲片的亮/中/暗层次分明、杂色最少；
16 色糊成一块，32 色局部有杂色拼接。SD1.5 画面本身更脏，仍推荐 32。

裁切是 base v1.5 的构图能力边界，提示词和分辨率都绕不过去。能动的只有：

- **换 SD1.5 社区微调基座**（同架构、同显存、同步数，只换权重）——机制已就绪，
  `models/checkpoints/<名字>/` 放进去即可选。这是唯一可能把 67% 压下来的单项。
- **候选 + 闸门**：单张可交付率 33%，出 3 张挑一张约 70%、5 张约 86%。
  换成更好的基座之前，这是唯一能保证产线跑通的机制。
- **换 SDXL**：贴边率 0%、可交付率 92%（768²），但 W4 显存要 15GB。

### 像素化与 Refiner：都由用户在界面上开关

**「像素化」是显式开关**（不是靠"像素块填 0"这种隐蔽方式关闭）：

- 关掉 = 普通插画/立绘；打开 = 像素化（默认块 6 → 128px 有效像素）
- 旁边实时显示 `有效像素 = 边长 / 像素块`，块过疏（<64px）或色数过少（<8）会告警
- 颜色数独立可调（关掉像素化时它就是"色调分层"，也常用）

**Refiner 精修已随 SD 一起删除。** 那是 SDXL 的第二段 UNet（把画面往写实质感拉），
Qwen 没有对应物，界面上的开关与四个参数都已移除（提交 `e92775e`）。
要「更清晰」，直接出更大尺寸（1024²）比两段式更省时间也更保真。

### 明确未做（以及为什么）

- **差分扩散 inpaint**（Fooocus 那条路）：要改每一步去噪的潜变量混合，收益（无接缝）必须靠
  人眼 A/B 判定，纯代码交付无法验证，留作 w5 的第二引擎单独做。
- **多图 IP-Adapter**：diffusers 要求"每张参考图一个 adapter"（实测报
  `must have same length as the number of IP Adapters`），要挂两份权重并与共享 UNet 的
  卸载/重挂机制协调，风险高于收益。

---

## 换基座：怎么跑、怎么判

画质的主因是**基座权重**，不是架构。原始 runwayml v1.5 在"骑士 + 武士刀 + 铠甲 + 红斗篷"
这类复合概念上会直接画成半身特写（跑一次 `tools/ab_base.py` 就能在
`assets/_ab_base_same/sheet_knight.png` 看到对照，产物不入库）。
换同架构的社区微调，显存与速度都不变，只换权重。

```bash
# 1) 放权重：带 model_index.json 的 diffusers 目录即可自动出现在下拉里
#    models/checkpoints/<任意名字>/...
# 2) 冒烟（2 张图，确认能跑）
python tools/ab_base.py --prompts knight,slime --seeds 1 --limit 1 --size 512
# 3) 正式对照：口径必须一致，否则尺寸会混进结论
python tools/ab_base.py --models sd15,mychar --size 768 --steps 20 --seeds 3
# 4) 只看已有产物、重建报告（改版式不用重跑）
python tools/ab_base.py --rebuild --out assets/_ab_base
```

输出 `report.md`（汇总表）、`summary.csv`（明细）、`sheet_<prompt>.png`（人眼打分用）。

**判读三个硬门槛**（其余都是参考）：

| 指标 | 门槛 | 说明 |
|---|---|---|
| 多主体率 | 必须 0% | >0 说明画面里挤了多个主体/道具，精灵管线持续出废片 |
| 贴边率 | 越低越好 | 脚/头被画布切断，对齐必然跳；>30% 基本不可用 |
| 背景脏率 | 参考 | 两个基座通常都很高——这是"必须抠图"的原因，不是选基座的依据 |

**别用这个工具比速度**：它是单次冷调用，同一配置两次跑能差 70%（实测 sd15 768²
分别是 1.33s 和 1.47s，SDXL 是 1.77s 和 3.07s）。要测速用 `python tools/bench.py`
（预热后取稳态）。所以"A/B 表里的耗时"只能看数量级，不能当结论——我先前据此写过
"SDXL 只慢 1.33 倍"，那个数字不成立。

---

## 像素路线是不是更简单？—— 判据更简单，达成更难

直觉是"像素化之后细节都被抹平了，选基座应该更省事"。前半句成立，后半句被数据否掉了。
四组对照全部 768²/20 步、6 prompt × 2 seed（`tools/ab_base.py`，可复现）：

| 条件 | 可交付率↑ | 多主体率↓ | 贴边率↓ | 块单色率 |
|---|---|---|---|---|
| sd15 单段 768² | 33% | 17% | 50% | 1.000 |
| sd15 + 像素 LoRA (0.75) | 17% | **42%** | 33% | 1.000 |
| sd15 + LoRA + 单主体负向词 | 25% | **8%** | **67%** | 1.000 |
| sd15 + 512² 出结构 → ×1.5 放大 | 17% | 0% | **83%** | 1.000 |
| **SDXL 单段 768²** | **92%** | 8% | **0%** | 1.000 |

可交付率 = 单主体 + 无贴边 + 前景占比 5%~60% 同时成立（精灵帧要进序列，三条都得满足）。

### 三条被数据逼出来的结论

**1. 细节确实被抹平了，所以判据可以简化。** 五个条件的块单色率都是 **1.000**——
后处理（网格对齐 + 量化）把"画得细不细"这件事完全吃掉了。所以不必再纠结"哪个基座画得好看"，
只看结构三项即可。

**2. 但结构三项互相拉扯，靠调参只能两头摆。**
- 加像素 LoRA → 多主体率从 17% 飙到 **42%**（它的训练集是角色图集，天生爱画多个小角色），
  但它同时是像素风格的唯一来源。
- 用负向词压多主体（42% → **8%**）→ 角色被画大 → 贴边率从 33% 飙到 **67%**。
- 降 LoRA 强度（0.75 → 0.5）→ 多主体 42%→25%，但贴边 33%→42%。

三个旋钮都在"多个主体"和"顶满画布"之间挪权重，可交付率始终在 17%~25%。
**根因是 SD1.5 base 的构图控制力**，不是参数没调对。

**3. 高清修复修不了构图。** 512² 出结构再放大到 768²，贴边率反而升到 **83%**——
因为第一段就把脚裁了，放大只是等比放大裁切。`hires` 补的是细节，不是构图。
（这条也说明 Q6 那个两段式在精灵管线上的收益有限，真正的用途是"结构已经对了、补边缘细节"。）

### 所以怎么选

**按"每帧要出几张"算成本，而不是按单张速度算。**

| 基座 | 单张可交付率 | 每帧候选数 | 期望可用率 | 每帧生成次数 |
|---|---|---|---|---|
| sd15 | 33% | 3 | ~70% | 3 |
| sd15 | 33% | 5 | ~86% | 5 |
| SDXL | 92% | 1 | 92% | 1 |

精灵管线的真实成本是"出一次可用帧要生成几次"，而 SD1.5 需要 3~5 次候选才能追平 SDXL 的 1 次。
**结论：像素路线不是更简单，是判据更硬、容错更低**——它同时要求单主体、全身不裁、留白合理，
而这恰恰是弱基座最做不到的三件事。

反过来说，这也解释了项目里"候选 + 闸门"为什么是核心机制而不是锦上添花：
在 SD1.5 上它把 33% 的可交付率乘成 70~86%，是产线能不能跑的分水岭。

**如果你就是要用 SD1.5**（授权、显存、LoRA 生态）：固定 `pixel=6/palette=32`、
`candidates≥3`、`gate=True`，并把 `character sheet, sprite sheet, multiple characters`
加进负向词——这是四组实验里唯一能把多主体率压到 8% 的配置，代价是必须靠候选数补回贴边损失。

---

## SDXL 在精灵管线上的真实成本（稳态实测）

上一节的"每帧要出几张"是估算。这里用 `tools/bench_sprite.py` 把它测成硬数字：
**预热 1 次 + 稳态跑 2~3 次取中位数**，显存同时记张量峰值与分配器保留峰值
（真实"占掉多少卡"看保留）。

> 为什么不用单次计时：同一配置两次跑能差 70%。本仓库已经因此翻过一次车
> （曾据单次数据写过"SDXL 同分辨率只慢 1.33 倍"，后已撤回）。要测速只能用这种口径。

### W1 文生图：每"可用帧"的成本

`p` = 单张可交付率（`ab_base` 实测）；**每可用帧 = 中位耗时 / (1-(1-p)^k)**

| 配置 | 生成 | 保留峰值 | p | 每可用帧 |
|---|---|---|---|---|
| sd15 768² ×1 | 1.26s | 4.55GB | 33% | **4.12s** |
| sd15 768² ×2 | 2.35s | 6.68GB | 33% | 4.44s |
| sd15 768² ×4 | 4.55s | 11.25GB | 33% | 5.82s |
| **sdxl 768² ×1** | 1.75s | 10.95GB | **92%** | **2.01s** |
| sdxl 768² ×2 | 3.16s | 13.93GB | 92% | 3.28s |
| sdxl 1024² ×1 | 3.43s | 14.13GB | **75%** | 4.57s |
| sdxl 1024² ×2 | 6.62s | 19.31GB | 75% | 7.37s |

三件事被数据推翻/确认：

1. **SDXL@768 单候选 2.01s，是 SD1.5 最优（4.12s）的 2 倍快。**
   单张只慢 39%（1.75 vs 1.26），真正拉开差距的是可交付率 92% vs 33%——
   **不需要靠候选堆数量**。
2. **SD1.5 加批量反而更亏**（4.12 → 4.44 → 5.82）。768² 下批量几乎没有并行红利
   （x2 用 2.35s，而串行两次是 2.52s，只省 7%），但 P(可用) 是边际递减的，
   所以成本反升。批量在这条线上不是"免费加速"。
3. **SDXL@1024 是陷阱。** 训练分辨率是 1024，但精灵场景下 1024² 让角色占比更大，
   可交付率从 92% 掉到 **75%**，加上生成慢一倍 → 4.57s/帧，比 768² 贵 2.3 倍。
   **精灵管线应该用 SDXL@768，不是 1024。**

### W4（姿态驱动 + IP-Adapter）：显存才是硬约束

| 配置 | 单帧 | 批量 2 | 批量 3 |
|---|---|---|---|
| sd15 768² | 6.62GB / 1.79s | — | — |
| **sdxl 768²** | **15.33GB** / 2.56s | 18.16GB / 4.55s | 21.76GB / 6.78s |
| sdxl 1024² | 18.45GB / 5.00s | **23.98GB / 17.47s** ⚠ | — |

**最危险的一行是最后那个 23.98GB**：它**不报 OOM**，但耗时从 5.0s 涨到 17.5s（3.5 倍）。
显存打满时分配器开始反复腾挪，性能崩掉却不报错——这比 OOM 难查得多。
所以 SDXL@1024 在 W4 上只能单帧串行；@768 可以批量 2，批量 3 已无余量。

### 后处理不是瓶颈

纯 CPU 的抠图 → 像素化 → 量化 → 清洗，在 768² 上 **0.1s**（连续调输入）。
之前我报过 0.8s——那是拿**已像素化的成品**当输入，命中了网格对齐检测（refine_one），
而产线真实输入色数成千上万，0.5ms 内就回退了。这个测量偏差已修正。

### 这些结论已经进代码

- `Engine.recommend(model)` 给出每个底模在精灵管线上的推荐参数，`GET /models`
  的 `recommend` 字段带出来（SDXL 推荐 768×768 / 24 步，不是 1024）。
  **前端已消费**：切换底模时自动套到「用户没改过」的生成节点上（判据是当前值
  仍等于旧底模的推荐值），手改过的节点保持原样；新建节点也按当前底模初始化。
  Sidebar 会显示「推荐 768×768 / 24 步」，切换时给一次 toast 说明。
- `Engine.max_candidates(model, size)` 把上限一路送到界面：`/models` 的 recommend
  带 `max_candidates`（Qwen@768 是 1，因为 768² 批量 2 要 23.5GB、会挤到 cpu offload
  上，耗时翻倍且不报错），前端拿它当候选输入框的 `max`；切换底模时已设的值会跟着
  夹到新上限。**超上限不会报错，只会静默变慢**，所以不能靠用户自己记。
  估算公式在 `vramopt.estimate_pose_gb`（本模块只剩这一个函数）。
- meta 里的 `candidates_note` 目前**恒为 None**：候选数在界面层就被 `max_candidates`
  夹住了，运行时不会再降级。字段留着是因为前端有 `candidatesNote` 的渲染分支
  （`⚠ 候选已降级`），将来若要接"运行时按空闲显存降档"可以直接写这里。
  SD 时代那条 `vramopt.plan_candidates()`（按空闲显存封顶候选数）**从未接线**，
  2026-10-31 已随 SD 管线一起删除——它的职责由上面这个上限 + `_infer` 的 OOM 重试
  分担，不要再按旧文档去找它。
- 序列帧结果区显示 `候选 N/帧 · 补提 X 次 · 闸门 M/K`，候选被降级时给出原因。
  图片视图（看完图最常见的状态）也显示，只是缩成一行 `⚠ 候选已降级`——
  只在展开视图里说，用户就不会展开。

### 结论

- **要精灵产线的单位成本最低：SDXL@768 + 单候选**（2.01s/可用帧，10.95GB）。
- **要继续用 SD1.5**：单候选 + 闸门（4.12s/可用帧），不要指望批量加速；
  显存只有 4.55GB，可以同时跑别的东西。
- **不要用 SDXL@1024 做精灵**：除非你要的是 1024 的立绘/宣传图，不是 128px 的帧。