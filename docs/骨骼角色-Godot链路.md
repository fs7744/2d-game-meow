# 骨骼角色 → Godot 链路（切件 / 建骨 / 动作 / 导出）

> 状态：**链路已打通并在 Godot 4.7.2 真机跑通**（2026-11-03）。
> 短板不在链路，在**几何切件的画质**——见 §4 的实测截图与边界说明，先读那节再决定能不能用。
> 给接手的 agent：本文所有「坑」都是真机验出来的，不是推测；每条都写了怎么复现。
>
> **本文是使用手册**（怎么跑、字段是什么、怎么改）。
> 想看重做一遍的选型依据、每个坑是怎么被发现的、完整复现步骤，见
> `docs/骨骼角色-Godot-完整实现记录.md`。

## 1. 这条链路解决什么

逐帧方案（`workflows/godotexport.py`，已有）把每一帧烘成 PNG。加一个动作 = 重渲一整套帧。
骨骼方案把「部件 + 每个部件每帧的变换」交给引擎，加一个动作只多几 KB 关键帧。

实测（`合子/sprite_pose_0001.png`，486×673）：

| 方案 | 25 帧走路 + 待机 | 加第 3 个动作 |
|---|---|---|
| 逐帧 PNG | ≈ **7386 KB**（25 张 × 295KB） | +295 KB/帧 × 帧数 |
| 骨骼 cutout | **321 KB**（部件 307 + 动画与场景 14.5） | **+2.9 KB** |

**约 23 倍**，且动作越多差距越大。代价是画质：部件是刚性旋转的，膝肘不会弯，
衣服不会跟着变形（§4 有实测图）。

## 2. 四段链路

```
角色图 ──①切件──▶ 6 个部件 ──②建骨──▶ skeleton/binding ──③动作──▶ motion.json
                                                                      │
                                                              ④导出 ▼
                                            Godot: Rig(Node2D) + 6×Sprite2D + AnimationPlayer
```

| 段 | 落点 | 关键函数 |
|---|---|---|
| ① 切件 | `workflows/rigbuild.py` | `split_image()` / `plan_regions()` |
| ② 建骨 | `workflows/rigbuild.py` | `build_rig()` |
| ③ 动作 | `workflows/rigbuild.py` | `build_action()` |
| ④ 导出 | `workflows/godotrig.py` | `export_bundle()` |
| 数据模型 | `workflows/rigmodel.py` | 校验 / 采样 / 世界变换 |
| 一键跑 | `tools/rig_godot_demo.py` | `--verify` 真机验，`--shots` 出截图 |

```bash
.venv/Scripts/python.exe tools/rig_godot_demo.py 角色.png --verify --shots
```

## 3. 数据模型：从 FrameBaker 移植了什么、砍了什么

源：`D:\code\gen\others\FrameBaker-main`（MIT），`packages/shared/src/animation.ts`。

| FrameBaker | 这里（`rigmodel.py`） | 理由 |
|---|---|---|
| `Transform{translation:Vec3, rotation:Quaternion, scale:Vec3}` | `{x, y, rotation}` | cutout 只有平面旋转 |
| 16 元素列优先矩阵 | 不存 | 求值交给引擎，预览直接合成 |
| `warp` / `deform` 轨道 | **不做** | 网格形变要蒙皮权重，是另一条技术线（§9） |
| `Vec3` 关键帧 | `float` / `[x, y]` | 同上 |

**保留的四条语义**（这几条才是 FrameBaker 值钱的地方）：

1. `schemaVersion` + `kind` + **未知字段直接报错**——不做静默兼容，否则版本漂移会安静导出错东西。
2. 骨用 `id` + `parentId` 表达层级，`rest` 是**相对父骨**的变换。
3. 轨道用 `targetId` + `property`，和 Godot `NodePath("骨:属性")` **一一对应**——导出是纯翻译，不做映射表。
4. 插值 `step` / `linear` / `cubic-bezier` → Godot `interp` 0/1/2。

坐标系刻意和 Godot 完全一致：x 向右、**y 向下**、旋转**顺时针**为正、单位弧度。
（中间只有 `rigbuild._rad()` 一处把「设计参数的逆时针度数」翻成 Godot 语义，
全链路其它地方不要再翻第二次。）

## 4. 切件：几何切件能做到什么程度（**先看这节**）

`split_image()` 按人形比例把整图切成六个矩形区域，相邻区域外扩 `overlap` 像素
防止露缝，再裁掉四周透明边。

### 4.1 能用的情况

四肢分明、贴身衣物（简笔小人、紧身衣、像素小人）——切出来的部件基本就是肢体本身。

### 4.2 不能用的真实情况

大裙摆 / 长发 / 飘带 / 宽袖的角色，**「腿部切区」切的其实是裙子**。转动时裙子裂开、
露出一块背景，观感直接破功。实测（`docs/img/rig-godot-frames.png`，Godot 真渲染）：

- 走路 4 帧：裙摆分成两块互相穿插，腿从裙子里"切"出来
- 手臂转动：腋下露黑缝
- 头部区域横跨全宽（486px），因为长发比肩宽

这是**几何切件的固有性质**，调参数救不回来：切区是矩形，而衣服结构不是。
想提升只能换切件方式（§9 待办 1）。

### 4.3 比例参数的默认值

`DEFAULT_PROPORTIONS`：`headRatio 0.26` / `shoulderDrop 0.035` / `hipRatio 0.55` /
`torsoWidth 0.46` / `shoulderHalf 0.23` / `hipHalf 0.10` / `armReach 0.10`。

**Q 版角色必须调 `headRatio`**（3 头身约 0.34）。切歪了的表现是「头被切掉半个」
或「脖子部位多出一块躯干」，`--head-ratio` 直接可调。

## 5. 建骨：枢轴是三个坑的总和

**枢轴 = 该部件的旋转中心 = 对应骨的世界位置。** 贴图的 `offset` 把「图内关节位置」
挪到节点原点，一条公式吃掉 overlap 和裁边的全部位移：

```
offset = 图尺寸/2 - (关节在世界坐标 - 切区左上角)
```

三个踩出来的坑：

| 坑 | 症状 | 正解 |
|---|---|---|
| 枢轴取**切区边** | 头/躯干枢轴差 `2×overlap`，人偶「缩脖子」（单帧看不出来，一播就明显） | 用解剖位置（颈线/肩线/髋线） |
| 枢轴取**切区中心** | 臂区从角色边缘算起，中心落在手臂中段 → 整条臂「绕肘甩」 | 用解剖横向位置（`shoulderHalf` / `hipHalf`） |
| 根部放在 `(cx, foot_y)` | 实例化后偏移半个身位 | 根骨恒在 **(0,0)**，位置由游戏 `set_position(地面线)` 决定 |

骨骼层级（7 根）：

```
root (0,0)
├─ torso (颈)      ── head / arm_l / arm_r
└─ leg_l / leg_r   (髋)
```

腿挂 `root` 而不是 `torso`：躯干一旦俯仰，腿不该跟着歪。代价是腿的
`position` 要自己带 bob（`build_action` 里已经这么做了）。

## 6. 动作：相位、采样、符号

`build_action(kind, frames, samples, fps, ...)`：

- `frames`（相位密度）只决定**一圈多长**：`duration = frames / fps`，默认 8/8 = 1 秒。
- `samples`（关键帧数）决定**平滑度**，默认 24。这就是骨骼方案的核心优势：
  **加密几乎不增包体**（25 个 float vs 25 张 PNG），8 帧的阶梯感可以被抹平。
- 相位必须 `ph = 2π·i/samples`。写成 `i/frames` 时 24 个采样会走 **3 圈**，
  播放起来是"抽搐式快进"，且 t=0 与 t=0.5 姿态重复（这个 bug 真出过）。
- 末帧复制首帧，循环才严丝合缝（`loop_mode = 1`）。
- 符号：设计参数是**逆时针为正的度数**（和 PIL 一致），写进轨道前
  `_rad()` 取负转成 Godot 的顺时针弧度。

轨道数（walk 9 条 / idle 3 条）：

```
torso.position, leg_l.position, leg_r.position,
leg_l.rotation, leg_r.rotation, leg_l.scale, leg_r.scale,
arm_l.rotation, arm_r.rotation
```

层级化把 rigwalk 时代的 18 条压到 9 条——躯干的 bob 会带着头/臂走，不用各自写。

## 7. 导出：三个致命细节（都是真机抓的）

`workflows/godotrig.py` → `Rig`(Node2D) → 骨节点 → Sprite2D → AnimationPlayer。

### 7.1 轨道路径必须是**完整路径**

```
path = NodePath("torso/arm_l:rotation")   ✓
path = NodePath("arm_l:rotation")         ✗
```

只写骨名时，`torso` 恰好在根下所以能命中，`arm_l` 在 `torso` 下就解析失败。
**Godot 只打一条 `couldn't resolve track` 警告，播放时那条轨道就是不动**，
编辑器里看起来"动画能播"，只有手臂不摆。

### 7.2 float 属性的关键帧必须写成 `0.0`，不能是 `0`

```
"values": [0, -0.160437, ...]      ✗ rotation 一动不动
"values": [0.0, -0.160437, ...]    ✓
```

裸整数解析成 `int`，`AnimationMixer` 不会把它写进 float 属性，**零警告**。
`Vector2` 的分量是 int 反而没事（构造器会转），但统一带小数点不会错。
`godotrig._f()` 负责这件事，`tests/test_rig.py` 有一条断言直接扫 `values` 里的裸数字。

### 7.3 根骨不建节点

场景根 `Rig` 就是根骨。多挂一个空 `Node2D` 会让所有轨道路径多一层。

### 7.4 offset 只在一处算

`build_rig` 算好 `slot.offset`，`build_scene` **直接用**，不再按 `pivot` 加 `±h/2`。
两套一叠加，部件会整体偏半个身高。

## 8. 验证

```bash
# 纯计算测试（0.5 秒，不加载模型、不读真实资产）
.venv/Scripts/python.exe -m pytest tests/test_rig.py -q

# 真机：建工程 → godot --import → 跑探针 → 可选截图
.venv/Scripts/python.exe tools/rig_godot_demo.py 角色.png --verify --shots
```

真机探针（`tools/rig_godot_demo.py` 里的 `PROBE_GD`）会输出：

```
RIG_ANIMS ["idle", "walk"]
RIG_NODES 12
RIG_ANIM walk len=1.000 tracks=9 moved_steps=2
```

`moved_steps` 是「4 段采样里有多少段与起点不同」；走路因为正弦过零会回到中点，
**2 是正常值**，0 才是坏（说明轨道没生效）。

必需的前置：`godot --headless --path <proj> --import`。
不导入 PNG 的话 `load("res://...png")` 直接失败（`.import` 是编辑器生成的）。

Godot 位置（本机）：`D:\GAMES_tool\Godot_v4.7.2-stable_mono_win64\..._console.exe`。
**用 console 版**，普通版不往 stdout 打日志。

## 9. 待办（按优先级）

1. **换切件来源**（决定画质上限，最值钱的一项）。当前几何切件只适合四肢分明的角色。
   可行路线：真拆件（See-Through，Apache-2.0，要 CUDA）/ 分层重绘 / 人工 PSD。
   **接口已经留好**：`build_rig(parts, joints, boxes)` 只认「图 + 枢轴 + 切区」，
   换来源不用改建骨和导出。
2. **关节补重叠**：在部件图上按关节画一圈同色，转动时不会露背景。纯图像处理，
   不需要新模型，但只能缓解、不能根治（衣服结构仍不跟骨骼）。
3. **接入 API + 前端节点**：现在只有命令行。要让用户在画布里点一下出包，
   需要 `/projects/{scope}/rig/*` 端点 + 一个节点（参照 `RigWalkNode.tsx` 的形态）。
4. **多朝向**：`Z_ORDER` 已有 front/back/side 三套层级，但没有对应的部件图。
   侧视要么另导一套拆件，要么用 `scale.x = -1` 翻转（只能做镜像）。
5. **动作库**：现在只有程序生成的 walk / idle。FrameBaker 的 `builtinMotions.ts`
   有现成的动作数据，可以转成 `motion-clip` 格式（字段几乎同构）。
6. **网格形变**（完全不同的一条路）：单图 + `Polygon2D` + `Skeleton2D` + 权重，
   能解决"衣服不跟骨骼"的问题，但权重自动生成没有可靠方案（FrameBaker 的
   warp/deform 也是手工调）。

## 10. 文件速查

| 关注点 | 位置 |
|---|---|
| 数据格式与校验 | `workflows/rigmodel.py`（`PROPERTIES` / `validate_*`） |
| 切件比例 | `workflows/rigbuild.py` `DEFAULT_PROPORTIONS` |
| 枢轴规则 | `workflows/rigbuild.py` `plan_regions()` 注释 |
| 贴图 offset | `workflows/rigbuild.py` `build_rig()` 里 `off_x/off_y` |
| 动作相位与符号 | `workflows/rigbuild.py` `build_action()` |
| Godot 文本格式 | `workflows/godotrig.py` `build_animation()` / `build_scene()` |
| 数字格式化（`0.0` 的坑） | `workflows/godotrig.py` `_f()` |
| 真机探针 | `tools/rig_godot_demo.py` `PROBE_GD` |
| 测试 | `tests/test_rig.py`（16 条，0.5 秒） |
| 实测截图 | `docs/img/rig-godot-frames.png`（4 帧走路）、`docs/img/rig-godot-zoom.png`（接缝特写） |

## 11. 与逐帧方案怎么选

| | 逐帧（`godotexport.py`） | 骨骼（本文） |
|---|---|---|
| 画质 | 模型画成什么样就是什么样 | 部件刚性旋转，衣服不跟 |
| 包体 | 帧数 × 单帧 | 部件 + KB 级动画 |
| 加动作 | 重渲整套 | +3 KB |
| 适合 | 出片质量优先、动作固定的角色 | 要换动作、包体敏感的批量角色 |

**两条并存，不要互相替代。** 都从「成片集 / 部件」出，导出格式一样是 Godot 工程。
