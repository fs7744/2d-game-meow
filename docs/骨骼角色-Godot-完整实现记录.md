# 骨骼角色 → Godot：完整实现记录

> 记录时间：2026-11-03　提交：`ca85092`（链路）、`05eea34`（前序脚底对齐）
>
> **这份文档和 `docs/骨骼角色-Godot链路.md` 的分工**：
> 那份是**使用手册**（怎么跑、怎么改、字段是什么）；
> 这份是**实现记录**（为什么这么做、每一步怎么推出来的、错在哪、怎么发现的）。
> 想复现或想重做一遍选型，看这份；只想用，看那份。

---

## 0. 一页速览

| 项 | 结果 |
|---|---|
| 走过的一段 | 角色图 → 6 部件 → 骨架/绑定 → 动作关键帧 → Godot 工程 |
| 代码量 | 模型 564 + 建骨 436 + 导出 592 + 工具 249 + 测试 303 行 |
| 包体（实测） | 25 帧走路 + 待机：逐帧 **7386 KB** ／ 骨骼 **321 KB** |
| 加动作成本 | 逐帧 +295 KB×帧数 ／ 骨骼 **+2.9 KB** |
| 验证方式 | Godot **4.7.2 真机**（headless 探针 + opengl3 渲染截图），不是文本自检 |
| 测试 | 16 条 / 0.5 秒，纯计算，不加载模型、不读真实资产 |
| 画质 | 四肢分明的角色可用；**大裙摆/长发角色会裂**（§10，有截图） |
| 未做 | API 端点、前端节点、多朝向、动作库、真拆件 |

一句话：**链路和格式全对了，卡在切件质量**。

---

## 1. 需求是怎么走到这一步的

这一段不是流水账，是为了说明**为什么最后是「Godot + 移植 FrameBaker」这个组合**——
中间否掉的每条路都有具体原因，重走一遍会浪费同样多时间。

### 1.1 起点：给 3D 人偶摆姿势，效果不好

用户拿一张瑜伽/舞蹈照片试「从图片提姿势」，人偶崩成麻花。查下来是两个独立问题：

1. `COCO_TO_OP` 映射表把 `13` 写成 `R_ANK`（应为 `L_ANK`）→ 左右脚踝取同一个点。
   直立站姿看不出来，抬腿姿势就叠成一点。**修了**（纯常量双射测试钉住）。
2. 官方 `convertOpenPoseToPose` 只有 2D 点，靠"髋线在画面里的角度"反推骨盆倾斜。
   侧身照片两髋投影只隔身高 1.5% → 判出 −81° → 人偶绕 Z 轴转倒。
   **接了 MediaPipe 3D 点**（`pose_world_landmarks`，9.4 MB，Apache-2.0）：
   算躯干 yaw → 绕 Y 转正 → 等比投影；前端把官方反推的 `modelRotation` 归零。

修完"不再躺倒"，但**肢体交叠（手抓脚）处 3D 精度不够**，仍需人工微调。

### 1.2 转向：与其估深度，不如直接读真实 3D

用户问「用 glTF / FBX 里的动画来做 3D 人偶姿势图生成呢」——**这是正解**：
动画里的 3D 是文件自带的，不存在深度歧义。

实现了节点上的「导入动作」（glb/gltf/fbx + 滑块选帧）。
坑在 `docs/人偶动作导入-glTF-FBX-调研与实现.md`，这里不重复。

### 1.3 目标升级：要"角色带骨骼、可换动作、包体小"

这时需求从"摆一个姿势"变成"**交付能在游戏里跑的角色**"。调研文档给的行业地图
是 Spine / Live2D 体系（拆件 → 对位 → 建骨）。已有的两个项目：

| 项目 | 有什么 | 缺什么 |
|---|---|---|
| `D:\code\pixelforge` | 拆件（6 部件）、像素化、`rigwalk` 算出每部件 `(关节, 角度)` | **算完就烘成 PNG 丢掉了**，没有骨骼数据 |
| `D:\code\gen\others\FrameBaker-main` | 完整骨骼/关键帧编辑、`.fbanim` 打包 | 自家格式，**没有引擎运行时** |

所以三个项目其实是一条链的前半段，缺最后一节：**引擎侧**。

### 1.4 格式之争：DragonBones 还是 Godot 原生

用户问「DragonBonesJS 不是现成的吗」。查完的结论：

- 它**没死**（2025-05 还在加 Pixi 8 支持，MIT）；
- 但**没有 Godot 官方 runtime**，官方 README 自己推荐用户转去 LoongBones（订阅制、纯 Web）；
- 格式没有正式规范，只能读 runtime 源码反推。

所以：目标引擎是 Godot 的话，用 DragonBones 是**绕路**——
还得自己写一个 Godot runtime 去读它的 JSON，等于把"半天能出第一帧"变成"写完 runtime 才能验证"。

**最终选 Godot 原生 cutout**：`Node2D` + 每部件一个 `Sprite2D` + `AnimationPlayer`。
零许可风险，格式是纯文本，Godot 自己就是规范。

### 1.5 项目的选定

原本想在 pixelforge 做（有 `rigwalk.py` 和 `godot2d.py` 现成）。
用户明确要求**在 `2d-game-meow` 做**。于是切割成：
`2d-game-meow` 已经有逐帧版 Godot 导出（`workflows/godotexport.py`），
这次补的是**骨骼那条线**，两条并存。

---

## 2. 最终方案与三处关键取舍

```
角色图 ─①切件─▶ 6 部件 ─②建骨─▶ skeleton.json + binding.json
                              ─③动作─▶ actions/*.json
                              ─④导出─▶ Godot 工程（tscn + tres + parts/）
```

### 取舍 1：移植 FrameBaker 的数据模型，但只移植 2D 需要的

`packages/shared/src/animation.ts`（1089 行）是 3D 风格的：`Vec3` 平移、
**四元数旋转**、16 元素列优先矩阵、`warp`/`deform` 轨道。

cutout 只有平面旋转，四元数在 2D 里退化成一个角度。所以：

| FrameBaker | 这里 | 理由 |
|---|---|---|
| `Transform{translation:Vec3, rotation:Quaternion, scale:Vec3}` | `{x, y, rotation}` | 2D 只需一个角 |
| 16 元素矩阵 | 不存 | 求值交给 Godot |
| `warp` / `deform` 轨道 | **不做** | 要蒙皮权重，是另一条技术线 |
| `Vec3` 关键帧 | `float` / `[x, y]` | 同上 |

**但保留了四条语义**（这才是它值钱的地方，别自己另发明一套）：

1. `schemaVersion` + `kind` 显式标在文件头上，**未知字段直接报错**，不做静默兼容。
2. 骨用 `id` + `parentId` 表达层级，`rest` 是**相对父骨**的变换。
3. 轨道用 `targetId` + `property`，和 Godot 的 `NodePath("骨:属性")` **一一对应**——
   导出是纯翻译，不做映射表（映射表是 bug 温床）。
4. 插值 `step` / `linear` / `cubic-bezier` → Godot `interp` 0/1/2。

### 取舍 2：几何切件，而不是先上 See-Through

真拆件（See-Through）要 CUDA + 专门权重，本项目产线只有 Qwen。
几何切件是**可用性兜底**：按人形比例切六个矩形区域。

这个取舍的代价在 §10 说清楚了。**接口按真拆件产物设计的**：
`build_rig(parts, joints, boxes)` 只认「图 + 枢轴 + 切区」，换来源不用改建骨和导出。

### 取舍 3：轨道存绝对值，不存相对 rest 的增量

Godot 的 `AnimationPlayer` 写的就是 `position` / `rotation` **绝对值**。
存增量的话导出时得把 rest 加回去，而「rest 改了、动画忘了同步」这种错误
没人能一眼看出来。存绝对值，导出是零转换。

---

## 3. 数据模型（`workflows/rigmodel.py`）

### 3.1 三份 JSON

**`skeleton.json`** —— 骨架。`bones[]` 里每根：

```json
{
  "id": "torso",
  "name": "躯干",
  "parentId": "root",
  "rest": { "x": 0.0, "y": -498.02, "rotation": 0.0 }
}
```

`rest` 是**相对父骨**的变换。根骨恒为 `(0,0)`（理由见 §5.4）。

**`binding.json`** —— 绑定。`slots[]` 把一个贴图挂到一根骨上：

```json
{
  "id": "torso", "boneId": "torso",
  "texture": "parts/torso.png",
  "pivot": "top",
  "offset": [0.0, 97.52],
  "z": 3
}
```

- `pivot`：语义记录（top / bottom / center），说明旋转中心在图的哪条边；
- `offset`：**真正干活的**，见 §5.2；
- `z`：绘制层级，`Z_ORDER` 给了 front/back/side 三套。

**`actions/<名>.json`** —— 动作：

```json
{
  "schemaVersion": 1, "kind": "motion-clip",
  "name": "walk", "fps": 8.0, "loop": true, "duration": 1.0,
  "tracks": [
    { "targetId": "arm_l", "property": "rotation", "interpolation": "linear",
      "keyframes": [{ "time": 0.0, "value": 0.0 }, ...] }
  ]
}
```

可用属性（`rigmodel.PROPERTIES`）只有四个：`position`（2 个值）、`rotation`（1）、
`scale`（2）、`visible`（1）。加属性前先确认 Godot 有同名属性，否则是死轨。

### 3.2 坐标系：和 Godot 完全一致

x 向右、**y 向下**、旋转**顺时针**为正、单位弧度。

不要改成数学坐标（y 向上）。那会在切件、动作生成、导出三处各写一次取反；
本项目在姿态图那条线已经因为 y 轴方向返工过。

**全链路只有一处翻符号**：`rigbuild._rad()` 把「设计参数里的逆时针度数」
（和 PIL 的 `rotate` 一致）翻成 Godot 语义。其它地方一律用 Godot 语义。

### 3.3 求值：为预览和验证服务

`local_transforms()` 把 rest + 轨道覆盖合成局部变换；`world_transforms()` 按拓扑序
累积父骨旋转。这两条**不参与导出**（Godot 自己算），作用是：
在 Python 里就能验证动作对不对，不用每次都开引擎。

采样器 `sample_track()` 用二分找区间（关键帧可能上千），支持三种插值。
`cubic-bezier` 用二分反解 `x(t)`——控制点限制在 `[0,1]`，保证时间单调。

---

## 4. 切件（`rigbuild.split_image` / `plan_regions`）

### 4.1 找主体包围盒

有 alpha 就用 alpha（>8 视为不透明）；没有（白底不透明图）就把亮度 >240 当背景。
不用 `point().getbbox()` 直接判定——它按非零像素算，必须先反相。

### 4.2 六条解剖线

```
y0 ──────────── 头顶
     headRatio  (默认 0.26)
neck ─────────── 颈线
     shoulderDrop (0.035)
shoulder ─────── 肩线
     hipRatio - headRatio
hip ──────────── 髋线
y1 ──────────── 脚底
```

横向：

- `torsoWidth`(0.46) → 躯干宽 = 全身宽 × 0.46
- `shoulderHalf`(0.23) → 肩关节在 `cx ± 0.23W`（≈ 躯干边缘）
- `hipHalf`(0.10) → 髋关节比肩窄
- `armReach`(0.10) → 手臂区向下延伸到髋线以下多少

### 4.3 六个切区

| 部件 | x 范围 | y 范围 |
|---|---|---|
| head | 全宽 | `y0` ~ `neck+ov` |
| torso | `cx ± 半躯干宽` | `neck-ov` ~ `hip+ov` |
| arm_l | `x0` ~ 躯干左边缘 | `shoulder-ov` ~ `arm_bottom` |
| arm_r | 躯干右边缘 ~ `x1` | 同上 |
| leg_l | `x0` ~ `cx` | `hip-ov` ~ `y1` |
| leg_r | `cx` ~ `x1` | 同上 |

`ov` = `overlap`（默认 3px）：相邻区域重叠几像素，转动时不露缝。

**切完再裁掉四周透明边**（`getbbox`），因为头这种"横跨全宽"的切区可能 486×178
而实际头发只占中间 300px。裁完要把 `box` 一起更新，否则贴图偏移算错。

### 4.4 参数校验（提前拦住"切出空部件"）

`plan_regions` 里有一条硬检查：`neck < shoulder < hip < y1`。
写成反的就直接抛 `RigBuildError`，而不是切出一堆空图让用户猜。

`split_image` 还会统计空部件，**≥3 个空就报错**：
`"这张图多半不是单个正面/侧面的全身角色图"`。侧面图里看不到另一只臂是正常的，
但一半以上为空说明输入就不对。

---

## 5. 建骨（`rigbuild.build_rig`）

### 5.1 骨骼树

```
root (0,0)  ← 脚底中心，场景根就是它
├─ torso (颈)
│   ├─ head   (与躯干同点，绕颈转)
│   ├─ arm_l  (肩)
│   └─ arm_r  (肩)
└─ leg_l / leg_r (髋)
```

**腿挂 `root` 而不是 `torso`**：躯干一旦俯仰，腿不该跟着歪。
代价是腿的 `position` 轨道要自己带 bob（§6.2 有推导）。

### 5.2 贴图偏移：一条公式吃掉所有位移

Sprite2D 默认 `centered = true`（贴图中心在节点原点）。我们要让**解剖关节**
落在原点，所以：

```
offset_x = 图宽/2 - (关节世界 x - 切区左上 x)
offset_y = 图高/2 - (关节世界 y - 切区左上 y)
```

这一条同时吸收了 `overlap` 外扩和裁透明边带来的位移。早期分成"按 pivot 加 ±h/2"
+ 额外补偿两套算法，一叠加部件就整体偏半个身高。

### 5.3 三个枢轴坑（都是实测撞出来的）

| 写法 | 症状 | 为什么 |
|---|---|---|
| 取**切区边** | 人偶"缩脖子" | 切区为盖缝上下各扩 `overlap`，头/躯干枢轴差 `2×ov`。单帧几乎看不出，一播就明显 |
| 取**切区中心** | 整条臂"绕肘甩" | 臂区从角色边缘算起，中心落在手臂中段，不是肩 |
| **不裁透明边** | 头部件 486 宽，包体虚高 | 长发横跨全宽，但两侧是透明的 |

正解：枢轴取**解剖位置**（颈线/肩线/髋线 × 中线/肩半宽/髋半宽），
与 `overlap` 和裁边完全解耦。测试 `test_overlap_does_not_move_joints`
专门钉住"改 overlap 不能移动关节"。

### 5.4 根骨为什么必须在 (0,0)

根骨的世界位置就是「角色站在哪」，那是游戏运行时 `set_position(地面线)` 决定的。

烘成 `(cx, foot_y)` 的话，实例化后角色会偏移半个身位，用户以为是导出错了。
所以 `build_rig` 里 `cx/foot_y` 只用来**换算**其它骨的相对位置，根骨恒 `(0,0)`。

---

## 6. 动作生成（`rigbuild.build_action`）

### 6.1 相位与采样是两件事

```
frames  = 相位密度（一圈几个姿态，默认 8）→ 决定 duration = frames / fps
samples = 关键帧数（默认 24）           → 决定平滑度
ph      = 2π · i / samples              ← 必须按 samples 走一整圈
```

**这是骨骼方案最大的价值**：加密几乎不增包体（25 个 float vs 25 张 PNG），
8 帧的阶梯感可以被抹平。逐帧方案做不到——加密 = 多 3 倍图。

这里踩过一个坑：相位写成 `ph = 2π·i/frames` 时，24 个采样会走 **3 圈**（i=24 → 6π）。
症状是"抽搐式快进"+ `t=0` 与 `t=0.5` 姿态完全重复。
真机探针的 `moved_steps` 指标（4 段采样里有几段与起点不同）就是为了抓这个。

### 6.2 层级化把 18 条轨道压到 9 条

逐帧时代（pixelforge 的 `rigwalk`）每个部件都要算绝对位置：6 个部件 × (位置+角度) ≈ 18 条。

层级化之后，躯干的 bob 会带着头/臂走，只有腿要自己带（挂在 root 上）:

```
torso.position      ← bob
leg_l/r.position    ← bob - lift
leg_l/r.rotation    ← ±leg_a
leg_l/r.scale       ← 1 - squash（屈膝）
arm_l/r.rotation    ← ∓arm_a
```

→ **9 条**。walk 的 `tracks` 正好 9 条，idle 3 条（只有 torso.position + 两臂）。

### 6.3 循环闭合

关键帧最后追加一帧：`{time: duration, value: 首帧的值}`。
不补的话，Godot 的 `LOOP_LINEAR` 会在末尾插值回第一帧之前，接缝处跳一下。

### 6.4 已知小瑕疵

`_rad(0.0)` 会产生 `-2.5e-17` 这类浮点噪声，写进 JSON 是科学计数法（难看但无害）：
导出时 `godotrig._f()` 会归成 `0.0`。要清理就在 `_rad()` 里 `round(v, 6)` 并
把 `abs(v) < 1e-9` 归零——一行的事，本次没动（避免在"暂停"前改代码）。

---

## 7. Godot 导出（`workflows/godotrig.py`）

### 7.1 产物

```
<stem>.tscn                场景：Rig(Node2D) + 骨节点 + Sprite2D + AnimationPlayer
anim.tres                  AnimationLibrary（动画名 → Animation）
actions/walk.tres          每段动作一个 Animation
actions/idle.tres
parts/*.png                6 张部件图（从部件名命名，不是 role 名）
rig/*.json                 原始数据（可改了再导，或给别的工具用）
README.txt                 三步导入说明
<stem>.zip                 条目带 <folder>/ 前缀
```

zip 条目带前缀是刻意的：`.tres` 里写的是 `res://<folder>/...`，
解到工程根目录后路径恰好成立。这是唯一能让"路径正确"由结构保证、而不是靠用户记的办法。

### 7.2 三个致命格式坑（**单测完全测不出，全是真机抓的**）

#### 坑 1：轨道路径必须写完整

```
path = NodePath("torso/arm_l:rotation")   ✓
path = NodePath("arm_l:rotation")         ✗
```

只写骨名时，`torso` 恰好在根下所以能命中，`arm_l` 在 `torso` 下就解析失败。
Godot 只打一条 `couldn't resolve track` 警告，**播放时那条轨道静默不动**。
在编辑器里看，动画在播、头在动、手臂不摆——非常容易误判成"参数没调好"。

**发现方式**：headless 探针里 `rig.get_node("arm_l")` 报 `Node not found`，
顺手 grep 了完整日志，看到 warning 才定位。

**防回归**：`godotrig.bone_paths()` 统一算路径；`tests/test_rig.py` 断言
`NodePath("torso/arm_l:rotation")` 必须存在；`verify_bundle` 从 `.tscn` 还原
完整节点路径后逐条比对轨道目标。

#### 坑 2：float 关键帧必须写 `0.0`，不能写 `0`

```
"values": [0, -0.160437, ...]      ✗ rotation 一动不动
"values": [0.0, -0.160437, ...]    ✓
```

裸整数解析成 `int`，`AnimationMixer` 不会把它写进 float 属性，**零警告**。
表现是最迷惑的一种：**position 在动、rotation 一动不动**，因为 `Vector2` 的分量
是 int 反而没事（构造器会转）。

**发现方式**：被逼到做对照实验——
① 从文件加载的轨道不生效；② 程序化建的 rotation 轨道也不生效；
③ `leg.rotation = 0.42` 手动 setter 生效。
再让 Godot 自己 `ResourceSaver.save()` 存一份同结构的动画，逐字符对比：

```
Godot 自己写的： "values": [0.0, 0.5]
我们生成的：    "values": [0, -0.160437, ...]
```

差异只有这一个，改完立刻生效。

**防回归**：`godotrig._f()` 保证输出一定带小数点；
`tests/test_rig.py` 有一条断言直接扫 `values` 里的裸整数。

#### 坑 3：根骨不能建节点

场景根 `Rig` 就是根骨。多挂一个空 `Node2D` 会让所有轨道路径多一层，
坑 1 的修复也会跟着失效。

### 7.3 自检查什么、查不出什么

`verify_bundle` 查三件事：

1. 每个 `res://` 引用在包内存在（防"路径差一层 → 打开一片空白"）；
2. 动画资源数与动作数一致（防 `play("walk")` 找不到）；
3. 轨道目标节点存在（从 `.tscn` 还原完整路径后比对）。

**查不出的**：坑 1 的"路径写法不完整"（如果节点在根下就查不出）、坑 2 的类型问题。
所以 `tools/rig_godot_demo.py --verify` 是**必需**的，不能只靠自检。

---

## 8. 真机验证（可复现）

### 8.1 一条命令

```bash
.venv/Scripts/python.exe tools/rig_godot_demo.py 角色.png --verify --shots
```

它做的事：切件 → 建骨 → 动作 → 导出 → 建临时 Godot 工程 → `--import` →
跑探针 → 可选渲染截图。

### 8.2 正常输出

```
源图 sprite_pose_0001.png (486, 673)
部件: {'head': (486, 178), 'torso': (224, 201), ...}
枢轴: {'head': [243.0, 174.98], 'torso': [243.0, 174.98],
       'arm_l': [131.22, 198.54], 'leg_l': [194.4, 370.15], ...}
导出 hezi_rig/hezi.tscn  骨 7  部件 6
体积: {'parts': '307 KB', 'scene': '14 KB', 'total': '362 KB'}
动作: walk(1.00s/225键), idle(1.00s/75键)

--- Godot 真机 ---
RIG_ANIMS ["idle", "walk"]
RIG_NODES 12
RIG_ANIM idle len=1.000 tracks=3 moved_steps=2
RIG_ANIM walk len=1.000 tracks=9 moved_steps=2
截图 4 张 → tests/_out/rigdemo2/shots
```

`moved_steps` = 4 段采样里有几段与起点不同。走路正弦过零会回到中点，
**2 是正常值**；0 才说明轨道没生效。

### 8.3 Godot 自身的坑（不在我们代码里，但每次都会撞）

| 现象 | 原因 | 处理 |
|---|---|---|
| `load("res://...png")` 返回 null | PNG 没导入 | 先 `--headless --path <proj> --import`（`.import` 由编辑器生成） |
| 没有任何输出 | 用的是普通版 Godot | **必须用 `_console.exe` 版**才有 stdout |
| `Can't save PNG at path` | 传了相对路径 | `save_png()` 要绝对路径 |
| 多帧截图背景不一致 | clear color 未统一 | `RenderingServer.set_default_clear_color()` 开头设一次 |
| 截图为空/未渲染 | headless 不渲染 | 截图要 `--rendering-driver opengl3`（不要 `--headless`） |

本机 Godot：`D:\GAMES_tool\Godot_v4.7.2-stable_mono_win64\Godot_v4.7.2-stable_mono_win64_console.exe`。

---

## 9. 实测数据

### 9.1 包体

| 方案 | 25 帧走路 + 待机 | 加第 3 个动作 |
|---|---|---|
| 逐帧 PNG | ≈ **7386 KB**（25 × 295KB） | +295 KB × 帧数 |
| 骨骼 cutout | **321 KB** = 部件 307 + 动画与场景 14.5 | **+2.9 KB** |

约 23 倍，且动作越多差距越大（逐帧线性增长，骨骼几乎不变）。

### 9.2 动作资源

| 动作 | 时长 | 轨道 | 关键帧 | 文件 |
|---|---|---|---|---|
| walk | 1.00 s | 9 | 225（25×9） | 4.3 KB |
| idle | 1.00 s | 3 | 75（25×3） | 2.9 KB |

### 9.3 部件

| 部件 | 尺寸 | 说明 |
|---|---|---|
| head | 486×178 | 长发横跨全宽（裁边后仍是全宽，因为头发确实铺满） |
| torso | 224×201 | |
| arm_l / arm_r | 134×241 | 对称 |
| leg_l / leg_r | 246×306 | **包含大块裙摆**——这是 §10 的问题根源 |

---

## 10. 质量边界（**最重要的一节**）

### 10.1 结论

> **几何切件对大裙摆 / 长发 / 飘带 / 宽袖的角色不成立。**

「腿部切区」切的实际是裙子。转动时裙子裂成两块互相穿插、露出背景。
截图：`docs/img/rig-godot-frames.png`（4 帧走路）、`docs/img/rig-godot-zoom.png`（接缝特写）。

### 10.2 为什么调参数救不回来

切区是**矩形**，而衣服结构不是。把切区调窄 → 裙子缺一块；调宽 → 带进更多别的东西；
加 overlap → 只是让重叠区域变大，转动时两层裙摆互相穿。

根本问题：**衣服不跟骨骼走**。cutout 的语义就是"刚性贴图绕关节转"，
它天生处理不了"布料形变"。

### 10.3 什么角色能用

四肢分明、贴身衣物：简笔小人、紧身衣角色、像素小人。

验证方法：跑 `--shots` 看 4 帧，**只看关节处有没有露出背景**。
没露就能用，露了就换切件来源。

### 10.4 提升路径（按代价排）

1. **真拆件**（推荐）：See-Through（Apache-2.0，要 CUDA）/ Qwen 分层重绘 / 人工 PSD。
   `build_rig` 不用改。
2. **关节补重叠**：在部件图上按关节画一圈同色。纯图像处理，不需要新模型。
   只能缓解，不能根治。
3. **网格形变**（另一条路）：单图 + `Polygon2D` + `Skeleton2D` + 权重。
   能解决"衣服不跟骨骼"，但权重自动生成没有可靠方案（FrameBaker 的 warp/deform 也是手工调）。

---

## 11. 未完成项与建议顺序

| 优先级 | 项 | 为什么排这个位置 |
|---|---|---|
| 1 | **换切件来源** | 决定画质上限。UI 做完了画质不行 = 给走不通的路修收费站 |
| 2 | 关节补重叠 | 便宜的缓解手段，可以在等拆件方案时先做 |
| 3 | API 端点 + 前端节点 | 让人能点按钮出包。做成 `/projects/{scope}/rig/*` + 一个节点 |
| 4 | 多朝向 | `Z_ORDER` 有 front/back/side 三套层级，但没有对应部件图；侧视目前只能用 `scale.x = -1` 镜像 |
| 5 | 动作库 | 只有程序生成的 walk/idle。FrameBaker 的 `builtinMotions.ts` 字段几乎同构，可以转 |
| 6 | 网格形变 | 完全不同的一条技术线 |

**为什么 UI 排第三**：链路已经能用命令行跑通，UI 只是把同样的调用包一层。
在没有解决画质之前，UI 只会让人更快地看到"裙子裂开"。

---

## 12. 从零复现

```bash
cd D:/code/2d-game-meow

# 1. 跑测试（0.5 秒，不需要 GPU / 模型）
.venv/Scripts/python.exe -m pytest tests/test_rig.py -q

# 2. 一张图跑完整条链（含 Godot 真机验证 + 4 帧截图）
.venv/Scripts/python.exe tools/rig_godot_demo.py \
    "assets/characters/合子/Sprites/Naked/neutral/sprite_pose_0001.png" \
    --out tests/_out/rigdemo2 --stem hezi --verify --shots

# 3. 只导出不验证（快）
.venv/Scripts/python.exe tools/rig_godot_demo.py 角色.png --stem hero

# 4. 在 Godot 里打开产物
#    把 <out>/exports/hero.zip 解到某个 Godot 工程根目录，双击 hero.tscn
```

产物结构：

```
<out>/rig/{skeleton.json, binding.json, actions/*.json, parts/*.png}
<out>/exports/<stem>/{<stem>.tscn, anim.tres, actions/, parts/, rig/, README.txt}
<out>/exports/<stem>.zip
<out>/shots/frame_*.png        （--shots 时）
<out>/_godot/                  （--verify 时的临时工程，可删）
```

依赖：`PIL`、`numpy`（已有）。**不需要** mediapipe / torch / CUDA。
Godot 4.4+ 即可（本机是 4.7.2）。

---

## 附录 A：踩坑速查表

| 现象 | 原因 | 修法 |
|---|---|---|
| 人偶"缩脖子" | 枢轴取了切区边（含 overlap） | 枢轴用解剖线，与 overlap 解耦 |
| 整条胳膊绕肘甩 | 枢轴取切区中心 | 用 `shoulderHalf`/`hipHalf` 的解剖横向位置 |
| 实例化后偏半个身位 | 根骨 `rest` 烘进了 `(cx, foot_y)` | 根骨恒 `(0,0)` |
| 播放"抽搐式快进" | 相位按 `frames` 走，采样 24 走 3 圈 | `ph = 2π·i/samples` |
| 子骨（臂/腿）不动，Godot 只警告 | 轨道路径只写了骨名 | 写完整路径 `torso/arm_l` |
| position 动、rotation 不动，零警告 | float 关键帧写成裸整数 `0` | `_f()` 强制 `0.0` |
| 部件整体偏半个身高 | offset 在两处各算了一次 | 只在 `build_rig` 算 |
| `load(png)` 返回 null | PNG 没导入 | 先 `--import` |
| 截图失败 | 相对路径 | `save_png()` 用绝对路径 |
| 切出空部件 | 输入不是单人全身图 / 比例参数反了 | 报错文案已提示 |

## 附录 B：文件清单

| 文件 | 行数 | 职责 |
|---|---|---|
| `workflows/rigmodel.py` | 564 | 三份 JSON 的结构、校验、采样、世界变换 |
| `workflows/rigbuild.py` | 436 | 切件 / 建骨 / 动作生成 / 落盘 |
| `workflows/godotrig.py` | 592 | Godot 文本资源生成、自检、zip |
| `tools/rig_godot_demo.py` | 249 | 一键跑 + 真机探针 + 截图 |
| `tests/test_rig.py` | 303 | 16 条纯计算测试 |
| `docs/骨骼角色-Godot链路.md` | 250 | 使用手册（怎么用、怎么改） |
| `docs/img/rig-godot-frames.png` | — | Godot 真渲染 4 帧 |
| `docs/img/rig-godot-zoom.png` | — | 接缝特写（问题证据） |
