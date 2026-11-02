# 3D 人偶姿势图：glTF / FBX 动画导入（调研 + 实现交接）

> 这份文档给**接手的 agent** 看。包含：为什么换路线、调研过哪些方案、
> 哪些被否决及理由、已经实现了什么、还差什么、怎么验证。
> 状态：**§6 的 6 项待办全部完成并真机验证通过**（2026-11-02 晚）。
> 相关 commit：`7fdc835`（主干）、`c2650a9`（本文件）、
> 本轮（loader 本地化 / 非 Mixamo 骨架 / 批量抽帧 / 明确报错 / 资产指引）。

---

## 1. 问题背景

「人偶姿势」节点（`poseedit`）的正文是官方 VNCCS PoseStudio 的内嵌 iframe
（`web/public/vnccs-pose/studio.html?embed=1`），摆的是一个 MakeHuman 网格 +
53 根骨的 3D 人偶。它的产出（白底灰人偶 PNG）当作「角色迁移」的动作参考图。

获取姿势有三条路：

| 路线 | 状态 | 问题 |
|---|---|---|
| 手动拖关节 | 一直有 | 复杂姿势（瑜伽/舞蹈）摆一帧要很久 |
| **从图片反推 2D → 3D** | 2026-11-02 刚修好 | 见 §2：侧身/悬空姿势仍摆不准 |
| **从动画资产（glTF/FBX）取帧** | **本文件的主题** | 已实现，见 §5 |

### 2. 为什么「从图片」这条路有天花板（别再重复调研）

官方 `pose_studio/imports/openpose.js` 的 `convertOpenPoseToPose` **只吃 2D
关节点**，靠两骨 IK + 「髋线在画面里的角度」反推骨盆倾角，**没有深度信息**。

一张侧身照片里两侧髋部在画面上几乎重合（实测只隔身高的 1.5%），转换器就把
这条被压扁的短线段当成「骨盆竖起来了」，判出 **−81° 倾斜**，整个人偶绕 Z 轴
转倒。这是 2D→3D 的固有歧义，不是调阈值能解决的。

**已做的缓解**（2026-11-02，见 `workflows/pose3d.py`）：
MediaPipe Pose Landmarker 给米制 3D 点 → 绕 Y 转正 → 等比重投影成 2D →
编辑器把官方反推的 `modelRotation` 归零。实测那张瑜伽照从「躺倒的麻花」变成
「直立抬腿」。**但肢体交叠处（手抓脚）仍差 0.2 个身高**，需要手动微调。

---

## 3. 调研过的 3D 方案（结论：只有动画资产这条路没有天花板）

### 3.1 HybrIK / SMPL-X —— **本机跑不起来**（已验证，别再试）

官方仓库已迁移到 `jeffffffli/HybrIK`（原 `Jeff-sjtu/HybrIK`）。权重在 HF
镜像上能拿到：`Arktische/HybrIK/hybrik_hrnet48_w3dpw.pth`、
`SceneWorks/hybrik-resnet34/model.safetensors`。

三条硬阻断（每条都实测过）：

1. **装不上**。`setup.py` 用 `CUDAExtension` + `BuildExtension`，安装时就要编译
   CUDA 扩展。本机 `where nvcc` → 无，`where cl` → 无，无 conda。
   `pip index versions pytorch3d` → *No matching distribution*（PyPI 没有
   Windows 轮子，PyTorch3D 是 HybrIK 的硬依赖）。
2. **SMPL 模型要注册**。`fetch_data.sh` 开头就是
   `read -p "Username (SMPLify):"`，从 `download.is.tue.mpg.de` 取
   `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`。没有它，HybrIK 输出的 SMPL
   参数变不成 3D 关节。
3. **依赖会破坏现有环境**：`install_requires` 钉了 `opencv-python==4.1.2.30`
   （本项目在用 5.0）和 `chumpy`（Python 2 时代的包）。

许可另说：SMPL / SMPL-X 是 MPI 注册制，**仅限非商业**，且禁止随包分发。

### 3.2 MotionBERT（ONNX）—— 可行但不是最优

`bukuroo/MotionBERT-3d-ONNX`（Apache-2.0，162MB，纯 onnxruntime，输出 17 个
3D 关节、**不需要 SMPL**）。本机试过：hf-mirror 偶发 429 限流，`huggingface.co`
直连不通、`hf-mirror.com` 通、`cdn.jsdelivr.net` 通。

没做的原因：MotionBERT 是**视频时序**模型，单帧效果不如动画资产直接采样；而且
它同样有「从 2D 估计 3D」的深度歧义天花板，只是比 MediaPipe 准。

### 3.3 MediaPipe Pose Landmarker —— 已接入（作为「从图片」那条路）

`pose_landmarker_full.task` **9.0MB**，Apache-2.0，输出 33 个点
（`pose_world_landmarks` 是米制 3D）。已装、已接、已验证。

### 3.4 glTF / FBX 动画 —— **本文件采用**

**动画里的 3D 是自带的**，不用估深度：骨长、朝向、遮挡全真实，套上去就是自然
人形。而且它直接对应产线需求（idle / walk / run / attack 整套动作）。

**关键优势：官方已经写好了 90%。** `pose_studio/imports/mixamo.js`（852 行）里
有 65 根 Mixamo 骨 → MakeHuman 骨的映射 `MIXAMO_TO_MH_BONE_MAP`、按层级的
重定向父链 `MIXAMO_RETARGET_PARENTS`，以及导出的
`buildMixamoWorldKeypoints(sourceBones, viewer)`。

**网络实测（本机，2026-11-02）**：

| 站点 | 结果 |
|---|---|
| `pypi.org` / `pypi.tuna` | 200 ✅ |
| `hf-mirror.com` | 200（偶发 429 限流）⚠️ |
| `huggingface.co` 直连 | 不通 ❌ |
| `drive.google.com` | 不通 ❌（HybrIK 权重因此下不来） |
| `cdn.jsdelivr.net` | 200 ✅ |
| `unpkg.com` | 200 ✅（但 three 的 npm 包不含 examples/models） |
| `esm.sh` | 200 ✅（**官方 mixamo.js 就是从这儿动态 import FBXLoader 的**） |

---

## 4. 官方现成能力（别重写）

`web/public/vnccs-pose/pose_studio/imports/mixamo.js`：

| 导出 | 用途 | 我们用了吗 |
|---|---|---|
| `buildMixamoWorldKeypoints(sourceBones, viewer)` | 从源骨架的**世界关键点**提取器 → 48 个点 | ✅ 核心 |
| `importMixamoFBXAnimation(file, viewer, options)` | 一次性抽 N 帧返回一整组 pose | ❌ 它是"批量导入"语义，不支持拖滑块实时改时间点 |

内部私有、本项目自己复刻了一份（10 行）：`collectSourceBones(root)` —— 把 root
下的骨骼按「原名 / 去掉 `mixamorig:` 前缀 / 归一化名」三种方式登记。

**骨骼映射的前提**：源文件的骨骼名得是 Mixamo 风格（`Hips` / `Spine` /
`LeftArm` / `LeftForeArm` / `LeftUpLeg` / `LeftFoot`…）。Mixamo 导出的
glb 与 fbx 都符合。**非 Mixamo 骨架（如 Quaternius / KayKit 的 CC0 资产）需要
另写一套映射表**——见 §6。

---

## 5. 已实现

### 5.1 文件

| 文件 | 作用 |
|---|---|
| `web/public/vnccs-pose/anim-import.js` | 加载 / 采样 / 切轨道，核心逻辑 |
| `web/public/vnccs-pose/anim-skeleton.js` | **新增**。任意骨架的骨名 → 官方认的 17 个锚点（§5.4） |
| `web/public/vnccs-pose/vendor/three/jsm/**` | **新增**。GLTFLoader / FBXLoader 及依赖的本地副本（r160） |
| `web/public/vnccs-pose/entry.js` | `animLoad` / `animClip` / `animTime` / `animSample` 四条消息；文件头有完整协议 |
| `web/public/vnccs-pose/studio.html` | 加了 **importmap**（把裸名 `three` 指到 vendor 的 three） |
| `web/src/components/PoseStudio.tsx` | 「导入动作」按钮、轨道下拉、时间滑块、「存入 N 帧」 |
| `web/src/canvas/Canvas.tsx`、`web/src/nodes/GenNodeCard.tsx` | 把 `onSavePoses` 透到编辑器 |
| `web/src/App.tsx` | `savePosesToLibrary`（一次请求） |
| `workflows/library.py` / `app.py` / `web/src/api.ts` | `add_poses` + `POST /library/poses/batch` + `addPosesApi` |
| `web/src/styles.css` | `.pose-anim*` / `.pose-frames*` |
| `tools/verify_anim_import.mjs` | **新增**。真机验证（§5.6） |

### 5.2 消息协议（父 ↔ 子）

```
父 → 子：
  { type:'animLoad', buffer: ArrayBuffer, name: 文件名 }
      → 回 { type:'animLoaded', clips:[{name,duration}], boneCount, format,
              rig:{ family, familyLabel, mapped, missing, anchors } }
  { type:'animClip', index }     切轨道（切完自动采第 0 帧）
  { type:'animTime', t }         采样 t 秒 → 回 { type:'pose', pose, yaw }
  { type:'animSample', count }   等分抽 N 帧
      → 回 { type:'animFrames', frames:[{t,pose,preview}], clip, yaw }
子 → 父：
  { type:'animLoaded', ... } | { type:'animFrames', ... }
  | { type:'pose', pose, yaw } | { type:'error', message }
```

`rig.anchors` 是 `{LeftArm: 'upperarm_l', ...}` —— 每个锚点对到了文件里的哪根骨头。
界面不显示，但用户报「套歪了」时能一眼分出是**映射错**还是 **IK 错**；
验证脚本也靠它断言「对到的是对的那根」，而不是「凑够 17 个」。

**为什么传 ArrayBuffer 而不是路径**：父子是同源 iframe，但子窗拿不到用户
`<input type=file>` 选的 `File` 对象（跨文档不行）。宿主读成字节发过去，
子窗自己造 Blob 解析。

### 5.3 关键实现点（改这块前先读）

- **解析器在本地**（`vendor/three/jsm/`，r160）：`anim-import.js` 先按
  `import.meta.url` 相对路径 import，**只有本地失败才回落 esm.sh**。
  上游 loader 写的是裸名 `from 'three'`，靠 `studio.html` 的 importmap 落到
  `./vendor/three/three.module.js` —— 与 `core.js` 动态 import 的是**同一个 URL**，
  所以 loader 与 viewer 现在共用**同一份 three 实例**（以前是两份）。
  importmap **必须写在 module 脚本之前**，`web/smoke.mjs` 有断言钉住这条。
- **FBX 依赖不止一个文件**：`FBXLoader` 还要 `libs/fflate.module.js`（解压）和
  `curves/NURBSCurve.js` + `curves/NURBSUtils.js`（NURBS 曲线）；`GLTFLoader`
  还要 `utils/BufferGeometryUtils.js`。缺一个就是「本地加载失败 → 悄悄回落 CDN」，
  断网时才炸。`smoke.mjs` 逐个文件断言存在。
- **每次采样前 `viewer.resetPose()`**：上一帧摆过、这一帧没被驱动的关节会停在
  旧位置，看起来像「动作有残留」。官方 FBX 导入也是每帧先 reset。
- **滑块要节流 60ms**：每帧要跑一次关键点提取 + 全骨 IK，不解节流会卡手。
- **`pose` 回传即存档点**：采样后 iframe 主动 `post({type:'pose'})`，宿主按
  拖动松手同样的语义存进节点（和姿势库一致）。
- **不能用官方的 `importMixamoFBXAnimation`**：它是「一次性抽 N 帧」语义，
  我们要的是「拖滑块实时改时间点」，必须自己持有 `AnimationMixer`。
  （**批量入库**是自己循环 `applyTime` + `renderOut`，见 `entry.js` 的 `animSample`；
  不用官方那个是因为它只出 pose、不出缩略图。）
- **抽帧要避开两端**：`t = duration * (i + 0.5) / count`，不取 0 和 duration ——
  很多循环动画（idle/walk）的首尾是过渡帧，抽进来一堆"站着的同一帧"。

### 5.4 非 Mixamo 骨架怎么认（`anim-skeleton.js`）

**不改官方 `mixamo.js`**：解析结果是"别名登记表"，把真实骨头挂到 Mixamo 名下，
官方 `findSourceBone` 天然就找到了：

```js
const rig = readSkeleton(root)            // 拍平 Object3D 树（**不只是 isBone**）
const map = resolveRig(rig)               // { family, alias:{Hips: 3, ...}, missing }
const sb  = buildSourceBones(rig, map)    // { bones:{Hips:Object3D,...}, normalizedBones }
buildMixamoWorldKeypoints(sb, viewer)     // 官方原样调用
```

**策略：结构优先、名字兜底。**

1. **名字**只用来取**根锚点**：Hips / 两条上臂 / 两条大腿（+ Neck / Head）。
   4 套词表覆盖 Mixamo、Unreal（`upperarm_l`）、VRM（`J_Bip_L_UpperArm`，靠
   **后缀**匹配）、KayKit（`upper_arm_l`）。短词（`arm`/`leg`/`palm`）是**弱词**，
   只在没有强候选时才用。
2. **肘腕膝踝靠层级**：人形骨架"肩→肘→腕、髋→膝→踝"一定是祖孙三代，
   从根锚点沿**子树最大的那个子节点**往下走两跳就是。名字强匹配时尊重名字，
   名字没认出来或只靠弱词粘上时才走链 —— 名字和层级都清楚时，层级不提供
   额外信息，改它只会引入风险。
   > 为什么必须有这一层：同一个词在不同骨架里意思相反。Mixamo 的 `LeftLeg`
   > 是**小腿**，KayKit 的 `Leg` 是**大腿**，RobotExpressive 的 `Leg.L` 根本不是
   > 关节。只有层级不会骗人。
3. **杂物要跳过**：子树大小排序时过滤 `skirt/cape/cloth/hair/weapon/socket/tail`
   这类挂在身上的骨（裙子/披风的子树常常比腿大，会把"膝"抢走）。
4. **末端骨也要读**：很多 glTF 的 `LowerLeg.L_end` / `Head_end` 不在 `skin.joints`
   里，GLTFLoader 建成普通 `Object3D`，但它的位置就是脚踝/头顶 —— 所以
   `readSkeleton` 遍历全部子节点，`isBone` 只作为加分项。
5. **认不出就明说**：`missing` 非空时 `describeRig()` 拼出「缺 Hips / LeftArm …」
   + 文件里的骨名样本，`loadAnimation` 直接抛这个错，宿主显示在 `.pose-err` 里。

### 5.5 批量抽帧入库

「一条 walk → 一整套姿势」：`animSample {count}` → iframe 等分 `applyTime` →
每帧 `getPose()` + `renderOut(192)` 缩略图 → `animFrames` → 宿主命名
`动作名 01/02/…`（重名自动加 `(2)`）→ `POST /library/poses/batch` **一次落盘**。

**为什么不用逐条 `POST /library/poses`**：那是 N 次整库重写（每帧还带一张预览图），
抽 16 帧就是 16 次。`library.add_poses` 合并完只写一次，
`tests/test_library.py::test_pose_add_batch_single_write` 直接数 `_write` 调用次数。

### 5.6 验证（真机，2026-11-02）

```bash
cd web && npm run build
node tools/verify_anim_import.mjs          # 一个浏览器进程跑完全部用例
node tools/verify_anim_import.mjs --only fbx
```

脚本做四件事 jsdom 做不到的：真解析文件、走 FBX 那条路、换骨名验映射、
**拦截全部外部请求**（＝断网）证明解析器是本地副本。

| 用例 | 素材 | 结果 |
|---|---|---|
| glb-mixamo | `Soldier.glb`（2.05MB） | 4 轨 / 49 骨 / 17 锚点 / 40% 处套出姿势 / 导出 PNG 20KB / 抽 4 帧且帧帧不同 |
| fbx-mixamo | `Samba Dancing.fbx`（3.6MB） | 2 轨 / **119 骨** / 17 锚点 / 姿势自然 / 导出 PNG 18.6KB |
| glb-unreal | 把 Soldier 的节点名改成 `pelvis/upperarm_l/thigh_l…` | `mapped=17`，`LeftArm→upperarm_l`、`LeftFoot→foot_l` |
| glb-kaykit | 改成 `hips/upper_arm_l/upper_leg_l…` | `mapped=17`，`LeftUpLeg→upper_leg_l`、`LeftToeBase→toe_l` |
| glb-junk | 节点名改成 `b0/b1/…` | 报错「这个文件的骨骼认不出：缺 Hips / Neck / …」 |

导出的两张人偶图肉眼复核过（`tests/_out/anim_import/*.png`）：走路的抬腿、
Samba 的张臂，肢体无扭曲无穿透。**`fbx` 这条以前只是"代码写了没跑过"，
现在真跑通了** —— 顺带证明了 fflate / NURBS 那几个依赖没漏。

---

## 6. 待办

**§6 原来的 6 项全部完成**（2026-11-02 晚），逐条对应：

| 原编号 | 事项 | 落点 |
|---|---|---|
| 1 | loader 本地化，断网可用 | `vendor/three/jsm/**` + `studio.html` 的 importmap；验证脚本**拦截全部外部请求**跑通 |
| 2 | FBX 实测 | `Samba Dancing.fbx` 走通（119 骨 / 2 轨 / 出图），见 §5.6 |
| 3 | 批量抽帧入库 | `animSample` + `/library/poses/batch` + 「存入 N 帧」按钮 |
| 4 | 非 Mixamo 骨架映射 | `anim-skeleton.js`，Unreal / VRM / KayKit 三种命名真机验过 |
| 5 | 动作资产获取指引 | README「姿势编辑界面」下的《动作从哪来》 |
| 6 | 骨名不匹配的明确报错 | `describeRig()`：缺哪几根 + 文件骨名样本 + 建议 |

**还敞着的（想继续做再看）：**

| # | 事项 | 备注 |
|---|---|---|
| A | ASCII FBX | `FBXLoader` 只吃二进制 FBX；ASCII 版本直接报解析错。Mixamo 默认导出就是二进制，影响不大 |
| B | 病态骨架仍是尽力而为 | 像 `RobotExpressive`（腿链没有独立脚骨、脚是独立的 IK 控制骨）能映射出来但姿势未必准；这类资产不在产线范围内 |
| C | 批量抽帧共用当前朝向 | 抽 N 帧时全部用当前机位，不能一帧一个角度。要 8 方向的同一动作，得抽完再逐个改朝向 |
| D | 抽帧上限 24 | 再多就得进库后手动删，姿势库整体上限 200 |

**别做的事**（已否决，理由见 §3）：
把 HybrIK/SMPL-X 接进来；为了"更准"去装 PyTorch3D。

---

## 7. 相关文件速查

| 关注点 | 文件:位置 |
|---|---|
| 3D 重投影（从图片那条路） | `workflows/pose3d.py`、`app.py:pose_extract` |
| 2D→3D 官方转换器 | `web/public/vnccs-pose/pose_studio/imports/openpose.js` |
| Mixamo 骨骼映射（官方，**不要改**） | `web/public/vnccs-pose/pose_studio/imports/mixamo.js` |
| 任意骨架 → Mixamo 锚点 | `web/public/vnccs-pose/anim-skeleton.js` |
| 动作导入实现 | `web/public/vnccs-pose/anim-import.js` |
| 解析器本地副本（r160） | `web/public/vnccs-pose/vendor/three/jsm/**`（见同目录 `SOURCE.md`） |
| importmap（裸名 `three` → vendor） | `web/public/vnccs-pose/studio.html` |
| iframe 消息协议 | `web/public/vnccs-pose/entry.js`（头注释里有完整协议表） |
| 宿主 UI | `web/src/components/PoseStudio.tsx` |
| 姿势库批量写入 | `workflows/library.py:add_poses`、`app.py:/library/poses/batch` |
| 真机验证（动作导入） | `tools/verify_anim_import.mjs`（需 `tools/shot` 里的 puppeteer-core） |
| 真机验证（人偶编辑器本体） | `tools/verify_pose_editor.py` |
| DWPose 2D 点（图片路线用） | `workflows/pose.py`（注意 `COCO_TO_OP` 表，见 changelog） |
