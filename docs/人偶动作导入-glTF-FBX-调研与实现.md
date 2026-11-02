# 3D 人偶姿势图：glTF / FBX 动画导入（调研 + 实现交接）

> 这份文档给**接手的 agent** 看。包含：为什么换路线、调研过哪些方案、
> 哪些被否决及理由、已经实现了什么、还差什么、怎么验证。
> 状态：**主干已实现并真机验证通过**（2026-11-02，commit 见 `git log --grep 动作导入`），剩余项见 §6。

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

## 5. 已实现（2026-11-02）

### 5.1 文件

| 文件 | 作用 |
|---|---|
| `web/public/vnccs-pose/anim-import.js` | **新增**。加载 / 采样 / 切轨道，核心逻辑 |
| `web/public/vnccs-pose/entry.js` | 加了 `animLoad` / `animClip` / `animTime` 三条消息；文件头注释里补了协议 |
| `web/src/components/PoseStudio.tsx` | 加了「导入动作」按钮、轨道下拉、时间滑块；读文件 → ArrayBuffer → postMessage |
| `web/src/styles.css` | `.pose-anim` / `.pose-anim-range` / `.pose-anim-t` |

### 5.2 消息协议（父 ↔ 子）

```
父 → 子：
  { type:'animLoad', buffer: ArrayBuffer, name: 文件名 }   → 回 { type:'animLoaded', clips:[{name,duration}], boneCount, format }
  { type:'animClip', index }                              切轨道（切完自动采第 0 帧）
  { type:'animTime', t }                                  采样 t 秒 → 回 { type:'pose', pose, yaw }
子 → 父：
  { type:'animLoaded', ... } | { type:'pose', pose, yaw } | { type:'error', message }
```

**为什么传 ArrayBuffer 而不是路径**：父子是同源 iframe，但子窗拿不到用户
`<input type=file>` 选的 `File` 对象（跨文档不行）。宿主读成字节发过去，
子窗自己造 Blob 解析。

### 5.3 关键实现点（改这块前先读）

- **解析器是动态 import 的**：`https://esm.sh/three@0.160.0` +
  `.../loaders/{FBXLoader,GLTFLoader}.js`。vendor 里只有 three 本体（r160），
  没有 loader。**断网会失败**，报错文案已写明「需要联网」。
  → §6 有一条「把 loader 本地化」待办。
- **three 实例混用**：loader 与 mixer 用 esm.sh 的 three，viewer 用 vendor 的
  three（都是 r160）。实测没问题——`buildMixamoWorldKeypoints` 只从源骨架读
  坐标矩阵，用 viewer 的 THREE 重新构造 Vector3，不碰跨实例的内部状态。
- **每次采样前 `viewer.resetPose()`**：上一帧摆过、这一帧没被驱动的关节会停在
  旧位置，看起来像「动作有残留」。官方 FBX 导入也是每帧先 reset。
- **滑块要节流 60ms**：每帧要跑一次关键点提取 + 全骨 IK，不解节流会卡手。
- **`pose` 回传即存档点**：采样后 iframe 主动 `post({type:'pose'})`，宿主按
  拖动松手同样的语义存进节点（和姿势库一致）。
- **不能用官方的 `importMixamoFBXAnimation`**：它是「一次性抽 N 帧」语义，
  我们要的是「拖滑块实时改时间点」，必须自己持有 `AnimationMixer`。

### 5.4 验证结果（真机，2026-11-02）

用 three 自带的 `Soldier.glb`（Mixamo 骨架，jsdelivr 取：
`cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/Soldier.glb`，2.05MB）：

- 49 根骨、**4 条动画轨道**：Idle / Run / TPose / Walk（Idle 时长 1.9667s）
- 载入后轨道下拉出现，拖滑块到 40% → 时间读数 0.79s，人偶实时跟帧
- 导出的人偶图姿态自然、无扭曲、无肢体穿透（截图见 `docs/img/pose-anim-ui.png`、
  `docs/img/pose-anim-scrub.png`）

**复现脚本**（不入库，在 `/d/tmp/shot/`）：

```bash
# 1) 静态服务（只要 dist，不需要后端）
cd web && python -m http.server 8796 --directory dist
# 2) puppeteer-core 驱动系统 Edge：
#    打开 /?demo=1 → 点侧栏「人偶姿势」→ input[type=file][accept*=".glb"] 上传
#    → 断言 .pose-anim-range 出现、.pose-anim select 的 options 是 4 条轨道
#    → 拖 range 到 40% → 断言 .pose-anim-t 读数变化
```

---

## 6. 待办（按优先级）

| # | 事项 | 验收标准 | 备注 |
|---|---|---|---|
| 1 | **把 loader 本地化** | 断网也能导入动作 | vendor 里放 `GLTFLoader.js` / `FBXLoader.js`（r160），需要 importmap 或改写它们的 `from 'three'` |
| 2 | **FBX 实测** | 能导入一个 .fbx 并出图 | 本次只实测了 glb；FBX 路径代码写了但没跑过。FBXLoader 还依赖 `fflate` |
| 3 | **批量抽帧入库** | 一条 walk 动画 → 一次往姿势库灌 N 帧 | 官方 `importMixamoFBXAnimation` 本来就是干这个的，宿主侧做个「导入为 N 个姿势」即可 |
| 4 | **非 Mixamo 骨架的映射表** | 能导入 Quaternius/KayKit 的 CC0 资产 | 需要另写 `骨名 → MakeHuman 骨` 的表；这类资产骨骼命名五花八门，可能要按「层级+位置」猜 |
| 5 | **动作资产获取指引** | README 写清楚从哪下 | Mixamo 免费账号可下 glb/fbx（**可商用**，但别再分发它的模型/动作）；CC0 库：Quaternius、KayKit |
| 6 | 骨骼名不匹配时的**明确报错** | 用户看到"这个文件的骨骼认不出"而不是"没反应" | 现在只有 `applyTime` 返回 false → 「这一帧套不到人偶上（骨骼映射不上）」 |

**别做的事**（已否决，理由见 §3）：
把 HybrIK/SMPL-X 接进来；为了"更准"去装 PyTorch3D。

---

## 7. 相关文件速查

| 关注点 | 文件:位置 |
|---|---|
| 3D 重投影（从图片那条路） | `workflows/pose3d.py`、`app.py:pose_extract` |
| 2D→3D 官方转换器 | `web/public/vnccs-pose/pose_studio/imports/openpose.js` |
| Mixamo 骨骼映射（官方） | `web/public/vnccs-pose/pose_studio/imports/mixamo.js` |
| 动作导入实现（新增） | `web/public/vnccs-pose/anim-import.js` |
| iframe 消息协议 | `web/public/vnccs-pose/entry.js`（头注释里有完整协议表） |
| 宿主 UI | `web/src/components/PoseStudio.tsx` |
| DWPose 2D 点（图片路线用） | `workflows/pose.py`（注意 `COCO_TO_OP` 表，见 changelog） |
