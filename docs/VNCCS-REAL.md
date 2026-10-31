# VNCCS 到底怎么做（源码实证，2026-10-31）

> 来源：`D:\code\gen\others\ComfyUI_VNCCS_Utils-main` 源码 +
> `workflows/VNCCS_Utils Pose Studio QWEN.json` 官方工作流 +
> `MIUProject/VNCCS_PoseStudio_QI2.1`（HF，读不到原站时以 PLAN §2.16b 记录的工作流为准）。

这页回答一个问题：**VNCCS 的姿态迁移，条件图（image1）到底是什么、prompt 怎么写、
参数怎么设。** 之前我们的实现"学了皮毛"，差在把条件图当成了线框骨架。

## 1. 条件图 = 3D 渲染的灰色人偶，不是线框骨架

`web/pose_studio/core.js` 里的渲染管线（逐条证据）：

- 网格是 **MakeHuman v2**（`web/pose_studio/assets/pose_studio_makehuman.v2.bin.gz`，
  37.8MB，CC0 授权），一个**完整的人体网格**（头、躯干、带锥度的四肢、手脚）。
- 材质（`core.js` ~3438 行）：

  ```js
  const material = new THREE.MeshPhongMaterial({
      color: 0xaaaaaa,      // skinMode: 'flat_color' → 中性灰，无贴图
      specular: 0x111111, shininess: 5, side: DoubleSide,
  });
  ```

  另加了一段 rim 变暗的 fragment 后处理（边缘轻微压暗，增强体积感）。
- 灯光（工作流 state `lights`）：`ambient #404040 @ 0.5` + `directional #ffffff @ 1.0
  位置(1,2,3)`。默认 `keepOriginalLighting: true` 时**跳过合成灯光**、光照 prompt 为空。
- 背景（`export.bg_color`）：`[255,255,255]` 纯白；`capture()` 里把
  `scene.background` 设成这个颜色再渲染。
- **输出尺寸 1024×1024**（`export.view_width/height`），相机用 `PerspectiveCamera(45°)`
  以 `cam_yaw_deg / cam_pitch_deg / cam_zoom` 取景——**角度长在相机里，不在 prompt 里**。
- 导出时**隐藏**所有骨骼、关节点、IK 标记、网格参考线（`capture()` 里把这些
  `visible=false`）。所以模型看到的是一张**纯白底上的灰色阴影人偶照片**。

一句话：**image1 = 一张"对着某个相机机位拍的灰色 3D 人体照片"**。

## 2. 两张参考图的顺序与 prompt

官方 QWEN 工作流（8 个节点）的连线：

```
VNCCS_PoseStudio.images  ->  编码节点的 image1
LoadImage "Character"    ->  编码节点的 image2
VNCCS_PoseStudio.lighting_prompt -> 编码节点的 prompt
LoraLoaderModelOnly (VNCCS LoRA, strength=1, model-only) -> patched_model
```

- `image1` = 人偶渲染（姿态来源）；`image2` = 角色图（身份来源）。
- **prompt 就是 `lighting_prompt`**。`generatePromptFromLights()` 在
  `keepOriginalLighting: true` 时**返回空串**；工作流的 `prompt_template` 是
  `"Draw character from image2\n<lighting>\n<user_prompt>"`，三项皆空时最终 prompt 就是
  **`Draw character from image2`**（QIE2511 那套）。
- **QI2.1 那套**（我们底模）的指令模板是（PLAN §2.16b 从 HF 的
  `VNCCS_PoseStudio_QI2.1.json` 读的，逐字）：

  ```
  replace the pose of <image 2> with the pose of <image 1>. keep the character of <image 2>
  ```

- **全程没有任何方向词**（没有 front/side/back view）。方向由人偶的相机机位决定。

## 3. 采样与 LoRA

- QI2.1：25 步、cfg 1、LoRA 1.0（model-only）、从空 latent 全量去噪。
- 姿态数据是**完整骨骼旋转**（`poses[0].bones` = pelvis / spine_01..03 /
  thigh_l/r / calf_l/r / upperarm_* / forearm_* …），需要时从一张照片用
  SAM 3D Body 反推再套到人偶上。

## 4. 我们之前错在哪（逐条对照）

| 项 | VNCCS 实际 | 我们之前 |
|---|---|---|
| 条件图 | 白底灰色阴影 3D 人偶（1024²，相机取景） | **黑白线框骨架**（768²） |
| 朝向 | 相机机位（长在图里） | 往 prompt 里塞 "viewing direction: ..." |
| prompt | 固定指令 + 光照描述，无视角词 | 指令 + 用户描述 + 视角词 |
| 姿态来源 | 完整骨骼旋转 → 蒙皮 | 18 个 2D 关节点 |

**结论：线框骨架不是这个 LoRA 的训练格式，所以姿态跟随弱、朝向不可控。**
"效果差"的根因在这里，不在步数、不在底色。

## 5. 实测（同 seed 967553171、同角色、20 步、白底）

| 条件图 | prompt | 结果 |
|---|---|---|
| 黑底线框 | 无 | 背影 |
| 白底线框 0° | 无 | 正面、姿势跟随 |
| 白底线框 90° | 无 | 正面（朝向没传过去） |
| 白底线框 + 分腿张臂 | `front view`（或更弱的说法） | **姿势被压回站立** |
| 3D 人偶（SVG 平涂）/ 人偶+线框叠加 / 灰阴影人偶 | 无 | 都正面（朝向仍没传过去） |

- 说明 ①：**任何"front view"文字都会压过骨架姿势** —— 所以默认一个字都不写。
- 说明 ②：**我们自己画的任何"人偶"都传不过朝向** —— 因为我们没有他们的 MakeHuman
  网格 + Phong 阴影 + 相机取景，画出来不在 LoRA 的分布内。

### 决定性验证（2026-10-31 晚，`tools/render_mannequin.py`）

用 **真 MakeHuman v2 网格**做 Phong 软光栅（白底、正交投影、单方向光），喂给 VNCCS
（无任何方向词、同 seed、同角色）：

| 人偶渲染机位 | 结果 |
|---|---|
| 0° 正面 | **正面** ✓ |
| 90° 侧面 | **侧面、脸朝左** ✓ |
| 180° 背面 | **背面** ✓ |

**一次就全对。** 这证明：线框骨架 / 手绘人偶失败，是因为它们不在 LoRA 的训练分布里；
只要把条件图换成"真 3D 人体网格的灰色阴影渲染"，朝向和姿态就都由相机决定，
prompt 里一个字都不用写——和 VNCCS 官方完全一致。

`tools/render_mannequin.py` 已固化（解析 `.bin.gz` + 软光栅）。**还差两块**：
① 18 关节 → 53 根 MakeHuman 骨的 IK；② 蒙皮（skin_indices/skin_weights 驱动顶点），
现在只渲染静息 A-pose。

## 6. 已落地：整套移植官方 PoseStudio（2026-11-01）

上面的「备选路线 1」不再需要自己造——官方 ComfyUI_VNCCS_Utils 里那套
PoseStudio 是现成的、调校好的，整套搬过来即可：

| 文件 | 作用 |
|---|---|
| `web/public/vnccs-pose/pose_studio/core.js` | PoseViewerCore（8938 行）：渲染 + 骨骼 + morph + IK + 相机 |
| `web/public/vnccs-pose/pose_studio/morph_runtime.mjs` | VNMHJS02 网格解析（MakeHuman CC0 网格） |
| `web/public/vnccs-pose/pose_studio/imports/openpose.js` | `convertOpenPoseToPose`：18 关键点 → 53 根骨的欧拉角 |
| `web/public/vnccs-pose/vendor/three/` | 官方 THREE 0.160（core.js 相对路径动态加载） |
| `web/public/vnccs-pose/entry.mjs` | 我们的最小入口（只做接线，无自有 IK/渲染） |

链路（**全是官方现成的**）：

```
loadMorphPack → solveMorph → buildStaticModelData → PoseViewerCore.loadData
→ convertOpenPoseToPose(18 关节) → setPose → capture(yawDeg=朝向)
```

三条之前踩过的坑，官方都已有正解：

1. **不缩放骨长**：官方用纯旋转（两骨解析解 + 极点约束）摆姿态。
   我们自写的「方向对齐 + 缩放骨长」会肩外展、肘外翻、手悬空（畸形）。
2. **18 关节就是 COCO 格式**：`convertOpenPoseToPose` 直接吃，我们不用自己映射。
3. **朝向用 capture 的 yawDeg**（相机机位），不是 modelRotation。
   人偶图的朝向**烘焙在图里**，所以下游不该再补方向词——补了是重复指令，
   还会压过姿势（实测 front view 会让分腿张臂退回站立）。

界面接入：骨骼编辑节点新增「**编辑 3D 人偶**」，打开官方的**编辑会话**（弹窗）。
交互与官方 UniCanvas 的 pose layer 一致：

- 3D 视口里**直接拖橙色关节点**摆姿势（官方 raycast + IK + 极点约束，
  肘/膝的弯向由官方保证）；
- 右键拖 = 轨道旋转视角、中键平移、滚轮缩放（官方 OrbitControls）；
  视角只是「检查视图」，**不影响输出**——导出用的是「朝向」下拉定的机位
  （官方同理：capture framing 与导航相机分离）；
- 点「导出人偶图」→ 官方 `capture()` 输出白底灰人偶 PNG。

产物存 `node.mannequin`（与线框骨架 `node.image` 并存）。角色迁移的
「动作参考」**优先用人偶图**，没有才退回线框骨架（`poseSourceOf`）。

后端**零改动**：`_vnccs_skeleton` 只把「低饱和 + 暗」的像素换成白底，
人偶图本就是白底灰人偶，判据 `mx < 70` 不会命中它。

## 7. 仍需注意的

- **没有 PoseStudio 渲染器时的兜底**：线框骨架仍可用（姿势能跟住，
  是精灵产线最需要的），只是姿态贴合比人偶图弱一截。
- **朝向**：人偶图自带朝向，线框骨架则分不清正/背面（2D 投影里互为镜像、
  90° 收成一条线）。用线框骨架时只能靠文字补，代价是姿势打折。
- **`tools/render_mannequin.py`**（我们早期写的 Python 软光栅）已被官方
  PoseStudio 取代，保留仅作无浏览器环境下的对照参考。
