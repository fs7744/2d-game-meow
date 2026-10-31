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

## 6. 我们的取舍（现状）

在没有"真 3D 人偶渲染器"之前：

- 条件图继续用线框骨架（它至少**姿势能跟住**，这是精灵产线最需要的）。
- 朝向：**默认不写任何方向词**（VNCCS 也从不写）。用户**手动选非正面**才补一句，
  并明确提示"会削弱姿势贴合"。这是止损，不是正确解。
- 已修的硬 bug：骨架转白底（否则画背影）、transparent 漏传后处理（否则白块）。

## 7. 要真正做好，需要补的（备选路线）

1. **在浏览器里用 three.js 渲染真的人偶**（`web/pose_studio/assets` 那套 CC0 网格 +
   皮肤贴图 + Phong 灯光），导出成 image1。工作量最大，但最忠实。
2. **Python 侧软渲染人偶**：解析 MakeHuman `.bin.gz`（含蒙皮权重）+ 骨骼旋转，
   用 numpy 做 Phong 栅格化。不引入浏览器依赖，但要写一个微型光栅器。
3. 保持线框 + 显式朝向（现状），接受"侧面/背面只能靠文字、姿势会打折"。

在选 1/2 之前，先把这里当"地面真相"，别再把方向词当万能药。
