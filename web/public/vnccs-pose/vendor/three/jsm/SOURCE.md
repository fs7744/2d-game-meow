# three.js jsm 片段（只放了动作导入用到的几个）

**为什么要放进来**：动作导入（glb / gltf / fbx）要用 `GLTFLoader` / `FBXLoader`。
以前这两个是从 `https://esm.sh/three@0.160.0` 动态 import 的，**断网就导不进动作**。
现在放本地，`anim-import.js` 优先用这里的副本，只有本地加载失败才回落 CDN。

| 文件 | 来源 |
|---|---|
| `loaders/GLTFLoader.js` | `three.js@r160/examples/jsm/loaders/GLTFLoader.js` |
| `loaders/FBXLoader.js` | `three.js@r160/examples/jsm/loaders/FBXLoader.js` |
| `utils/BufferGeometryUtils.js` | GLTFLoader 的依赖（`toTrianglesDrawMode`） |
| `libs/fflate.module.js` | FBXLoader 的依赖（解压 FBX 里的压缩块） |
| `curves/NURBSCurve.js` + `curves/NURBSUtils.js` | FBXLoader 的依赖（NURBS 曲线） |

**文件内容一个字没改**（方便直接对着上游 diff / 版本升级），所以它们里面还是
`import { ... } from 'three'` 这种裸模块名。裸名靠 `studio.html` 里的 **importmap**
解析：

```html
<script type="importmap">
  { "imports": { "three": "./vendor/three/three.module.js" } }
</script>
```

**importmap 必须写在 `<script type="module" src="./entry.js">` 之前**，否则解析不到。
映射目标和 `pose_studio/core.js` 里 `new URL("../vendor/three/three.module.js", import.meta.url)`
指向的是同一个 URL —— 于是 loader 与 viewer 用**同一份 three 实例**，
`instanceof` / 内部状态都不再有跨实例问题。

**升级 three 时**：`three.module.js` 与这几个 jsm 文件的 **r 版本必须一致**
（r160 的 loader 配 r161 的 core 会在 `BufferGeometry` / `Object3D` 的字段上炸）。
