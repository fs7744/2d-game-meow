/**
 * 无浏览器渲染冒烟：用 jsdom 加载真实构建产物 dist/index.html + bundle，
 * 断言空状态、节点卡、生成提交路径。npm run smoke
 */
import { existsSync, readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { JSDOM, VirtualConsole } from 'jsdom'

const here = dirname(fileURLToPath(import.meta.url))
const dist = resolve(here, 'dist')

const calls = []
// 用例可临时注入额外的 meta 字段（模拟后端返回的序列帧统计）
let extraMeta = null
let jobSeq = 0
/**
 * 服务端「磁盘」：模拟 `assets/projects/<id>/project.json`。
 *
 * **每个 boot 一份**（挂在返回的 fetch 上）：全局共享会污染后续用例——
 * 前一个用例往磁盘上写了项目，后一个用例启动时就会拉到它，画布不再是空的，
 * 于是「新建出 X 节点」这类断言全错（而且错得莫名其妙）。
 */
function makeFetch() {
  const serverProjects = new Map()
  let serverCurrent = ''
  // 任务查询是否一直失败（模拟服务端异常）
  let jobsFailMode = false
  /** 服务端「磁盘」上的用户资料库（library/*.json）。**每个 fetch 一份**。 */
  const library = { prompts: [], poses: [], prefs: {} }
  /** 服务端「磁盘」上的角色库（assets/characters/）。**每个 fetch 一份**。 */
  const serverCharacters = new Set()
  // 分配项目 id 的计数器
  let serverSeq = 0
  // 表情批量任务的模拟进度（null = 没有在跑）
  let emotionJob = null
  // 看图反推是否失败（模拟缺权重 / 装错 fork）
  let analyzeFails = false
  // 多姿势批量的模拟进度
  let poseJob = null
  // 内衣底图任务的模拟进度
  let underwearJob = null
  // 图集导出了没有（GET /atlas 用）
  let atlasState = { exists: false }
  // 图集导出任务 / 一键出全套任务
  let atlasJob = null
  let pipeJob = null
  // 多姿势批量的模拟进度
  // 保存时是否报冲突（模拟别的窗口改了同一个项目）
  let saveConflictMode = false
  const fetchFn = async (url, init = {}) => {
    calls.push({ url: String(url), method: init.method || 'GET', body: init.body })
    if (String(url).endsWith('/health')) {
      return { ok: true, json: async () => ({ ok: true, engine: { device: 'Fake GPU 24GB' } }) }
    }
    if (String(url).endsWith('/assets') && (init.method || 'GET') === 'DELETE') {
      return {
        ok: true,
        json: async () => ({ deleted: ['D:////x////out.png'], skipped: [], not_found: [], count: 1 }),
      }
    }
    if (String(url).endsWith('/pose/extract')) {
      return {
        ok: true,
        json: async () => ({
          points: Array.from({ length: 18 }, (_, i) =>
            i === 0 ? [0.6, 0.05] : [0.5, 0.2]),
          joints: 18, azimuth_guess: 90,
        }),
      }
    }
    // ---- 多姿势批量（官方 pose_generation）----
    if (String(url).includes('/characters/') && String(url).endsWith('/poses/generate')) {
      jobSeq += 1
      poseJob = { polls: 0 }
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued', costume: 'Naked',
          total: 4, position: 1, defaults: { steps: 20, cfg: 1, target_size: 1024 },
        }),
      }
    }

    // ---- 一键出全套（编排：内衣 → 姿势 → 表情）----
    if (String(url).includes('/characters/') && String(url).endsWith('/pipeline')) {
      const b = JSON.parse(init.body || '{}')
      jobSeq += 1
      pipeJob = { polls: 0 }
      const steps = []
      if (b.underwear) steps.push('underwear')
      if (b.pose_dir || (b.poses && b.poses.length)) steps.push('poses')
      if (b.emotions && b.emotions.length) steps.push('emotions')
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued', costume: b.costume || 'Naked',
          steps, total: steps.length * 2,
          labels: { underwear: '换白内衣底图', poses: '出多姿势', emotions: '出表情' },
          position: 1,
        }),
      }
    }

    // ---- 内衣底图（官方 remove_clothes）----
    if (String(url).includes('/characters/') && String(url).endsWith('/clothes/underwear')) {
      jobSeq += 1
      underwearJob = { polls: 0 }
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued', costume: 'Underwear',
          position: 1,
          prompt: 'Dress character: White underwear',
          defaults: { target_size: 1024, steps: 20, cfg: 1, costume: 'Underwear' },
        }),
      }
    }

    // ---- 精灵图集导出 ----
    // GET 查状态、POST 导出，**按 method 分开**：不然前端一进页面就把导出跑了
    if (String(url).includes('/characters/') && String(url).endsWith('/atlas')) {
      const method = (init.method || 'GET')
      if (method === 'GET') {
        if (!atlasState.exists) {
          return { ok: true, json: async () => ({ character: 'Alice', exists: false }) }
        }
        return {
          ok: true,
          json: async () => ({
            character: 'Alice', exists: true,
            image: '/files/characters/Alice/Atlas/atlas.png',
            frames: atlasState.frames, animations: atlasState.animations,
            size: atlasState.size, canvas: atlasState.canvas,
            bytes: atlasState.bytes, modified: 1,
          }),
        }
      }
      // **走队列**：提交返回 job_id，真正的结果在 /jobs 里
      jobSeq += 1
      atlasJob = { polls: 0 }
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued', frames: 4, position: 1,
        }),
      }
    }

    // ---- 内衣底图（官方 remove_clothes）----
    if (String(url).includes('/characters/') && String(url).endsWith('/clothes/underwear')) {
      jobSeq += 1
      underwearJob = { polls: 0 }
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued', costume: 'Underwear',
          position: 1,
          prompt: 'Dress character: White underwear',
          defaults: { target_size: 1024, steps: 20, cfg: 1, costume: 'Underwear' },
        }),
      }
    }

    // ---- 换装（官方 Clothes Designer）----
    if (String(url).includes('/characters/') && String(url).endsWith('/clothes/wizard')) {
      return {
        ok: true,
        json: async () => ({
          fields: { top: 'white nurse uniform, mint trim', bottom: 'pleated skirt',
            shoes: 'white shoes', head: 'nurse cap', face: 'goggles' },
          seconds: 21.0, raw: '{}', description: '护士',
        }),
      }
    }
    if (String(url).includes('/characters/') && String(url).endsWith('/clothes/design')) {
      jobSeq += 1
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued', costume: 'Nurse',
          position: 1, defaults: { steps: 25, cfg: 3, target_size: 1024 },
        }),
      }
    }

    // ---- AI 向导：一句话造角色（纯文本，不需要 mmproj）----
    if (String(url).endsWith('/characters/wizard')) {
      const b = JSON.parse(init.body || '{}')
      return {
        ok: true,
        json: async () => ({
          fields: { sex: 'female', age: 25, race: 'elf', skin_color: 'pale skin',
            hair: 'silver hair', eyes: 'red eyes', body: 'slim build, small_breasts' },
          seconds: 40.8, raw: '```json {...} ```',
          description: b.description || '',
        }),
      }
    }

    // ---- 看图反推属性（官方 Cloner 旁路）----
    if (String(url).includes('/characters/') && String(url).endsWith('/analyze')) {
      if (analyzeFails) {
        // 后端把官方错误码塞在 detail 里，前端按码给不同话术
        return { ok: false, status: 503,
          json: async () => ({ detail: { error: 'MMPROJ_MISSING',
            message: '缺视觉投影器：models/llm/Qwen3.5-4B/mmproj-F16.gguf' } }) }
      }
      return {
        ok: true,
        json: async () => ({
          character: 'Alice',
          image: '/files/characters/Alice/Sprites/Naked/neutral/sprite_pose_0001.png',
          fields: { sex: 'female', age: 21, race: 'human',
            hair: 'silver hair, long hair', eyes: 'red eyes', body: 'slim' },
          seconds: 22.4, raw: '```json {...} ```',
        }),
      }
    }

    // ---- 表情批量生成（官方 Emotion Studio：只改脸）----
    //
    // **必须排在角色出图那条前面**：`/emotions/generate` 也以 `/generate` 结尾，
    // 顺序反了会被「角色出图」分支抢走，然后测试怎么都等不到 progress。
    if (String(url).includes('/emotions/detect')) {
      return {
        ok: true,
        json: async () => ({
          character: 'Alice', costume: 'Naked', count: 2,
          items: [
            { index: 1, url: '/files/a.png', size: [512, 768],
              box: [180, 90, 300, 210], detected: true, error: '' },
            // 第二张故意检不出脸：官方此时整图重画，前端要能把这事说出来
            { index: 2, url: '/files/b.png', size: [512, 768],
              box: null, detected: false, error: '' },
          ],
          defaults: { target_size: 2048, bbox_threshold: 0.5, bbox_dilation: 50,
            drop_size: 10, feather: 50, face_denoise: 1 },
          max_crop_side: 1024,
        }),
      }
    }
    if (String(url).includes('/emotions/generate')) {
      jobSeq += 1
      emotionJob = { polls: 0 }
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued',
          total: 4, sources: 2, emotions: ['angry', 'smile'], position: 1,
        }),
      }
    }
    if (String(url).includes('/vnccs/emotions/emotions.json')) {
      return {
        ok: true,
        json: async () => JSON.parse(readFileSync(
          resolve(here, 'public/vnccs/emotions/emotions.json'), 'utf8')),
      }
    }

    // ---- 角色中心（assets/characters/）----
    if (String(url).endsWith('/characters') && (init.method || 'GET') === 'POST') {
      const b = JSON.parse(init.body || '{}')
      serverCharacters.add(b.name)
      return { ok: true, json: async () => ({ name: b.name, config: { character_info: b.info } }) }
    }
    if (String(url).endsWith('/characters') && (init.method || 'GET') === 'GET') {
      return {
        ok: true,
        json: async () => ({
          count: serverCharacters.size,
          items: [...serverCharacters].map((n) => ({
            name: n, cover_url: '', sex: 'female', style: 'ghibli_miyazaki',
            nsfw: false, costumes: ['Naked'], sprites: 0, config_version: '2.0',
          })),
        }),
      }
    }
    // 角色出图：拼 prompt（不跑模型）/ 提交生成
    if (String(url).includes('/characters/') && String(url).endsWith('/prompt')) {
      const b = JSON.parse(init.body || '{}')
      return {
        ok: true,
        json: async () => ({
          name: 'Alice', style: b.style || 'ghibli_miyazaki',
          // 官方结构：每个字段一行
          prompt: 'Draw an image of a character in the following style: Test.\n'
            + 'The image contains one character.\nGender: female\nAge: 18 years old',
          negative: 'bad quality',
          resolution: [792, 1408],
          defaults: { steps: 25, cfg: 3, sampler: 'euler', scheduler: 'simple' },
        }),
      }
    }
    if (String(url).includes('/characters/') && String(url).endsWith('/generate')) {
      jobSeq += 1
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`, status: 'queued',
          prompt: JSON.parse(init.body || '{}').prompt || '', resolution: [792, 1408],
        }),
      }
    }
    // 精灵帧序列：**先 meta 后单帧**（官方 vnccs_common.js:739 的顺序）
    if (String(url).includes('/characters/') && String(url).includes('/poses/meta')) {
      // 注意：这里的 url 是**相对路径**（/characters/...），new URL 必须给 base
      const ps = new URL(String(url), 'http://x').searchParams
      const emo = ps.get('emotion') || 'neutral'
      // 用不同表情给出不同数量，好验证「只有一张时不显示翻页按钮」
      const n = emo === 'angry' ? 1 : 3
      return {
        ok: true,
        json: async () => ({
          count: n,
          files: Array.from({ length: n }, (_, i) => `/files/frame_${emo}_${i}.png`),
        }),
      }
    }
    if (String(url).includes('/characters/') && (init.method || 'GET') === 'GET') {
      const name = decodeURIComponent(String(url).split('/characters/')[1].split('?')[0])
      return {
        ok: true,
        json: async () => ({
          name,
          config: { character_info: {
            name, sex: 'female', age: 18,
            // 官方写的是空格形式，但用户手填常写成下划线；两处都要能认
            eyes: 'blue_eyes, my own trait',
          },
            costumes: {}, costume_ids: {}, config_version: '2.0' },
          costumes: ['Naked'],
          // 有素材才会有可选表情：Neutral 3 张、angry 1 张
          assets: { Naked: {
            neutral: ['/files/a.png', '/files/b.png', '/files/c.png'],
            angry: ['/files/d.png'],
          } },
          cover_url: '',
        }),
      }
    }
    // 静态目录里的官方素材：**读磁盘上的真 json**，分类数量也是真的
    if (String(url).includes('/vnccs/character_template/character_presets_v2.json')) {
      return {
        ok: true,
        json: async () => JSON.parse(readFileSync(
          resolve(here, 'public/vnccs/character_template/character_presets_v2.json'), 'utf8')),
      }
    }
    // ---- 用户资料库（提示词收藏 / 姿势库 / 偏好）----
    //
    // **权威副本在服务端**：前端只在启动时拉一次，之后的增删改都是发请求、
    // 用返回值覆盖本地。mock 因此在 fetch 实例上存一份「磁盘」。
    if (/\/library\/prompts\/[^/]+$/.test(String(url))
        && (init.method || 'GET') === 'DELETE') {
      const id = decodeURIComponent(String(url).split('/').pop())
      library.prompts = library.prompts.filter((p) => p.id !== id)
      return { ok: true, json: async () => ({ prompts: library.prompts }) }
    }
    if (String(url).endsWith('/library/prompts') && (init.method || 'GET') === 'POST') {
      const b = JSON.parse(init.body || '{}')
      const t = String(b.text || '').trim()
      const n = String(b.negative || '').trim()
      const key = `${t}\u0000${n}`
      if (t && !library.prompts.some((p) => `${p.text}\u0000${p.negative}` === key)) {
        library.prompts.unshift({ id: `p${library.prompts.length + 1}`, text: t,
          negative: n, at: Date.now() })
      }
      return { ok: true, json: async () => ({ prompts: library.prompts }) }
    }
    if (/\/library\/poses\/[^/]+$/.test(String(url))
        && (init.method || 'GET') === 'DELETE') {
      const id = decodeURIComponent(String(url).split('/').pop())
      library.poses = library.poses.filter((p) => p.id !== id)
      return { ok: true, json: async () => ({ poses: library.poses }) }
    }
    if (/\/library\/poses\/[^/]+$/.test(String(url))) {           // 改名（PUT）
      const id = decodeURIComponent(String(url).split('/').pop())
      const nm = String(JSON.parse(init.body || '{}').name || '').trim()
      library.poses = library.poses.map((p) => (p.id === id ? { ...p, name: nm } : p))
      return { ok: true, json: async () => ({ poses: library.poses }) }
    }
    if (String(url).endsWith('/library/poses') && (init.method || 'GET') === 'POST') {
      const b = JSON.parse(init.body || '{}')
      const nm = String(b.name || '').trim() || '姿势'
      const hit = library.poses.find((p) => p.name === nm)
      if (hit) Object.assign(hit, { pose: b.pose, yaw: b.yaw, preview: b.preview })
      else {
        library.poses.unshift({ id: `z${library.poses.length + 1}`, name: nm,
          pose: b.pose, yaw: Number(b.yaw) || 0, preview: b.preview || '', at: Date.now() })
      }
      return { ok: true, json: async () => ({ poses: library.poses }) }
    }
    if (String(url).endsWith('/library/prefs') && (init.method || 'GET') === 'PUT') {
      library.prefs = { ...library.prefs, ...JSON.parse(init.body || '{}') }
      return { ok: true, json: async () => ({ prefs: library.prefs }) }
    }
    // 迁移：浏览器里剩下的旧存档（只补不覆盖）
    if (String(url).endsWith('/library/seed')) {
      const b = JSON.parse(init.body || '{}')
      for (const p of (b.prompts || [])) {
        if (!library.prompts.some((x) => x.text === p.text && x.negative === p.negative)) {
          library.prompts.push({ ...p, id: `s${library.prompts.length + 1}` })
        }
      }
      for (const p of (b.poses || [])) library.poses.push(p)
      library.prefs = { ...(b.prefs || {}), ...library.prefs }
      return { ok: true, json: async () => ({ ...library }) }
    }
    if (String(url).endsWith('/library')) {
      return { ok: true, json: async () => ({ ...library }) }
    }
    // 随项目发行的 VNCCS 官方姿势清单（只给路径，不给本体）
    if (String(url).includes('/pose_library')) {
      return {
        ok: true,
        json: async () => ({
          count: 3,
          items: [
            { id: 'a.json', name: 'Arabesque', category: 'Ballet', tags: ['ballet'],
              yaw: 0, json: '/vnccs-pose/PoseLibrary/a.json',
              preview: '/vnccs-pose/PoseLibrary/a.webp' },
            { id: 'b.json', name: 'Crawling', category: 'Crawling',
              tags: ['side view'], yaw: 90, json: '/vnccs-pose/PoseLibrary/b.json',
              preview: '/vnccs-pose/PoseLibrary/b.webp' },
            { id: 'c.json', name: 'Fighting Stance', category: 'Fighting',
              tags: ['rear view'], yaw: 180, json: '/vnccs-pose/PoseLibrary/c.json',
              preview: '' },
          ],
          categories: ['Ballet', 'Crawling', 'Fighting'],
        }),
      }
    }
    // 取一条官方姿势的本体（PoseStudio 原生 json）
    if (String(url).includes('/vnccs-pose/PoseLibrary/') && String(url).endsWith('.json')) {
      return { ok: true, json: async () => ({ bones: { upperarm_r: [10, 20, 30] } }) }
    }
    if (String(url).includes('/upload')) {
      // **按文件名返回不同的磁盘路径**：骨架图与 3D 人偶图要能区分开，
      // 否则「提交的动作参考用的是人偶图」这条断言写不出来（两条 path 相同）。
      const nm = decodeURIComponent(
        /[?&]name=([^&]*)/.exec(String(url))?.[1] || 'abc_pic.png')
      return {
        ok: true,
        json: async () => ({
          path: `D:////x////uploads////${nm}`, url: `/files/uploads/${nm}`,
          w: 256, h: 192, format: 'PNG',
        }),
      }
    }
    if (String(url).endsWith('/generate')) {
      // 每次提交给不同的 id：同 id 会让前端以为旧任务还 done 着、
      // 不再轮询，第二次提交的结果就永远拿不到（这条用例踩过）
      jobSeq += 1
      return { ok: true, json: async () => ({ job_id: `job-${jobSeq}` }) }
    }
    if (String(url).includes('/jobs/')) {
      // 表情批量：第一次查询返回「跑到一半」，第二次返回完成。
      // 进度是这一批任务的**唯一**可见信号，必须能在界面上看到。
      if (atlasJob) {
        atlasJob.polls += 1
        // **三轮才完成**：两轮的话 800ms 就结束了，测试抓不到"打包中"那一帧。
        if (atlasJob.polls < 3) {
          return {
            ok: true,
            json: async () => ({
              job_id: 'job-atlas', status: 'running',
              progress: { done: 1, total: 3, current: '拼接 4 帧' },
            }),
          }
        }
        atlasState = {
          exists: true, frames: 4, animations: 2,
          size: [192, 288], canvas: [96, 144], bytes: 24 * 1024,
        }
        return {
          ok: true,
          json: async () => ({
            job_id: 'job-atlas', status: 'done',
            progress: { done: 3, total: 3, current: '' },
            result: { frames: 4, animations: 2, size: [192, 288],
              canvas: [96, 144], bytes: 24 * 1024 },
          }),
        }
      }
      if (pipeJob) {
        pipeJob.polls += 1
        if (pipeJob.polls < 3) {
          return {
            ok: true,
            json: async () => ({
              job_id: 'job-pipe', status: 'running',
              progress: { done: 2, total: 6, step: 'poses',
                current: '出多姿势：第 2/4 张姿势' },
            }),
          }
        }
        return {
          ok: true,
          json: async () => ({
            job_id: 'job-pipe', status: 'done',
            progress: { done: 6, total: 6, step: '', current: '' },
            result: { steps: ['underwear', 'poses', 'emotions'], poses: { count: 4 } },
          }),
        }
      }
      if (underwearJob) {
        underwearJob.polls += 1
        if (underwearJob.polls < 2) {
          return {
            ok: true,
            json: async () => ({
              job_id: 'job-uw', status: 'running',
              progress: { done: 0, total: 1, current: '换内衣底图中' },
            }),
          }
        }
        return {
          ok: true,
          json: async () => ({
            job_id: 'job-uw', status: 'done',
            progress: { done: 1, total: 1, current: '' },
            result: { costume: 'Underwear', sprite: 'x' },
          }),
        }
      }
      if (poseJob) {
        poseJob.polls += 1
        if (poseJob.polls < 2) {
          return {
            ok: true,
            json: async () => ({
              job_id: 'job-pose', status: 'running',
              progress: { done: 1, total: 4, current: '第 2/4 张姿势' },
            }),
          }
        }
        return {
          ok: true,
          json: async () => ({
            job_id: 'job-pose', status: 'done',
            progress: { done: 4, total: 4, current: '' },
            result: { count: 4, sprites: ['a', 'b', 'c', 'd'], failed: [] },
          }),
        }
      }
      if (emotionJob) {
        emotionJob.polls += 1
        if (emotionJob.polls < 2) {
          return {
            ok: true,
            json: async () => ({
              job_id: 'job-emo', status: 'running',
              progress: { done: 2, total: 4, current: 'smile / 第 1 帧' },
            }),
          }
        }
        return {
          ok: true,
          json: async () => ({
            job_id: 'job-emo', status: 'done',
            progress: { done: 4, total: 4, current: '' },
            result: {
              count: 4, total: 4, failed: [],
              items: [{ index: 1, emotion: 'angry' }, { index: 2, emotion: 'smile' }],
            },
          }),
        }
      }
      // **服务端异常模式**：模拟任务查询一直失败（服务挂了 / 网络断）。
      // 节点必须能自己从「执行中」退出来，不能一直转圈。
      if (jobsFailMode) {
        return { ok: false, status: 500, json: async () => ({ detail: '服务端内部错误' }) }
      }
      return {
        ok: true,
        json: async () => ({
          job_id: `job-${jobSeq}`,
          status: 'done',
          result: {
            file: 'D:\\x\\out.png',
            // extraMeta 由用例临时注入，模拟序列帧结果里的候选/闸门统计
            meta: { seed: 42, seconds: 1.5, peak_vram_gb: 2.6, ...(extraMeta || {}) },
          },
        }),
      }
    }
    if (String(url).endsWith('/train') && (init.method || 'GET') === 'POST') {
      return { ok: true, json: async () => ({ train_id: 't1' }) }
    }
    if (String(url).includes('/train/t1')) {
      return {
        ok: true,
        json: async () => ({
          train_id: 't1', status: 'running', stage: 'train',
          step: 20, total: 100, loss: 0.0512, elapsed: 8.4,
          peak_vram_gb: 4.67, trainable_params: 1594368, name: 'my_lora',
        }),
      }
    }
    if (String(url).endsWith('/models')) {
      return {
        ok: true,
        json: async () => ({
          current: 'sd15',
          available: ['sd15', 'sdxl', 'sdxl-anime'],
          // sdxl-anime 的架构是 sdxl：LoRA 过滤必须按架构比，按名字比会把
          // 挂在 sdxl-anime 上的 SDXL LoRA 全过滤掉
          arch: { sd15: 'sd15', sdxl: 'sdxl', 'sdxl-anime': 'sdxl' },
          refiner: true,
          // 与后端 Engine.RECOMMEND 一致：SDXL 推荐 768/24，不是 1024
          recommend: {
            sd15: { width: 768, height: 768, steps: 30, cfg: 7.0, pixel: 6, palette: 32,
              max_candidates: 6 },
            sdxl: { width: 768, height: 768, steps: 24, cfg: 6.5, pixel: 6, palette: 32,
              max_candidates: 3 },
            'sdxl-anime': { width: 1024, height: 1024, steps: 28, cfg: 8.0, pixel: 0,
              palette: 0, clip_skip: 2, preset: 'anime', max_candidates: 3 },
          },
        }),
      }
    }
    if (String(url).endsWith('/loras/scan')) {
      return {
        ok: true,
        json: async () => ({
          message: '扫描 13 个权重\n不可用（已隐藏）：octopath_traverl, wan_t2v_h_unipixel-e6',
          count: 7,
          added: ['octopath_traverl', 'pixel-art-xl'],
          filled: [],
          hidden: ['octopath_traverl', 'wan_t2v_h_unipixel-e6'],
          duplicates: [],
          stale: [],
          total: 13,
          written: true,
        }),
      }
    }
    if (String(url).endsWith('/loras/apply')) {
      const items = JSON.parse(init.body || '{}').items || []
      return {
        ok: true,
        json: async () => ({
          active: items.map((x) => `qwen_${x.name}`),
          items: items.map((x) => ({ name: x.name, scale: x.scale, adapter: `qwen_${x.name}` })),
        }),
      }
    }
    if (String(url).endsWith('/loras/unload')) {
      return { ok: true, json: async () => ({ active: [], items: [] }) }
    }
    if (String(url).endsWith('/fast_mode')) {
      const on = JSON.parse(init.body || '{}').on === true
      return {
        ok: true,
        json: async () => ({ fast_mode: on, lora: on ? 'qwen21_lightning' : null,
          max_steps: on ? 8 : null }),
      }
    }
    if (String(url).endsWith('/loras')) {
      // Qwen LoRA：只列 diffusers 原生格式的（engine.available_loras）
      return {
        ok: true,
        json: async () => ({
          items: [
            { name: 'qwen21_anyangle', file: 'models/lora/qwen21_anyangle.safetensors',
              size_mb: 114, tensors: 384 },
            { name: 'qwen21_lightning', file: 'models/lora/qwen21_lightning.safetensors',
              size_mb: 64, tensors: 256 },
          ],
          active: [],
        }),
      }
    }
    if (String(url).endsWith('/pose_dirs')) {
      return {
        ok: true,
        json: async () => ({
          count: 2,
          items: [
            { path: 'D:////x////assets////_pose', count: 4, sample: ['_pose_walk_00.png'],
              name: '_pose' },
            { path: 'D:////x////assets////pose_video', count: 8, sample: ['f00.png'], name: 'pose_video' },
          ],
        }),
      }
    }
    if (String(url).startsWith('/files/')) {
      return { ok: true, json: async () => ({}) }
    }
    // ---- 项目导入 / 导出 / 模板 ----
    // **顺序要紧**：`/projects/export` 与 `/projects/import` 也含 `/projects/`，
    // 必须先判掉，否则会被下面的「按 id 读项目」分支抢走。
    if (String(url).endsWith('/projects/export')) {
      return { ok: true, blob: async () => new Blob([new Uint8Array([80, 75, 3, 4])]) }
    }
    if (String(url).endsWith('/projects/import')) {
      return {
        ok: true,
        json: async () => ({
          meta: { name: '导入的项目' },
          payload: { nodes: [{ id: 'in1', kind: 'gen', x: 0, y: 0, view: 'expanded',
            params: { prompt: '导入节点' } }], viewport: { x: 0, y: 0, z: 1 } },
          imported_files: 3,
        }),
      }
    }
    // ---- 打开项目文件夹 ----
    // 必须排在「按 id 读项目」前面：`/projects/<id>/reveal` 也含 `/projects/`
    if (/\/projects\/[^/]+\/reveal$/.test(String(url))) {
      return {
        ok: true,
        json: async () => ({ opened: 'D:/repo/assets/projects/pdemo' }),
      }
    }
    // ---- 项目落盘存储（服务端磁盘）----
    // **分配 id**：服务端保证唯一（随机后缀 + 存在性检查）。
    // 必须排在 `/projects` 列表分支前面吗？——`/projects/allocate` 不以
    // `/projects` 结尾，所以不冲突；但为了可读性放在这里。
    if (String(url).endsWith('/projects/allocate')) {
      let id
      do {
        id = `pm${(serverSeq++).toString(36)}${Math.random().toString(36).slice(2, 8)}`
      } while (serverProjects.has(id))
      return { ok: true, json: async () => ({ id }) }
    }
    if (String(url).endsWith('/projects') && (init.method || 'GET') === 'GET') {
      return {
        ok: true,
        json: async () => ({
          count: serverProjects.size,
          items: [...serverProjects.values()].map((p) => ({
            id: p.id, name: p.name, created: p.created, updated: p.updated,
            count: (p.nodes || []).length,
          })),
          current: serverCurrent,
        }),
      }
    }
    if (String(url).endsWith('/workspace')) {
      return { ok: true, json: async () => ({ ok: true }) }
    }
    if (String(url).includes('/projects/') && (init.method || 'GET') === 'PUT') {
      const id = decodeURIComponent(String(url).split('/projects/')[1].split('?')[0])
      const doc = JSON.parse(init.body || '{}')
      // **冲突模式**：模拟别的窗口改了同一个项目 → 409
      if (saveConflictMode) {
        return { ok: false, status: 409,
          json: async () => ({ detail: '项目已被其他窗口修改' }) }
      }
      serverProjects.set(id, { ...doc, id })
      return { ok: true, json: async () => ({ id, name: doc.name, count: (doc.nodes || []).length }) }
    }
    if (String(url).includes('/projects/') && (init.method || 'GET') === 'GET') {
      const id = decodeURIComponent(String(url).split('/projects/')[1].split('?')[0])
      const doc = serverProjects.get(id)
      if (!doc) return { ok: false, status: 404, json: async () => ({}) }
      return { ok: true, json: async () => doc }
    }
    if (String(url).includes('/projects/') && (init.method || 'GET') === 'DELETE') {
      const id = decodeURIComponent(String(url).split('/projects/')[1].split('?')[0])
      serverProjects.delete(id)
      return { ok: true, json: async () => ({ deleted: id }) }
    }
    if (String(url).endsWith('/templates') && (init.method || 'GET') === 'GET') {
      return {
        ok: true,
        json: async () => ({
          count: 1,
          items: [{ id: 'pixel_line', name: '像素精灵产线', created: 1, bytes: 2048, nodes: 3 }],
        }),
      }
    }
    if (String(url).endsWith('/templates') && (init.method || 'GET') === 'POST') {
      return { ok: true, json: async () => ({ id: 'pixel_line', name: '像素精灵产线' }) }
    }
    if (String(url).includes('/templates/') && String(url).includes('/instantiate')) {
      return {
        ok: true,
        json: async () => ({
          meta: { name: '像素精灵产线' },
          payload: { nodes: [{ id: 'tn1', kind: 'gen', x: 0, y: 0, view: 'expanded',
            params: { prompt: '模板节点' } }], viewport: { x: 0, y: 0, z: 1 } },
          imported_files: 2,
        }),
      }
    }
    if (String(url).includes('/templates/') && (init.method || 'GET') === 'DELETE') {
      return { ok: true, json: async () => ({ deleted: 'x' }) }
    }
    return { ok: false, status: 404, json: async () => ({}) }
  }
  // 让用例能检查「服务端磁盘」上的东西
  fetchFn.serverProjects = serverProjects
  fetchFn.setServerCurrent = (v) => { serverCurrent = v }
  fetchFn.setJobsFail = (v) => { jobsFailMode = v }
  fetchFn.setSaveConflict = (v) => { saveConflictMode = v }
  fetchFn.setAnalyzeFail = (v) => { analyzeFails = v }
  return fetchFn
}

async function boot(search, seed) {
  const html = readFileSync(resolve(dist, 'index.html'), 'utf8')
  const vc = new VirtualConsole()
  const errors = []
  vc.on('jsdomError', (e) => {
    const msg = String(e)
    if (msg.includes('Could not load link')) return   // jsdom 无静态服务器，CSS 拉不到属正常
    // 同理：PoseStudio 宿主 iframe 也拉不到（真实浏览器里有 vite 静态服务）
    if (msg.includes('Could not load iframe')) return
    errors.push(msg)
  })
  const dom = new JSDOM(html, {
    url: `http://localhost/${search}`,
    runScripts: 'dangerously',
    resources: 'usable',
    pretendToBeVisual: true,
    virtualConsole: vc,
    beforeParse(win) {
      win.fetch = makeFetch()
      win.matchMedia = () => ({ matches: false, addListener() {}, removeListener() {} })
      // 记录捕获调用：卡头拖拽会 setPointerCapture，按钮上则不应调用。
      // 真实浏览器里正是这个捕获把 click 重定向到卡头，导致按钮点不动。
      win.__captured = []
      win.HTMLElement.prototype.setPointerCapture = () => win.__captured.push(1)
      win.HTMLElement.prototype.releasePointerCapture = () => {}
      win.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    },
  })

  // 老存档迁移测试用：先塞一份老格式的 localStorage
  if (seed) {
    for (const [k, v] of Object.entries(seed)) {
      dom.window.localStorage.setItem(k, v)
    }
  }

  // 手动注入 bundle（jsdom 的 resources:usable 在本地文件下不可靠）
  const bundle = readFileSync(resolve(dist, 'index.html'), 'utf8')
    .match(/src="([^"]+\.js)"/)[1]
  const code = readFileSync(resolve(dist, bundle.replace(/^\//, '')), 'utf8')
  const script = dom.window.document.createElement('script')
  script.textContent = code
  dom.window.document.body.appendChild(script)
  await new Promise((r) => setTimeout(r, 600))
  return { dom, errors, fetch: dom.window.fetch }
}

/** 按分组标题找侧栏分组（分组会增减，别用下标写死） */
function section(doc, title) {
  return [...doc.querySelectorAll('.rail-sec')].find(
    (sec) => (sec.querySelector('.rail-sec-title')?.textContent || '') === title,
  )
}

const results = []
function check(name, cond, extra = '') {
  results.push({ name, ok: !!cond, extra })
  console.log(`${cond ? '✓' : '✗'} ${name}${extra ? '  ' + extra : ''}`)
}

// 1) 空状态
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  check('渲染出顶栏品牌', doc.querySelector('.brand .title')?.textContent?.includes('喵助手'))
  check('空状态引导可见', !!doc.querySelector('.canvas-empty .empty-card'))
  // 不写死数量：加节点类型时忘改这里会误报。只断言「有人偶姿势」这项入口
  check('侧栏存在（节点分组含骨骼编辑）',
    !!section(doc, '节点')?.querySelectorAll('.rail-item').length
    && [...section(doc, '节点').querySelectorAll('.rail-item')]
      
      .some((x) => (x.textContent || '').includes('人偶姿势')),
    String(section(doc, '节点')?.querySelectorAll('.rail-item').length))
  check('有「项目」分组（独立画布）', !!section(doc, '项目'))
  check('缩放显示 100%', doc.querySelector('.zoom-val')?.textContent === '100%')
  check('无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 2) demo 节点 + 自动生成
{
  calls.length = 0
  const { dom, errors } = await boot('?demo=gen')
  const win = dom.window
  const doc = win.document
  await new Promise((r) => setTimeout(r, 1200))
  check('生成卡渲染', !!doc.querySelector('.node.gen'))
  check('卡片标题是「图生成」', doc.querySelector('.node-kind')?.textContent === '图生成')
  check('提示词已填入', (doc.querySelector('.prompt')?.value || '').includes('猫骑士'))
  const genCall = calls.find((c) => c.url.endsWith('/generate'))
  check('提交了 w1 任务', !!genCall && genCall.method === 'POST')
  if (genCall) {
    const body = JSON.parse(genCall.body)
    check('workflow=w1', body.workflow === 'w1', body.workflow)
    check('参数带 pixel/palette', body.params.pixel === 6 && body.params.palette === 32,
        JSON.stringify({ p: body.params.pixel, c: body.params.palette }))
  }
  check('轮询了任务状态', calls.some((c) => c.url.includes('/jobs/')))
  check('出图后显示预览图', !!doc.querySelector('.preview img, .node-image-only img'))

  // ---- 提示词放大编辑 + 尺寸预设（2026-11-02） ----
  // 用户反馈：节点里的 prompt 只有两三行高，写长描述看不全；尺寸还得自己
  // 记「768 是精灵甜点」「竖版立绘填什么」。
  const zoomBtn = doc.querySelector('.prompt-wrap .pzoom')
  check('提示词旁有放大编辑按钮', !!zoomBtn, zoomBtn?.title)
  zoomBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  const zoomArea = doc.querySelector('.pzoom-area')
  check('点开后出现放大编辑弹窗', !!zoomArea)
  check('弹窗里带的就是这段提示词',
    (zoomArea?.value || '').includes('猫骑士'), (zoomArea?.value || '').slice(0, 40))
  check('弹窗显示字数统计',
    /字/.test(doc.querySelector('.pzoom-meta')?.textContent || ''),
    doc.querySelector('.pzoom-meta')?.textContent)

  // 在弹窗里改 → 原输入框要跟着变（受控组件，不能各改各的）
  Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value')
    .set.call(zoomArea, '白色猫骑士，蓝披风')
  zoomArea?.dispatchEvent(new win.Event('input', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  check('弹窗里改了就同步回原输入框',
    (doc.querySelector('.prompt')?.value || '') === '白色猫骑士，蓝披风',
    doc.querySelector('.prompt')?.value)
  ;[...doc.querySelectorAll('.pzoom-head button')]
    .find((b) => (b.textContent || '').includes('完成'))
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  check('点「完成」关闭弹窗', !doc.querySelector('.pzoom-mask'))

  const sizeSel = doc.querySelector('.size-preset select')
  const sizeVals = [...(sizeSel?.options || [])].map((o) => o.value)
  check('宽高旁边有尺寸预设下拉', !!sizeSel, sizeVals.join(','))
  check('尺寸预设含常见档位（方形 / 竖 2:3 / 横 3:2）',
    ['1024x1024', '768x768', '832x1216', '1216x832'].every((k) => sizeVals.includes(k)),
    sizeVals.join(','))
  Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value')
    .set.call(sizeSel, '832x1216')
  sizeSel?.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const numIns = [...doc.querySelectorAll('.row input[type=number]')]
  check('选了竖版预设后宽高跟着变',
    numIns[0]?.value === '832' && numIns[1]?.value === '1216',
    `${numIns[0]?.value}×${numIns[1]?.value}`)
  check('下拉自身回显为选中那一档', sizeSel?.value === '832x1216', sizeSel?.value)

  check('无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


// 2b) 服务端异常：节点不能永远卡在「执行中」
{
  const { dom, errors, fetch: f } = await boot('?demo=gen')
  const doc = dom.window.document
  // 让任务查询一直失败（模拟后端挂掉 / 网络断）
  f.setJobsFail(true)
  await new Promise((r) => setTimeout(r, 1200))

  const btn = doc.querySelector('.gen-btn')
  check('节点有生成按钮', !!btn)
  // 提交成功后节点进入「生成中…」
  const busyText = () => (btn?.textContent || '')
  await new Promise((r) => setTimeout(r, 3000))

  // **关键断言**：查询失败 3 次后（约 2 秒）节点必须自己退出「执行中」
  check('查询失败后节点不再显示「生成中…」',
    !busyText().includes('生成中'), busyText())
  check('节点不再显示「排队中…」',
    !busyText().includes('排队中'), busyText())
  check('按钮恢复可点（不再 disabled）', btn && !btn.disabled, String(btn?.disabled))
  check('节点上有失败提示', (doc.querySelector('.node')?.textContent || '').includes('失败')
    || (doc.querySelector('.node')?.textContent || '').includes('无法查询'),
    (doc.querySelector('.node')?.textContent || '').slice(0, 120))

  check('服务端异常场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 3) 交互：只看图切换 / 复制 / 加号新建 / 删除
{
  const { dom, errors } = await boot('?demo=gen')
  const doc = dom.window.document
  await new Promise((r) => setTimeout(r, 1400))

  const toggle = doc.querySelector('.icon-btn.view-toggle')
  check('存在「只看图」按钮', !!toggle && (toggle.textContent || '').includes('▣'))
  toggle?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('切换为只展示图', !!doc.querySelector('.node-image-only'))

  // 切回按钮的文本是「▤ 参数」，用 class 定位而非精确匹配符号
  const back = doc.querySelector('.icon-btn.view-toggle')
  back?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('切回参数视图', !!doc.querySelector('.node .prompt'))

  const copyBtn = [...doc.querySelectorAll('.icon-btn')].find((b) => (b.getAttribute('title') || '').includes('复制'))
  copyBtn?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('复制后 2 个节点', doc.querySelectorAll('.node').length === 2,
        String(doc.querySelectorAll('.node').length))

  const addBtn = doc.querySelector('.rail-add')
  addBtn?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('加号新建后 3 个节点', doc.querySelectorAll('.node').length === 3,
        String(doc.querySelectorAll('.node').length))

  const delBtn = [...doc.querySelectorAll('.icon-btn')].find((b) => b.textContent === '✕')
  delBtn?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('删除先弹确认框', !!doc.querySelector('.modal-mask'))
  check('确认框有三个出口', doc.querySelectorAll('.modal-actions button').length === 3,
        String(doc.querySelectorAll('.modal-actions button').length))
  // 「仅移除节点」：只从画布拿掉，不动磁盘
  const onlyRemove = [...doc.querySelectorAll('.modal-actions button')]
    .find((b) => (b.textContent || '').includes('仅移除'))
  onlyRemove?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('仅移除后 2 个节点', doc.querySelectorAll('.node').length === 2,
        String(doc.querySelectorAll('.node').length))
  check('仅移除不调用删除接口', !calls.some((c) => c.method === 'DELETE'))

  // 键盘删除：先选中一个节点，再按 Delete -> 也弹确认框
  const node = doc.querySelector('.node')
  node?.dispatchEvent(new dom.window.Event('pointerdown', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 40))
  dom.window.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Delete', bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('Delete 键也弹确认框', !!doc.querySelector('.modal-mask'))
  const cancel = [...doc.querySelectorAll('.modal-actions button')]
    .find((b) => (b.textContent || '').includes('取消'))
  cancel?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('取消后节点还在', doc.querySelectorAll('.node').length === 2,
        String(doc.querySelectorAll('.node').length))

  check('交互场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


// 4) 从节点列表拖拽到画布：落点换算成世界坐标
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const viewport = doc.querySelector('.canvas-viewport')
  const ev = new dom.window.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 400 },
    clientY: { value: 300 },
    dataTransfer: {
      value: { types: ['application/x-node-type'], getData: () => 'gen' },
    },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 120))
  const node = doc.querySelector('.node')
  check('拖拽后新建节点', !!node)
  // 默认视口 {x:120,y:90,z:1}，jsdom 里 canvas 矩形为 0 点
  check('落点换算正确 (280,210)', node?.style.left === '280px' && node?.style.top === '210px',
        `${node?.style.left},${node?.style.top}`)
  check('拖拽无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


// 5) 语言提示 + 提示词收藏
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))

  // 受控 textarea：必须走原生 setter + input 事件（下面的断言要用，先定义）
  const ta = doc.querySelector('textarea.prompt')
  const setValue = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  // demo 画布里已经有中文提示词，所以先清空才能看到"空提示"那一版文案
  setValue(ta, '')
  await new Promise((r) => setTimeout(r, 80))
  check('空提示时说明中英都行',
        (doc.querySelector('.lang-hint')?.textContent || '').includes('中英文都行'))

  setValue(ta, '背剑侠客')
  await new Promise((r) => setTimeout(r, 80))
  // Qwen 的文本编码器是 Qwen3-VL，中文可以直接写。SD 时代这里会转红告警
  // （「会被拆成乱 token」），那条断言随 SD 管线一起作废了。
  check('中文不告警', !doc.querySelector('.lang-hint.bad'))
  check('中文提示说明可直接写',
        (doc.querySelector('.lang-hint')?.textContent || '').includes('中文没问题'))

  setValue(ta, 'white cat knight, red cape, front view')
  await new Promise((r) => setTimeout(r, 80))
  check('英文不告警', !doc.querySelector('.lang-hint.bad'))

  // 负向词在「更多设置」里（jsdom 不裁剪关闭的 details，能直接取到）
  const negTa = doc.querySelector('details.fold textarea')
  check('找到反向提示词输入框', !!negTa)
  setValue(negTa, 'watermark, low quality')
  await new Promise((r) => setTimeout(r, 80))

  const star = doc.querySelector('.star')
  check('星标可用', !!star && !star.disabled)
  star.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  const favItems = () => section(doc, '资料库').querySelectorAll('.rail-item')
  check('收藏后出现在列表', favItems().length === 1)
  check('星标点亮', !!doc.querySelector('.star.on'))
  check('列表文本正确',
    (favItems()[0]?.querySelector('.rail-item-main')?.textContent || '')
      .includes('white cat knight'))
  // 负向词必须跟着一起收、一起显示（缩成第二行灰字）
  check('收藏项带上负向词',
    (favItems()[0]?.querySelector('.rail-item-main em')?.textContent || '')
      .includes('watermark'),
    favItems()[0]?.querySelector('.rail-item-main em')?.textContent || '（无）')

  // 正向与负向都改掉，再点收藏条目 —— 两项都必须被覆盖回来
  setValue(ta, 'blue slime')
  setValue(negTa, 'changed by test')
  await new Promise((r) => setTimeout(r, 80))
  doc.querySelector('.node').dispatchEvent(new win.Event('pointerdown', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  favItems()[0].querySelector('.rail-item-main').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('点收藏条目回填到节点',
    doc.querySelector('textarea.prompt').value.includes('white cat knight'),
    doc.querySelector('textarea.prompt').value)
  check('点收藏条目同时回填负向词',
    (doc.querySelector('details.fold textarea')?.value || '').includes('watermark'),
    doc.querySelector('details.fold textarea')?.value || '（空）')

  // 重复收藏不应出现两条
  doc.querySelector('.star').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('重复收藏去重（已收藏则取消）', favItems().length === 0, String(favItems().length))

  // 再收藏一次后删除
  doc.querySelector('.star').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  section(doc, '资料库').querySelector('.rail-del')
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('删除收藏', favItems().length === 0)

  check('收藏场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


// 6) 视图切换的根因回归：卡头上的按钮不能被 pointer capture 吃掉
{
  const { dom, errors } = await boot('?demo=gen')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 1400))

  const btn = () => doc.querySelector('.icon-btn.view-toggle')
  const nodeLeft = () => doc.querySelector('.node').style.left

  // 切到图片模式（这是一次普通 click，不经过 pointerdown）
  const left0 = nodeLeft()
  btn().dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('图片模式：只剩画面', !!doc.querySelector('.node-image-only'))

  // 关键：真实浏览器是 pointerdown → setPointerCapture → pointerup → click，
  // capture 会把 click 派发到卡头而非按钮。这里模拟该序列并断言按钮仍可点。
  const capturedBefore = (win.__captured || []).length
  const b = btn()
  b.dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  b.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true, button: 0 }))
  b.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))

  check('按钮上 pointerdown 不触发拖拽捕获',
    (win.__captured || []).length === capturedBefore,
    `captured ${capturedBefore} -> ${(win.__captured || []).length}`)
  check('图片模式可切回参数视图', !!doc.querySelector('.node .prompt'))
  check('切回后节点未被拖走', nodeLeft() === left0, `${left0} -> ${nodeLeft()}`)

  // 卡头空白处仍应能拖拽
  const head = doc.querySelector('.node-head')
  head.dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  check('卡头空白处仍可拖拽', (win.__captured || []).length === capturedBefore + 1)

  check('视图切换场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 7) 与 8) 原本是 LoRA 下拉 / 触发词 / 训练面板的用例。
// **已随 SD 一起删除**（提交 e92775e）：Qwen 不吃 SD LoRA，那两块 UI 也没了。
// 用例一起删掉，别留一组永远红的断言。

// 9) 画布平移：空白处直接拖（曾经因为 .canvas-grid 是 pointer-events:none 而完全拖不动）
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))

  const vp = doc.querySelector('.canvas-viewport')
  const world = () => doc.querySelector('.canvas-world')
  const before = world().style.transform

  const ptr = (el, type, x, y) => {
    const ev = new win.Event(type, { bubbles: true })
    Object.defineProperties(ev, {
      clientX: { value: x }, clientY: { value: y },
      button: { value: 0 }, pointerId: { value: 1 },
    })
    el.dispatchEvent(ev)
  }

  // target 就是 viewport 本身（网格层 pointer-events:none，点空白必然是它）
  ptr(vp, 'pointerdown', 500, 400)
  ptr(vp, 'pointermove', 560, 450)
  await new Promise((r) => setTimeout(r, 80))
  const after = world().style.transform
  check('拖空白可平移', before !== after, `${before} -> ${after}`)
  check('平移量正确 (+60,+50)', /translate\(180px, 140px\)/.test(after), after)

  ptr(vp, 'pointerup', 560, 450)
  await new Promise((r) => setTimeout(r, 60))

  // 节点上拖不能平移（那是移动节点）
  const node = doc.querySelector('.node')
  const before2 = world().style.transform
  ptr(node, 'pointerdown', 300, 300)
  ptr(node, 'pointermove', 400, 400)
  await new Promise((r) => setTimeout(r, 80))
  check('拖节点不平移画布', world().style.transform === before2, world().style.transform)
  ptr(node, 'pointerup', 400, 400)

  check('平移场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


// 10) 左边栏：菜单层级 + 收起/展开 + 底模设置
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))

  check('侧栏存在', !!doc.querySelector('.rail'))
  const titles = [...doc.querySelectorAll('.rail-sec-title')].map((x) => x.textContent)
  // 底模选择已移到每个节点自己的卡头上，侧栏不再有「底模」分组；
  // 「LoRA」是新增的（Qwen 现在支持 LoRA，见 engine.available_loras）；
  // 「模板」是项目预设模板区（存模板 / 从模板新建）。
  check('六个菜单分组：项目/模板/节点/资料库/LoRA/设置',
    JSON.stringify(titles) === JSON.stringify(['项目', '模板', '节点', '资料库', 'LoRA', '设置']),
    JSON.stringify(titles))

  // 分组可折叠
  const head = doc.querySelector('.rail-sec-head')
  head.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('分组可折叠', !doc.querySelector('.rail-sec').classList.contains('open'))
  head.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('分组可展开', doc.querySelector('.rail-sec').classList.contains('open'))

  // 整条收起 / 展开
  doc.querySelector('.rail-toggle').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 100))
  check('侧栏可收起', !!doc.querySelector('.rail.collapsed'))
  doc.querySelector('.rail-toggle').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 100))
  check('侧栏可展开', !doc.querySelector('.rail.collapsed'))

  // 底模已经下放到**节点卡头**，侧栏不再有「底模」分组
  doc.querySelector('details.fold').open = true
  await new Promise((r) => setTimeout(r, 80))
  check('侧栏不再有「底模」分组（每个节点自己选）', !section(doc, '底模'), JSON.stringify(titles))
  const nodeModelSel = doc.querySelector('.node-model select')
  check('节点卡头上有自己的底模下拉', !!nodeModelSel)
  // 不写死数量：mock 里加一个底模就会误报。只断言"列出来的都是本机已装的"
  check('节点底模下拉列出本机已装的',
    !!nodeModelSel && [...nodeModelSel.options]
      .filter((o) => o.value).every((o) => ['sd15', 'sdxl', 'sdxl-anime', 'qwen-image-2.1'].includes(o.value)),
    String(nodeModelSel?.options.length))

  const setSelect = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSelect(nodeModelSel, 'sdxl')
  await new Promise((r) => setTimeout(r, 300))

  check('侧栏场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 11a) 项目：导入 / 导出 / 预设模板
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))

  // ---- 模板列表（后端 GET /templates）----
  const tplSec = [...doc.querySelectorAll('.rail-sec')]
    .find((s) => s.querySelector('.rail-sec-title')?.textContent === '模板')
  check('侧栏有「模板」分组', !!tplSec)
  check('模板列表显示模板名', (tplSec?.textContent || '').includes('像素精灵产线'),
    (tplSec?.textContent || '').slice(0, 60))
  check('模板列表显示节点数', (tplSec?.textContent || '').includes('3 个节点'),
    (tplSec?.textContent || '').slice(0, 80))

  // ---- 从模板新建项目 ----
  const before = doc.querySelectorAll('.project-item').length
  const tplBtn = [...(tplSec?.querySelectorAll('.rail-item-main') || [])][0]
  tplBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  const instCall = calls.find((c) => c.url.includes('/instantiate'))
  check('从模板新建调用了 /instantiate', !!instCall,
    JSON.stringify(calls.slice(-3).map((c) => c.url)))
  check('实例化请求带了新项目 id',
    !!instCall && JSON.parse(instCall.body || '{}').scope?.length > 0,
    instCall?.body || '')
  check('导入后多了一个项目', doc.querySelectorAll('.project-item').length === before + 1,
    `${before} -> ${doc.querySelectorAll('.project-item').length}`)

  // ---- 打开项目文件夹 ----
  // 按钮是 ⌂，点了要 POST /projects/<id>/reveal（后端用 explorer/open/xdg-open 唤起）
  const revealBtn = doc.querySelector('.project-item .rail-act[title*="文件夹"]')
  check('项目行有「打开文件夹」按钮', !!revealBtn)
  const revealBefore = calls.length
  revealBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  const revealCall = calls.slice(revealBefore).find((c) => /\/reveal$/.test(c.url))
  check('打开文件夹调用了 POST /projects/<id>/reveal',
    !!revealCall && revealCall.method === 'POST',
    revealCall ? `${revealCall.method} ${revealCall.url}`
      : JSON.stringify(calls.slice(-3).map((c) => `${c.method} ${c.url}`)))
  check('打开文件夹后给了提示',
    (doc.querySelector('.toast')?.textContent || '').includes('已打开'),
    doc.querySelector('.toast')?.textContent || '（无 toast）')

  // ---- 导出项目 ----
  const dlBtn = doc.querySelector('.project-item .rail-act[title*="导出"]')
  check('项目行有「导出」按钮', !!dlBtn)
  dlBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))
  const expCall = calls.find((c) => c.url.includes('/projects/export'))
  check('导出调用了 /projects/export', !!expCall,
    JSON.stringify(calls.slice(-3).map((c) => `${c.method} ${c.url}`)))
  const expBody = expCall ? JSON.parse(expCall.body || '{}') : {}
  check('导出请求带了 meta（项目名）', typeof expBody.meta?.name === 'string',
    JSON.stringify(expBody.meta || {}))
  check('导出请求带了 payload（节点）', Array.isArray(expBody.payload?.nodes),
    JSON.stringify(Object.keys(expBody.payload || {})))

  // ---- 存模板 ----
  const starBtn = doc.querySelector('.project-item .rail-act[title*="模板"]')
  check('项目行有「存为模板」按钮', !!starBtn)
  starBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))
  check('存模板调用了 POST /templates',
    calls.some((c) => c.url.endsWith('/templates') && c.method === 'POST'),
    JSON.stringify(calls.slice(-3).map((c) => `${c.method} ${c.url}`)))

  // ---- 导入项目（文件选择）----
  const importBtn = [...doc.querySelectorAll('.rail-new')]
    .find((b) => (b.textContent || '').includes('导入项目'))
  check('侧栏有「导入项目」按钮', !!importBtn)
  const fileInput = doc.querySelector('input[type="file"][accept=".mproj"]')
  check('有 .mproj 文件选择框', !!fileInput)
  if (fileInput) {
    // **jsdom 没有 DataTransfer 构造器**：造一个只带 files 的假对象，
    // 用 defineProperty 塞进 input.files（这个属性在 jsdom 里是只读的）。
    const fakeFile = new win.File([new Uint8Array([80, 75, 3, 4])], '别人的项目.mproj',
      { type: 'application/octet-stream' })
    const fakeList = { 0: fakeFile, length: 1, item: (i) => fakeFile }
    Object.defineProperty(fileInput, 'files', {
      configurable: true,
      get: () => fakeList,
    })
    fileInput.dispatchEvent(new win.Event('change', { bubbles: true }))
    await new Promise((r) => setTimeout(r, 400))
  }
  check('导入调用了 /projects/import',
    calls.some((c) => c.url.includes('/projects/import')),
    JSON.stringify(calls.slice(-3).map((c) => `${c.method} ${c.url}`)))

  check('项目导入导出场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


// 12) 图生图节点：派生、源图解析、提交 w2
{
  calls.length = 0
  const { dom, errors } = await boot('?demo=gen')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 1400))

  // 生成节点出图后出现「↳ 图生图」派生按钮
  const spawn = [...doc.querySelectorAll('button.mini.wide')]
    .find((b) => (b.textContent || '').includes('图生图'))
  check('生成节点下方有「↳ 图生图」按钮', !!spawn)

  spawn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('派生出一个图生图节点', doc.querySelectorAll('.node').length === 2,
    String(doc.querySelectorAll('.node').length))
  const i2i = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '').includes('图生图'))
  check('新节点标题是「图生图」', !!i2i)
  check('图生图卡有源图区', !!i2i?.querySelector('.src-box'))
  check('源图下拉已自动选中上游节点', !!i2i?.querySelector('.src-box select')?.value)
  check('源图缩略图已显示', !!i2i?.querySelector('.src-thumb img'))
  check('图生图卡片不再要求宽高（尺寸随源图）',
    ![...(i2i?.querySelectorAll('.row label') || [])].some((l) => /^宽/.test(l.textContent || '')))

  // 提交 → 必须是 w2 且带 init_image / strength
  i2i.querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  check('提交 workflow=w2', body.workflow === 'w2', body.workflow)
  check('带 init_image 绝对路径', typeof body.params.init_image === 'string'
    && body.params.init_image.includes('out.png'), body.params.init_image)
  // Qwen 没有 strength 语义（走「指令改图」）。多发这个键后端会丢掉它，
  // 但每张产物都会挂一条 params_ignored 警告 —— 所以前端不该发。
  check('不带 strength（Qwen 没有这个参数）',
    body.params.strength === undefined, String(body.params.strength))
  check('不带 width/height（由源图决定）',
    body.params.width === undefined && body.params.height === undefined)
  check('提交带上了项目作用域（scope）',
    typeof body.params.scope === 'string' && body.params.scope.length > 0,
    String(body.params.scope))

  check('图生图场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 12) 图生图没有源图时：按钮禁用 + 不提交
{
  calls.length = 0
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  const addBtns = [...doc.querySelectorAll('.rail-add')]
  addBtns[1].dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))

  const i2i = doc.querySelector('.node')
  check('侧栏 + 号能新建图生图节点',
    (i2i?.querySelector('.node-kind')?.textContent || '') === '图生图')
  const ta = i2i.querySelector('textarea.prompt')
  const setValue = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setValue(ta, 'white cat knight')
  await new Promise((r) => setTimeout(r, 80))
  const btn = doc.querySelector('.gen-btn')
  check('没有源图时生成按钮禁用', btn?.disabled === true)
  check('没有源图时不提交任务', !calls.some((c) => c.url.endsWith('/generate')))

  check('空源图场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 13) 拖本机图片进画布 -> 图片节点（可改大小、可当图生图的源）
{
  calls.length = 0
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  const viewport = doc.querySelector('.canvas-viewport')

  const file = new win.File([new Uint8Array([1, 2, 3])], 'pic.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))

  check('拖入图片生成图片节点', !!doc.querySelector('.node.is-img'))
  check('图片节点标题是「图片」',
    (doc.querySelector('.node.is-img .node-kind')?.textContent || '') === '图片')
  check('上传请求已发出', calls.some((c) => c.url.includes('/upload')))
  check('上传带上了项目作用域',
    (calls.find((c) => c.url.includes('/upload'))?.url || '').includes('project='),
    calls.find((c) => c.url.includes('/upload'))?.url)
  check('图片节点渲染了图', !!doc.querySelector('.node.is-img .img-holder img'))
  check('图片节点默认宽度按原图上限', doc.querySelector('.node.is-img')?.style.width === '256px',
    doc.querySelector('.node.is-img')?.style.width)

  // 改展示宽度
  const wInput = doc.querySelector('.node.is-img .img-tools input')
  const setter = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set
  setter.call(wInput, '140')
  wInput.dispatchEvent(new win.Event('input', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 100))
  check('可自由改变展示大小', doc.querySelector('.node.is-img')?.style.width === '140px',
    doc.querySelector('.node.is-img')?.style.width)

  // 派生图生图：源就是这张本地图
  const spawn = [...doc.querySelectorAll('.node.is-img button.mini.wide')][0]
  spawn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  const i2i = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图生图')
  check('图片节点能派生图生图节点', !!i2i)
  check('图生图节点接上了图片节点', !!i2i?.querySelector('.src-thumb img'))

  // 提交时 init_image 必须是上传返回的磁盘路径
  const ta = i2i.querySelector('textarea.prompt')
  const setTa = (el, v) => {
    const s2 = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    s2.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setTa(ta, 'white cat knight')
  await new Promise((r) => setTimeout(r, 80))
  i2i.querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  check('提交 workflow=w2 且用本地图当源',
    body.workflow === 'w2' && body.params.init_image.includes('pic.png'),
    JSON.stringify({ w: body.workflow, i: body.params.init_image }))
  check('提交带上了项目作用域（scope）',
    typeof body.params.scope === 'string' && body.params.scope.length > 0,
    String(body.params.scope))

  check('图片节点场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 14) 连线：从图片节点的输出点拖到图生图节点
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  // 先造一个图片节点
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'pic.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))

  // 再加一个空的图生图节点
  const addBtns = [...doc.querySelectorAll('.rail-add')]
  addBtns[1].dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  const target = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图生图')
  check('有可作为连线目标的图生图节点', !!target)

  // 从图片节点的输出点按下
  const out = doc.querySelector('.node.is-img .conn.out')
  check('节点有输出连接点', !!out)
  out.dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 80))
  check('连线中出现待定线', !!doc.querySelector('svg.links path.pending'))

  // 悬停到目标节点上再松手
  target.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const sel = target.querySelector('.src-box select')
  check('连线后图生图节点的源被设成图片节点',
    !!sel && sel.value && sel.value.length > 0, sel?.value)
  check('画布上画出连线', doc.querySelectorAll('svg.links path.link').length >= 1,
    String(doc.querySelectorAll('svg.links path.link').length))

  check('连线场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 15) 真删：确认框里选「真删」-> 调 DELETE /assets，节点消失
{
  calls.length = 0
  const { dom, errors } = await boot('?demo=gen')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 1400))

  const delBtn = [...doc.querySelectorAll('.icon-btn')].find((b) => b.textContent === '✕')
  delBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('出图节点删除也弹确认框', !!doc.querySelector('.modal-mask'))
  const files = [...doc.querySelectorAll('.modal-files li')].map((li) => li.textContent)
  check('确认框列出要删的文件', files.length > 0, JSON.stringify(files))

  const hard = [...doc.querySelectorAll('.modal-actions button')]
    .find((b) => (b.textContent || '').includes('真删'))
  hard?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))

  const del = calls.find((c) => c.method === 'DELETE' && c.url.endsWith('/assets'))
  check('调用了 DELETE /assets', !!del)
  if (del) {
    const body = JSON.parse(del.body)
    check('提交了要删的路径', Array.isArray(body.paths) && body.paths.length > 0,
        JSON.stringify(body.paths))
  }
  check('真删后节点消失', doc.querySelectorAll('.node').length === 0,
        String(doc.querySelectorAll('.node').length))

  check('真删场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 16) 项目：每个项目是独立画布，互不干涉
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))

  const projSec = () => section(doc, '项目')
  const projRows = () => [...(projSec()?.querySelectorAll('.project-item') || [])]
  const nodes = () => doc.querySelectorAll('.node').length

  check('项目分组初始有 1 个项目', projRows().length === 1, String(projRows().length))
  check('当前项目有标记', !!projSec().querySelector('.project-item.current'))
  check('demo 节点在画布上', nodes() === 1, String(nodes()))

  // 新建项目 -> 空白画布
  projSec().querySelector('.rail-new').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  check('新建后项目数变 2', projRows().length === 2, String(projRows().length))
  check('新项目是空画布', nodes() === 0, String(nodes()))
  check('顶栏显示当前项目名',
    (doc.querySelector('.brand .sub')?.textContent || '').includes('项目'),
    doc.querySelector('.brand .sub')?.textContent)

  // 在新项目里加一个节点
  section(doc, '节点').querySelector('.rail-add')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('新项目里能加节点', nodes() === 1, String(nodes()))

  // 切回第一个项目 -> 原来的节点回来，新项目的节点不在这
  projRows()[0].querySelector('.rail-item-main')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  check('切回旧项目后画布内容换回来了', nodes() === 1, String(nodes()))
  const promptA = doc.querySelector('textarea.prompt')?.value || ''
  check('切回的是旧项目的节点', promptA.includes('猫骑士'), promptA)

  // 再切到新项目 -> 它的节点还在（没被旧项目覆盖）
  projRows()[1].querySelector('.rail-item-main')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  check('切回新项目仍是它自己的节点', nodes() === 1, String(nodes()))
  const promptB = doc.querySelector('textarea.prompt')?.value || ''
  check('两个项目的节点内容互不相同', promptB !== promptA, `${promptA} | ${promptB}`)

  // 删项目（先只从列表移除）
  projRows()[1].querySelector('.rail-del')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('删项目弹确认框', !!doc.querySelector('.modal-mask'))
  const onlyRemove = [...doc.querySelectorAll('.modal-actions button')]
    .find((b) => (b.textContent || '').includes('仅从列表移除'))
  onlyRemove?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  check('项目数回到 1', projRows().length === 1, String(projRows().length))

  check('项目场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 16c) id 冲突：服务端分配 id + 冲突要提示
{
  const { dom, errors, fetch: f } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 700))

  // ---- 新建项目：id 来自服务端，不是本地生成 ----
  const projSec = () => section(doc, '项目')
  const before = projSec().querySelectorAll('.project-item').length
  projSec().querySelector('.rail-new').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 700))
  check('新建项目时向服务端要 id',
    calls.some((c) => c.url.includes('/projects/allocate')),
    JSON.stringify(calls.slice(-4).map((c) => `${c.method} ${c.url}`)))
  check('新建后项目数 +1',
    projSec().querySelectorAll('.project-item').length === before + 1,
    String(projSec().querySelectorAll('.project-item').length))
  // 服务端分配到的 id 真的落盘了
  check('服务端分配的 id 被用来落盘',
    [...f.serverProjects.keys()].length >= 1,
    JSON.stringify([...f.serverProjects.keys()]))

  // ---- 保存冲突：要让用户看见，不能静默失败 ----
  // **先切回有节点的项目**：新建的项目是空画布（没有节点可改，
  // 也就不会触发自动保存）。
  const rows = [...projSec().querySelectorAll('.project-item')]
  rows[0]?.querySelector('.rail-item-main')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))

  f.setSaveConflict(true)
  const ta = doc.querySelector('textarea.prompt')
  check('切回后有节点可以改', !!ta)
  if (ta) {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    setter.call(ta, '冲突测试提示词')
    ta.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  await new Promise((r) => setTimeout(r, 1200))
  // 提示渲染在 .toast 里（见 App.tsx 的 uploadMsg / notice）
  const toast = doc.querySelector('.toast')
  check('保存冲突时弹出了提示条', !!toast, '没有 .toast')
  check('提示内容说明了是冲突', (toast?.textContent || '').includes('冲突'),
    (toast?.textContent || '').slice(0, 120))
  check('提示里带上了服务端的原因',
    (toast?.textContent || '').includes('其他窗口'),
    (toast?.textContent || '').slice(0, 160))
  f.setSaveConflict(false)

  check('id 冲突场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 16b) 项目落盘：编辑后写到服务端磁盘
{
  const { dom, errors, fetch: f } = await boot('?demo=1')
  const serverProjects = f.serverProjects
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 600))

  const projSec = () => section(doc, '项目')
  const projRows = () => [...(projSec()?.querySelectorAll('.project-item') || [])]

  // 启动时会拉 /projects —— 空列表时把缓存里的项目写上去
  check('启动时拉了项目列表',
    calls.some((c) => c.url.endsWith('/projects') && (c.method || 'GET') === 'GET'),
    JSON.stringify(calls.slice(0, 5).map((c) => `${c.method} ${c.url}`)))
  check('服务端没有项目时前端会建一个并落盘',
    serverProjects.size >= 1, String(serverProjects.size))
  // **浏览器里不留任何项目内容**：localStorage 的每个键都不含节点数据
  const lsKeys = Object.keys(win.localStorage)
  const lsDump = lsKeys.map((k) => `${k}=${win.localStorage.getItem(k) || ''}`).join('|')
  check('浏览器 localStorage 里没有项目内容',
    !lsDump.includes('nodes') && !lsDump.includes('viewport'),
    lsKeys.join(','))
  // 改内容 -> 防抖后 PUT 到服务端
  const before = serverProjects.size
  const ta = doc.querySelector('textarea.prompt')
  if (ta) {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    setter.call(ta, '落盘测试提示词')
    ta.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  await new Promise((r) => setTimeout(r, 800))
  const putCalls = calls.filter((c) => c.method === 'PUT' && c.url.includes('/projects/'))
  check('编辑后调用了 PUT /projects/<id>', putCalls.length > 0,
    JSON.stringify(calls.slice(-4).map((c) => `${c.method} ${c.url}`)))
  const saved = [...serverProjects.values()].find((p) =>
    JSON.stringify(p.nodes || []).includes('落盘测试提示词'))
  check('提示词真的写进了服务端项目定义', !!saved,
    JSON.stringify([...serverProjects.keys()]))
  check('保存的项目带节点', (saved?.nodes || []).length > 0, String(saved?.nodes?.length))

  // 新建项目 -> 立刻落盘（空项目也要在磁盘上占位）
  serverProjects.clear()
  projSec().querySelector('.rail-new').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 600))
  check('新建项目也落盘了', serverProjects.size >= 1, String(serverProjects.size))

  // 删项目 -> 服务端也要删（否则刷新后项目又冒出来）
  serverProjects.clear()
  await new Promise((r) => setTimeout(r, 300))
  const delTarget = projRows()[0]
  if (delTarget) {
    delTarget.querySelector('.rail-del')
      .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
    await new Promise((r) => setTimeout(r, 150))
    const onlyRemove = [...doc.querySelectorAll('.modal-actions button')]
      .find((b) => (b.textContent || '').includes('仅从列表移除'))
    onlyRemove?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
    await new Promise((r) => setTimeout(r, 400))
  }
  check('删项目调用了 DELETE /projects/<id>',
    calls.some((c) => c.method === 'DELETE' && c.url.includes('/projects/')),
    JSON.stringify(calls.slice(-4).map((c) => `${c.method} ${c.url}`)))

  check('项目落盘场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 17) 老存档迁移：meow.canvas.v1 的一份 nodes+viewport 要变成第一个项目
{
  const legacy = {
    nodes: [{
      id: 'legacy1', kind: 'gen', x: 40, y: 60, view: 'expanded',
      params: {
        prompt: 'legacy knight', negative: '', width: 768, height: 768, steps: 30,
        cfg: 7, seed: 1, preset: 'sprite', pixel: 6, palette: 32, matte: false,
        clipSkip: 1, strength: 0.55, lora: '', loraScale: 0.8,
      },
    }],
    viewport: { x: 33, y: 44, z: 1.5 },
  }
  const { dom, errors, fetch: f } = await boot('', { 'meow.canvas.v1': JSON.stringify(legacy) })
  const doc = dom.window.document
  await new Promise((r) => setTimeout(r, 500))

  const projSec = () => section(doc, '项目')
  const rows = () => [...(projSec()?.querySelectorAll('.project-item') || [])]
  check('老存档被迁移成 1 个项目', rows().length === 1, String(rows().length))
  check('迁移后的项目名是「我的项目」',
    (rows()[0]?.querySelector('b')?.textContent || '') === '我的项目',
    rows()[0]?.querySelector('b')?.textContent)
  check('老节点被带过来了',
    (doc.querySelector('textarea.prompt')?.value || '') === 'legacy knight',
    doc.querySelector('textarea.prompt')?.value)
  check('老视口被带过来了（缩放 1.5）',
    (doc.querySelector('.zoom-val')?.textContent || '') === '150%',
    doc.querySelector('.zoom-val')?.textContent)
  // **迁移完要从浏览器删掉**：项目内容不该继续留在浏览器里
  check('老存档已从浏览器删除',
    dom.window.localStorage.getItem('meow.canvas.v1') === null,
    String(dom.window.localStorage.getItem('meow.canvas.v1')))
  // 迁移后的项目要落到服务端磁盘
  check('迁移的项目落到了服务端',
    f.serverProjects.size >= 1, String(f.serverProjects.size))
  const migrated = [...f.serverProjects.values()][0]
  check('服务端存下了迁移过来的节点',
    JSON.stringify(migrated?.nodes || []).includes('legacy knight'),
    JSON.stringify(migrated?.nodes || []).slice(0, 80))

  check('迁移场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 18) 全屏预览 + 下载
{
  const { dom, errors } = await boot('?demo=gen')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 1400))

  const fullBtn = [...doc.querySelectorAll('.icon-btn')].find(
    (b) => (b.getAttribute('title') || '').includes('全屏'))
  check('卡头有全屏预览按钮', !!fullBtn)

  // 点缩略图 -> 灯箱
  doc.querySelector('.preview img')?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('点图片打开全屏预览', !!doc.querySelector('.lightbox'))
  check('灯箱里有大图', !!doc.querySelector('.lightbox-img'))
  const dl = doc.querySelector('.lightbox a.mini.wide')
  check('灯箱里有下载按钮', !!dl && (dl.textContent || '').includes('下载'))
  check('下载链接带 download=1', (dl?.getAttribute('href') || '').includes('download=1'),
    dl?.getAttribute('href'))

  // ---- 缩放 / 平移（图片预览要看细节）----
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))
  const pct = () => (doc.querySelector('.lightbox-pct')?.textContent || '').trim()
  check('灯箱有缩放工具条', !!doc.querySelector('.lightbox-zoomctl'))
  check('显示当前缩放百分比', /^\d+%$/.test(pct()), pct())
  const zoomBtn = (kw) => [...doc.querySelectorAll('.lightbox-zoomctl .mini')]
    .find((b) => (b.getAttribute('title') || '').startsWith(kw))
  const plus = zoomBtn('放大')
  const minus = zoomBtn('缩小')
  const fitBtn = zoomBtn('适应')
  const oneBtn = [...doc.querySelectorAll('.lightbox-zoomctl .mini')]
    .find((b) => (b.textContent || '').trim() === '1:1')
  check('有放大/缩小/适应/1:1 四个按钮',
    !!plus && !!minus && !!fitBtn && !!oneBtn,
    String([!!plus, !!minus, !!fitBtn, !!oneBtn]))

  const atStart = parseInt(pct(), 10)
  plus.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(120)
  const zoomed = parseInt(pct(), 10)
  check('点「+」后百分比变大', zoomed > atStart, `${atStart}% → ${zoomed}%`)
  check('放大后换成最近邻（像素方块要方）',
    (doc.querySelector('.lightbox-img')?.style.imageRendering || '') === 'pixelated',
    doc.querySelector('.lightbox-img')?.style.imageRendering)
  check('放大后「适应」可点（能缩回去）', fitBtn?.disabled === false)

  minus.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(120)
  check('点「−」能缩回去', parseInt(pct(), 10) === atStart, pct())

  // 滚轮缩放：jsdom 没有布局，clientX/clientY 取 0 也行（锚点算不出但缩放要生效）
  doc.querySelector('.lightbox-stage')?.dispatchEvent(new win.WheelEvent('wheel', {
    deltaY: -240, bubbles: true, cancelable: true,
  }))
  await wait(120)
  check('滚轮向上能放大', parseInt(pct(), 10) > atStart, pct())

  // 双击：适应 ↔ 4 倍
  const stage = doc.querySelector('.lightbox-stage')
  const fitNow = await (async () => {
    fitBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
    await wait(120)
    return parseInt(pct(), 10)
  })()
  stage.dispatchEvent(new win.MouseEvent('dblclick', { bubbles: true, clientX: 0, clientY: 0 }))
  await wait(120)
  check('双击放大到 4 倍', parseInt(pct(), 10) === fitNow * 4, `${pct()}（适应 ${fitNow}%）`)
  stage.dispatchEvent(new win.MouseEvent('dblclick', { bubbles: true, clientX: 0, clientY: 0 }))
  await wait(120)
  check('再双击回适应窗口', parseInt(pct(), 10) === fitNow, pct())

  // ---- 拖拽平移不能把预览关掉（用户报："一拖预览就关了"）----
  // 舞台里的 click 以前会冒泡到 .lightbox 的"点背景关闭"，拖动松手补的那个
  // click 也被算进去 → 拖一下就没了。现在舞台先 stopPropagation 再判断。
  stage.querySelector('.lightbox-img')
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(120)
  check('点图片本身不关闭预览', !!doc.querySelector('.lightbox'))

  stage.dispatchEvent(new win.PointerEvent('pointerdown', {
    bubbles: true, button: 0, clientX: 100, clientY: 100,
  }))
  stage.dispatchEvent(new win.PointerEvent('pointermove', {
    bubbles: true, clientX: 180, clientY: 160,
  }))
  stage.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  stage.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(120)
  check('拖动平移后预览还在（松手补的 click 不算点背景）',
    !!doc.querySelector('.lightbox'))
  check('拖动时舞台有 dragging 类（光标变抓手）',
    !!doc.querySelector('.lightbox-stage.dragging') === false, '松手后应清掉')

  // Esc 关闭
  win.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('Esc 关掉灯箱', !doc.querySelector('.lightbox'))

  // 卡头的 ⤢ 也能开
  fullBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('卡头按钮也能打开灯箱', !!doc.querySelector('.lightbox'))
  // 点背景关闭
  doc.querySelector('.lightbox')?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('点背景关闭灯箱', !doc.querySelector('.lightbox'))

  check('灯箱场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 19) 抠图节点：拖图当源 + 三种背景方式 + 提交 wbg
{
  calls.length = 0
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  // 侧栏新建一个抠图节点
  const nodeItems = [...section(doc, '节点').querySelectorAll('.rail-item')]
  const bgItem = nodeItems.find((it) => (it.querySelector('b')?.textContent || '') === '抠图')
  check('侧栏有「抠图」节点类型', !!bgItem)
  bgItem?.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const node = doc.querySelector('.node')
  check('新建出抠图节点',
    (node?.querySelector('.node-kind')?.textContent || '') === '抠图',
    node?.querySelector('.node-kind')?.textContent)
  check('抠图节点有源图区', !!node?.querySelector('.src-box'))
  check('抠图节点有三种背景方式',
    node?.querySelectorAll('.bg-opts input[type=radio]').length === 3,
    String(node?.querySelectorAll('.bg-opts input[type=radio]').length))
  check('没有源图时不能提交', node?.querySelector('.gen-btn')?.disabled === true)

  // 直接把本地图拖到这个节点上 -> 变成源
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'pic.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))
  // 拖放落在空白处会建图片节点；这里检查源图下拉/提交体
  const bgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '抠图')
  check('抠图节点仍在', !!bgNode)

  // 给抠图节点接上源：把图片节点的连线接过去（用下拉不方便，这里用拖拽连线）
  const imgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图片')
  check('拖进来的图片成了独立图片节点', !!imgNode)

  const out = imgNode?.querySelector('.conn.out')
  out?.dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 60))
  bgNode.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const sel = bgNode.querySelector('.src-box select')
  check('连线后抠图节点接上了源', !!sel && !!sel.value, sel?.value)
  check('有源图后可以提交了', bgNode.querySelector('.gen-btn')?.disabled === false)

  bgNode.querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  check('提交 workflow=wbg', body.workflow === 'wbg', body.workflow)
  check('带上了源图路径', typeof body.params.init_image === 'string'
    && body.params.init_image.length > 0, body.params.init_image)

  check('抠图场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 20) 局部重绘节点：蒙版笔刷 + 提交体
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  const nodeItems = [...section(doc, '节点').querySelectorAll('.rail-item')]
  const ipItem = nodeItems.find((it) => (it.querySelector('b')?.textContent || '') === '局部重绘')
  check('侧栏有「局部重绘」节点类型', !!ipItem)
  ipItem?.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const node = doc.querySelector('.node')
  check('新建出局部重绘节点',
    (node?.querySelector('.node-kind')?.textContent || '') === '局部重绘',
    node?.querySelector('.node-kind')?.textContent)
  check('有蒙版区', !!node?.querySelector('.mask-box'))
  check('没有蒙版时显示占位', (node?.querySelector('.mask-thumb')?.textContent || '').includes('无蒙版'))
  check('没源图/蒙版/提示词时不能提交', node?.querySelector('.gen-btn')?.disabled === true)
  check('没有源图时不能画蒙版',
    node?.querySelector('.mask-box .mini')?.disabled === true)

  // 接一个图片节点当源
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'pic.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))
  const imgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图片')
  const ipNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '局部重绘')
  imgNode?.querySelector('.conn.out')
    .dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 60))
  ipNode.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('连线后局部重绘节点接上了源',
    !!ipNode.querySelector('.src-box select')?.value)

  // 打开蒙版笔刷
  ipNode.querySelector('.mask-box .mini')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('打开蒙版笔刷', !!doc.querySelector('.mask-editor'))
  check('笔刷有三种工具',
    doc.querySelectorAll('.mask-tools .mini').length >= 3,
    String(doc.querySelectorAll('.mask-tools .mini').length))
  check('有笔刷大小滑杆', !!doc.querySelector('.mask-tools input[type=range]'))
  check('有蒙版颜色色板', doc.querySelectorAll('.mask-swatches .swatch').length === 6,
    String(doc.querySelectorAll('.mask-swatches .swatch').length))
  // 关掉（jsdom 没有 canvas 2d，不去点保存）
  const cancel = [...doc.querySelectorAll('.mask-bar .mini')].find(
    (b) => (b.textContent || '').includes('取消'))
  cancel?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('可以关掉笔刷', !doc.querySelector('.mask-editor'))

  check('局部重绘场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 21) 局部 AI 重绘节点：裁剪重画 + 提交体
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  const nodeItems = [...section(doc, '节点').querySelectorAll('.rail-item')]
  const arItem = nodeItems.find((it) => (it.querySelector('b')?.textContent || '') === '局部AI重绘')
  check('侧栏有「局部AI重绘」节点类型', !!arItem)
  arItem?.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const node = doc.querySelector('.node')
  check('新建出局部AI重绘节点',
    (node?.querySelector('.node-kind')?.textContent || '') === '局部AI重绘',
    node?.querySelector('.node-kind')?.textContent)
  check('有蒙版区', !!node?.querySelector('.mask-box'))
  check('有源图区', !!node?.querySelector('.src-box'))
  check('没源图/蒙版/提示词时不能提交', node?.querySelector('.gen-btn')?.disabled === true)
  check('有更多设置（上下文/外扩/羽化/校正/吸附）',
    !!node?.querySelector('.adv') &&
    node.querySelectorAll('.adv .check').length === 2,
    String(node?.querySelectorAll('.adv .check').length))

  // 接一个图片节点当源
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'pic.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))
  const imgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图片')
  const arNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '局部AI重绘')
  imgNode?.querySelector('.conn.out')
    .dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 60))
  arNode.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('连线后局部AI重绘节点接上了源',
    !!arNode.querySelector('.src-box select')?.value)

  // 打开笔刷（jsdom 没 canvas 2d，只验证能开能关）
  arNode.querySelector('.mask-box .mini')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('能打开蒙版笔刷', !!doc.querySelector('.mask-editor'))
  const cancel = [...doc.querySelectorAll('.mask-bar .mini')].find(
    (b) => (b.textContent || '').includes('取消'))
  cancel?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('能关掉笔刷', !doc.querySelector('.mask-editor'))

  check('局部AI重绘场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 22) 像素转换节点：源图 + 参数 + 提交 wpx
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  const nodeItems = [...section(doc, '节点').querySelectorAll('.rail-item')]
  const pxItem = nodeItems.find((it) => (it.querySelector('b')?.textContent || '') === '像素转换')
  check('侧栏有「像素转换」节点类型', !!pxItem)
  pxItem?.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const node = doc.querySelector('.node')
  check('新建出像素转换节点',
    (node?.querySelector('.node-kind')?.textContent || '') === '像素转换',
    node?.querySelector('.node-kind')?.textContent)
  check('有源图区', !!node?.querySelector('.src-box'))
  check('没有源图时不能提交', node?.querySelector('.gen-btn')?.disabled === true)
  check('有边长/色数/吸附/取色/两个开关',
    node?.querySelectorAll('.px-opts .strength input[type=number]').length === 3 &&
    !!node?.querySelector('.px-opts select') &&
    node.querySelectorAll('.px-opts .check').length === 2,
    String(node?.querySelectorAll('.px-opts .check').length))

  // 接一个图片节点当源
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'pic.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))
  const imgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图片')
  const pxNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '像素转换')
  imgNode?.querySelector('.conn.out')
    .dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 60))
  pxNode.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('连线后像素转换节点接上了源',
    !!pxNode.querySelector('.src-box select')?.value)
  check('有源图后可以提交了', pxNode.querySelector('.gen-btn')?.disabled === false)

  pxNode.querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  check('提交 workflow=wpx', body.workflow === 'wpx', body.workflow)
  check('带上了源图路径', typeof body.params.init_image === 'string'
    && body.params.init_image.length > 0, body.params.init_image)
  check('带上了色数与取色方式',
    typeof body.params.colors === 'number' && typeof body.params.sample_method === 'string',
    JSON.stringify({ c: body.params.colors, s: body.params.sample_method }))

  check('像素转换场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 22b) 转像素节点（topixel）：源图必须真的传给后端
//
// 回归：这个节点能建、能连线、界面还提示"有源图可留空提示词"，但提交时
// `init` 从来没被赋值过（判断列表里漏了 topixel），后端只收到空 prompt，
// 直接抛「需要 prompt 或 init_image 之一」——用户看到的就是"转像素失败"。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))

  const items = [...section(doc, '节点').querySelectorAll('.rail-item')]
  const tpItem = items.find((it) => (it.querySelector('b')?.textContent || '') === '转像素')
  check('侧栏有「转像素」节点类型', !!tpItem)
  tpItem?.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(150)

  const tpNode = () => [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '转像素')
  check('能新建转像素节点', !!tpNode())
  check('有源图区（可留空）',
    (tpNode()?.querySelector('.src-note')?.textContent || '').includes('有源图'),
    tpNode()?.querySelector('.src-note')?.textContent)

  // 拖一张图进画布当源
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'pic.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 260 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await wait(400)
  const imgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图片')
  check('画布上出现了源图节点', !!imgNode)

  imgNode?.querySelector('.conn.out')
    .dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await wait(60)
  tpNode()?.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await wait(60)
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await wait(150)
  check('连线后转像素节点接上了源', !!tpNode()?.querySelector('.src-box select')?.value)

  calls.length = 0
  tpNode()?.querySelector('.gen-btn')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(300)
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  check('转像素提交了', !!sub)
  const body = sub ? JSON.parse(sub.body) : {}
  check('workflow=topixel', body.workflow === 'topixel', body.workflow)
  // **核心断言**：源图路径必须传过去（以前这里是 undefined）
  check('把源图路径传给了后端（否则后端只会看到空 prompt）',
    typeof body.params?.init_image === 'string' && body.params.init_image.length > 0,
    String(body.params?.init_image))

  check('转像素场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 23) 角色迁移节点：两个源 + 提交 wpc
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  const nodeItems = [...section(doc, '节点').querySelectorAll('.rail-item')]
  const pcItem = nodeItems.find((it) => (it.querySelector('b')?.textContent || '') === '角色迁移')
  check('侧栏有「角色迁移」节点类型', !!pcItem)
  pcItem?.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const node = doc.querySelector('.node')
  check('新建出角色迁移节点',
    (node?.querySelector('.node-kind')?.textContent || '') === '角色迁移',
    node?.querySelector('.node-kind')?.textContent)
  check('有动作参考图区', !!node?.querySelector('.src-box'))
  check('有角色图区（第二个源）', !!node?.querySelector('.char-box'))
  check('两个源各有一个下拉', node?.querySelectorAll('.src-box select').length === 2,
    String(node?.querySelectorAll('.src-box select').length))
  check('没有源图时不能提交', node?.querySelector('.gen-btn')?.disabled === true)
  // 姿态/身份两根 SD 强度滑块已随 SD 删除，只剩「占高」
  check('只剩「占高」一根滑块（姿态/身份强度已删除）',
    node?.querySelectorAll('.adv .strength input[type=range]').length === 1,
    String(node?.querySelectorAll('.adv .strength input[type=range]').length))
  check('不再有「姿态」「身份」两个滑块',
    ![...(node?.querySelectorAll('.adv .strength span') || [])]
      .some((s) => /^(姿态|身份)$/.test((s.textContent || '').trim())))

  // 拖一个本地图进来 -> 动作参考图（第一个源）
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'pose.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))
  const imgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图片')
  const pcNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '角色迁移')
  check('拖进来的图片成了独立图片节点', !!imgNode)

  // 连线 -> 动作参考图
  imgNode?.querySelector('.conn.out')
    .dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 60))
  pcNode.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('连线后接上了动作参考图',
    !!pcNode.querySelector('.src-box select')?.value)

  // 角色图：第二个下拉选图片节点
  const charSel = pcNode.querySelectorAll('.src-box select')[1]
  const charSet = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
  charSet.call(charSel, imgNode.getAttribute('data-node-id') || '')
  charSel.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  // 填提示词
  const ta = pcNode.querySelector('textarea.prompt')
  const setTa = (el, v) => {
    const s2 = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    s2.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setTa(ta, 'a blue slime monster')
  await new Promise((r) => setTimeout(r, 80))

  pcNode.querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  // 默认引擎是 VNCCS（保角色一致性更好）：走 wvnccs + char_images 数组
  check('默认走 VNCCS 引擎', body.workflow === 'wvnccs', body.workflow)
  check('带上了骨架图与角色图（多图数组）',
    typeof body.params.pose_image === 'string' && body.params.pose_image.length > 0 &&
    Array.isArray(body.params.char_images) && body.params.char_images.length > 0,
    JSON.stringify({ p: body.params.pose_image, c: body.params.char_images }))

  // 引擎下拉存在，且能选到旧路线（旧路径的载荷由后端 pytest 覆盖：
  // 这里再点一次生成拿不到新请求——首次生成后节点处于 running，按钮是禁用的）
  const engineSel = pcNode.querySelector('.pc-engine select')
  check('节点上有引擎下拉', !!engineSel)
  check('引擎下拉有旧路线选项',
    !!engineSel && [...engineSel.options].some((o) => o.value === 'wpc'),
    engineSel ? [...engineSel.options].map((o) => o.value).join(',') : '')
  check('额外参考图入口存在',
    !!pcNode.querySelector('.pc-extra'),
    pcNode.querySelector('.pc-extra') ? '有' : '没有')
  // 双图条件没有强度旋钮：这两个键不该出现在载荷里
  check('不带姿态/身份强度（双图条件没有强度旋钮）',
    body.params.cn_scale === undefined && body.params.ipa_scale === undefined,
    JSON.stringify({ cn: body.params.cn_scale, ipa: body.params.ipa_scale }))

  check('角色迁移场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 24) 序列帧替换节点：精灵图 + 角色图 → 提交 wss
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  const nodeItems = [...section(doc, '节点').querySelectorAll('.rail-item')]
  const ssItem = nodeItems.find((it) => (it.querySelector('b')?.textContent || '') === '序列帧替换')
  check('侧栏有「序列帧替换」节点类型', !!ssItem)
  ssItem?.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  const node = doc.querySelector('.node')
  check('新建出序列帧替换节点',
    (node?.querySelector('.node-kind')?.textContent || '') === '序列帧替换',
    node?.querySelector('.node-kind')?.textContent)
  check('有序列帧源区与角色图区',
    !!node?.querySelector('.src-box') && !!node?.querySelector('.char-box'))
  check('没有源图时不能提交', node?.querySelector('.gen-btn')?.disabled === true)
  // 方式选择：AI 重绘 / 换皮
  const modeSel = [...node.querySelectorAll('select')].find((s) =>
    [...s.options].some((o) => o.textContent.includes('换皮')))
  check('有 AI重绘/换皮 方式选择', !!modeSel)
  // 逐帧引擎：VNCCS（帧间角色一致） / 旧版。
  // AnimateDiff 与"显存档位"两个下拉已随 SD 一起删掉，别再写它们的断言。
  const engineSel = [...node.querySelectorAll('select')].find((s) =>
    [...s.options].some((o) => o.value === 'wvnccs'))
  check('有逐帧引擎选择（VNCCS/旧版）', !!engineSel)
  check('引擎默认 VNCCS',
    !!engineSel && engineSel.value === 'wvnccs',
    engineSel ? engineSel.value : '')

  // 拖一个本地图进来 -> 序列帧源
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'sheet.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 300 }, clientY: { value: 200 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))
  const imgNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '图片')
  const ssNode = [...doc.querySelectorAll('.node')].find(
    (n) => (n.querySelector('.node-kind')?.textContent || '') === '序列帧替换')

  // 连线 -> 序列帧源
  imgNode?.querySelector('.conn.out')
    .dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 60))
  ssNode.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('连线后接上了序列帧源', !!ssNode.querySelector('.src-box select')?.value)

  // 角色图
  const charSel = ssNode.querySelectorAll('.src-box select')[1]
  const charSet = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
  charSet.call(charSel, imgNode.getAttribute('data-node-id') || '')
  charSel.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))

  // 填提示词（generate 模式需要）
  const ta = ssNode.querySelector('textarea.prompt')
  const setTa = (el, v) => {
    const s2 = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    s2.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setTa(ta, 'a blue slime monster')
  await new Promise((r) => setTimeout(r, 80))

  // 先切到换皮模式：此时不需要提示词也能提交
  // （必须放在提交之前——提交后任务进队列，按钮本来就会禁用）
  const modeSet = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
  modeSet.call(modeSel, 'recolor')
  modeSel.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('换皮模式下不需要提示词也能提交',
    ssNode.querySelector('.gen-btn')?.disabled === false)

  // 结果区要显示「实际候选 / 补提 / 闸门通过」——这几个数决定要不要重跑，
  // 而且候选数不等于设定值时说明被显存降级了，必须让用户看见。
  extraMeta = { candidates: 2, attempts: 11, frames: 4, gate_passed: 3,
                candidates_note: '显存不足：候选从 4 降到 2（预计 23.9GB > 可用 21.6GB）' }
  ssNode.querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  check('提交 workflow=wss', body.workflow === 'wss', body.workflow)
  check('带上了序列帧与角色图两个路径',
    typeof body.params.sheet_image === 'string' && body.params.sheet_image.length > 0 &&
    typeof body.params.char_image === 'string' && body.params.char_image.length > 0,
    JSON.stringify({ s: body.params.sheet_image, c: body.params.char_image }))
  check('带上了网格（0=自动）与替换方式',
    body.params.cols === undefined && body.params.rows === undefined &&
    typeof body.params.mode === 'string',
    JSON.stringify({ cols: body.params.cols, mode: body.params.mode }))

  // 结果区要显示「实际候选 / 补提 / 闸门通过」——这几个数决定要不要重跑，
  // 而且候选数不等于设定值时说明被显存降级了，必须让用户看见。
  extraMeta = { candidates: 2, attempts: 11, frames: 4, gate_passed: 3,
                candidates_note: '显存不足：候选从 4 降到 2（预计 23.9GB > 可用 21.6GB）' }
  // React 每次重渲染都会换 DOM 节点，必须重新查，不能复用旧引用
  const ssNow = () => [...doc.querySelectorAll('.node')]
    .find((n) => (n.querySelector('.node-kind')?.textContent || '') === '序列帧替换')
  // App 的 job 轮询间隔 700ms，等固定时长会脆；轮询到出现结果为止（最多 4s）
  for (let i = 0; i < 20; i++) {
    if (ssNow().querySelector('.meta')) break
    await new Promise((r) => setTimeout(r, 200))
  }
  const metaLine = ssNow().querySelector('.meta')?.textContent || ''
  check('结果显示候选数（2/帧）', metaLine.includes('候选 2/帧'), metaLine)
  check('结果显示补提次数（11 - 4×2 = 3）', metaLine.includes('补提 3 次'), metaLine)
  check('结果显示闸门通过帧数（3/4）', metaLine.includes('闸门 3/4'), metaLine)
  // 展开视图说明原因；图片视图只给简短提示（见 GenNodeCard 的 img-warn）
  const warnText = ssNow().querySelector('.warn')?.textContent
    || ssNow().querySelector('.img-warn')?.getAttribute('title') || ''
  check('候选被降级时给出原因', warnText.includes('候选从 4 降到 2'), warnText)
  extraMeta = null

  check('序列帧替换场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 22) 切换底模：自动套用后端推荐参数，但不覆盖手改过的节点
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window

  const genItem = [...section(doc, '节点').querySelectorAll('.rail-item')]
    .find((it) => (it.querySelector('b')?.textContent || '') === '图生成')
  const addGen = async () => {
    genItem.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
    await new Promise((r) => setTimeout(r, 120))
  }
  await addGen()
  await addGen()

  const num = (node, prefix) => {
    const lab = [...node.querySelectorAll('label')]
      .find((l) => (l.textContent || '').trim().startsWith(prefix))
    return lab?.querySelector('input')?.value
  }
  let nodes = [...doc.querySelectorAll('.node')]
  check('新节点默认用 SD1.5 的推荐值（768 / 30 步）',
    num(nodes[0], '宽') === '768' && num(nodes[0], '步数') === '30',
    JSON.stringify({ w: num(nodes[0], '宽'), s: num(nodes[0], '步数') }))

  // 手改第二个节点的宽度，模拟"用户调过参"
  const setNum = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set
    setter.call(el, String(v))
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  const wLab = [...nodes[1].querySelectorAll('label')]
    .find((l) => (l.textContent || '').trim().startsWith('宽'))
  setNum(wLab.querySelector('input'), 640)
  await new Promise((r) => setTimeout(r, 80))

  const modelSel = doc.querySelector('.node-model select')
  const setSel = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSel(modelSel, 'sdxl')
  await new Promise((r) => setTimeout(r, 200))

  nodes = [...doc.querySelectorAll('.node')]
  // 底模现在是**节点级**：侧栏那个只决定「新节点的默认值」。
  // 仍处于「跟随默认」的节点要跟着走（否则会出现「底模 SDXL、步数 30」
  // 这种自相矛盾的配置）；手改过参数的节点不被覆盖。
  check('节点切 SDXL 后套用推荐步数（30 → 24）',
    num(nodes[0], '步数') === '24', num(nodes[0], '步数'))
  check('手改过参数的节点不被覆盖（宽仍是 640）',
    num(nodes[1], '宽') === '640', num(nodes[1], '宽'))
  check('切换给出推荐参数提示',
    (doc.querySelector('.toast')?.textContent || '').includes('推荐'),
    doc.querySelector('.toast')?.textContent)

  // 节点自己的底模下拉：只影响这一个节点
  nodes = [...doc.querySelectorAll('.node')]
  const nodeModelSel = nodes[0].querySelector('.node-model select')
  check('节点上有自己的底模下拉', !!nodeModelSel)
  // 先把这个节点单独钉到 sd15，再钉到 sdxl，验「只动自己」
  setSel(nodeModelSel, 'sd15')
  await new Promise((r) => setTimeout(r, 250))
  nodes = [...doc.querySelectorAll('.node')]
  check('节点单独选 SD1.5 后套用推荐步数（24 → 30）',
    num(nodes[0], '步数') === '30', num(nodes[0], '步数'))
  check('该节点值变成具体的底模（不再跟随默认）',
    nodes[0].querySelector('.node-model select')?.value === 'sd15',
    nodes[0].querySelector('.node-model select')?.value)
  check('另一个节点不受影响',
    num(nodes[1], '宽') === '640' && num(nodes[1], '步数') === '30',
    JSON.stringify({ w: num(nodes[1], '宽'), s: num(nodes[1], '步数') }))
  // 切回「跟随默认」：只改字段，不动参数
  setSel(nodes[0].querySelector('.node-model select'), '')
  await new Promise((r) => setTimeout(r, 200))
  check('切回「跟随默认」步数仍是 30（不重置参数）',
    num([...doc.querySelectorAll('.node')][0], '步数') === '30',
    num([...doc.querySelectorAll('.node')][0], '步数'))

  check('底模切换场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 23) 候选数上限跟随底模显存（超了要让用户看见，而不是静默变慢）
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))

  // 加一个序列帧节点
  const item = [...section(doc, '节点').querySelectorAll('.rail-item')]
    .find((it) => (it.querySelector('b')?.textContent || '') === '序列帧替换')
  item.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(150)

  const node = () => doc.querySelector('.node')
  const candInput = () => {
    const lab = [...node().querySelectorAll('label')]
      .find((l) => (l.textContent || '').trim().startsWith('候选'))
    return lab?.querySelector('input')
  }
  const setNum = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set
    setter.call(el, String(v))
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  check('SD1.5 下候选上限是 6（后端按总显存算）',
    candInput()?.getAttribute('max') === '6', candInput()?.getAttribute('max'))

  // 超出上限的输入被夹住
  setNum(candInput(), 9)
  await wait(120)
  check('输入超过上限时被夹到 6，不会提交跑不动的配置',
    candInput()?.value === '6', candInput()?.value)
  check('候选旁说明上限来源',
    (node().textContent || '').includes('由当前底模的显存占用算出'),
    (node().textContent || '').slice(-60))

  // 用**这个节点自己的**底模下拉切到 SDXL：上限收到 3，已设的 6 要跟着降
  const nodeModelSel = node().querySelector('.node-model select')
  const setSel = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSel(nodeModelSel, 'sdxl')
  await wait(300)
  check('节点切 SDXL 后候选上限收到 3',
    candInput()?.getAttribute('max') === '3', candInput()?.getAttribute('max'))
  check('已设的 6 张跟着降到 3（界面上不能写着跑不动的值）',
    candInput()?.value === '3', candInput()?.value)

  check('候选上限场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 26) 姿态序列节点（W4）：骨架目录 → 逐帧生成，能提交也能看到候选统计
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))

  const item = [...section(doc, '节点').querySelectorAll('.rail-item')]
    .find((it) => (it.querySelector('b')?.textContent || '') === '姿态序列')
  check('侧栏有「姿态序列」节点', !!item)
  item.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(150)

  const nd = () => doc.querySelector('.node')
  check('新建出姿态序列节点',
    (nd().querySelector('.node-kind')?.textContent || '') === '姿态序列',
    nd().querySelector('.node-kind')?.textContent)

  // 未选目录时不能提交：W4 没有源图这条路，目录就是它的全部输入
  check('未选骨架目录时按钮禁用', nd().querySelector('.gen-btn')?.disabled === true)
  check('给出缺骨架帧的提示',
    (nd().textContent || '').includes('缺骨架帧'), (nd().textContent || '').slice(-60))

  // 选目录（选项来自后端 /pose_dirs）
  const sel = nd().querySelector('.src-box select')
  const setSel = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSel(sel, 'D:////x////assets////_pose')
  await wait(150)
  check('选目录后显示帧数（4 帧）',
    (nd().textContent || '').includes('这个目录有 4 帧'),
    (nd().textContent || '').slice(-60))

  // 手填路径也要能生效（后端接受目录或通配）
  check('有手填路径的输入框', !!nd().querySelector('.dir-input'))

  // 填提示词后提交
  const ta = nd().querySelector('textarea.prompt')
  const setTa = (el, v) => {
    const st = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    st.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setTa(ta, 'white cat knight, red cape')
  await wait(120)

  extraMeta = { candidates: 2, attempts: 9, frames: 4, gate_passed: 4 }
  nd().querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(250)

  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  check('提交 workflow=w4', body.workflow === 'w4', body.workflow)
  check('pose_images 传的是目录（后端展开成排序后的帧列表）',
    body.params.pose_images === 'D:////x////assets////_pose', body.params.pose_images)
  check('带上候选与补提（W4 的产线机制）',
    body.params.candidates === 2 && body.params.gate_retries === 1,
    JSON.stringify({ c: body.params.candidates, r: body.params.gate_retries }))
  check('cols=0 表示按帧数自动排',
    body.params.cols === 0, body.params.cols)

  // 结果统计（W4 分支和序列帧分支是两段独立 JSX，必须各验一次）
  for (let i = 0; i < 20; i++) {
    if (nd().querySelector('.meta')) break
    await wait(200)
  }
  const metaLine = nd().querySelector('.meta')?.textContent || ''
  check('W4 结果显示候选与闸门', metaLine.includes('候选 2/帧') && metaLine.includes('闸门 4/4'),
    metaLine)
  extraMeta = null

  check('姿态序列场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 27) 节点可拖宽 + 参数 tooltip
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))

  const item = [...section(doc, '节点').querySelectorAll('.rail-item')]
    .find((it) => (it.querySelector('b')?.textContent || '') === '图生成')
  item.querySelector('.rail-add').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(150)

  const nd = () => doc.querySelector('.node')
  check('节点默认宽 300', nd().style.width === '300px', nd().style.width)
  check('有拖宽手柄', !!nd().querySelector('.node-resize'))

  // 拖右边缘 100px：宽度应变成 400
  const h = nd().querySelector('.node-resize')
  h.dispatchEvent(new win.MouseEvent('pointerdown', { bubbles: true, clientX: 300 }))
  win.dispatchEvent(new win.MouseEvent('pointermove', { clientX: 400 }))
  await wait(120)
  check('拖动后节点变宽（300 → 400）',
    parseInt(nd().style.width, 10) === 400, nd().style.width)
  win.dispatchEvent(new win.MouseEvent('pointerup', {}))

  // 宽度有上下限：拖到很窄也不会崩成 0
  check('宽度不会被拖到不可用（>=260）',
    parseInt(nd().style.width, 10) >= 260, nd().style.width)

  // 参数 tooltip：每一条都要说清"是什么/影响/推荐"
  const lab = (name) => [...nd().querySelectorAll('label')]
    .find((l) => (l.textContent || '').trim().startsWith(name))
  const stepsTip = lab('步数')?.getAttribute('title') || ''
  check('步数有 tooltip', stepsTip.length > 10, stepsTip)
  check('tooltip 含「影响」与「推荐」',
    stepsTip.includes('影响：') && stepsTip.includes('推荐：'), stepsTip)
  check('推荐值写的是具体数字（不是"视情况"）',
    /\d/.test(stepsTip.split('推荐：')[1] || ''), stepsTip)

  // 像素块/色数**不再让用户填**（2026-11-02）：那两个数强耦合，填错不报错、
  // 只是出图看着不对；而且「只填色数」以前还会硬切 alpha、在透明图边上压出白边。
  // 现在一档「产出风格」决定全部，这里钉住「不许再暴露回去」。
  check('参数区没有「像素块」输入', !lab('像素块'), '像素块不该再暴露')
  check('参数区没有「色数」输入', !lab('色数'), '色数不该再暴露')
  const allTips = [...nd().querySelectorAll('label[title]')]
    .map((l) => l.getAttribute('title') || '')
  check('没有「未定义说明」的占位',
    !allTips.some((t) => t.includes('未定义说明')),
    allTips.find((t) => t.includes('未定义说明')) || '')

  // 参数区整体覆盖率：至少 10 个 label 带说明
  const withTip = [...nd().querySelectorAll('label[title]')]
  check('参数区大部分控件都有 tooltip（>=10 个）', withTip.length >= 10, String(withTip.length))

  check('拖宽与 tooltip 场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 28) 原本是 LoRA 扫描用例，已随 LoRA 一起删除（见 7) 的说明）。

// 29) 原本是「叠加多个 LoRA」用例，已随 LoRA 一起删除。

// 30) 像素块要显示有效像素，并在极端值时告警
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))
  await wait(500)
  doc.querySelector('details.fold').open = true
  await wait(120)

  const setNum = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set
    setter.call(el, String(v))
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  const lab = (name) => [...doc.querySelectorAll('label')]
    .find((l) => (l.textContent || '').trim().startsWith(name))

  // ---- 风格档位真的能切（回归：连调 setParam 会互相覆盖）----
  // 这一档一次要改 4 个字段（契约 + 像素块 + 色数 + 档位）。以前用连调
  // setParam 实现，每次展开源都是渲染时闭包的旧 params，只有最后一次生效，
  // 结果档位立刻弹回「自定义参数」——用户看到的就是「根本选不了」。
  // 下面必须测**会改多个字段的档位**，只测「自定义」档会漏掉这个 bug。
  const stylePick = () => [...doc.querySelectorAll('select')].find((x) =>
    [...x.options].some((o) => o.textContent.includes('像素精灵·高清')))
  const setSelValue = Object.getOwnPropertyDescriptor(
    win.HTMLSelectElement.prototype, 'value').set
  {
    const s0 = stylePick()
    check('存在「产出风格」下拉', !!s0)
    setSelValue.call(s0, 'spriteHi')
    s0.dispatchEvent(new win.Event('change', { bubbles: true }))
    await wait(150)
    check('切到「像素精灵·高清」后下拉不回弹',
      stylePick()?.value === 'spriteHi', stylePick()?.value)

    // 连切两次，确认第二次同样生效（覆盖「第一次碰巧没坏」的情况）
    const s1 = stylePick()
    setSelValue.call(s1, 'pixel8')
    s1.dispatchEvent(new win.Event('change', { bubbles: true }))
    await wait(150)
    check('连续切到「8-bit」也生效', stylePick()?.value === 'pixel8', stylePick()?.value)

    // 参数要真的落地，不只是下拉显示对。旧写法是切到「自定义」看像素块输入值，
    // 但自定义档已经不给手选了 —— 改成看**尺寸**：各档自带尺寸，切过去必须跟着变
    // （spriteIll 是 1024²，spriteHi 是 768²）。这同样是"一次改多个字段"的证据。
    const sizeSel = () => [...doc.querySelectorAll('select')]
      .find((x) => [...x.options].some((o) => o.textContent.includes('1024×1024')))
    setSelValue.call(stylePick(), 'spriteIll')
    stylePick().dispatchEvent(new win.Event('change', { bubbles: true }))
    await wait(200)
    check('切到「像素精灵·Illustrious」后尺寸跟着变 1024²',
      (sizeSel()?.value || '').includes('1024'), sizeSel()?.value)
    // 切回像素精灵·高清，为后续用例保持默认态
    const s3 = stylePick()
    setSelValue.call(s3, 'spriteHi')
    s3.dispatchEvent(new win.Event('change', { bubbles: true }))
    await wait(150)
    check('切回后尺寸回到 768²',
      (sizeSel()?.value || '').includes('768'), sizeSel()?.value)
  }

  // 「自定义参数」不再是可选档位：像素块/色数这两个手填入口已经删掉。
  // 旧存档里参数偏离档位时下拉会显示这个只读条目，但不该能选。
  const styleSel = stylePick()
  const customOpt = [...(styleSel?.options || [])]
    .find((o) => o.textContent.includes('自定义参数'))
  check('「产出风格」下拉还在', !!styleSel)
  check('「自定义参数」只读、不可选', !!customOpt && customOpt.disabled === true,
    String(customOpt?.disabled))
  check('风格档位仍是 7 档可选项',
    [...(styleSel?.options || [])].filter((o) => !o.disabled).length === 7,
    String([...(styleSel?.options || [])].filter((o) => !o.disabled).length))
  check('默认不暴露像素块（由风格档位决定）', !lab('像素块'), '像素块本该隐藏')
  check('默认不暴露色数（由风格档位决定）', !lab('色数'), '色数本该隐藏')

  check('像素块/色数入口场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 31) 色数 / 像素块不再暴露给用户（画风由档位 + 提示词决定）
//
// 2026-11-02：用户反馈「业界其他 AI 生图没让人填色数和像素块」，而且
// **「只填色数」以前会硬切 alpha** —— 白底预乘的过渡带（RGB 220~240）被
// 实化成完全不透明的白，深色背景上就是一圈白边。两个入口一起删掉：
// 手填的参数不再存在，后端的量化也只跟着「像素化」走（见 test_paint_path）。
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))
  await wait(500)
  doc.querySelector('details.fold').open = true
  await wait(120)

  const lab = (name) => [...doc.querySelectorAll('label')]
    .find((l) => (l.textContent || '').trim().startsWith(name))
  check('「更多设置」展开后没有像素化开关', !lab('像素化'), '像素化开关本该删除')
  check('没有色数输入框', !lab('色数'))
  check('没有像素块输入框', !lab('像素块'))

  // 档位仍在，且都带一句「这一档出什么」的说明
  const styleSel = [...doc.querySelectorAll('select')].find((x) =>
    [...x.options].some((o) => o.textContent.includes('像素精灵（16-bit）')))
  check('有「产出风格」下拉', !!styleSel)
  check('风格说明可见',
    (doc.querySelector('.node .src-note')?.textContent || '').length > 6,
    doc.querySelector('.node .src-note')?.textContent)

  // 契约兜底提示还在（选 none 时警告"没有全身/居中约束"）
  check('色数/像素块入口场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 32) 原本是 Refiner 精修开关用例，已随 SD 一起删除（Qwen 没有精修器）。

// 37) 人偶姿势节点：正文是官方 3D 人偶编辑器（iframe），朝向 + 导出入口
//
// **这里不放 SVG 骨架了**：2D 伪 3D 骨架编辑器已经整个删掉，节点正文换成官方
// PoseStudio 的内嵌壳（浏览器里是 WebGL 渲染的 3D 人偶）。jsdom 里 iframe 拉不到，
// 所以这里只断言**宿主这一层**：编辑器挂上了、朝向有 8 个、导出按钮在。
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))
  const add = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('人偶姿势'))
  check('侧栏有「人偶姿势」入口', !!add)
  add?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const node = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
  check('能创建人偶姿势节点', !!node)

  // 编辑器本体：官方 PoseStudio 的嵌入模式（只留画布，控制条在宿主这边）
  const frame = node?.querySelector('.pose-stage iframe')
  check('节点正文挂着 3D 人偶编辑器', !!frame,
    frame?.getAttribute('src') || '（没有 iframe）')
  check('编辑器走官方 PoseStudio 的嵌入模式',
    (frame?.getAttribute('src') || '').includes('/vnccs-pose/studio.html?embed=1'),
    frame?.getAttribute('src'))

  // **嵌入模式要真的生效**：studio.html 里那套 `html.embed #bar { display:none }`
  // 一直在，但以前没人给 <html> 加 `embed` class —— 于是节点卡片里上下两套一模一样的
  // 工具条（iframe 的朝向/重置/导出 + 宿主的），而且 iframe 那个导出点了没反应。
  const studioHtml = readFileSync(resolve(dist, 'vnccs-pose/studio.html'), 'utf8')
  check('iframe 页会按 ?embed=1 打上 embed 标记',
    studioHtml.includes("classList.add('embed')"), 'studio.html 少了加 class 的脚本')
  // **动作导入解析器必须在本地**：这三个文件一旦缺失，导入会悄悄回落到
  // esm.sh（断网就不能用了），而且 JS 里没有任何报错能看出这件事。
  check('three 的 glb/fbx 解析器随包发布（断网也能导入动作）',
    ['loaders/GLTFLoader.js', 'loaders/FBXLoader.js', 'libs/fflate.module.js',
      'curves/NURBSCurve.js', 'utils/BufferGeometryUtils.js']
      .every((f) => existsSync(resolve(dist, 'vnccs-pose/vendor/three/jsm', f))),
    'vnccs-pose/vendor/three/jsm 下缺文件')
  // 上游 loader 里写的是裸名 `from 'three'`，全靠这条 importmap 落到 vendor 的
  // three 上；而且必须写在 module 脚本**之前**（规范要求，晚了解析不到）。
  const mapAt = studioHtml.indexOf('type="importmap"')
  const modAt = studioHtml.indexOf('type="module"')
  check('studio.html 有 importmap 把 three 指到 vendor',
    mapAt > 0 && studioHtml.includes('"three": "./vendor/three/three.module.js"'),
    '没有 importmap，loader 会去联网找 three')
  check('importmap 写在 module 脚本之前',
    mapAt > 0 && modAt > mapAt, `importmap@${mapAt} module@${modAt}`)
  // 工具条从 DOM 里删掉（只保留宿主一份），不是靠 CSS 藏——藏起来的按钮
  // 仍然能被点、仍然会把人绕进"点了没反应"的死路。
  check('iframe 里不再有自带的工具条（按钮只有宿主一份）',
    !studioHtml.includes('id="bar"') && !studioHtml.includes('id="hint"'),
    'studio.html 里还留着 #bar / #hint')

  // 宿主控制条：四个动作按钮文案要完整（以前写成"视角/站立/从图"，
  // 加上 .mini 只有 28px 宽，两字被挤成竖排）
  const actBtns = [...(node?.querySelectorAll('.pose-actions .mini') || [])]
    .map((b) => (b.textContent || '').trim())
  check('动作按钮文案完整（重置视角/重置姿势/从图片提姿势/导出人偶图）',
    actBtns.includes('重置视角') && actBtns.includes('重置姿势')
    && actBtns.includes('从图片提姿势') && actBtns.includes('导出人偶图'),
    actBtns.join(' | '))
  check('导出按钮在最后（独占一行）',
    actBtns[actBtns.length - 1] === '导出人偶图', actBtns.join(' | '))
  check('加载中给出提示（37MB 人偶模型要等）',
    (node?.querySelector('.pose-status')?.textContent || '').includes('人偶模型'),
    node?.querySelector('.pose-status')?.textContent || '（无）')

  // 朝向 = 导出机位。8 个精灵常用方向，正面是默认。
  const yawSel = node?.querySelector('.pose-row select')
  check('朝向有 8 个方向', yawSel?.options.length === 8, String(yawSel?.options.length))
  check('朝向默认正面', yawSel?.value === '0', yawSel?.value)
  const setSel = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value')
      .set.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSel(yawSel, '90')
  await new Promise((r) => setTimeout(r, 120))
  check('朝向切到 90° 后回显', yawSel?.value === '90', yawSel?.value)

  // 导出按钮：**没就绪时必须是禁用的**（不然点了没反应，也没人告诉用户为什么）
  const buttons = [...(node?.querySelectorAll('button') || [])]
  check('有「导出人偶图」按钮',
    buttons.some((b) => (b.textContent || '').includes('导出人偶图')))
  const exportBtn = buttons.find((b) => (b.textContent || '').includes('导出人偶图'))
  check('渲染器没就绪时导出按钮禁用（不给无声的失败）', exportBtn?.disabled === true,
    String(exportBtn?.disabled))
  check('有「从图片」提姿势的入口', !!node?.querySelector('input[type=file]'))
  check('说明提到导出的是 3D 人偶',
    (node?.querySelector('.src-note')?.textContent || '').includes('3D 人偶'),
    (node?.querySelector('.src-note')?.textContent || '').slice(0, 40))
  check('人偶节点场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 38) 人偶姿势：从图片自动提姿势（DWPose 由后端 mock，2D 点交给官方转换器）
//
// 断言的是**投递**：提取到的 18 个点要进 `node.poseImport`，节点里的编辑器
// 收到后走官方 convertOpenPoseToPose 解成 3D 骨骼。3D 反解必须在渲染器里做
// （要用官方骨骼的静止长度），所以前端这里只负责把点送进去。
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))
  const add = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('人偶姿势'))
  add?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const node = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))

  // **按 accept 精确选**：节点里现在有两个 file input（从图片提姿势 / 导入动作），
  // 只按 `input[type=file]` 会抓到先出现的那个（导入动作），用例就测错东西了。
  const input = node?.querySelector('input[type=file][accept*="image"]')
  check('编辑器有「从图片」的文件入口', !!input)
  // 导入动作的入口（glb/gltf/fbx）
  check('编辑器有「导入动作」的文件入口',
    !!node?.querySelector('input[type=file][accept*=".glb"]'))
  check('有「导入动作」按钮',
    [...(node?.querySelectorAll('button') || [])]
      .some((b) => (b.textContent || '').includes('导入动作')))
  // 抽帧那行**只在载入动作之后**才该出现：没见过文件就摆「存为姿势集」，
  // 用户点了只会得到"还没载入动作"。
  check('没导入动作时不摆「存为姿势集」（避免点了必然失败）',
    !node?.querySelector('.pose-frames'))

  // **单张和批量是两条独立的路**：角色迁移只管一张，「姿态批量」是另一个节点。
  // 以前批量塞在角色迁移的折叠区里，出现过"选了姿势集却没接角色图"的错位状态。
  check('角色迁移节点里不再有批量区（模式互斥）',
    !node?.querySelector('.pose-batch-box'),
    node?.querySelector('.pose-batch-box') ? '还留着批量折叠块' : '')

  const batchAdd = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('姿态批量'))
  check('侧栏有「姿态批量」入口', !!batchAdd)
  batchAdd?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const batchNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
  check('能创建姿态批量节点', !!batchNode)
  check('姿态批量节点有姿势集下拉', !!batchNode?.querySelector('.pose-batch select'))
  check('姿态批量节点有输出文件夹名输入',
    !!batchNode?.querySelector('.pose-batch-out'))
  check('姿态批量节点有「全部生成」按钮',
    [...(batchNode?.querySelectorAll('button') || [])]
      .some((b) => (b.textContent || '').includes('全部生成')))
  // 主操作要和别的节点一致：**整行绿色大按钮**（`gen-btn`）。
  // 用 `.pose-batch-btn` 时底色 `--bg` 比卡片还暗，看着像"透明的洞"。
  const pbGo = [...(batchNode?.querySelectorAll('button') || [])]
    .find((b) => (b.textContent || '').includes('全部生成'))
  check('「全部生成」是绿色主按钮（gen-btn）',
    !!pbGo && pbGo.classList.contains('gen-btn'),
    pbGo?.className || '（没有按钮）')
  // 底色三档：白 / 深 / **透明**。少「透明」就是用户报的那个"没有透明选项"。
  const pbBgSel = [...(batchNode?.querySelectorAll('select') || [])]
    .find((s2) => [...s2.options].some((o) => o.value === 'dark'))
  const pbBgVals = pbBgSel ? [...pbBgSel.options].map((o) => o.value) : []
  check('底色有「透明底」选项',
    pbBgVals.includes('transparent'), pbBgVals.join('|') || '（没找到底色下拉）')
  check('姿态批量节点接了角色图端口（0/1 未接时给提示）',
    !!batchNode?.querySelector('.conn.in.char'),
    batchNode?.querySelector('.conn.in')?.className || '（没有端口）')

  // 宿主控制条：四个动作按钮文案要完整（以前写成"视角/站立/从图"，
  // 加上 .mini 只有 28px 宽，两字被挤成竖排）
  const actBtns = [...(node?.querySelectorAll('.pose-actions .mini') || [])]
    .map((b) => (b.textContent || '').trim())
  check('动作按钮文案完整（重置视角/重置姿势/从图片提姿势/导出人偶图）',
    actBtns.includes('重置视角') && actBtns.includes('重置姿势')
    && actBtns.includes('从图片提姿势') && actBtns.includes('导出人偶图'),
    actBtns.join(' | '))
  check('导出按钮在最后（独占一行）',
    actBtns[actBtns.length - 1] === '导出人偶图', actBtns.join(' | '))
  check('加载中给出提示（37MB 人偶模型要等）',
    (node?.querySelector('.pose-status')?.textContent || '').includes('人偶模型'),
    node?.querySelector('.pose-status')?.textContent || '（无）')

  // 朝向 = 导出机位。8 个精灵常用方向，正面是默认。
  const yawSel = node?.querySelector('.pose-row select')
  check('朝向有 8 个方向', yawSel?.options.length === 8, String(yawSel?.options.length))
  check('朝向默认正面', yawSel?.value === '0', yawSel?.value)
  const setSel = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value')
      .set.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSel(yawSel, '90')
  await new Promise((r) => setTimeout(r, 120))
  check('朝向切到 90° 后回显', yawSel?.value === '90', yawSel?.value)

  // 导出按钮：**没就绪时必须是禁用的**（不然点了没反应，也没人告诉用户为什么）
  const buttons = [...(node?.querySelectorAll('button') || [])]
  check('有「导出人偶图」按钮',
    buttons.some((b) => (b.textContent || '').includes('导出人偶图')))
  const exportBtn = buttons.find((b) => (b.textContent || '').includes('导出人偶图'))
  check('渲染器没就绪时导出按钮禁用（不给无声的失败）', exportBtn?.disabled === true,
    String(exportBtn?.disabled))
  check('有「从图片」提姿势的入口', !!node?.querySelector('input[type=file]'))
  check('说明提到导出的是 3D 人偶',
    (node?.querySelector('.src-note')?.textContent || '').includes('3D 人偶'),
    (node?.querySelector('.src-note')?.textContent || '').slice(0, 40))
  check('人偶节点场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 38) 人偶姿势：从图片自动提姿势（DWPose 由后端 mock，2D 点交给官方转换器）
//
// 断言的是**投递**：提取到的 18 个点要进 `node.poseImport`，节点里的编辑器
// 收到后走官方 convertOpenPoseToPose 解成 3D 骨骼。3D 反解必须在渲染器里做
// （要用官方骨骼的静止长度），所以前端这里只负责把点送进去。
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))
  const add = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('人偶姿势'))
  add?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const node = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))

  // **按 accept 精确选**：节点里现在有两个 file input（从图片提姿势 / 导入动作），
  // 只按 `input[type=file]` 会抓到先出现的那个（导入动作），用例就测错东西了。
  const input = node?.querySelector('input[type=file][accept*="image"]')
  check('编辑器有「从图片」的文件入口', !!input)
  // 导入动作的入口（glb/gltf/fbx）
  check('编辑器有「导入动作」的文件入口',
    !!node?.querySelector('input[type=file][accept*=".glb"]'))
  check('有「导入动作」按钮',
    [...(node?.querySelectorAll('button') || [])]
      .some((b) => (b.textContent || '').includes('导入动作')))
  // 抽帧那行**只在载入动作之后**才该出现：没见过文件就摆「存为姿势集」，
  // 用户点了只会得到"还没载入动作"。
  check('没导入动作时不摆「存为姿势集」（避免点了必然失败）',
    !node?.querySelector('.pose-frames'))

  // 角色迁移节点的**批量面板**：默认收在 <details> 里（单张是主用法），
  // 但必须能打开 —— 它是"姿势集 → 一整套成片"的唯一入口。
  const cloneAdd = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('角色迁移'))
  check('侧栏有「角色迁移」入口', !!cloneAdd)
  cloneAdd?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const cloneNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('角色迁移'))
  check('能创建角色迁移节点', !!cloneNode)
  // **批量不在这个节点里**：单张 / 批量是两种模式，批量已经独立成
  // 「姿态批量」节点。角色迁移里再留一个折叠的批量区，就会出现
  // "选了姿势集却没接角色图"这种半截状态（实测踩过）。
  check('角色迁移节点里没有批量区（模式互斥）',
    !cloneNode?.querySelector('.pose-batch-box'),
    cloneNode?.querySelector('.pose-batch-box') ? '还留着批量折叠块' : '')
  const file = new win.File([new Uint8Array([1, 2, 3])], 'pose.png', { type: 'image/png' })
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  input.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 500))

  const up = calls.filter((c) => c.url.includes('/upload')).pop()
  check('先把原图传上去', !!up, up?.url)
  const ex = calls.find((c) => c.url.endsWith('/pose/extract'))
  check('调了 /pose/extract', !!ex, ex?.url)
  // 画布上是 demo 节点 + 新建的人偶节点两个，别按下标取，**按有没有 poseImport 找**
  const saved = JSON.parse(
    calls.filter((c) => c.url.includes('/projects/') && c.method === 'PUT').pop()
      ?.body || '{}')
  const savedPose = (saved?.nodes || []).find((n) => n?.poseImport?.points)
  check('提取结果写进节点（poseImport，18 个点）',
    savedPose?.poseImport?.points?.length === 18,
    String(savedPose?.poseImport?.points?.length))
  check('提示带上提取到的 18/18 关节',
    (doc.querySelector('.toast')?.textContent || '').includes('18/18'),
    doc.querySelector('.toast')?.textContent)
  check('从图片提姿势场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 39) 侧栏 LoRA 面板：列出可用 LoRA、勾选即挂载
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 600))
  // 排除快速模式那一行（它也是 .lora-row）
  const rows = [...doc.querySelectorAll('.lora-row:not(.fast-toggle)')]
  check('侧栏列出可用的 Qwen LoRA', rows.length === 2, String(rows.length))
  check('LoRA 行显示名称与体积',
    (rows[0]?.textContent || '').includes('qwen21_anyangle')
    && (rows[0]?.textContent || '').includes('114MB'),
    rows[0]?.textContent)
  const cb = rows[0]?.querySelector('input[type=checkbox]')
  check('LoRA 有勾选框', !!cb)
  cb?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  const apply = calls.find((c) => c.url.endsWith('/loras/apply'))
  check('勾选后调用了 /loras/apply', !!apply, apply?.url)
  if (apply) {
    const body = JSON.parse(apply.body)
    check('apply 请求带 LoRA 名与强度',
      body.items?.length === 1 && body.items[0].name === 'qwen21_anyangle',
      JSON.stringify(body))
  }
  check('LoRA 面板无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 40) 快速模式：侧栏开关 → 调 /fast_mode
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 600))
  const row = doc.querySelector('.lora-row.fast-toggle')
  check('侧栏有「快速模式」开关', !!row, row?.textContent)
  const fastNote = row?.parentElement?.querySelector('.src-note')
  check('快速模式说明提到步数与提速',
    (fastNote?.textContent || '').includes('8')
    && (fastNote?.textContent || '').includes('9.3s'),
    fastNote?.textContent)
  const cb = row?.querySelector('input[type=checkbox]')
  cb?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  const fm = calls.find((c) => c.url.endsWith('/fast_mode'))
  check('勾选后调用了 /fast_mode', !!fm, fm?.url)
  if (fm) {
    check('/fast_mode 请求带 on=true', JSON.parse(fm.body).on === true, fm.body)
  }
  check('快速模式无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 41) 扩图 / 换装 / 换脸：三个新节点能创建、正文渲染正确
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 600))

  const addOf = (name) => [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes(name))

  check('侧栏有「扩图」「换装」「换脸」三个入口',
    !!addOf('扩图') && !!addOf('换装') && !!addOf('换脸'),
    [...doc.querySelectorAll('.rail-item b')].map((b) => b.textContent).join('/'))

  // 扩图：5 个方向按钮 + 比例滑块
  addOf('扩图')?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const outNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('扩图'))
  check('能创建扩图节点', !!outNode)
  check('扩图有 5 个方向按钮（右/左/下/上/四周）',
    outNode?.querySelectorAll('.skel-dir').length === 5,
    String(outNode?.querySelectorAll('.skel-dir').length))
  const allBtn = [...(outNode?.querySelectorAll('.skel-dir') || [])]
    .find((b) => (b.textContent || '') === '四周')
  allBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('点「四周」后高亮', !!allBtn?.classList.contains('on'), allBtn?.className)

  // 换装：有 char 输入端口
  addOf('换装')?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const outfitNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('换装'))
  check('能创建换装节点', !!outfitNode)
  check('换装有第二路输入端口（服装图）',
    !!outfitNode?.querySelector('.conn.in.char'))
  check('换装提示提到「身份/姿势/背景」保留',
    (outfitNode?.textContent || '').includes('保留图 1'),
    (outfitNode?.querySelector('.src-note')?.textContent || '').slice(0, 40))

  // 换脸
  addOf('换脸')?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const faceNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('换脸'))
  check('能创建换脸节点', !!faceNode)
  check('换脸提示提到「脸与发型」',
    (faceNode?.textContent || '').includes('脸与发型'),
    (faceNode?.textContent || '').slice(0, 60))
  check('新节点场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 42) 人偶姿势 → 角色迁移：人偶图连上后「迁移角色」必须可点
//
// 用户报的原话：「骨骼编辑连线，无法迁移角色」。根因是卡片判断"有没有源图"时
// 只认 `job` 和 `kind === 'image'`，而**人偶编辑器把导出的人偶图放在 `image` 上、
// 却不是 image 类型、也没有 job** → 明明连上了，卡片显示「无源图」、
// 「迁移角色」按钮一直是禁用的。这条用例把那个组合钉死。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))

  // 1. 建人偶姿势节点
  const skelAdd = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('人偶姿势'))
  check('有「人偶姿势」入口', !!skelAdd)
  skelAdd?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))

  const skelNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))

  // 2. 导出人偶图：走的是**节点里那块 iframe 的 postMessage 链路**。
  //    jsdom 不加载 iframe 内容，所以把 contentWindow 换成空壳（吞掉父页的 init），
  //    再手动喂官方那侧的两条消息：ready（模型加载完）→ result（用户点了导出）。
  //    组件按 e.source 认自己的子窗，所以 source 要带上这个空壳。
  const frame = skelNode?.querySelector('.pose-stage iframe')
  const fakeWin = { postMessage: () => {} }
  if (frame) {
    Object.defineProperty(frame, 'contentWindow', {
      value: fakeWin, configurable: true,
    })
  }
  win.dispatchEvent(new win.MessageEvent('message',
    { data: { type: 'ready' }, source: fakeWin }))
  await new Promise((r) => setTimeout(r, 150))
  const exportBtn = [...(skelNode?.querySelectorAll('button') || [])]
    .find((b) => (b.textContent || '').includes('导出人偶图'))
  check('渲染器就绪后导出按钮可点', exportBtn?.disabled === false,
    String(exportBtn?.disabled))
  exportBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  win.dispatchEvent(new win.MessageEvent('message', {
    data: { type: 'result', dataURL: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==' },
    source: fakeWin,
  }))
  await new Promise((r) => setTimeout(r, 800))
  check('导出人偶图后节点上有产物（image 字段）',
    !!skelNode?.querySelector('.pose-preview img'),
    skelNode?.querySelector('.pose-preview')?.textContent || '（没有 .pose-preview）')

  // 3. 建角色迁移节点
  const pcAdd = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('角色迁移'))
  pcAdd?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const pcNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '') === '角色迁移')
  check('新建出角色迁移节点', !!pcNode)
  check('一开始「迁移角色」是禁用的（还没连）',
    pcNode?.querySelector('.gen-btn')?.disabled === true)

  // 4. 连一条线：人偶姿势节点的输出 → 角色迁移的「动作参考」
  skelNode?.querySelector('.conn.out')
    .dispatchEvent(new win.PointerEvent('pointerdown', { bubbles: true, button: 0 }))
  await new Promise((r) => setTimeout(r, 60))
  pcNode.dispatchEvent(new win.PointerEvent('pointerover', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  win.dispatchEvent(new win.PointerEvent('pointerup', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))

  const srcSel = pcNode.querySelector('.src-box select')
  check('连线上后动作参考选了人偶节点',
    srcSel?.value === skelNode?.getAttribute('data-node-id'),
    `select=${srcSel?.value} skel=${skelNode?.getAttribute('data-node-id')}`)
  // **关键断言**：骨架图在 `image` 上，卡片必须认它，缩略图要出来
  check('动作参考的缩略图出来了（不再显示「无源图」）',
    !!pcNode?.querySelector('.src-thumb img'),
    pcNode?.querySelector('.src-thumb')?.textContent || '（空）')
  check('提示说明用的是导出的 3D 人偶图',
    (pcNode?.querySelector('.src-note')?.textContent || '').includes('3D 人偶图'),
    pcNode?.querySelector('.src-note')?.textContent)

  // 5. 角色图：拖一张本地图进来当身份来源
  const viewport = doc.querySelector('.canvas-viewport')
  const file = new win.File([new Uint8Array([1])], 'char.png', { type: 'image/png' })
  const ev = new win.Event('drop', { bubbles: true })
  Object.defineProperties(ev, {
    clientX: { value: 320 }, clientY: { value: 220 },
    dataTransfer: { value: { types: ['Files'], files: [file], getData: () => '' } },
  })
  viewport.dispatchEvent(ev)
  await new Promise((r) => setTimeout(r, 400))

  const imgNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '') === '图片')
  check('拖进来的图成了图片节点（角色图来源）', !!imgNode)
  const charSel = pcNode.querySelectorAll('.src-box select')[1]
  const charSet = Object.getOwnPropertyDescriptor(
    win.HTMLSelectElement.prototype, 'value').set
  charSet.call(charSel, imgNode.getAttribute('data-node-id') || '')
  charSel.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))

  // 6. **这就是用户要的**：两个源都齐了，按钮必须能点
  const runBtn = pcNode.querySelector('.gen-btn')
  check('两个源齐了之后「迁移角色」可以点',
    runBtn?.disabled === false,
    runBtn?.disabled ? `仍然禁用（title=${runBtn?.title}）` : 'ok')

  // 7. 朝向：人偶图的朝向**已经烘焙在图里**（相机机位渲染的），
  //    所以默认「自动」一个字都不写——补方向词是重复指令，还会压过姿势。
  const viewSel = [...pcNode.querySelectorAll('label.bg-mode')]
    .find((l) => (l.textContent || '').trim().startsWith('朝向'))?.querySelector('select')
  check('节点上有「朝向」下拉', !!viewSel,
    viewSel ? [...viewSel.options].map((o) => o.value).join(',') : '（没有）')
  check('朝向已改为「由图决定」且禁用（不再写方向词）',
    viewSel?.value === 'auto' && viewSel?.disabled,
    `value=${viewSel?.value} disabled=${viewSel?.disabled}`)

  // 自动：上游是人偶编辑器（默认机位 0° 正面）
  const genCalls = () => calls.filter((c) => c.url.endsWith('/generate'))
  const submitAndRead = async () => {
    // 上一次提交后节点会短暂处于 running（按钮禁用），必须等**新的一条**请求，
    // 否则读到的是上一发的 payload——这里踩过一次，断言会假阳性。
    const before = genCalls().length
    for (let i = 0; i < 40; i++) {
      if (!runBtn?.disabled) break
      await new Promise((r) => setTimeout(r, 25))
    }
    runBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
    for (let i = 0; i < 60 && genCalls().length === before; i++) {
      await new Promise((r) => setTimeout(r, 25))
    }
    const last = genCalls().pop()
    return last ? JSON.parse(last.body) : { params: {} }
  }

  // 自动 + 骨骼 0°：**不该写方位词**。
  // 实测「front view」这类词会压过骨架姿势（分腿张臂的骨架会被画成站立），
  // 而正面本来就是模型的默认——写了只有害处。
  let body = await submitAndRead()

  check('自动 + 人偶 0° 不写方位词（正面是默认，写了会压过姿势）',
    !String(body.params.prompt || '').includes('viewing direction'),
    body.params.prompt)

  // 把人偶的导出机位切到「右侧」（90°）→ 「自动」仍然不写方向词
  // （VNCCS 官方从不写方向词，方向长在渲染的机位里；写文字会压过姿势）
  const yawSel2 = skelNode?.querySelector('.pose-row select')
  Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value')
    .set.call(yawSel2, '90')
  yawSel2?.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  check('人偶机位切到 90°（界面回显）', yawSel2?.value === '90', yawSel2?.value)
  body = await submitAndRead()
  check('自动（即使人偶 90°）也不写方向词',
    !String(body.params.prompt || '').includes('viewing direction'),
    body.params.prompt)

  // 9. 人偶图已经是这个节点**唯一的产物**（线框骨架随 2D 编辑器一起删了），
  //    所以只看它落地之后下游怎么用。
  const prevs = [...(skelNode?.querySelectorAll('.pose-preview') || [])]
  check('节点上只有人偶图这一个产物', prevs.length === 1, `previews=${prevs.length}`)
  check('产物标注了「3D 人偶图」',
    (prevs[0]?.textContent || '').includes('3D 人偶图'),
    prevs[0]?.textContent || '（没有预览条）')
  // **核心断言**：提交的动作参考必须是 3D 人偶图（VNCCS LoRA 的训练分布），
  // 不是线框骨架——两者都在同一个节点上，靠 poseSourceOf 决定用哪个。
  check('提交的动作参考用的是 3D 人偶图（不是线框骨架）',
    String(body.params.pose_image || '').includes('mannequin.png'),
    String(body.params.pose_image))
  check('提交走的是 VNCCS 路线', body.workflow === 'wvnccs', String(body.workflow))
  // 人偶图的朝向已经烘焙在图里，所以**不该**再补方向词
  check('用了人偶图就不再补方向词（方向已在图里）',
    !String(body.params.prompt || '').includes('viewing direction'),
    String(body.params.prompt))
  // 8. 透明底：VNCCS 支持原生 alpha，但节点上以前没有这个开关
  //    （用户看到的是「底色」只有白/深两档，问「怎么没有透明」）
  const alphaBox = [...pcNode.querySelectorAll('details.adv label.check')]
    .find((l) => (l.textContent || '').includes('透明背景'))
  const alphaInput = alphaBox?.querySelector('input[type=checkbox]')
  check('节点上有「透明背景」勾选框', !!alphaInput,
    alphaBox?.textContent || '（没有）')
  check('VNCCS 引擎下透明底可勾（不是禁用）', alphaInput?.disabled === false)
  check('默认不勾', alphaInput?.checked === false)

  const bgSel = [...pcNode.querySelectorAll('details.adv label.bg-mode')]
    .find((l) => (l.textContent || '').trim().startsWith('底色'))?.querySelector('select')
  check('没勾透明底时「底色」可选', bgSel?.disabled === false)

  alphaInput?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('勾上之后「底色」被禁用（模型直接出 alpha，底色没意义）',
    bgSel?.disabled === true, `disabled=${bgSel?.disabled}`)

  body = await submitAndRead()
  check('勾透明底后提交带上 transparent=true',
    body.params.transparent === true, String(body.params.transparent))

  check('骨骼→迁移场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 44) 资料库：提示词与姿势的浏览面板 + **localStorage 一个字不留**
//
// 用户明确要求：项目内容与收藏都不许留在浏览器里（清一次站点数据就全没了）。
// 现在提示词收藏 / 姿势库 / 界面偏好全在服务端磁盘（library/*.json），
// 这一组用例把「落点」钉死：收藏走 /library/prompts，界面里不留 meow.* 的键。
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))

  // 1. 收藏一条 → 必须打到服务端，不能只写 localStorage
  const ta = doc.querySelector('textarea.prompt')
  const setValue = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setValue(ta, 'red panda ranger')
  await new Promise((r) => setTimeout(r, 80))
  doc.querySelector('.star').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  const addCall = calls.find((c) => c.url.endsWith('/library/prompts') && c.method === 'POST')
  check('收藏打到服务端 /library/prompts（不写 localStorage）', !!addCall, addCall?.url)

  // 2. 打开资料库面板：两个 tab（提示词 / 姿势）
  const openLib = [...doc.querySelectorAll('.rail-lib .rail-item-main')]
    .find((b) => (b.textContent || '').includes('提示词收藏'))
  check('侧栏有资料库入口', !!openLib)
  openLib?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  const modal = doc.querySelector('.lib-modal')
  check('打开资料库浏览面板', !!modal)
  check('面板上有两个 tab（提示词 / 姿势）',
    [...(doc.querySelectorAll('.lib-tab') || [])].length === 2,
    String(doc.querySelectorAll('.lib-tab').length))
  const cards = () => [...doc.querySelectorAll('.lib-modal .lib-card')]
  check('提示词页显示刚收藏的那条',
    cards().some((c) => (c.textContent || '').includes('red panda ranger')),
    `cards=${cards().length}`)
  check('每条提示词都有「应用」与「删除」',
    !!cards()[0]?.textContent?.includes('应用') && !!cards()[0]?.textContent?.includes('删除'))

  // 3. 搜索框：按关键词过滤。
  //    **必须走原生 setter**：受控 input 直接赋 .value 会被 React 改回去，
  //    那样两边断言都"恰好"通过（收藏本来就只有一条），等于没测。
  const search = doc.querySelector('.lib-search')
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setInput(search, 'red panda')
  await new Promise((r) => setTimeout(r, 120))
  check('搜索命中的是那一条', cards().length === 1, `cards=${cards().length}`)
  setInput(search, 'zzz-不存在')
  await new Promise((r) => setTimeout(r, 120))
  check('搜不到时一条都不显示', cards().length === 0, `cards=${cards().length}`)
  check('搜不到时给出空态说明',
    (doc.querySelector('.lib-empty')?.textContent || '').includes('zzz-不存在'),
    doc.querySelector('.lib-empty')?.textContent || '（无空态）')
  setInput(search, '')
  await new Promise((r) => setTimeout(r, 100))

  // 4. 切到姿势页 → 官方库清单是**懒加载**的（只有真要看才拉）
  const poseTab = [...doc.querySelectorAll('.lib-tab')]
    .find((b) => (b.textContent || '').includes('姿势'))
  poseTab?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))
  check('切到姿势页才去取官方库清单',
    calls.some((c) => c.url.includes('/pose_library')), calls.map((c) => c.url).slice(-2).join())
  const srcBtns = [...doc.querySelectorAll('.lib-src-btn')]
  check('姿势页有「VNCCS 官方库 / 我存的」两个来源',
    srcBtns.length === 2 && srcBtns[0].textContent.includes('VNCCS'),
    srcBtns.map((b) => b.textContent).join(' | '))
  check('官方库按分类列出（带预览缩略图）',
    doc.querySelectorAll('.lib-modal .lib-thumb img').length === 2,
    String(doc.querySelectorAll('.lib-modal .lib-thumb img').length))
  check('卡片上带标签（挑姿势时知道是不是要的角度）',
    (doc.querySelector('.lib-tags')?.textContent || '').length > 0,
    doc.querySelector('.lib-tags')?.textContent || '（无标签）')
  const catSel = doc.querySelector('.lib-cat')
  check('有分类筛选下拉', !!catSel && catSel.options.length === 4,
    String(catSel?.options.length))

  // 5. 应用一条官方姿势：先取本体 json，再写进**选中的人偶节点**。
  //    没有选中人偶节点时按钮必须是禁用的（不然点了什么都没发生，也没人说明原因）。
  const applyBtn0 = [...doc.querySelectorAll('.lib-modal .lib-card')]
    .find((c) => (c.textContent || '').includes('爬行'))
    ?.querySelector('.lib-card-actions button')
  check('没选人偶节点时「应用」禁用并说明原因', applyBtn0?.disabled === true,
    applyBtn0?.title || String(applyBtn0?.disabled))

  // 建一个「人偶姿势」节点并选中它
  ;[...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('人偶姿势'))
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const poseNode = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
  poseNode?.dispatchEvent(new win.Event('pointerdown', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  const applyBtn = [...doc.querySelectorAll('.lib-modal .lib-card')]
    .find((c) => (c.textContent || '').includes('爬行'))
    ?.querySelector('.lib-card-actions button')
  applyBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  check('应用官方姿势时先去取它的 json',
    calls.some((c) => c.url.includes('/vnccs-pose/PoseLibrary/') && c.url.endsWith('.json')),
    calls.map((c) => c.url).slice(-3).join())
  // 自动保存有 400ms 防抖，等它落盘再看服务端收到了什么
  await new Promise((r) => setTimeout(r, 900))
  const applied = JSON.parse(
    calls.filter((c) => c.url.includes('/projects/') && c.method === 'PUT').pop()?.body || '{}')
  const appliedNode = (applied?.nodes || []).find((n) => n?.poseApply)
  check('取回来的官方姿势写进人偶节点（带朝向）',
    appliedNode?.poseApply?.yaw === 90 && !!appliedNode?.pose,
    JSON.stringify(appliedNode?.poseApply || {}).slice(0, 80))

  // 6. Esc 能关掉浮层（临时的面板不该只能去找 ×）
  win.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('Esc 关闭资料库面板', !doc.querySelector('.lib-modal'))

  // 7. **核心**：浏览器里不该再有任何 meow.* 的键
  const left = Object.keys(win.localStorage).filter((k) => k.startsWith('meow.'))
  check('localStorage 里不剩任何 meow.* 的键', left.length === 0, left.join(','))
  check('资料库场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 45) 旧存档迁移：浏览器里有老数据 → 搬进磁盘（/library/seed）后才清掉
{
  const legacy = {
    'meow.prompts.v1': JSON.stringify([
      { id: 'lp1', text: '旧收藏的提示词', negative: 'blurry', at: 1 },
    ]),
    'meow.maskColor': '#ff3b6b',
  }
  const { dom, errors } = await boot('', legacy)
  const doc = dom.window.document
  await new Promise((r) => setTimeout(r, 600))
  const seedCall = calls.find((c) => c.url.endsWith('/library/seed'))
  check('把浏览器里的旧存档交给服务端 /library/seed', !!seedCall, seedCall?.url)
  if (seedCall) {
    const b = JSON.parse(seedCall.body)
    check('迁的是旧提示词（含负向词）',
      b.prompts?.[0]?.text === '旧收藏的提示词' && b.prompts?.[0]?.negative === 'blurry',
      JSON.stringify(b.prompts?.[0]))
    check('旧偏好也一起迁', b.prefs?.['meow.maskColor'] === '#ff3b6b',
      JSON.stringify(b.prefs))
  }
  const left = Object.keys(dom.window.localStorage).filter((k) => k.startsWith('meow.'))
  check('服务端确认后浏览器里的旧键被清掉', left.length === 0, left.join(','))
  check('迁过来的提示词出现在侧栏',
    (section(doc, '资料库')?.textContent || '').includes('旧收藏的提示词'),
    (section(doc, '资料库')?.textContent || '').slice(0, 40))
  check('迁移场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46) 角色中心：与画布同级的一级页面（移植自 VNCCS 的 Character Creator）
//
// 角色库不属于任何一个画布——同一个角色要在多个项目里复用，所以它跟画布**同级**，
// 不是画布里的一个节点。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))

  const tabs = [...doc.querySelectorAll('.view-tab')]
  check('顶栏有「画布 / 角色中心」两个一级视图',
    tabs.length === 2 && tabs[0].textContent.includes('画布')
    && tabs[1].textContent.includes('角色中心'),
    tabs.map((t) => t.textContent).join('/'))

  tabs[1]?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))
  check('切到角色中心后画布节点不见了', !doc.querySelector('.node'))
  check('角色中心页面挂上了', !!doc.querySelector('.chars'))
  check('去服务端读了角色列表',
    calls.some((c) => c.url.endsWith('/characters')),
    calls.map((c) => c.url).slice(-2).join())
  check('空库时给出可照做的说明',
    (doc.querySelector('.chars-empty')?.textContent || '').includes('assets/characters'),
    (doc.querySelector('.chars-empty')?.textContent || '').slice(0, 40))

  // 建一个角色：名字只能含字母/数字/空格/-/_（后端 ensure_safe_name 会挡路径穿越）
  const input = doc.querySelector('.chars-new input')
  check('有新建角色的输入框', !!input)
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setInput(input, 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  ;[...doc.querySelectorAll('.chars-new button')][0]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  const created = calls.find((c) => c.url.endsWith('/characters') && c.method === 'POST')
  check('建角色打到服务端 POST /characters', !!created, created?.body)
  check('角色出现在库里', [...doc.querySelectorAll('.chars-card-name')]
    .some((n) => n.textContent === 'Alice'),
  [...doc.querySelectorAll('.chars-card-name')].map((n) => n.textContent).join())

  // 选中后右侧出详情：官方那套属性 + 四个 tab
  const card = [...doc.querySelectorAll('.chars-card')]
    .find((c) => (c.textContent || '').includes('Alice'))
  card?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  check('选中角色后拉详情',
    calls.some((c) => c.url.includes('/characters/Alice')))
  const subTabs = [...doc.querySelectorAll('.chars-tab')]
  check('详情有 属性/出图/服装/表情库/素材 五个 tab',
    subTabs.length === 5, subTabs.map((t) => t.textContent).join('/'))
  check('属性页有画风区（官方 358 条风格库）',
    (doc.querySelector('.chars-style')?.textContent || '').includes('画风'),
    (doc.querySelector('.chars-style')?.textContent || '').slice(0, 30))

  // 切回画布
  ;[...doc.querySelectorAll('.view-tab')][0]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))
  check('切回画布后角色中心收起', !doc.querySelector('.chars'))
  check('角色中心场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 47) 角色出图：官方结构化 prompt + 提交后自动收进角色目录
//
// 官方造角色不是一句自然语言，是**每个字段一行**的结构化 prompt
// （character_creator_v2.py:1124）——风格 LoRA 就是照这个格式调的。
// 这条用例钉住两件事：prompt 是后端按官方结构拼的；出完图后端直接收进
// Sprites/<服装>/<表情>/，不用前端再搬一次。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))

  // 进角色中心并建一个角色
  ;[...doc.querySelectorAll('.view-tab')][1]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  ;[...doc.querySelectorAll('.chars-new button')][0]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))

  // 切到「出图」页
  const genTab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('出图'))
  check('详情有「出图」页', !!genTab)
  genTab?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  check('向后端要了拼好的 prompt',
    calls.some((c) => c.url.includes('/characters/Alice/prompt')))
  const ta = doc.querySelector('textarea.chars-prompt')
  check('prompt 显示出来了（官方结构化格式）',
    (ta?.value || '').includes('The image contains one character.')
    && (ta?.value || '').includes('Gender: female'),
  (ta?.value || '').slice(0, 60))
  check('显示官方默认参数（25 步 / cfg 3）',
    (doc.querySelector('.chars-body')?.textContent || '').includes('25 步'),
    (doc.querySelector('.chars-body')?.textContent || '').slice(0, 60))
  check('显示官方公式算出的分辨率 792×1408',
    (doc.querySelector('.chars-body')?.textContent || '').includes('792×1408'))

  // Prompt Enhancement：官方用**图像模型自己的 Qwen3-VL** 改写（不新增权重）
  const peBox = doc.querySelector('.chars-check input[type=checkbox]')
  check('有 Prompt Enhancement 开关', !!peBox)
  check('PE 默认开（官方默认开）', peBox?.checked === true)
  check('PE 说明里写清了用的是图像模型自己的 Qwen3-VL',
    (doc.querySelector('.chars-check')?.textContent || '').includes('Qwen3-VL'),
    (doc.querySelector('.chars-check')?.textContent || '').slice(0, 50))
  peBox?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  const peCall = calls.filter((c) => c.url.includes('/characters/Alice/prompt')).pop()
  check('关掉 PE 后重新拼 prompt 且带 enhance=false',
    !!peCall && JSON.parse(peCall.body).enhance === false,
    peCall?.body)
  peBox?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))

  const genBtn = [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('生成底图'))
  check('有「生成底图」按钮', !!genBtn)
  genBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 600))
  const gen = calls.find((c) => c.url.includes('/characters/Alice/generate'))
  check('出图提交到 /characters/Alice/generate', !!gen, gen?.url)
  if (gen) {
    const b = JSON.parse(gen.body)
    check('提交的是 w1 文生图', b.workflow === 'w1', b.workflow)
    check('带了拼好的 prompt 与负向词',
      String(b.prompt || '').includes('Gender:') && b.negative === 'bad quality',
      String(b.prompt || '').slice(0, 40))
  }
  check('轮询了任务状态', calls.some((c) => c.url.includes('/jobs/')))
  check('出完图自动切到素材页',
    [...doc.querySelectorAll('.chars-tab')].some(
      (t) => t.classList.contains('on') && (t.textContent || '').includes('素材')))
  check('出图场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46b) 看图反推属性：官方 Cloner 旁路（本地 4B 模型）
//
// 它跟「手填属性」是两条路：模型读图给出 11 个字段，**先摆给用户看**，
// 用户点「用这些属性」才写进 config。属性错了会一路影响后面所有出图
// （prompt 是它拼的），所以绝不能拿到就自动保存。
{
  const { dom, errors, fetch: f } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  const body = () => doc.querySelector('.chars-body')?.textContent || ''
  const btn = () => [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('反推'))

  check('属性页有「从图片反推属性」按钮', !!btn())
  click(btn())
  await new Promise((r) => setTimeout(r, 500))
  check('调用了 /characters/Alice/analyze',
    calls.some((c) => c.url.includes('/characters/Alice/analyze')))
  check('结果先摆出来给人看（不是直接保存）',
    !!doc.querySelector('.chars-analyze'), '没有 .chars-analyze')
  check('结果里有模型读出的字段',
    (doc.querySelector('.chars-analyze')?.textContent || '').includes('silver hair'),
    (doc.querySelector('.chars-analyze')?.textContent || '').slice(0, 80))
  check('结果里有「用这些属性」按钮',
    [...doc.querySelectorAll('.chars-analyze button')]
      .some((b) => (b.textContent || '').includes('用这些属性')))

  // 点「用这些属性」→ 才写进 config
  const applyBtn = [...doc.querySelectorAll('.chars-analyze button')]
    .find((b) => (b.textContent || '').includes('用这些属性'))
  click(applyBtn)
  await new Promise((r) => setTimeout(r, 400))
  const put = calls.filter((c) => c.method === 'PUT'
    && c.url.includes('/characters/Alice/info')).pop()
  check('确认后才保存属性', !!put, put?.url)
  if (put) {
    const b = JSON.parse(put.body)
    check('保存的内容是模型读出来的字段',
      JSON.stringify(b).includes('silver hair'), JSON.stringify(b).slice(0, 90))
  }
  check('应用后结果面板收起', !doc.querySelector('.chars-analyze'))

  // ---- 失败路径：缺权重时要说清是缺什么，而不是"未知错误" ----
  f.setAnalyzeFail(true)
  click(btn())
  await new Promise((r) => setTimeout(r, 500))
  const txt = body()
  check('反推失败时把错误码/原因显示出来',
    txt.includes('MMPROJ_MISSING') || txt.includes('投影器'), txt.slice(-120))
  check('失败不影响手填（属性字段还在）',
    !!doc.querySelector('.chars-body select'))
  f.setAnalyzeFail(false)

  check('看图反推场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46c) AI 向导：一句话造角色（纯文本，官方 character_creator_v2.py:1740）
//
// 跟「看图反推」是一对：一个从图读、一个从话推。都是**生成完给用户看**，
// 确认才写进 config——属性错了会一路影响后面所有出图。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  const input = doc.querySelector('.chars-wizard-input')
  check('属性页有一句话输入框', !!input)
  const btn = [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('一句话生成'))
  check('有「一句话生成」按钮', !!btn)
  check('没输入时按钮是禁用的（避免空提交）', btn?.disabled === true)

  setInput(input, '一个苍白的精灵女剑士，银发红瞳，身材纤细')
  await new Promise((r) => setTimeout(r, 80))
  check('输入后按钮可用', btn.disabled === false)
  click(btn)
  await new Promise((r) => setTimeout(r, 500))

  const call = calls.find((c) => c.url.endsWith('/characters/wizard'))
  check('提交到 /characters/wizard', !!call, call?.url)
  if (call) {
    const b = JSON.parse(call.body)
    check('带上了用户那句话', String(b.description || '').includes('精灵'), b.description)
  }
  check('结果摆出来给人看', !!doc.querySelector('.chars-analyze'))
  check('结果里有模型推出来的字段',
    (doc.querySelector('.chars-analyze')?.textContent || '').includes('silver hair'),
    (doc.querySelector('.chars-analyze')?.textContent || '').slice(0, 80))
  check('跟官方一样补上了 breast 标签',
    (doc.querySelector('.chars-analyze')?.textContent || '').includes('small_breasts'))

  const applyBtn = [...doc.querySelectorAll('.chars-analyze button')]
    .find((b) => (b.textContent || '').includes('用这些属性'))
  click(applyBtn)
  await new Promise((r) => setTimeout(r, 400))
  const put = calls.filter((c) => c.method === 'PUT'
    && c.url.includes('/characters/Alice/info')).pop()
  check('确认后才保存（向导也不落盘）', !!put, put?.url)

  check('AI 向导场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46d) 换装：五个字段 → 一张换装底图（官方 Clothes Designer）
//
// **换装不是"只存描述"**：它把 top/bottom/head/shoes/face 拼成
// `Dress the character:` 指令，拿裸装底图重画一张，存成这套服装的底图。
// 这条用例钉住：AI 拆服装能把字段灌进表单；出图前会先把表单存进 config
// （不然换装读到的是上一次的内容）。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  // 进「服装」页
  const tab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('服装'))
  check('详情有「服装」页', !!tab)
  click(tab)
  await new Promise((r) => setTimeout(r, 400))
  const body = () => doc.querySelector('.chars-body')?.textContent || ''
  check('说明里写清了换装是「把描述变成一张图」',
    body().includes('把描述变成一张图') && body().includes('Dress the character'),
    body().slice(0, 160))

  // ---- 1. AI 拆服装 ----
  const input = doc.querySelector('.chars-wizard-input')
  check('服装页有一句话输入框', !!input)
  setInput(input, '护士，白大褂 + 荧光绿内衬 + 护目镜')
  await new Promise((r) => setTimeout(r, 80))
  const wizBtn = [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('AI 拆服装'))
  check('有「AI 拆服装」按钮', !!wizBtn)
  click(wizBtn)
  await new Promise((r) => setTimeout(r, 500))
  check('调用了 /clothes/wizard',
    calls.some((c) => c.url.includes('/characters/Alice/clothes/wizard')))
  const inputs = [...doc.querySelectorAll('.chars-costume input')]
  check('拆出来的字段填进了表单（顶部第一个）',
    (inputs[0]?.value || '').includes('nurse uniform'), inputs[0]?.value)
  check('face 字段也填了（护目镜这种穿戴物）',
    (inputs[4]?.value || '').includes('goggles'), inputs[4]?.value)

  // ---- 2. 选一套具体服装（Naked 不能换装）----
  const sel = doc.querySelector('.chars-row select')
  const setSelect = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  // mock 的角色只有 Naked，这里直接改成目标服装名（真实环境是新建后选中）
  setSelect(sel, sel.options[0].value)
  await new Promise((r) => setTimeout(r, 120))
  const designBtn = () => [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('换装底图'))
  check('有「出换装底图」按钮', !!designBtn())
  check('Naked 时按钮禁用（基础集不换装）', designBtn()?.disabled === true)

  // ---- 3. 一句话拆完再保存，才算数 ----
  const saveBtn = [...doc.querySelectorAll('.chars-costume button')]
    .find((b) => (b.textContent || '').includes('保存服装'))
  check('有「保存服装」按钮', !!saveBtn)
  click(saveBtn)
  await new Promise((r) => setTimeout(r, 400))
  const put = calls.filter((c) => c.method === 'PUT'
    && c.url.includes('/characters/Alice/costumes/')).pop()
  check('保存服装发到 /costumes/<服装>', !!put, put?.url)
  if (put) {
    const b = JSON.parse(put.body)
    check('存的是表单里的五个字段',
      JSON.stringify(b).includes('nurse uniform'), JSON.stringify(b).slice(0, 120))
  }

  check('换装场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46e) 多姿势批量：一个骨架帧目录 → 一整批精灵帧（官方 pose_generation）
//
// 三件事要钉住：整批**共用一个种子**（不是每张随机）、发布是**整批替换**
// （旧帧进 V1，跟表情那条逐张追加不一样）、**有进度条**（一批几十张要跑很久）。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  // 进「素材」页（多姿势生成放在这里，产出就是这一套帧）
  const tab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('素材'))
  check('详情有「素材」页', !!tab)
  click(tab)
  await new Promise((r) => setTimeout(r, 500))

  const body = () => doc.querySelector('.chars-body')?.textContent || ''
  check('素材页有多姿势批量区', !!doc.querySelector('.chars-posegen'))
  check('说明里写清了会整批替换（旧帧进 V1）',
    body().includes('整批替换') && body().includes('V1'), body().slice(-200))
  check('说明里写清了整批共用一个种子', body().includes('共用一个种子'))

  const sel = doc.querySelector('.chars-posegen select')
  check('骨架帧目录下拉有内容（来自 /pose_dirs）',
    (sel?.options.length || 0) >= 1, String(sel?.options.length))

  const btn = [...doc.querySelectorAll('.chars-posegen button')]
    .find((b) => (b.textContent || '').includes('按这批姿势生成'))
  check('有「按这批姿势生成」按钮', !!btn)
  check('有目录时按钮可用', btn?.disabled === false)
  click(btn)
  await new Promise((r) => setTimeout(r, 500))

  const call = calls.find((c) => c.url.includes('/characters/Alice/poses/generate'))
  check('提交到 /poses/generate', !!call, call?.url)
  if (call) {
    const b = JSON.parse(call.body)
    check('带上了骨架帧目录（不是逐张路径）', !!b.pose_dir, JSON.stringify(b).slice(0, 80))
    check('带上了服装', b.costume === 'Naked', b.costume)
  }
  check('批量任务有进度条', !!doc.querySelector('.chars-prog'))
  check('进度里带当前第几张',
    (doc.querySelector('.chars-prog')?.parentElement?.textContent || '').includes('张姿势'),
    (doc.querySelector('.chars-prog')?.parentElement?.textContent || '').slice(-80))

  await new Promise((r) => setTimeout(r, 1600))
  check('多姿势场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46f) 导出图集：帧 → 引擎能直接读的 atlas + 坐标表
//
// 这一层**不是官方功能**，但它是"接进引擎"的关键一步。要钉住三件事：
//   1. 进页面只**查状态**（GET），不会顺手把导出跑一遍；
//   2. 导出后统计（帧数/组数/尺寸）要摆给用户看；
//   3. 对齐规则要在界面上写清楚（底边对齐 —— 不然动画里角色上下抖）。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  const tab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('素材'))
  click(tab)
  await new Promise((r) => setTimeout(r, 600))

  const body = () => doc.querySelector('.chars-body')?.textContent || ''
  const exportBtn = () => [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('导出图集'))

  check('素材页有「导出图集」按钮', !!exportBtn())
  check('说明里写清了底边对齐（不然动画会抖）', body().includes('底边对齐'))
  check('说明里写清了输出位置（Atlas/）', body().includes('Atlas'))
  // `calls` 是全局累积的（前面用例的请求都在里面），只统计**本用例开始后**的
  const mark = calls.length
  const atlasCalls = () => calls.slice(mark)
    .filter((c) => c.url.endsWith('/atlas'))
  check('进页面只 GET 查状态，没有自动导出',
    atlasCalls().every((c) => (c.method || 'GET') === 'GET'),
    JSON.stringify(atlasCalls().map((c) => `${c.method} ${c.url}`)))

  click(exportBtn())
  // **走队列**：提交 → 轮询（running → done），前端轮询间隔 800ms。
  // 进度反馈要在**提交的瞬间**就出现（以前是同步的，界面只能干等）。
  await new Promise((r) => setTimeout(r, 120))
  const post = atlasCalls().filter((c) => c.method === 'POST')
  check('点按钮才 POST 导出', post.length === 1,
    JSON.stringify(atlasCalls().map((c) => `${c.method} ${c.url}`)))
  check('提交后立刻有进度反馈',
    body().includes('打包中') || body().includes('已提交'),
    body().slice(-160))
  await new Promise((r) => setTimeout(r, 3000))
  check('导出后显示帧数与组数', body().includes('4 帧') && body().includes('2 组动画'),
    body().slice(-160))
  check('导出后显示图集尺寸与单帧画布',
    body().includes('192×288') && body().includes('96×144'), body().slice(-160))
  check('导出后出现「现有图集」统计', body().includes('现有图集'))
  // **必须给文件入口**：只显示一行路径的话，用户导出完拿不到东西
  const atlasImg = doc.querySelector('.chars-atlas-link img')
  check('图集有缩略图预览', !!atlasImg, atlasImg?.getAttribute('src'))
  check('缩略图指向真实文件', (atlasImg?.getAttribute('src') || '').includes('atlas.png'),
    atlasImg?.getAttribute('src'))
  const jsonLink = [...doc.querySelectorAll('a')]
    .find((a) => (a.textContent || '').includes('atlas.json'))
  check('有 atlas.json 的入口', !!jsonLink, jsonLink?.getAttribute('href'))

  check('图集导出场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46h) 一键出全套：换内衣 → 出姿势 → 出表情（顺序照官方）
//
// 这一条要钉住的是**顺序**和**顺序的依赖**：
//   · 姿势必须在表情之前（表情是在姿势帧上改脸）；
//   · 勾了内衣，姿势就得从内衣那张底图出发（后端 source_costume）；
//   · 三步的进度要合成一条（以前是三个独立按钮，用户得自己等三次）。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  const tab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('素材'))
  click(tab)
  await new Promise((r) => setTimeout(r, 500))

  const body = () => doc.querySelector('.chars-body')?.textContent || ''
  const pipeBtn = () => [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('一键出全套'))

  check('素材页有「一键出全套」按钮', !!pipeBtn())
  check('说明里写清了顺序（内衣→姿势→表情）',
    body().includes('换白内衣') && body().includes('出姿势') && body().includes('出表情'))
  check('说明里写清了顺序不能乱', body().includes('顺序不能乱'))

  // 表情多选：默认空（只出姿势）
  const multi = doc.querySelector('.chars-body select[multiple]')
  check('有表情多选下拉', !!multi)
  check('默认不选表情（只出姿势，别让人等半天）',
    (multi?.selectedOptions.length || 0) === 0,
    String(multi?.selectedOptions.length))

  click(pipeBtn())
  await new Promise((r) => setTimeout(r, 600))

  const call = calls.find((c) => c.url.includes('/characters/Alice/pipeline'))
  check('提交到 /pipeline', !!call, call?.url)
  if (call) {
    const b = JSON.parse(call.body)
    check('默认勾了「先换白内衣」', b.underwear === true, String(b.underwear))
    check('带上了骨架帧目录', !!b.pose_dir, b.pose_dir)
    check('没选表情就不传 emotions',
      !b.emotions || b.emotions.length === 0, JSON.stringify(b.emotions))
  }

  // 轮询到 running（前端间隔 1200ms，mock 前两轮都返回 running）：
  // 进度条 + 当前步骤名必须在
  await new Promise((r) => setTimeout(r, 1500))
  check('流水线有进度条', !!doc.querySelector('.chars-prog'))
  check('进度里带当前步骤与细节',
    (doc.querySelector('.chars-body')?.textContent || '').includes('出多姿势'),
    (doc.querySelector('.chars-body')?.textContent || '').slice(-160))

  await new Promise((r) => setTimeout(r, 3000))
  check('完成后给出可执行的下一步（导出图集）',
    body().includes('导出图集'), body().slice(-160))
  check('一键出全套场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 46g) 换白内衣底图（官方 remove_clothes）
//
// 官方的预处理一步：把角色换成白色内衣，得到一张干净底图给后续每套服装当起点。
// **它是一条换装指令 + Clothes LoRA，不是"去掉衣服变裸体"** —— 界面上也要这么说，
// 否则用户会以为点下去会得到一张裸图。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  const tab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('服装'))
  click(tab)
  await new Promise((r) => setTimeout(r, 400))

  const body = () => doc.querySelector('.chars-body')?.textContent || ''
  const uwBtn = () => [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('换白内衣'))

  check('服装页有「换白内衣底图」按钮', !!uwBtn())
  check('说明里写清了官方指令原文', body().includes('Dress character: White underwear'))
  check('说明里写清了产物是「干净底图」而不是裸体', body().includes('干净底图'))

  click(uwBtn())
  await new Promise((r) => setTimeout(r, 500))
  const call = calls.find((c) => c.url.includes('/clothes/underwear'))
  check('提交到 /clothes/underwear', !!call, call?.url)
  if (call) {
    const b = JSON.parse(call.body || '{}')
    // 默认参数走官方：20 步 / cfg 1.0
    check('默认不带覆盖参数（用官方 20 步 / cfg 1.0）',
      b.steps === undefined && b.cfg === undefined, JSON.stringify(b))
  }

  await new Promise((r) => setTimeout(r, 1500))
  check('完成后提示存成了哪套服装', body().includes('白内衣'), body().slice(-140))
  check('内衣场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 47b) 表情批量生成：官方 Emotion Studio 的「只改脸」
//
// 表情不是"再出一张整图"——官方把脸框裁出来放大重画，再按羽化蒙版贴回原图，
// 所以脖子/衣服/动作逐像素不变（character_generator.py:1591-1708）。
// 这一组钉住三件事：
//   1. 先检脸框（不跑生成模型），把「几张源帧、脸框多大」摆给用户看；
//   2. 选中的表情按 safe_name 提交，不是把 157 条全发过去；
//   3. 批量任务的**进度**真的显示出来了（一批几十张，没进度只能干等）。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))
  const click = (el) => el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  // 进角色中心并建一个角色
  ;[...doc.querySelectorAll('.view-tab')][1]?.dispatchEvent(
    new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  click([...doc.querySelectorAll('.chars-new button')][0])
  await new Promise((r) => setTimeout(r, 300))

  // 切到「表情」页
  const tab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('表情'))
  check('详情有「表情」页', !!tab)
  click(tab)
  await new Promise((r) => setTimeout(r, 500))

  const body = () => doc.querySelector('.chars-body')?.textContent || ''
  check('表情库是真数据（157 条）',
    body().includes('157 条'), body().slice(0, 80))
  check('说明里写清了「只改脸」',
    body().includes('只改脸') && body().includes('羽化'),
    body().slice(0, 120))

  // ---- 1. 先检脸框 ----
  const detectBtn = [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('检出脸框'))
  check('有「检出脸框」按钮', !!detectBtn)
  click(detectBtn)
  await new Promise((r) => setTimeout(r, 500))
  check('调用了 /emotions/detect',
    calls.some((c) => c.url.includes('/characters/Alice/emotions/detect')))
  check('检完显示源帧数与脸框尺寸',
    body().includes('源帧 2 张') && body().includes('120×120'),
    body().slice(0, 200))

  // ---- 2. 选表情 ----
  const cards = () => [...doc.querySelectorAll('.chars-emo')]
  check('表情卡可点选（是 button，不是死图）',
    cards().length > 0 && cards()[0].tagName === 'BUTTON', cards()[0]?.tagName)
  click(cards()[0])
  click(cards()[1])
  await new Promise((r) => setTimeout(r, 150))
  check('选中状态看得见（.on）',
    doc.querySelectorAll('.chars-emo.on').length === 2,
    String(doc.querySelectorAll('.chars-emo.on').length))
  check('显示已选数量', body().includes('已选 2 个'))

  // ---- 3. 提交批量生成 ----
  const genBtn = [...doc.querySelectorAll('.chars-body button')]
    .find((b) => (b.textContent || '').includes('批量生成'))
  check('有「批量生成」按钮', !!genBtn)
  check('选之前/没选时按钮是禁用的语义（有选中才可用）',
    genBtn && !genBtn.disabled)
  click(genBtn)
  // **一次轮询之内**就要看到进度条：mock 的第一次 /jobs 查询返回 running 2/4，
  // 第二次才 done。等太久的话任务已经结束、界面切到素材页，就再也断不到进度了。
  await new Promise((r) => setTimeout(r, 450))

  const genCall = calls.find((c) => c.url.includes('/characters/Alice/emotions/generate'))
  check('提交到 /emotions/generate', !!genCall, genCall?.url)
  if (genCall) {
    const b = JSON.parse(genCall.body)
    check('只提交选中的两个表情（按 safe_name）',
      Array.isArray(b.emotions) && b.emotions.length === 2, JSON.stringify(b.emotions))
    check('带上官方默认参数（target_size 2048 / 步数 25）',
      b.target_size === 2048 && b.steps === 25, JSON.stringify(b))
    check('带上服装', b.costume === 'Naked', b.costume)
  }
  check('批量任务有进度条', !!doc.querySelector('.chars-prog'),
    String(doc.querySelectorAll('.chars-prog').length))
  const ptext = doc.querySelector('.chars-prog')?.textContent || ''
  check('进度显示 done/total', ptext.includes('/'), ptext)
  check('进度里带当前在做哪一张',
    (doc.querySelector('.chars-body')?.textContent || '').includes('smile'),
    (doc.querySelector('.chars-body')?.textContent || '').slice(-80))

  // 等到完成
  await new Promise((r) => setTimeout(r, 1600))
  check('完成后切到素材页（能马上翻刚出的图）',
    [...doc.querySelectorAll('.chars-tab')].some(
      (t) => t.classList.contains('on') && (t.textContent || '').includes('素材')))
  check('表情批量场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 48) 素材页：官方那套精灵帧翻看器（先看数量 → 再按帧号单张取）
//
// 用户要翻的是"这一套角色用过哪些帧"，而不是把所有图一次性塞进列表——
// 精灵帧一多（几十上百张）首屏就卡死。所以按官方的顺序做两步。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))

  ;[...doc.querySelectorAll('.view-tab')][1]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  ;[...doc.querySelectorAll('.chars-new button')][0]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))
  ;[...doc.querySelectorAll('.chars-card')].find((c) =>
    (c.textContent || '').includes('Alice'))
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))

  const assetsTab = [...doc.querySelectorAll('.chars-tab')]
    .find((t) => (t.textContent || '').includes('素材'))
  assetsTab?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))

  // 服装 + 表情两个是翻看器的；多姿势批量一个骨架帧目录；
  // 「一键出全套」再两个（骨架帧目录 + 表情多选）
  check('素材页下拉齐全（翻看器 2 + 多姿势 1 + 一键出全套 2）',
    doc.querySelectorAll('.chars-body select').length === 5,
    String(doc.querySelectorAll('.chars-body select').length))
  check('先去问这个序列有多少帧（不是一次性把图拉回来）',
    calls.some((c) => c.url.includes('/characters/Alice/poses/meta')),
    calls.map((c) => c.url).slice(-2).join())
  check('翻看器挂上了', !!doc.querySelector('.sprite-preview'))

  const img = doc.querySelector('.sprite-stage img')
  check('按帧号单张取图（src 带 index=0）',
    (img?.getAttribute('src') || '').includes('index=0'),
    img?.getAttribute('src'))
  check('src 带时间戳（同名文件被替换后浏览器不能拿旧图）',
    (img?.getAttribute('src') || '').includes('t='))
  check('多帧时显示计数 1/3',
    (doc.querySelector('.sprite-count')?.textContent || '') === '1/3',
    doc.querySelector('.sprite-count')?.textContent)

  // 翻页：采样 index 前进，且相邻帧会被预取
  ;[...doc.querySelectorAll('.sprite-nav button')][1]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  check('点下一帧后计数变 2/3',
    (doc.querySelector('.sprite-count')?.textContent || '') === '2/3',
    doc.querySelector('.sprite-count')?.textContent)
  check('翻页后 src 的 index 跟着变',
    (doc.querySelector('.sprite-stage img')?.getAttribute('src') || '').includes('index=1'),
    doc.querySelector('.sprite-stage img')?.getAttribute('src'))

  // 素材页也能放大看细节：点图 → 灯箱（滚轮缩放 / 拖动都在灯箱里）
  doc.querySelector('.sprite-stage img')
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('点精灵帧能打开灯箱（放大看细节）', !!doc.querySelector('.lightbox'))
  check('灯箱里有缩放工具条', !!doc.querySelector('.lightbox-zoomctl'))
  win.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('Esc 关掉角色中心的灯箱', !doc.querySelector('.lightbox'))

  // 全部帧的缩略图条
  check('缩略图条列出全部 3 帧',
    (doc.querySelector('.chars-asset-row')?.querySelectorAll('img').length || 0) === 3,
    String(doc.querySelector('.chars-asset-row')?.querySelectorAll('img').length))

  // 只有一帧时不给翻页按钮 —— 否则用户点了没反应，也没人告诉为什么
  const emoSel = doc.querySelectorAll('.chars-body select')[1]
  Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value')
    .set.call(emoSel, 'angry')
  emoSel.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 450))
  check('只有一帧时不显示翻页按钮（不给无声的失败）', !doc.querySelector('.sprite-nav'),
    doc.querySelector('.sprite-count')?.textContent || '（有导航条）')
  check('单帧时明说只有一帧',
    (doc.querySelector('.chars-body')?.textContent || '').includes('只有这一帧'),
    (doc.querySelector('.chars-body')?.textContent || '').slice(-40))
  check('素材翻看场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 49) 属性标签选择器：官方 Character Creator 的「Choose Presets」
//
// 手填"blonde hair / blonde_hair / Blonde"三种写法混着用，出来的图会飘。
// 官方的解法是给每个属性字段配一组标签，点选从动绑定 tag 里的标准写法
// （移植自 character_presets.mjs 的 presetGroups / presetSelection）。
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 400))

  ;[...doc.querySelectorAll('.view-tab')][1]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const setInput = (el, v) => {
    Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  setInput(doc.querySelector('.chars-new input'), 'Alice')
  await new Promise((r) => setTimeout(r, 80))
  ;[...doc.querySelectorAll('.chars-new button')][0]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 300))
  ;[...doc.querySelectorAll('.chars-card')].find((c) =>
    (c.textContent || '').includes('Alice'))
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 350))

  // 取的是 **creator_v2 那份**（15 个分类），不是 SDXL 时代那份 legacy 的
  check('取的是 creator_v2 标签库（不是 SDXL 的 legacy tags）',
    calls.some((c) => c.url.includes('character_presets_v2.json')),
    calls.map((c) => c.url).filter((u) => u.includes('.json')).join())

  const pickBtns = [...doc.querySelectorAll('.chars-field')]
    .filter((f) => f.querySelector('button'))
  // 官方 FIELD_CATEGORIES 有 7 个字段，breast_size 是调用处开的**第 8 个入口**
  // （character_creator_v2.js:4018：body 弹窗只给体型、胸围单独一个弹窗）
  check('8 个字段入口有「选」按钮（FIELD_CATEGORIES 7 个 + 胸围特例）',
    pickBtns.length === 8, pickBtns.map((f) => f.textContent.slice(0, 3)).join('/'))
  check('画质词/负向词/LoRA 不给标签选择（官方同样是手填）',
    ![...doc.querySelectorAll('.chars-field')].find((f) =>
      (f.textContent || '').startsWith('画质词'))?.querySelector('button'))

  // 头发：官方把 6 个分类合成一个弹窗（hair_color / pattern / length /
  // texture / hairstyles / framing）
  const hairBtn = [...doc.querySelectorAll('.chars-field')]
    .find((f) => (f.textContent || '').includes('发型发色'))?.querySelector('button')
  hairBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  check('点「选」弹出标签面板', !!doc.querySelector('.tag-modal'))
  const heads = [...doc.querySelectorAll('.tag-group-head')].map((h) => h.textContent)
  check('头发分成 6 组（颜色/挑染/长度/质感/发型/刘海）', heads.length === 6, heads.join('/'))
  check('chip 有内容（显示的是中文译文）',
    (doc.querySelector('.tag-chip')?.textContent || '').trim().length > 0,
    doc.querySelector('.tag-chip')?.textContent || '')

  // 眼睛：预置值是 'blue_eyes, my own trait' —— 验别名 + 保留自定义文本
  ;[...doc.querySelectorAll('.tag-mask')][0]
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  // 眼睛字段现在是 chips 展示：找「眼睛」那一行里的「改」按钮（打开标签库）
  const eyeField = [...doc.querySelectorAll('.chars-field')]
    .find((f) => (f.textContent || '').includes('眼睛'))
  const eyeBtn = [...(eyeField?.querySelectorAll('button') || [])]
    .find((b) => ['改', '选'].includes((b.textContent || '').trim()))
  eyeBtn?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const chips = [...doc.querySelectorAll('.tag-chip')]
  const blueChip = chips.find((c) => (c.textContent || '').trim() === '蓝眼')
  check('下划线写法 blue_eyes 也能匹配到「蓝眼」这个 chip（别名）',
    blueChip?.classList.contains('on'),
    blueChip?.className || '（没找到 chip）')
  const shown = doc.querySelector('.tag-result input')?.value || ''
  check('打开时把现有值照原样带出来（不偷偷改写用户手写的描述）',
    shown === 'blue_eyes, my own trait', shown)

  // 点一个 chip：写入的是**空格形式**（官方 toggle 的写法）
  const greenChip = chips.find((c) => (c.textContent || '').trim() === '绿眼')
  greenChip?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('选一项后写入框更新（含新增的 green eyes）',
    (doc.querySelector('.tag-result input')?.value || '').includes('green eyes'),
    doc.querySelector('.tag-result input')?.value)

  // 确定 → 写回角色属性
  ;[...doc.querySelectorAll('.tag-act button')]
    .find((b) => (b.textContent || '').includes('确定'))
    ?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 400))
  check('确定后关闭面板', !doc.querySelector('.tag-modal'))
  const put = calls.filter((c) => c.url.includes('/characters/Alice/info')).pop()
  check('把选好的内容写回服务端角色属性',
    !!put && JSON.parse(put.body).eyes.includes('green eyes'),
    put?.body)
  check('标签选择场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} 通过`)
process.exit(failed.length ? 1 : 0)
