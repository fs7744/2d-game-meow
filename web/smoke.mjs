/**
 * 无浏览器渲染冒烟：用 jsdom 加载真实构建产物 dist/index.html + bundle，
 * 断言空状态、节点卡、生成提交路径。npm run smoke
 */
import { readFileSync } from 'node:fs'
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
  // 分配项目 id 的计数器
  let serverSeq = 0
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
    if (String(url).includes('/upload')) {
      return {
        ok: true,
        json: async () => ({
          path: 'D:////x////uploads////abc_pic.png', url: '/files/uploads/abc_pic.png',
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
  return fetchFn
}

async function boot(search, seed) {
  const html = readFileSync(resolve(dist, 'index.html'), 'utf8')
  const vc = new VirtualConsole()
  const errors = []
  vc.on('jsdomError', (e) => {
    const msg = String(e)
    if (msg.includes('Could not load link')) return   // jsdom 无静态服务器，CSS 拉不到属正常
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
  // 不写死数量：加节点类型时忘改这里会误报。只断言「有骨骼编辑」这项新入口
  check('侧栏存在（节点分组含骨骼编辑）',
    !!section(doc, '节点')?.querySelectorAll('.rail-item').length
    && [...section(doc, '节点').querySelectorAll('.rail-item')]
      .some((x) => (x.textContent || '').includes('骨骼编辑')),
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
  const doc = dom.window.document
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

  const star = doc.querySelector('.star')
  check('星标可用', !!star && !star.disabled)
  star.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  const favItems = () => section(doc, '提示词收藏').querySelectorAll('.rail-item')
  check('收藏后出现在列表', favItems().length === 1)
  check('星标点亮', !!doc.querySelector('.star.on'))
  check('列表文本正确',
    (favItems()[0]?.querySelector('.rail-item-main')?.textContent || '')
      .includes('white cat knight'))

  // 改成别的再点收藏条目，应能回填
  setValue(ta, 'blue slime')
  await new Promise((r) => setTimeout(r, 80))
  doc.querySelector('.node').dispatchEvent(new win.Event('pointerdown', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  favItems()[0].querySelector('.rail-item-main').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('点收藏条目回填到节点',
    doc.querySelector('textarea.prompt').value.includes('white cat knight'),
    doc.querySelector('textarea.prompt').value)

  // 重复收藏不应出现两条
  doc.querySelector('.star').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('重复收藏去重（已收藏则取消）', favItems().length === 0, String(favItems().length))

  // 再收藏一次后删除
  doc.querySelector('.star').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  section(doc, '提示词收藏').querySelector('.rail-del')
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
  check('六个菜单分组：项目/模板/节点/收藏/LoRA/设置',
    JSON.stringify(titles) === JSON.stringify(['项目', '模板', '节点', '提示词收藏', 'LoRA', '设置']),
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
    body.workflow === 'w2' && body.params.init_image.includes('abc_pic.png'),
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

  // 「像素块」曾经漏过一条：hint 查不到键时返回空串，界面上只是没说明、不报错。
  // 现在既补了说明，也让 hint 返回显眼文案 + 加了静态检查。
  // 像素块默认藏在「自定义参数」风格下——先切过去再查 tooltip
  const stSel3 = [...doc.querySelectorAll('select')].find((x) =>
    [...x.options].some((o) => o.textContent.includes('自定义参数')))
  if (stSel3) {
    const s3 = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    s3.call(stSel3, 'custom')
    stSel3.dispatchEvent(new win.Event('change', { bubbles: true }))
    await new Promise((r) => setTimeout(r, 120))
  }
  const pxTip = lab('像素块')?.getAttribute('title') || ''
  check('像素块有 tooltip（自定义档下）', pxTip.includes('有效像素'), pxTip)
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

    // 参数要真的落地，不只是下拉显示对：切到自定义档看像素块的输入值
    // （8-bit 档的 pixel=8；连调 bug 下这里会是切换前的 6）
    const s2 = stylePick()
    setSelValue.call(s2, 'custom')
    s2.dispatchEvent(new win.Event('change', { bubbles: true }))
    await wait(150)
    check('切档位时像素块真的写进去了（8-bit → 8）',
      Number(lab('像素块')?.querySelector('input')?.value) === 8,
      lab('像素块')?.querySelector('input')?.value)
    // 切回像素精灵·高清，为后续用例保持默认态
    const s3 = stylePick()
    setSelValue.call(s3, 'spriteHi')
    s3.dispatchEvent(new win.Event('change', { bubbles: true }))
    await wait(150)
  }

  // 像素块/色数只在「自定义参数」风格下暴露——先切过去
  const styleSel = [...doc.querySelectorAll('select')].find((x) =>
    [...x.options].some((o) => o.textContent.includes('自定义参数')))
  check('有「产出风格」下拉（含 7 档）', !!styleSel && styleSel.options.length >= 7,
    styleSel?.options.length)
  check('默认不暴露像素块（由风格档位决定）', !lab('像素块'), '像素块本该隐藏')
  const selSetter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
  selSetter.call(styleSel, 'custom')
  styleSel.dispatchEvent(new win.Event('change', { bubbles: true }))
  await wait(150)
  check('切到自定义后出现像素块与色数', !!lab('像素块') && !!lab('色数'))

  // 768² 配像素块 6 → 128px 有效像素（参考图那种密度）
  setNum(lab('像素块').querySelector('input'), 6)
  await wait(120)
  check('显示有效像素（768/6 = 128px）',
    (lab('像素块').textContent || '').includes('128px'),
    lab('像素块').textContent)
  check('正常密度不告警', !doc.querySelector('.node .warn'), doc.querySelector('.warn')?.textContent)

  // 手输 64 会被夹到 16（max 挡不住手输），且给出告警
  setNum(lab('像素块').querySelector('input'), 64)
  await wait(120)
  check('手输超过上限时被夹住（64 → 16）',
    lab('像素块').querySelector('input').value === '16',
    lab('像素块').querySelector('input').value)
  check('极端密度给出告警并说明后果',
    (doc.querySelector('.node .warn')?.textContent || '').includes('有效像素'),
    doc.querySelector('.node .warn')?.textContent)

  check('像素块告警场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 31) 像素化是显式开关（由用户控制要不要）
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  const wait = (ms) => new Promise((r) => setTimeout(r, ms))
  await wait(500)
  doc.querySelector('details.fold').open = true
  await wait(120)

  const lab = (name) => [...doc.querySelectorAll('label')]
    .find((l) => (l.textContent || '').trim().startsWith(name))
  // React 的 checkbox 走 click（ChangeEventPlugin 对 checkbox/radio 监听 click），
  // 派发 change 不会触发 onChange——这是测试写法问题，不是功能问题。
  const clickCheck = (el) =>
    el.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))

  // 像素化开关只在「自定义参数」风格下暴露——先切过去
  const stSel2 = [...doc.querySelectorAll('select')].find((x) =>
    [...x.options].some((o) => o.textContent.includes('自定义参数')))
  if (stSel2) {
    const set2 = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    set2.call(stSel2, 'custom')
    stSel2.dispatchEvent(new win.Event('change', { bubbles: true }))
    await wait(150)
  }

  const pxToggle = lab('像素化')?.querySelector('input')
  check('有「像素化」开关（自定义档下）', !!pxToggle)
  check('默认是开的（推荐值 6）', pxToggle?.checked === true, String(pxToggle?.checked))
  check('像素块输入框可用', lab('像素块')?.querySelector('input')?.disabled === false)

  // 关掉 → 像素块置 0 并禁用
  clickCheck(pxToggle)
  await wait(120)
  check('关掉后像素块变 0', lab('像素块')?.querySelector('input')?.value === '0',
    lab('像素块')?.querySelector('input')?.value)
  check('关掉后像素块输入框禁用',
    lab('像素块')?.querySelector('input')?.disabled === true)
  check('关掉后不再显示有效像素', (lab('像素块')?.textContent || '').includes('关'),
    lab('像素块')?.textContent)
  check('关掉后不再告警', !doc.querySelector('.node .warn'), doc.querySelector('.warn')?.textContent)

  // 再打开 → 恢复默认 6
  clickCheck(lab('像素化').querySelector('input'))
  await wait(120)
  check('再打开恢复成 6（128px 有效像素）',
    lab('像素块')?.querySelector('input')?.value === '6',
    lab('像素块')?.querySelector('input')?.value)

  check('像素化开关场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 32) 原本是 Refiner 精修开关用例，已随 SD 一起删除（Qwen 没有精修器）。

// 37) 骨骼编辑节点：可拖关节 + 8 方向 + 导出入口
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))
  const add = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('骨骼编辑'))
  check('侧栏有「骨骼编辑」入口', !!add)
  add?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const node = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('骨骼编辑'))
  check('能创建骨骼编辑节点', !!node)
  // 18 关节 × 2（可见圆点 + 透明大热区）+ 可能的热区
  check('SVG 里渲染出 18 个关节点',
    (node?.querySelectorAll('.skel-canvas circle').length || 0) >= 36,
    String(node?.querySelectorAll('.skel-canvas circle').length))
  check('有 8 个方向按钮', node?.querySelectorAll('.skel-dir').length === 8,
    String(node?.querySelectorAll('.skel-dir').length))
  const side = [...(node?.querySelectorAll('.skel-dir') || [])]
    .find((b) => (b.textContent || '') === '右侧')
  side?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 150))
  check('点「右侧」后高亮（方位角切到 90°）', !!side?.classList.contains('on'),
    side?.className)
  check('有导出骨架图按钮',
    [...(node?.querySelectorAll('button') || [])]
      .some((b) => (b.textContent || '').includes('导出骨架图')))
  check('骨骼编辑场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 38) 骨骼编辑：从图片自动提取 18 个关节（DWPose 由后端 mock）
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))
  const add = [...doc.querySelectorAll('.rail-add')].find((b) =>
    (b.closest('.rail-item')?.textContent || '').includes('骨骼编辑'))
  add?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 250))
  const node = [...doc.querySelectorAll('.node')].find((n) =>
    (n.querySelector('.node-kind')?.textContent || '').includes('骨骼编辑'))
  const input = node?.querySelector('input[type=file]')
  check('编辑器有「从图片提取」的文件入口', !!input)
  const file = new win.File([new Uint8Array([1, 2, 3])], 'pose.png', { type: 'image/png' })
  Object.defineProperty(input, 'files', { value: [file], configurable: true })
  input.dispatchEvent(new win.Event('change', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 500))
  // **要抓关节点**：`.skel-canvas circle` 现在第一个是人形的眼睛
  const first = node?.querySelector('.skel-canvas circle.skel-joint')
  check('导入后关节坐标被替换（鼻 y=0.05 → SVG cy=5）',
    first?.getAttribute('cy') === '5', first?.getAttribute('cy'))
  check('提示带上提取到的 18/18 关节与朝向猜测',
    (doc.querySelector('.toast')?.textContent || '').includes('18/18')
    && (doc.querySelector('.toast')?.textContent || '').includes('右侧'),
    doc.querySelector('.toast')?.textContent)
  check('从图片提取场景无脚本错误', errors.length === 0, errors[0] || '')
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

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} 通过`)
process.exit(failed.length ? 1 : 0)
