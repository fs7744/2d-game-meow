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
function makeFetch() {
  return async (url, init = {}) => {
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
      return { ok: true, json: async () => ({ job_id: 'job-1' }) }
    }
    if (String(url).includes('/jobs/')) {
      return {
        ok: true,
        json: async () => ({
          job_id: 'job-1',
          status: 'done',
          result: { file: 'D:\\x\\out.png', meta: { seed: 42, seconds: 1.5, peak_vram_gb: 2.6 } },
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
          available: ['sd15', 'sdxl'],
          // 与后端 Engine.RECOMMEND 一致：SDXL 推荐 768/24，不是 1024
          recommend: {
            sd15: { width: 768, height: 768, steps: 30, cfg: 7.0, pixel: 6, palette: 32 },
            sdxl: { width: 768, height: 768, steps: 24, cfg: 6.5, pixel: 6, palette: 32 },
          },
        }),
      }
    }
    if (String(url).endsWith('/loras')) {
      return {
        ok: true,
        json: async () => ({
          count: 3,
          items: [
            { name: 'wuxia_v2', label: '武侠 Wuxia V2', trigger: 'wuxia',
              scale: 0.8, preset: 'wuxia', note: '水墨武侠', base: 'sd15' },
            { name: 'guofeng', label: '国风插画', trigger: 'guofeng, chinese style',
              scale: 0.7, preset: null, note: '工笔重彩', base: 'sd15' },
            { name: 'xuer_wuxia', label: '隐藏项', trigger: '', scale: 0.8,
              preset: null, hidden: true },
          ],
        }),
      }
    }
    if (String(url).startsWith('/files/')) {
      return { ok: true, json: async () => ({}) }
    }
    return { ok: false, status: 404, json: async () => ({}) }
  }
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
  return { dom, errors }
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
  check('渲染出顶栏品牌', doc.querySelector('.brand .title')?.textContent?.includes('喵叫'))
  check('空状态引导可见', !!doc.querySelector('.canvas-empty .empty-card'))
  check('侧栏存在（节点 9 项：图生成 / 图生图 / 局部重绘 / 局部AI重绘 / 角色迁移 / 序列帧替换 / 像素转换 / 抠图 / 图片）',
    section(doc, '节点')?.querySelectorAll('.rail-item').length === 9,
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

  const hint = doc.querySelector('.lang-hint')
  check('默认提示英文约束', (hint?.textContent || '').includes('英文'))

  // 受控 textarea：必须走原生 setter + input 事件
  const ta = doc.querySelector('textarea.prompt')
  const setValue = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLTextAreaElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }

  setValue(ta, '背剑侠客')
  await new Promise((r) => setTimeout(r, 80))
  check('中文触发警告', !!doc.querySelector('.lang-hint.bad'))

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

// 7) LoRA：下拉、自动切契约、触发词填入、提交带 lora
{
  const { dom, errors } = await boot('?demo=1')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))

  doc.querySelector('details.fold').open = true
  await new Promise((r) => setTimeout(r, 80))

  const sel = [...doc.querySelectorAll('select')].find(
    (s) => s.parentElement?.textContent?.startsWith('LoRA'))
  check('存在 LoRA 下拉', !!sel)
  check('隐藏项被过滤（不挂 + 2 个可见）', sel?.options.length === 3,
    String(sel?.options.length))

  const setSelect = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSelect(sel, 'wuxia_v2')
  await new Promise((r) => setTimeout(r, 100))

  check('选中后显示触发词', (doc.querySelector('.lora-trigger')?.textContent || '').includes('wuxia'))
  const presetSel = [...doc.querySelectorAll('select')].find(
    (s) => s.parentElement?.textContent?.startsWith('风格'))
  check('自动切到武侠契约', presetSel?.value === 'wuxia', presetSel?.value)

  const before = doc.querySelector('textarea.prompt').value
  doc.querySelector('.lora-trigger .mini')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('一键填入触发词', doc.querySelector('textarea.prompt').value.startsWith('wuxia'))
  check('触发词加在提示词之前', doc.querySelector('textarea.prompt').value !== before)

  // 再点一次不应重复添加
  const once = doc.querySelector('textarea.prompt').value
  doc.querySelector('.lora-trigger .mini')
    .dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('重复填入不重复触发词', doc.querySelector('textarea.prompt').value === once)

  doc.querySelector('.gen-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 200))
  const sub = calls.filter((c) => c.url.endsWith('/generate')).pop()
  const body = JSON.parse(sub.body)
  check('提交带 lora 与强度',
    body.params.lora === 'wuxia_v2' && body.params.lora_scale === 0.8,
    JSON.stringify({ l: body.params.lora, s: body.params.lora_scale }))
  check('提交带 wuxia 契约', body.params.preset === 'wuxia', body.params.preset)

  check('LoRA 场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


// 8) LoRA 训练面板：打开、提交、进度渲染
{
  const { dom, errors } = await boot('')
  const doc = dom.window.document
  const win = dom.window
  await new Promise((r) => setTimeout(r, 500))

  const openBtn = doc.querySelector('.train-btn')
  check('顶栏有「训练 LoRA」入口', !!openBtn && (openBtn.textContent || '').includes('训练'))
  openBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 120))
  check('训练面板打开', !!doc.querySelector('.train-panel'))
  check('面板有数据集输入', !!doc.querySelector('.train-panel input'))

  const setInput = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('input', { bubbles: true }))
  }
  const inputs = [...doc.querySelectorAll('.train-panel input')]
  setInput(inputs[1], 'mk_knight')          // 输出名
  setInput(inputs[2], 'mk_knight')          // 触发词
  await new Promise((r) => setTimeout(r, 80))

  const startBtn = doc.querySelector('.train-panel .gen-btn')
  check('开始训练按钮可用', !!startBtn && !startBtn.disabled)
  startBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 260))

  const sub = calls.filter((c) => c.url.endsWith('/train') && c.method === 'POST').pop()
  const body = JSON.parse(sub.body)
  check('提交带输出名与触发词',
    body.name === 'mk_knight' && body.trigger === 'mk_knight',
    JSON.stringify({ n: body.name, t: body.trigger }))

  // 轮询一次后会渲染进度条
  await new Promise((r) => setTimeout(r, 1500))
  check('显示进度条', !!doc.querySelector('.train-bar-fill'))
  check('显示步数与 loss',
    (doc.querySelector('.train-line')?.textContent || '').includes('20/100'),
    doc.querySelector('.train-line')?.textContent)
  check('显示可训练参数量',
    (doc.querySelector('.train-status')?.textContent || '').includes('1.59M'))

  doc.querySelector('.train-head .icon-btn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 100))
  check('面板可关闭', !doc.querySelector('.train-panel'))

  check('训练面板无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}


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
  check('四个菜单分组：项目/节点/收藏/设置',
    JSON.stringify(titles) === JSON.stringify(['项目', '节点', '提示词收藏', '设置']),
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

  // 设置里的底模下拉
  doc.querySelector('details.fold').open = true
  await new Promise((r) => setTimeout(r, 80))
  const setSec = section(doc, '设置')
  const msel = setSec.querySelector('select')
  check('设置里有底模下拉', !!msel)
  check('底模下拉列出本机已装的', msel && msel.options.length === 2,
    String(msel?.options.length))

  const setSelect = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSelect(msel, 'sdxl')
  await new Promise((r) => setTimeout(r, 300))
  // 切到 SDXL 后，SD1.5 的 LoRA 必须被过滤掉（装上去是废图）
  const nodeLora = [...doc.querySelectorAll('details select')].find(
    (x) => x.parentElement?.textContent?.startsWith('LoRA'))
  check('切到 SDXL 后该底模无可用 LoRA',
    nodeLora && nodeLora.options.length === 1, String(nodeLora?.options.length))

  check('侧栏场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

// 11) 图生图节点：派生、源图解析、提交 w2
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
  check('带 strength', typeof body.params.strength === 'number', String(body.params.strength))
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
  const { dom, errors } = await boot('', { 'meow.canvas.v1': JSON.stringify(legacy) })
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
  check('有姿态/身份/占高三档参数',
    node?.querySelectorAll('.adv .strength input[type=range]').length === 3,
    String(node?.querySelectorAll('.adv .strength input[type=range]').length))

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
  check('提交 workflow=wpc', body.workflow === 'wpc', body.workflow)
  check('带上了动作参考图与角色图两个路径',
    typeof body.params.pose_image === 'string' && body.params.pose_image.length > 0 &&
    typeof body.params.char_image === 'string' && body.params.char_image.length > 0,
    JSON.stringify({ p: body.params.pose_image, c: body.params.char_image }))
  check('带上了姿态/身份强度',
    typeof body.params.cn_scale === 'number' && typeof body.params.ipa_scale === 'number',
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
  // 引擎选择：AnimateDiff 时序一致性 / 逐帧多候选
  const engineSel = [...node.querySelectorAll('select')].find((s) =>
    [...s.options].some((o) => o.textContent.includes('AnimateDiff')))
  check('有 AnimateDiff/逐帧 引擎选择', !!engineSel)
  // 显存档位：auto / low / medium / high
  const vramSel = [...node.querySelectorAll('select')].find((s) =>
    [...s.options].some((o) => o.textContent.includes('省显存')))
  check('有显存档位选择', !!vramSel)

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

  const modelSel = [...doc.querySelectorAll('select')]
    .find((s2) => (s2.closest('label')?.textContent || '').includes('底模'))
  const setSel = (el, v) => {
    const setter = Object.getOwnPropertyDescriptor(win.HTMLSelectElement.prototype, 'value').set
    setter.call(el, v)
    el.dispatchEvent(new win.Event('change', { bubbles: true }))
  }
  setSel(modelSel, 'sdxl')
  await new Promise((r) => setTimeout(r, 200))

  nodes = [...doc.querySelectorAll('.node')]
  check('切 SDXL 后未改动的节点套用推荐步数（30 → 24）',
    num(nodes[0], '步数') === '24', num(nodes[0], '步数'))
  check('推荐尺寸是 768 而不是 1024（1024 反而更差）',
    num(nodes[0], '宽') === '768', num(nodes[0], '宽'))
  check('手改过参数的节点不被覆盖',
    num(nodes[1], '宽') === '640' && num(nodes[1], '步数') === '30',
    JSON.stringify({ w: num(nodes[1], '宽'), s: num(nodes[1], '步数') }))
  check('切换时给出提示',
    (doc.querySelector('.toast')?.textContent || '').includes('sdxl'),
    doc.querySelector('.toast')?.textContent)
  check('切回 SD1.5 也能还原',
    (setSel(modelSel, 'sd15'), true))
  await new Promise((r) => setTimeout(r, 200))
  check('还原后步数回到 30', num([...doc.querySelectorAll('.node')][0], '步数') === '30',
    num([...doc.querySelectorAll('.node')][0], '步数'))

  check('底模切换场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} 通过`)
process.exit(failed.length ? 1 : 0)
