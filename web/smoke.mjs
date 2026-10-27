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
    if (String(url).startsWith('/files/')) {
      return { ok: true, json: async () => ({}) }
    }
    return { ok: false, status: 404, json: async () => ({}) }
  }
}

async function boot(search) {
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
      win.HTMLElement.prototype.setPointerCapture = () => {}
      win.HTMLElement.prototype.releasePointerCapture = () => {}
      win.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    },
  })

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
  check('节点列表存在', doc.querySelectorAll('.palette-item').length === 1)
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
    check('参数带 pixel/palette', body.params.pixel === 4 && body.params.palette === 16)
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

  const toggle = [...doc.querySelectorAll('.icon-btn')].find((b) => b.textContent === '▣')
  check('存在「只看图」按钮', !!toggle)
  toggle?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('切换为只展示图', !!doc.querySelector('.node-image-only'))

  const back = [...doc.querySelectorAll('.icon-btn')].find((b) => b.textContent === '▤')
  back?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('切回参数视图', !!doc.querySelector('.node .prompt'))

  const copyBtn = [...doc.querySelectorAll('.icon-btn')].find((b) => (b.getAttribute('title') || '').includes('复制'))
  copyBtn?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('复制后 2 个节点', doc.querySelectorAll('.node').length === 2,
        String(doc.querySelectorAll('.node').length))

  const addBtn = doc.querySelector('.palette-item .add')
  addBtn?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('加号新建后 3 个节点', doc.querySelectorAll('.node').length === 3,
        String(doc.querySelectorAll('.node').length))

  const delBtn = [...doc.querySelectorAll('.icon-btn')].find((b) => b.textContent === '✕')
  delBtn?.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('删除后 2 个节点', doc.querySelectorAll('.node').length === 2,
        String(doc.querySelectorAll('.node').length))

  // 键盘删除：先选中一个节点，再按 Delete
  const node = doc.querySelector('.node')
  node?.dispatchEvent(new dom.window.Event('pointerdown', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 40))
  dom.window.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Delete', bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  check('Delete 键删除节点', doc.querySelectorAll('.node').length === 1,
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
  check('收藏后出现在列表', doc.querySelectorAll('.book-item').length === 1)
  check('星标点亮', !!doc.querySelector('.star.on'))
  check('列表文本正确',
    (doc.querySelector('.book-text')?.textContent || '').includes('white cat knight'))

  // 改成别的再点收藏条目，应能回填
  setValue(ta, 'blue slime')
  await new Promise((r) => setTimeout(r, 80))
  doc.querySelector('.node').dispatchEvent(new win.Event('pointerdown', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 60))
  doc.querySelector('.book-text').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('点收藏条目回填到节点',
    doc.querySelector('textarea.prompt').value.includes('white cat knight'),
    doc.querySelector('textarea.prompt').value)

  // 重复收藏不应出现两条
  doc.querySelector('.star').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('重复收藏去重（已收藏则取消）', doc.querySelectorAll('.book-item').length === 0,
    String(doc.querySelectorAll('.book-item').length))

  // 再收藏一次后删除
  doc.querySelector('.star').dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  doc.querySelector('.book-del')?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await new Promise((r) => setTimeout(r, 80))
  check('删除收藏', doc.querySelectorAll('.book-item').length === 0)

  check('收藏场景无脚本错误', errors.length === 0, errors[0] || '')
  dom.window.close()
}

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} 通过`)
process.exit(failed.length ? 1 : 0)
