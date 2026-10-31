import { loadMannequin, renderMannequin } from './src/vnccs/mannequin'

const status = document.getElementById('status')!
const out = document.getElementById('out') as HTMLCanvasElement

async function run() {
  try {
    status.textContent = 'probe icon…'
    const p = await fetch('/icon.png')
    status.textContent = `icon fetch: ${p.status} ${p.headers.get('content-length')}`
    await new Promise((r) => setTimeout(r, 100))
    status.textContent = '加载网格…'
    const m = await loadMannequin()
    status.textContent = '渲染…'
    const urls = [0, 90, 180].map((yaw) => renderMannequin(m, { yawDeg: yaw, width: 512, height: 512 }))
    const imgs = await Promise.all(urls.map((u) => {
      return new Promise<HTMLImageElement>((res, rej) => {
        const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = u
      })
    }))
    out.width = 512 * 3
    out.height = 512
    const ctx = out.getContext('2d')!
    imgs.forEach((im, k) => ctx.drawImage(im, k * 512, 0))
    status.textContent = 'done'
  } catch (e) {
    status.textContent = 'ERROR: ' + String(e) + ' | ' + ((e as Error)?.stack?.split('\n')[0] || '')
  }
}

void run()
