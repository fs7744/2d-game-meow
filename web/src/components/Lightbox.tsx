/**
 * 全屏预览（灯箱）。
 *
 * 卡片里的图只是缩略图，看不清像素细节；点一下就全屏看原图，顺便能下载。
 * 关掉有三种方式：点背景、按 Esc、点右上角 ×。
 */
type Props = {
  url: string
  name?: string
  onClose: () => void
}

export default function Lightbox({ url, name, onClose }: Props) {
  return (
    <div className="lightbox" onClick={onClose} role="dialog" aria-label="图片预览">
      <div className="lightbox-bar" onClick={(e) => e.stopPropagation()}>
        <span className="lightbox-name">{name || '预览'}</span>
        <span className="spacer" />
        <a
          className="mini wide"
          href={`${url}${url.includes('?') ? '&' : '?'}download=1`}
          download={name}
          title="下载原图"
        >
          ⭳ 下载
        </a>
        <button className="mini wide" onClick={onClose} title="关闭 (Esc)">
          ✕ 关闭
        </button>
      </div>
      <img className="lightbox-img" src={url} alt={name || '预览'} draggable={false} />
    </div>
  )
}
