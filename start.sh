#!/usr/bin/env bash
# 喵叫工作台 · 启动脚本（Git Bash / macOS / Linux）
#
#   ./start.sh                 构建（必要时）并启动，自动打开浏览器
#   ./start.sh --port 8010     换端口
#   ./start.sh --rebuild       强制重建前端
#   ./start.sh --no-build      不碰前端（用现有 dist）
#   ./start.sh --dev           只起 FastAPI，前端用 vite dev（另开终端 npm run dev）
#   ./start.sh --check         只做体检，不启动
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT=8000
BUILD=auto          # auto | force | never
DEV=0
CHECK_ONLY=0
HOST=127.0.0.1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)     PORT="${2:?需要端口号}"; shift 2 ;;
    --host)     HOST="${2:?需要地址}"; shift 2 ;;
    --rebuild)  BUILD=force; shift ;;
    --no-build) BUILD=never; shift ;;
    --dev)      DEV=1; shift ;;
    --check)    CHECK_ONLY=1; shift ;;
    -h|--help)  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$1（--help 看用法）"; exit 2 ;;
  esac
done

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*"; exit 1; }

echo "喵叫工作台 · 启动检查"
echo

# ---------- 1. Python ----------
PY=""
for cand in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python" python3 python; do
  if [[ "$cand" == */* ]]; then
    [[ -x "$cand" ]] && PY="$cand" && break
  else
    command -v "$cand" >/dev/null 2>&1 && PY="$(command -v "$cand")" && break
  fi
done
[[ -n "$PY" ]] || die "找不到 Python。先建虚拟环境：python -m venv .venv"
ok "Python：$PY（$("$PY" -V 2>&1)）"

"$PY" - <<'EOF' || die "缺少依赖。执行：$PY -m pip install -r requirements.txt"
import importlib.util, sys
missing = [m for m in ("torch", "diffusers", "transformers", "fastapi", "uvicorn")
           if importlib.util.find_spec(m) is None]
if missing:
    print("       缺包：" + ", ".join(missing))
    sys.exit(1)
EOF
ok "依赖齐全（torch / diffusers / transformers / fastapi / uvicorn）"

if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  GPU="$("$PY" -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null || echo GPU)"
  ok "CUDA 可用（$GPU）"
else
  warn "CUDA 不可用，会在 CPU 上跑（极慢）"
fi

# ---------- 2. 模型权重 ----------
# 现在只用 Qwen-Image-2.1；SD 的权重在 models/_archive/sd/（不再参与运行）
if [[ -f "models/checkpoints/qwen-image-2.1/model_index.json" ]]; then
  n=$(find models/checkpoints/qwen-image-2.1 -name "*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
  ok "Qwen-Image-2.1 权重就绪（$n 个 safetensors）"
else
  warn "Qwen-Image-2.1 权重缺失，生成会失败"
  warn "  → 放到 models/checkpoints/qwen-image-2.1/（需要 model_index.json）"
  warn "  → 目录结构见 README「安装」"
fi

# ---------- 3. 前端 ----------
need_build=0
if [[ "$BUILD" == "force" ]]; then
  need_build=1
elif [[ "$BUILD" == "auto" ]]; then
  if [[ ! -f web/dist/index.html ]]; then
    need_build=1
    warn "未找到前端构建产物，需要构建"
  elif [[ -n "$(find web/src web/index.html -newer web/dist/index.html -type f 2>/dev/null | head -1)" ]]; then
    need_build=1
    warn "前端源码有更新，需要重新构建"
  fi
fi

if [[ "$DEV" == "1" ]]; then
  ok "开发模式：前端请另开终端  cd web && npm run dev  （:5173 代理到 :$PORT）"
elif [[ "$need_build" == "1" ]]; then
  if [[ "$CHECK_ONLY" == "1" ]]; then
    warn "需要重新构建前端（体检模式不构建）：./start.sh --rebuild"
  elif [[ "$BUILD" == "never" ]]; then
    warn "--no-build 已指定，跳过构建"
  elif command -v npm >/dev/null 2>&1; then
    echo "  … 构建前端（npm ci + build）"
    ( cd web && { [[ -d node_modules ]] || npm ci --no-audit --no-fund; } && npm run build ) \
      | tail -4 | sed 's/^/    /'
    ok "前端构建完成"
  else
    die "需要 npm 构建前端，但找不到 npm。装 Node 18+，或用 --no-build"
  fi
else
  ok "前端产物是最新的"
fi

[[ "$CHECK_ONLY" == "1" ]] && { echo; echo "体检完成。"; exit 0; }

# ---------- 4. 端口 ----------
if command -v netstat >/dev/null 2>&1 && netstat -ano 2>/dev/null | grep -q ":$PORT .*LISTENING"; then
  die "端口 $PORT 已被占用（换个端口：--port 8010）"
fi

# ---------- 5. 启动 ----------
URL="http://$HOST:$PORT"
echo
echo "启动：$URL"
echo "（Ctrl+C 停止）"
echo

open_browser() {
  sleep 2
  if command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$URL" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  fi
}

if [[ "${NO_OPEN:-0}" != "1" ]] && [[ "$DEV" != "1" ]]; then
  open_browser &
fi

exec "$PY" -m uvicorn app:app --host "$HOST" --port "$PORT" --log-level info
