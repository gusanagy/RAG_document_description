#!/bin/bash
# ==========================================================
#  book_rag_launch.sh
#  Automação para lançar o app Streamlit do RAG + backend FAISS
# ==========================================================
# Uso:
#   bash book_rag_launch.sh
#
# Opções:
#   --port <num>         → Porta customizada (padrão 8501)
#   --env <env_name>     → Nome do ambiente Conda
#   --index <dir>        → Diretório do índice FAISS
#   --gpu-mem <limit>    → VRAM máxima (p.ex. "10GiB")
# ==========================================================

# ---- PARÂMETROS PADRÃO ----
PORT=8501
ENV_NAME="rag_book"
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$BASE_DIR/src"
SCRIPT_PY="$SRC_DIR/query_faiss_index.py"
APP_PY="$SRC_DIR/app_chat_runner.py"
INDEX_DIR="$BASE_DIR/books_rag/index"
GPU_MEM="10GiB"

# ---- PARSE ARGS ----
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift ;;
        --env) ENV_NAME="$2"; shift ;;
        --index) INDEX_DIR="$2"; shift ;;
        --gpu-mem) GPU_MEM="$2"; shift ;;
        *) echo "❌ Parâmetro desconhecido: $1"; exit 1 ;;
    esac
    shift
done

# ---- CHECAGENS ----
echo "📁 Diretório base: $BASE_DIR"
echo "📚 Índice FAISS:  $INDEX_DIR"
echo "⚙️  Script backend: $SCRIPT_PY"
echo "💬 App Streamlit:  $APP_PY"
echo "🧠 GPU Memória limite: $GPU_MEM"
echo "🌐 Porta: $PORT"
echo "🐍 Ambiente Conda: $ENV_NAME"
echo

if [ ! -d "$INDEX_DIR" ]; then
  echo "❌ ERRO: diretório de índice não encontrado: $INDEX_DIR"
  exit 1
fi
if [ ! -f "$SCRIPT_PY" ]; then
  echo "❌ ERRO: arquivo não encontrado: $SCRIPT_PY"
  exit 1
fi
if [ ! -f "$APP_PY" ]; then
  echo "❌ ERRO: arquivo não encontrado: $APP_PY"
  exit 1
fi

# ---- ATIVAÇÃO DO AMBIENTE ----
if command -v conda &>/dev/null; then
    echo "🔹 Ativando ambiente Conda: $ENV_NAME"
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME" || { echo "⚠️  Falha ao ativar conda env: $ENV_NAME"; }
else
    echo "⚠️  Conda não detectado. Continuando com Python padrão do sistema."
fi

# ---- TESTE DE DEPENDÊNCIAS ----
REQUIRED=("streamlit" "faiss" "sentence-transformers" "transformers")
echo "🔍 Verificando pacotes necessários..."
for pkg in "${REQUIRED[@]}"; do
    python -c "import $pkg" 2>/dev/null || {
        echo "⚠️  Pacote ausente: $pkg — instalando..."
        pip install "$pkg" -q
    }
done

# ---- VARIÁVEIS DE AMBIENTE ----
export RAG_INDEX_DIR="$INDEX_DIR"
export RAG_GPU_MEM="$GPU_MEM"
export HF_HOME="$BASE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$BASE_DIR/hf_cache"
export CUDA_VISIBLE_DEVICES=0

# ---- MENSAGEM ----
echo "✅ Tudo pronto!"
echo "🌍 Iniciando Streamlit na porta $PORT..."
echo "👉 Acesse: http://localhost:$PORT"
echo

# ---- EXECUÇÃO ----
cd "$SRC_DIR"
streamlit run "$APP_PY" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
