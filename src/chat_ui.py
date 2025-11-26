# chat_ui.py
import streamlit as st
from pathlib import Path
from query_engine import run_query

# ===============================
# Config básica da página
# ===============================
st.set_page_config(page_title="Chat Book RAG", page_icon="📚", layout="wide")

st.title("📚 Chat Book RAG — Consulta ao seu dataset")
st.caption("Consulta seus chunks de livros com FAISS, com e sem LLM.")

# ===============================
# Estado global do chat
# ===============================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_hits" not in st.session_state:
    st.session_state["last_hits"] = []

# ===============================
# Sidebar: Configurações agrupadas
# ===============================

st.sidebar.header("⚙️ Configurações do RAG")

# --- Caminho do índice ---
index_dir = st.sidebar.text_input(
    "📁 Caminho do índice FAISS",
    "/home/pdi_4/Documents/Documentos/rag/books_rag/index"
)

st.sidebar.markdown("---")

# --- Retrieval ---
st.sidebar.subheader("🔍 Retrieval")
embed_model = st.sidebar.text_input(
    "Modelo de Embeddings",
    "sentence-transformers/all-MiniLM-L6-v2"
)
k = st.sidebar.number_input("Top-K", min_value=1, max_value=50, value=7, step=1)
rerank = st.sidebar.checkbox("Re-ranking (Cross-Encoder)", False)
ce_model_name = st.sidebar.text_input(
    "Modelo de Re-ranking",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

st.sidebar.markdown("---")

# --- Structured Answer ---
st.sidebar.subheader("📘 Resposta (Structured)")
sentences_per_doc = st.sidebar.slider("Frases por documento", 1, 10, 4)
group_by_doc = st.sidebar.checkbox("Agrupar por documento", True)
wrap = st.sidebar.number_input("Wrap (colunas)", min_value=60, max_value=140, value=100)

st.sidebar.markdown("---")

# --- LLM Generation ---
st.sidebar.subheader("🤖 Resposta com LLM")
use_llm = st.sidebar.checkbox("Usar LLM", True)

gen_model_name = st.sidebar.text_input(
    "Modelo LLM",
    "Qwen/Qwen2.5-3B-Instruct"
)
max_context_chars = st.sidebar.number_input(
    "Máx. caracteres do contexto", min_value=500, max_value=20000, value=5000, step=500
)
limit_context_hits = st.sidebar.number_input(
    "Máx. chunks no contexto", min_value=1, max_value=20, value=6
)
max_new_tokens = st.sidebar.number_input(
    "Máx. tokens gerados", min_value=32, max_value=2048, value=500
)

dtype_str = st.sidebar.selectbox("dtype", ["fp16", "bf16", "fp32"], index=0)
bits = st.sidebar.selectbox("Quantização (bitsandbytes)", ["none", "8", "4"], index=0)
gpu_mem = st.sidebar.text_input("GPU mem limit (ex: 10GiB)", value="")

st.sidebar.markdown("---")
st.sidebar.markdown("🔥 Rode com:\n```\nstreamlit run chat_ui.py\n```")

st.divider()

# ===============================
# Renderiza histórico do chat
# ===============================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===============================
# Entrada do usuário
# ===============================
q = st.chat_input("Digite uma pergunta sobre seus livros...")

if q:
    # Mostra pergunta no chat
    st.session_state["messages"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # Valida índice
    index_path = Path(index_dir).expanduser().resolve()
    if not (index_path / "faiss.index").exists() or not (index_path / "meta.json").exists():
        err = f"Índice inválido em `{index_path}` (faiss.index ou meta.json ausentes)."
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state["messages"].append({"role": "assistant", "content": err})

    else:
        # Consulta o motor RAG
        with st.spinner("Consultando FAISS e gerando respostas..."):
            out = run_query(
                index_dir=str(index_path),
                query=q,
                rerank_enabled=rerank,
                use_llm=use_llm,

                # RETRIEVAL
                embed_model=embed_model,
                k=k,
                ce_model_name=ce_model_name,

                # STRUCTURED
                sentences_per_doc=sentences_per_doc,
                group_by_doc=group_by_doc,
                wrap=wrap,

                # LLM
                gen_model_name=gen_model_name,
                max_context_chars=max_context_chars,
                limit_context_hits=limit_context_hits,
                dtype_str=dtype_str,
                bits=bits,
                gpu_mem=gpu_mem if gpu_mem.strip() else None,
                max_new_tokens=max_new_tokens,
            )

        # Resposta structured
        msg_struct = "### 🧩 Resposta (Structured)\n" + (out["structured_answer"] or "_Sem resposta structured._")
        with st.chat_message("assistant"):
            st.markdown(msg_struct)
        st.session_state["messages"].append({"role": "assistant", "content": msg_struct})

        # Resposta LLM
        if use_llm:
            msg_llm = "### 🤖 Resposta (LLM)\n" + (out["llm_answer"] or "_LLM não retornou resposta._")
            with st.chat_message("assistant"):
                st.markdown(msg_llm)
            st.session_state["messages"].append({"role": "assistant", "content": msg_llm})

        # Armazena hits
        st.session_state["last_hits"] = out["hits"]

# ===============================
# Fontes (bottom section)
# ===============================
hits = st.session_state.get("last_hits", [])
with st.expander("📄 Fontes (chunks recuperados)"):
    if not hits:
        st.write("Nenhum hit disponível ainda.")
    else:
        for i, h in enumerate(hits, start=1):
            title = h.get("doc_title") or "Documento sem título"
            pg = h.get("page_start")
            sec = h.get("section") or h.get("part")
            cite = f"**{title}**"
            if sec:
                cite += f" — _{sec}_"
            if pg:
                cite += f" (p.{pg})"

            st.markdown(f"**[{i}]** {cite}")
            txt = h.get("text", "")
            st.write(txt[:800] + "…" if len(txt) > 800 else txt)
            st.markdown("---")
