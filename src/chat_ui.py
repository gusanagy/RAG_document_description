

# chat_ui.py
import streamlit as st
from pathlib import Path
from query_engine import run_query

# Config básica da página
st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")

st.title("📚 Chat RAG — Consulta ao seu dataset")
st.caption("Consulta seus chunks de livros com FAISS, com e sem LLM.")

# ---------- Estado global do chat ----------
if "messages" not in st.session_state:
    # Cada mensagem: {"role": "user"|"assistant", "content": str}
    st.session_state["messages"] = []

if "last_hits" not in st.session_state:
    st.session_state["last_hits"] = []

# ---------- Sidebar: configuração ----------
index_dir = st.sidebar.text_input(
    "Caminho do índice",
    "/home/pdi_4/Documents/Documentos/rag/books_rag/index"
)

rerank = st.sidebar.checkbox("Re-ranking", False)
use_llm = st.sidebar.checkbox("Usar LLM", True)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Rode com:\n\n```bash\nstreamlit run chat_ui.py\n```")

st.divider()

# ---------- Renderiza histórico de mensagens ----------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Entrada do usuário ----------
q = st.chat_input("Digite uma pergunta sobre seus livros...")

if q:
    # 1) registra & mostra mensagem do usuário
    st.session_state["messages"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # 2) valida índice
    index_path = Path(index_dir).expanduser().resolve()
    if not (index_path / "faiss.index").exists() or not (index_path / "meta.json").exists():
        err = f"Índice inválido em `{index_path}` (faiss.index ou meta.json ausentes)."
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state["messages"].append({"role": "assistant", "content": f"❌ {err}"})
    else:
        # 3) chama o motor de busca / geração
        with st.spinner("Consultando FAISS e gerando respostas..."):
            out = run_query(
                index_dir=str(index_path),
                query=q,
                rerank_enabled=rerank,
                use_llm=use_llm,
                embed_model="sentence-transformers/all-MiniLM-L6-v2",
                k=7,
                ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                # structured
                sentences_per_doc=4,
                group_by_doc=True,
                wrap=100,
                # llm
                gen_model_name="Qwen/Qwen2.5-3B-Instruct",
                max_context_chars=5000,
                limit_context_hits=6,
                dtype_str="fp16",
                bits="none",
                gpu_mem=None,
                max_new_tokens=500,
            )

        # 4) resposta structured
        structured_text = out.get("structured_answer") or "_Sem resposta structured._"
        structured_block = "### 🧩 Resposta (Structured)\n" + structured_text

        with st.chat_message("assistant"):
            st.markdown(structured_block)
        st.session_state["messages"].append({"role": "assistant", "content": structured_block})

        # 5) resposta LLM (se habilitado)
        if use_llm:
            llm_raw = out.get("llm_answer") or "_LLM não retornou resposta ou foi desativada._"
            llm_block = "### 🤖 Resposta (LLM)\n" + llm_raw

            with st.chat_message("assistant"):
                st.markdown(llm_block)
            st.session_state["messages"].append({"role": "assistant", "content": llm_block})

        # 6) guarda hits para a seção de fontes
        st.session_state["last_hits"] = out.get("hits", [])

# ---------- Fontes (chunks recuperados) ----------
hits = st.session_state.get("last_hits", [])
with st.expander("📄 Fontes (chunks recuperados)"):
    if not hits:
        st.write("Nenhum hit disponível ainda. Faça uma pergunta no chat.")
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
            text = h.get("text", "")
            if len(text) > 800:
                text = text[:800] + "…"
            st.write(text)
            st.markdown("---")
