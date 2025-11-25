#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_chat_runner.py
Streamlit que executa seu query_faiss_index.py ao perguntar e
preenche automaticamente as respostas (structured e llm),
além de exibir as fontes (hits).

Requisitos:
    pip install streamlit
"""

import json
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st


# ===================== UI Helpers =====================

def render_header():
    st.set_page_config(page_title="RAG Chat Runner", layout="wide", page_icon="💬")
    st.title("💬 RAG Chat Runner")
    st.caption("Pergunte e deixe o app rodar seu script para preencher as respostas com e sem LLM.")


def color_badge(text: str, color: str = "#3b82f6"):
    st.markdown(
        f"""
        <span style="
            background:{color};
            color:white;
            padding:3px 8px;
            border-radius:999px;
            font-size:12px;
            margin-right:8px;
            white-space:nowrap;
        ">{text}</span>
        """,
        unsafe_allow_html=True
    )


def bubble(content: str, sender: str = "user", width_px: int = 900):
    align = "flex-end" if sender == "assistant" else "flex-start"
    bg = "#111827" if sender == "assistant" else "#1f2937"
    st.markdown(
        f"""
        <div style="display:flex; justify-content:{align}; margin:8px 0;">
          <div style="
            max-width: {width_px}px;
            background:{bg};
            color:#e5e7eb;
            padding:12px 14px;
            border-radius:14px;
            line-height:1.4;
            font-size:15px;
            white-space:pre-wrap;
            border:1px solid #374151;">
            {content}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ===================== Backend Helpers =====================

def run_cli(cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Executa um comando e retorna CompletedProcess com stdout/stderr."""
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )


def extract_answer_from_stdout(stdout: str) -> str:
    """
    Fallback: tenta extrair a resposta do stdout quando --answer-out não foi escrito
    (mantido por segurança, mas normalmente não será usado).
    """
    m = re.search(r"---\s*RESPOSTA\s*---\s*\n(.*)$", stdout, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"^\[2/2\].*?(?:Respost[ao].*?:)?\s*\n(.*)$", stdout, flags=re.S | re.I | re.M)
    if m:
        return m.group(1).strip()
    return stdout.strip()


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cli_args(
    script_path: Path,
    index_dir: Path,
    embed_model: str,
    query: str,
    k: int,
    rerank: bool,
    m: int,
    preview_chars: int,
    answer_mode: str,
    group_by_doc: bool,
    sentences: int,
    wrap: int,
    gen_model: str,
    limit_context_hits: int,
    max_context_chars: int,
    max_new_tokens: int,
    dtype: str,
    bits: str,
    gpu_mem: Optional[str],
    export_hits_path: Optional[Path] = None,
    answer_out_path: Optional[Path] = None,   # << novo
) -> List[str]:
    """Monta a linha de comando p/ seu script."""
    cmd = [
        "python", str(script_path),
        "--index-dir", str(index_dir),
        "--model", embed_model,
        "--query", query,
        "--k", str(k),
        "--preview-chars", str(preview_chars),
        "--answer-mode", answer_mode,
    ]
    if rerank:
        cmd += ["--rerank", "--m", str(m)]
    if answer_mode == "structured":
        if group_by_doc:
            cmd += ["--group-by-doc"]
        cmd += ["--sentences", str(sentences), "--wrap", str(wrap)]
    else:
        cmd += [
            "--gen-model", gen_model,
            "--limit-context-hits", str(limit_context_hits),
            "--max-context-chars", str(max_context_chars),
            "--max-new-tokens", str(max_new_tokens),
            "--dtype", dtype,
            "--bits", bits,
        ]
        if gpu_mem:
            cmd += ["--gpu-mem", gpu_mem]

    # integrações com o viewer
    if export_hits_path is not None:
        cmd += ["--export-hits", str(export_hits_path)]
    if answer_out_path is not None:
        cmd += ["--answer-out", str(answer_out_path)]
    return cmd


def run_both_modes(
    script_path: Path,
    index_dir: Path,
    embed_model: str,
    query: str,
    k: int,
    rerank: bool,
    m: int,
    preview_chars: int,
    # structured
    group_by_doc: bool,
    sentences: int,
    wrap: int,
    # llm
    gen_model: str,
    limit_context_hits: int,
    max_context_chars: int,
    max_new_tokens: int,
    dtype: str,
    bits: str,
    gpu_mem: Optional[str],
) -> Dict[str, Any]:
    """
    Executa o script duas vezes (structured + llm) gravando/resgatando arquivos.
    Retorna:
      {
        "structured_text": str,
        "llm_text": str,
        "hits": [ ... ],
        "stdout_structured": str,
        "stderr_structured": str,
        "stdout_llm": str,
        "stderr_llm": str
      }
    """
    result: Dict[str, Any] = {
        "structured_text": "",
        "llm_text": "",
        "hits": [],
        "stdout_structured": "",
        "stderr_structured": "",
        "stdout_llm": "",
        "stderr_llm": "",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        hits_json = tmp / "hits.json"
        ans_struct = tmp / "answer_structured.txt"
        ans_llm = tmp / "answer_llm.txt"

        # 1) structured
        cmd_struct = build_cli_args(
            script_path, index_dir, embed_model, query, k, rerank, m, preview_chars,
            answer_mode="structured", group_by_doc=group_by_doc, sentences=sentences, wrap=wrap,
            gen_model=gen_model, limit_context_hits=limit_context_hits, max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens, dtype=dtype, bits=bits, gpu_mem=gpu_mem,
            export_hits_path=hits_json, answer_out_path=ans_struct
        )
        cp_s = run_cli(cmd_struct)
        result["stdout_structured"] = cp_s.stdout
        result["stderr_structured"] = cp_s.stderr

        if ans_struct.exists():
            result["structured_text"] = ans_struct.read_text(encoding="utf-8").strip()
        else:
            result["structured_text"] = extract_answer_from_stdout(cp_s.stdout)

        hits = load_json_if_exists(hits_json)
        if isinstance(hits, list):
            result["hits"] = hits

        # 2) llm
        cmd_llm = build_cli_args(
            script_path, index_dir, embed_model, query, k, rerank, m, preview_chars,
            answer_mode="llm", group_by_doc=group_by_doc, sentences=sentences, wrap=wrap,
            gen_model=gen_model, limit_context_hits=limit_context_hits, max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens, dtype=dtype, bits=bits, gpu_mem=gpu_mem,
            export_hits_path=hits_json, answer_out_path=ans_llm
        )
        cp_l = run_cli(cmd_llm)
        result["stdout_llm"] = cp_l.stdout
        result["stderr_llm"] = cp_l.stderr

        if ans_llm.exists():
            result["llm_text"] = ans_llm.read_text(encoding="utf-8").strip()
        else:
            result["llm_text"] = extract_answer_from_stdout(cp_l.stdout)

        # se ainda não carregou hits, tenta novamente
        if not result["hits"]:
            hits = load_json_if_exists(hits_json)
            if isinstance(hits, list):
                result["hits"] = hits

    return result


# ===================== App =====================

def main():
    render_header()

    # Sidebar: configurações do script
    with st.sidebar:
        st.subheader("Configuração do backend")
        script_path = st.text_input(
            "Caminho do query_faiss_index.py",
            value=str(Path.cwd() / "query_faiss_index.py")
        )
        index_dir = st.text_input(
            "index-dir (faiss.index + meta.json)",
            value="/home/pdi_4/Documents/Documentos/rag/books_rag/index"
        )
        embed_model = st.text_input(
            "Modelo de embeddings (o MESMO da indexação)",
            value="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.markdown("---")
        st.subheader("Busca e ranking")
        k = st.number_input("Top-k", min_value=1, max_value=100, value=6, step=1)
        rerank = st.checkbox("Re-ranking (cross-encoder)", value=False)
        m = st.number_input("Candidatos p/ re-ranking (m)", min_value=1, max_value=200, value=50, step=1)
        preview_chars = st.number_input("Preview por hit (chars)", min_value=100, max_value=2000, value=300, step=50)

        st.markdown("---")
        st.subheader("Structured (sem LLM)")
        group_by_doc = st.checkbox("Agrupar por documento", value=True)
        sentences = st.number_input("Frases por documento", min_value=1, max_value=10, value=3, step=1)
        wrap = st.number_input("Largura do texto", min_value=50, max_value=140, value=90, step=2)

        st.markdown("---")
        st.subheader("LLM (VRAM-friendly)")
        gen_model = st.text_input("Modelo HF (LLM)", value="Qwen/Qwen2.5-3B-Instruct")
        limit_context_hits = st.number_input("Max hits no contexto", min_value=1, max_value=20, value=3, step=1)
        max_context_chars = st.number_input("Max chars do contexto", min_value=500, max_value=20000, value=2500, step=250)
        max_new_tokens = st.number_input("Max tokens gerados", min_value=32, max_value=2048, value=180, step=16)
        dtype = st.selectbox("dtype", ["fp16", "bf16", "fp32"], index=0)
        bits = st.selectbox("quantização", ["none", "8", "4"], index=2)
        gpu_mem = st.text_input('gpu-mem (ex. "10GiB", vazio = sem limite)', value="10GiB")

    # Campo do chat
    query = st.text_input("Pergunte aqui (pressione Enter)", value="", placeholder="Ex: EKF em navegação robótica", key="query_text")

    if query:
        color_badge("consulta")
        bubble(f"Pergunta: {query}", sender="user")

        # Validação de caminhos
        script_file = Path(script_path).expanduser().resolve()
        index_path = Path(index_dir).expanduser().resolve()
        if not script_file.exists():
            st.error(f"Script não encontrado: {script_file}")
            return
        if not (index_path / "faiss.index").exists() or not (index_path / "meta.json").exists():
            st.error(f"Índice inválido: {index_path} (faiss.index/meta.json ausentes)")
            return

        # Executa o backend
        with st.spinner("Executando o backend…"):
            try:
                results = run_both_modes(
                    script_path=script_file,
                    index_dir=index_path,
                    embed_model=embed_model,
                    query=query,
                    k=int(k),
                    rerank=bool(rerank),
                    m=int(m),
                    preview_chars=int(preview_chars),
                    group_by_doc=bool(group_by_doc),
                    sentences=int(sentences),
                    wrap=int(wrap),
                    gen_model=gen_model,
                    limit_context_hits=int(limit_context_hits),
                    max_context_chars=int(max_context_chars),
                    max_new_tokens=int(max_new_tokens),
                    dtype=dtype,
                    bits=bits,
                    gpu_mem=gpu_mem if gpu_mem.strip() else None,
                )
            except subprocess.TimeoutExpired:
                st.error("Tempo limite excedido ao executar o script.")
                return
            except Exception as e:
                st.error(f"Erro ao executar: {e}")
                return

        # Mostra respostas
        structured_text = results.get("structured_text", "").strip()
        llm_text = results.get("llm_text", "").strip()

        if structured_text:
            color_badge("structured")
            bubble(structured_text, sender="user")
        else:
            st.warning("Sem saída structured detectada. Abra 'stdout/stderr (structured)' abaixo.")

        if llm_text:
            color_badge("llm")
            bubble(llm_text, sender="assistant")
        else:
            st.warning("Sem saída llm detectada. Abra 'stdout/stderr (llm)' abaixo.")

        # Fontes (hits)
        hits = results.get("hits", [])
        with st.expander("Fontes (hits)"):
            if hits:
                for i, h in enumerate(hits[:20], 1):
                    title = h.get("doc_title") or Path(h.get("source_file","")).stem
                    p = h.get("page_start")
                    sec = h.get("section") or h.get("part")
                    cite = f"{title} — {sec}" if sec else title
                    if p: cite = f"{cite} (p.{p})"
                    st.markdown(f"**[{i}] {cite}**  \n`chunk_id`: `{h.get('chunk_id','')}`  •  `{h.get('format','')}`")
                    st.write(h.get("text",""))
            else:
                st.info("Sem hits exportados. Verifique se o script gravou o JSON de hits.")

        # Debug (stdout/stderr)
        with st.expander("stdout/stderr (structured)"):
            st.code(results.get("stdout_structured",""))
            if results.get("stderr_structured","").strip():
                st.error(results["stderr_structured"])
        with st.expander("stdout/stderr (llm)"):
            st.code(results.get("stdout_llm",""))
            if results.get("stderr_llm","").strip():
                st.error(results["stderr_llm"])


if __name__ == "__main__":
    main()
