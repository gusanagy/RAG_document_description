#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_engine.py
Módulo de backend para RAG com FAISS + Sentence-Transformers + LLM opcional.

Funções principais:
    - load_index(...)          -> carrega FAISS + metadados
    - run_retrieval(...)       -> retorna hits (com re-ranking opcional)
    - build_structured_answer(...) -> monta resposta "âncorada" por documento
    - generate_answer_llm(...) -> gera resposta com LLM local, com controle de VRAM
    - run_query(...)           -> pipeline completo usado pelo Streamlit / main.py

Este arquivo NÃO usa argparse, NÃO imprime nada sozinho e
foi pensado para ser chamado por outros módulos (chat_ui, main, etc.).
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple, DefaultDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pprint import pprint
from collections import defaultdict, Counter


# =========================
# Cache simples de modelos
# =========================

_EMBED_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_CE_MODEL_CACHE: Dict[str, CrossEncoder] = {}


def _get_embed_model(name: str) -> SentenceTransformer:
    """Carrega e cacheia o modelo de embeddings."""
    if name not in _EMBED_MODEL_CACHE:
        _EMBED_MODEL_CACHE[name] = SentenceTransformer(name)
    return _EMBED_MODEL_CACHE[name]


def _get_ce_model(name: str) -> CrossEncoder:
    """Carrega e cacheia o cross-encoder p/ re-ranking."""
    if name not in _CE_MODEL_CACHE:
        _CE_MODEL_CACHE[name] = CrossEncoder(name)
    return _CE_MODEL_CACHE[name]


# =========================
# FAISS / IO
# =========================

def load_index(index_dir: str | Path) -> tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Carrega o índice FAISS e o meta.json gerado na indexação.

    Espera:
        index_dir/faiss.index
        index_dir/meta.json
    """
    index_dir = Path(index_dir)
    faiss_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.json"

    if not faiss_path.exists():
        raise FileNotFoundError(f"Índice FAISS não encontrado: {faiss_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadados não encontrados: {meta_path}")

    index = faiss.read_index(str(faiss_path))
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metas


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Gera embedding normalizado para a consulta."""
    model = _get_embed_model(model_name)
    v = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")


def retrieve(
    index: faiss.Index,
    metas: List[Dict[str, Any]],
    query: str,
    model_name: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Busca top-k no FAISS e retorna lista de hits:
      [{
         ...metadados do chunk...,
         "score": float,
         "rank": int
       }, ...]
    """
    qv = embed_query(query, model_name)
    D, I = index.search(qv, k)
    hits: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        rec = dict(metas[idx])
        rec["score"] = float(score)
        rec["rank"] = rank
        hits.append(rec)
    return hits


# =========================
# Citação / formatação
# =========================

def _human_title_from_source(source_file: Optional[str]) -> str:
    """Transforma o nome do arquivo em algo legível quando doc_title não existir."""
    if not source_file:
        return "Documento"
    stem = Path(source_file).stem
    title = re.sub(r"[_\-]+", " ", stem)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:1].upper() + title[1:] if title else "Documento"




# =========================
# Utilidades de texto
# =========================

def _tokenize_terms(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", text.lower())


def _score_sentence(sent: str, q_terms: Counter) -> Tuple[int, int]:
    """
    Heurística simples para ranquear frases:
      - mais termos da query => melhor
      - em caso de empate, menor comprimento é melhor
    """
    s_terms = _tokenize_terms(sent)
    if not s_terms:
        return (0, 10**9)
    cnt = sum(q_terms[t] for t in s_terms if t in q_terms)
    return (cnt, len(s_terms))


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


def select_relevant_sentences(
    text: str,
    query: str,
    max_sentences: int = 3,
) -> List[str]:
    """
    Seleciona as frases mais relevantes de um trecho de texto
    com base nos termos da query.
    """
    sents = _SENT_SPLIT.split(text.strip())
    q_terms = Counter(_tokenize_terms(query))

    ranked = sorted(
        sents,
        key=lambda s: (_score_sentence(s, q_terms)[0],
                       -_score_sentence(s, q_terms)[1]),
        reverse=True,
    )

    ranked = [s.strip() for s in ranked if len(s.strip()) > 0]
    top = ranked[:max_sentences]

    # mantém ordem original das top-N para não virar sopa de frase aleatória
    order = {s: i for i, s in enumerate(sents)}
    top_sorted = sorted(top, key=lambda s: order.get(s, 10**9))
    return top_sorted


def group_hits_by_doc(hits: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Agrupa hits por documento (doc_title + source_file).
    Mantém a ordem pelo melhor rank de cada doc.
    """
    groups: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in hits:
        key = f"{h.get('doc_title') or _human_title_from_source(h.get('source_file'))}||{h.get('source_file')}"
        groups[key].append(h)

    ordered_groups = sorted(groups.values(), key=lambda g: min(h["rank"] for h in g))
    return ordered_groups


# =========================
# Structured answer
# =========================
def format_citation(h: Dict[str, Any]) -> str:
    """
    Monta uma citação consistente no formato:

        <Título> — <Seção> (p.X)
        <Título> — <Seção> (pp.X–Y)
        <Título> — <Seção>
        <Título> (p.X)
        <Título>

    Regras:
        - Prioriza doc_title; fallback: nome humanizado do arquivo fonte.
        - Se houver página/início/fim, formata conforme PDF.
        - EPUB normalmente não tem paginação, então só retorna título + seção.
    """

    # --------- Título ---------
    title = h.get("doc_title") or _human_title_from_source(h.get("source_file"))

    # --------- Seção/capítulo ---------
    section = h.get("section") or h.get("part")

    # --------- Páginas ---------
    fmt = (h.get("format") or "").lower()
    p_start = h.get("page_start")
    p_end = h.get("page_end")

    pages = None
    if isinstance(p_start, int) and p_start > 0:
        if isinstance(p_end, int) and p_end and p_end > p_start:
            pages = f"pp.{p_start}–{p_end}"
        else:
            pages = f"p.{p_start}"

    # --------- PDF: todas combinações ---------
    if fmt == "pdf":
        if section and pages:
            return f"{title} — {section} ({pages})"
        if section:
            return f"{title} — {section}"
        if pages:
            return f"{title} ({pages})"
        return title

    # --------- EPUB/OUTROS ---------
    if section:
        return f"{title} — {section}"
    return title

def build_structured_answer(
    hits: List[Dict[str, Any]],
    query: str,
    sentences_per_doc: int = 3,
    group_by_doc: bool = True,
    wrap: int = 100,
) -> str:
    """
    Monta resposta textual em blocos no formato:

    De acordo com: <citação>
    - frase relevante 1
    - frase relevante 2
    ...

    Ideal para:
        - exibir sem LLM
        - servir como "texto base" no Streamlit
    """
    blocks: List[str] = []

    doc_sets = group_hits_by_doc(hits) if group_by_doc else [[h] for h in hits]

    for hs in doc_sets:
        best = sorted(hs, key=lambda x: x["rank"])[0]
        header = f"De acordo com: {format_citation(best)}"

        candidate_sentences: List[str] = []
        for h in hs:
            candidate_sentences.extend(
                select_relevant_sentences(h["text"], query, max_sentences=sentences_per_doc)
            )
            if len(candidate_sentences) >= sentences_per_doc:
                break

        candidate_sentences = candidate_sentences[:sentences_per_doc]

        if not candidate_sentences:
            snippet = textwrap.fill(best["text"].strip()[:300], width=wrap)
            block = f"{header}\n- {snippet}"
        else:
            lines = [f"- {textwrap.fill(s, width=wrap)}" for s in candidate_sentences]
            block = header + "\n" + "\n".join(lines)

        blocks.append(block)

    return "\n\n".join(blocks)


# =========================
# Debug: print_full_text
# =========================

def _highlight_terms(text: str, terms: Iterable[str]) -> str:
    """
    Realça termos com marcação simples [TERM] (sem depender de ANSI),
    útil para logs / terminal genérico.
    """
    if not terms:
        return text
    toks = sorted({t.strip() for t in terms if len(t.strip()) >= 2}, key=len, reverse=True)
    for t in toks:
        text = re.sub(re.escape(t), f"[{t}]", text, flags=re.I)
    return text


def print_full_text(
    hits: List[Dict[str, Any]],
    ranks_to_show: Iterable[int],
    wrap: int = 100,
    highlight_terms_from_query: Optional[str] = None,
):
    """
    Função utilitária de debug:
      - mostra metadados do chunk
      - mostra texto completo com wrap
      - opcionalmente realça termos da query
    """
    ranks_set = set(ranks_to_show)
    for h in hits:
        if h["rank"] not in ranks_set:
            continue

        header_info = {
            "Rank": h["rank"],
            "Título": format_citation(h),
            "Chunk ID": h.get("chunk_id", "—"),
            "Tipo": h.get("format", "—"),
            "Fonte": h.get("source_file", "—"),
            "Score": f"{h.get('score', 0):.4f}",
        }

        print("\n" + "=" * 120)
        pprint("📘  Informações do Documento")
        pprint(header_info, sort_dicts=False, width=120)
        print("-" * 120)

        txt = h["text"]
        if highlight_terms_from_query:
            terms = re.split(r"\s+", highlight_terms_from_query.strip())
            txt = _highlight_terms(txt, terms)

        wrapped = textwrap.fill(
            txt.strip(),
            width=wrap,
            replace_whitespace=False,
            drop_whitespace=False,
        )

        pprint("📝  Texto:")
        print(wrapped)
        print("=" * 120 + "\n")


# =========================
# Re-ranking
# =========================

def rerank_hits(
    query: str,
    hits: List[Dict[str, Any]],
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict[str, Any]]:
    """
    Re-ranking usando cross-encoder (query, passage) -> score_ce.

    Mantém a estrutura dos hits e adiciona:
      - "score_ce": float
      - "rank": reatribuído após ordenação
    """
    ce = _get_ce_model(ce_model_name)
    pairs = [(query, h["text"]) for h in hits]
    scores = ce.predict(pairs)

    for h, s in zip(hits, scores):
        h["score_ce"] = float(s)

    hits = sorted(hits, key=lambda x: x["score_ce"], reverse=True)
    for i, h in enumerate(hits, start=1):
        h["rank"] = i

    return hits
def build_header_info(h: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constrói um dicionário padronizado contendo todas as informações
    relevantes do chunk, incluindo a citação formatada.
    """

    title = h.get("doc_title") or _human_title_from_source(h.get("source_file"))
    section = h.get("section") or h.get("part")

    return {
        "rank": h.get("rank"),
        "citation": format_citation(h),
        "chunk_id": h.get("chunk_id", "—"),
        "format": h.get("format", "—"),
        "source_file": h.get("source_file", "—"),
        "score": float(h.get("score", 0.0)) if h.get("score") is not None else None,

        # Informações detalhadas
        "doc_title": title,
        "section": section,
        "page_start": h.get("page_start"),
        "page_end": h.get("page_end"),
    }



# =========================
# Geração com LLM (VRAM-friendly)
# =========================

def _parse_gpu_mem(gpu_mem_str: Optional[str], num_gpus: int) -> Optional[dict]:
    """
    Converte '11GiB' em dict p/ max_memory:
        {0: '11GiB', 1: '11GiB', 'cpu': '30GiB'}
    """
    if not gpu_mem_str:
        return None
    mm = {i: gpu_mem_str for i in range(num_gpus)}
    mm["cpu"] = "30GiB"
    return mm


def _build_bnb_conf(bits: str, dtype_str: str):
    """Cria configuração bitsandbytes para 8-bit ou 4-bit."""
    from transformers import BitsAndBytesConfig
    import torch

    if bits == "8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif bits == "4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(torch.bfloat16 if dtype_str == "bf16" else torch.float16),
        )
    return None


def generate_answer_llm(
    question: str,
    hits: List[Dict[str, Any]],
    gen_model_name: str,
    max_context_chars: int = 4000,
    limit_context_hits: Optional[int] = 4,
    dtype_str: str = "fp16",       # 'fp16' | 'bf16' | 'fp32'
    bits: str = "none",            # 'none' | '8' | '4'
    gpu_mem: Optional[str] = None, # ex. "10GiB"
    max_new_tokens: int = 300,
) -> str:
    """
    Gera resposta com LLM local, controlando o consumo de VRAM.

    Estratégias:
      - bits: quantização 8/4-bit com bitsandbytes
      - gpu_mem: limita VRAM por GPU via max_memory
      - limit_context_hits: quantos chunks entram no contexto
      - max_context_chars: corta o tamanho total do contexto
      - max_new_tokens: controla o comprimento da resposta
    """
    # lazy imports para não forçar dependências em quem só faz retrieval
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    # monta contexto estruturado com citação
    ctx_parts, used = [], 0
    selected_hits = hits if not limit_context_hits else hits[:limit_context_hits]
    for h in selected_hits:
        cite = format_citation(h)
        block = f"De acordo com: {cite}\n{h['text']}\n"
        if used + len(block) > max_context_chars:
            break
        ctx_parts.append(block)
        used += len(block)
    context = "\n---\n".join(ctx_parts)
    #melhorar aqui se precisar truncar o contexto
    system = (
        "Você é um assistente técnico. Responda ESTRITAMENTE com base no contexto abaixo.\n"
        "FORMATO SUGERIDO:\n"
        "De acordo com:\n <Título> (p.X|pp.X–Y) — <Seção>\n\n"
        "Explicação clara e objetiva. Explicando e complementando as informações do texto\n"
        "Se o contexto não contiver a informação necessária, deixe isso explícito."
        "Linguagem clara e formal."
        "Termos tecnicos quando apropriado."

    )
    user = f"Pergunta: {question}\n\nContexto:\n{context}\n\nResposta:"

    # dtype
    if dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device_map = "cuda:0" if torch.cuda.is_available() else None
    num_gpus = torch.cuda.device_count()
    max_memory = _parse_gpu_mem(gpu_mem, num_gpus) if num_gpus > 0 else None

    bnb_conf = None
    if bits in {"8", "4"} and num_gpus > 0:
        try:
            bnb_conf = _build_bnb_conf(bits, dtype_str)
        except Exception:
            bnb_conf = None

    tok = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)

    if bnb_conf is not None:
        mdl = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            quantization_config=bnb_conf,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    try:
        out = gen(f"{system}\n\n{user}")[0]["generated_text"]
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError(
                "CUDA OOM durante a geração. Reduza max_new_tokens, "
                "limit_context_hits ou use bits=8/4 + gpu_mem."
            )
        raise

    return out.strip()


# =========================
# Pipeline completo
# =========================
def run_retrieval(
    index_dir: str | Path,
    query: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 6,
    rerank_enabled: bool = False,
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Dict[str, Any]:
    """
    Executa apenas a parte de recuperação:
      - carrega índice
      - busca top-k
      - aplica re-ranking (opcional)

    Retorna:
      {
        "hits": [...],
        "index_dir": str,
        "query": str,
      }
    """
    index, metas = load_index(index_dir)
    hits = retrieve(index, metas, query, embed_model, k=k)

    if rerank_enabled:
        hits = rerank_hits(query, hits, ce_model_name=ce_model_name)
        hits = hits[:k]

    # ✅ Anexa o header em cada hit
    for h in hits:
        h["header"] = build_header_info(h)

    return {
        "index_dir": str(Path(index_dir).resolve()),
        "query": query,
        "hits": hits,
    }




def run_query(
    index_dir: str | Path,
    query: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 6,
    rerank_enabled: bool = False,
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # structured
    sentences_per_doc: int = 3,
    group_by_doc: bool = True,
    wrap: int = 100,
    # llm
    use_llm: bool = False,
    gen_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    max_context_chars: int = 4000,
    limit_context_hits: int = 4,
    dtype_str: str = "fp16",
    bits: str = "none",
    gpu_mem: Optional[str] = None,
    max_new_tokens: int = 300,
) -> Dict[str, Any]:
    """
    Pipeline completo para ser chamado pelo Streamlit ou main.py.

    Retorna:
      {
        "query": str,
        "hits": [...],
        "structured_answer": str,
        "llm_answer": Optional[str],
      }
    """
    retr = run_retrieval(
        index_dir=index_dir,
        query=query,
        embed_model=embed_model,
        k=k,
        rerank_enabled=rerank_enabled,
        ce_model_name=ce_model_name,
    )
    hits = retr["hits"]

    structured = build_structured_answer(
        hits,
        query=query,
        sentences_per_doc=sentences_per_doc,
        group_by_doc=group_by_doc,
        wrap=wrap,
    )

    llm_answer: Optional[str] = None
    if use_llm:
        llm_answer = generate_answer_llm(
            question=query,
            hits=hits,
            gen_model_name=gen_model_name,
            max_context_chars=max_context_chars,
            limit_context_hits=limit_context_hits,
            dtype_str=dtype_str,
            bits=bits,
            gpu_mem=gpu_mem,
            max_new_tokens=max_new_tokens,
        )

    return {
        "query": query,
        "hits": hits,
        "structured_answer": structured,
        "llm_answer": llm_answer,
    }
