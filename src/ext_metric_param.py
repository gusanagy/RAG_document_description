#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_eval_predictions.py

Script para automatizar a criação de um JSON de predições do seu RAG.

Fluxo:
  - Lê um arquivo de avaliação (eval_gold.json) com perguntas (e opcionalmente gold answers/docs).
  - Para cada pergunta:
      * chama run_query(...) do query_engine
      * extrai:
          - pergunta
          - resposta structured
          - resposta da LLM
          - lista de livros/documentos recuperados
          - metadata resumida dos chunks recuperados
  - Ao final, grava um único arquivo JSON com todas as predições.

Esse JSON depois pode ser usado por outro script para calcular métricas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from query_engine import run_query
from tqdm import tqdm
import torch
import gc


# =========================
# Função utilitária: limpar GPU
# =========================

def clear_gpu() -> None:
    """
    Libera memória da GPU entre iterações.

    - gc.collect(): limpa objetos Python sem referência.
    - torch.cuda.empty_cache(): libera cache de memória da CUDA.
    - torch.cuda.ipc_collect(): limpa IPC handles (às vezes ajuda em loops longos).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =========================
# CONFIGURAÇÕES BÁSICAS
# =========================

# Arquivo com as perguntas + gabarito (gold)
GOLD_PATH = Path("src/metrics/eval_gold.json")

# Arquivo de saída com as predições do RAG
OUT_JSON_PATH = Path("src/metrics/eval_predictions.json")

# Caminho do índice FAISS
DEFAULT_INDEX_DIR = "/home/pdi_4/Documents/Documentos/rag/books_rag/index"

# Modelo de embeddings usado na indexação
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo LLM para geração
DEFAULT_GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"


# =========================
# Helpers para resumir recuperação
# =========================

def _extract_retrieved_docs(hits: List[Dict[str, Any]], top_n: int = 5) -> List[str]:
    """
    Extrai uma lista de títulos de documentos a partir dos hits do RAG,
    removendo duplicados e limitando a top_n.

    Usa:
      - doc_title se existir
      - caso contrário, stem do source_file
    """
    docs: List[str] = []
    seen = set()

    for h in hits:
        title = h.get("doc_title")
        if not title:
            src = h.get("source_file") or ""
            title = Path(src).stem if src else "Documento"

        norm = title.strip()
        if not norm:
            continue

        key = norm.lower()
        if key in seen:
            continue

        seen.add(key)
        docs.append(norm)

        if len(docs) >= top_n:
            break

    return docs


def _extract_chunk_summaries(
    hits: List[Dict[str, Any]],
    max_chunks: int = 10
) -> List[Dict[str, Any]]:
    """
    Cria um resumo dos chunks recuperados para inspeção posterior.

    Não salva o texto inteiro pra não explodir o JSON:
      - preview (até 500 chars)
      - metadados relevantes (rank, doc_title, página, etc.)
    """
    out: List[Dict[str, Any]] = []

    for h in hits[:max_chunks]:
        text = h.get("text", "") or ""
        preview = text[:500] + ("…" if len(text) > 500 else "")

        out.append(
            {
                "rank": h.get("rank"),
                "doc_title": h.get("doc_title"),
                "source_file": h.get("source_file"),
                "format": h.get("format"),
                "section": h.get("section") or h.get("part"),
                "page_start": h.get("page_start"),
                "page_end": h.get("page_end"),
                "score": h.get("score"),
                "preview": preview,
            }
        )

    return out


# =========================
# PIPELINE PRINCIPAL
# =========================

def main() -> None:
    # 1) Carrega arquivo de perguntas + gold
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Arquivo de perguntas não encontrado: {GOLD_PATH}")

    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        raise ValueError("Nenhum 'qa_pairs' encontrado em eval_gold.json")

    print(f"Carregado {len(qa_pairs)} exemplos de avaliação.")
    print(f"Usando índice: {DEFAULT_INDEX_DIR}")

    predictions: List[Dict[str, Any]] = []

    # 2) Loop sobre os exemplos de avaliação
    for item in tqdm(qa_pairs, desc="Rodando RAG nas perguntas"):
        ex_id = int(item.get("id"))
        question = (item.get("question") or "").strip()

        if not question:
            print(f"[AVISO] Exemplo id={ex_id} sem 'question' válido. Pulando.")
            continue

        print(f"\n=== Executando id={ex_id} ===")
        print(f"Q: {question}")

        # 3) Tentativa inicial: rodar na GPU com quantização 4-bit
        try:
            print("Tentando na GPU...")
            out = run_query(
                index_dir=DEFAULT_INDEX_DIR,
                query=question,
                embed_model=DEFAULT_EMBED_MODEL,
                k=5,
                rerank_enabled=False,  # tiramos cross-encoder por simplicidade / VRAM
                ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",

                # structured
                sentences_per_doc=4,
                group_by_doc=True,
                wrap=100,

                # LLM (modo GPU)
                use_llm=True,
                gen_model_name=DEFAULT_GEN_MODEL,
                max_context_chars=3000,
                limit_context_hits=3,
                dtype_str="fp16",   # meio termo em GPU
                bits="4",           # quantização 4-bit (se bitsandbytes disponível)
                gpu_mem="8GiB",     # limite de VRAM
                max_new_tokens=256,
            )

        except RuntimeError as e:
            # Se a mensagem contiver CUDA OOM, cai para CPU
            if "CUDA out of memory" in str(e):
                print("⚠️ CUDA OOM DETECTADO — Limpando GPU e tentando na CPU...")

                clear_gpu()

                # Tentativa em CPU: dtype fp32, sem bits, sem gpu_mem
                out = run_query(
                    index_dir=DEFAULT_INDEX_DIR,
                    query=question,
                    embed_model=DEFAULT_EMBED_MODEL,
                    k=5,
                    rerank_enabled=False,
                    ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",

                    # structured
                    sentences_per_doc=4,
                    group_by_doc=True,
                    wrap=100,

                    # LLM (modo CPU)
                    use_llm=True,
                    gen_model_name=DEFAULT_GEN_MODEL,
                    max_context_chars=2000,  # contexto menor na CPU
                    limit_context_hits=2,
                    dtype_str="fp32",        # CPU = fp32
                    bits="none",             # sem quantização em CPU
                    gpu_mem=None,
                    max_new_tokens=180,
                )
            else:
                # Qualquer outro erro, a gente não tenta ser herói
                raise e

        # Limpa GPU mesmo após sucesso, para evitar fragmentação acumulada
        clear_gpu()

        # 4) Extração de dados da saída do run_query
        hits = out.get("hits", [])
        structured_ans = out.get("structured_answer") or ""
        llm_ans = out.get("llm_answer") or ""

        retrieved_docs = _extract_retrieved_docs(hits, top_n=5)
        chunk_summaries = _extract_chunk_summaries(hits, max_chunks=10)

        pred_record: Dict[str, Any] = {
            "id": ex_id,
            "question": question,

            # Gold (se quiser usar depois para métricas)
            "gold_answer": item.get("gold_answer"),
            "gold_docs": item.get("gold_docs"),

            # Saídas do sistema
            "structured_answer": structured_ans,
            "llm_answer": llm_ans,

            # Info de recuperação
            "retrieved_docs": retrieved_docs,
            "retrieved_chunks": chunk_summaries,
        }

        predictions.append(pred_record)

    # 5) Salva tudo em um único JSON
    out_data = {
        "predictions": predictions,
        "num_examples": len(predictions),
    }

    OUT_JSON_PATH.write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=======================================")
    print(f"✔ Arquivo de predições salvo em: {OUT_JSON_PATH.resolve()}")
    print(f"Total de entradas: {len(predictions)}")
    print("=======================================")


if __name__ == "__main__":
    main()
