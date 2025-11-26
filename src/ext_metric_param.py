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
          - metadata resumida dos chunks (opcional)
  - Ao final, grava um único arquivo JSON com todas as predições.

Esse JSON depois pode ser usado por outro script para calcular métricas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from query_engine import run_query
from tqdm import tqdm
# --------- CONFIGURAÇÕES BÁSICAS ---------

# Arquivo com as perguntas + gabarito
GOLD_PATH = Path("src/metrics/eval_gold.json")

# Arquivo de saída com as predições do RAG
OUT_JSON_PATH = Path("src/metrics/eval_predictions.json")

# Caminho do índice FAISS
DEFAULT_INDEX_DIR = "/home/pdi_4/Documents/Documentos/rag/books_rag/index"

# Modelo de embeddings usado na indexação
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo LLM para geração
DEFAULT_GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"


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


def _extract_chunk_summaries(hits: List[Dict[str, Any]], max_chunks: int = 10) -> List[Dict[str, Any]]:
    """
    Cria um resumo dos chunks recuperados para inspeção posterior.
    Não salva o texto inteiro pra não ficar gigante, só um preview + metadados.
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


def main():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Arquivo de perguntas não encontrado: {GOLD_PATH}")

    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        raise ValueError("Nenhum qa_pairs encontrado em eval_gold.json")

    predictions: List[Dict[str, Any]] = []

    print(f"Carregado {len(qa_pairs)} exemplos de avaliação.")
    print(f"Usando índice: {DEFAULT_INDEX_DIR}")

    for item in tqdm(qa_pairs):
        ex_id = int(item.get("id"))
        question = item.get("question", "").strip()
        if not question:
            print(f"[AVISO] Exemplo id={ex_id} sem 'question' válido. Pulando.")
            continue

        print(f"\n=== Rodando exemplo id={ex_id} ===")
        print(f"Q: {question}")

        # Chama o RAG (run_query é o motor principal)
        out = run_query(
            index_dir=DEFAULT_INDEX_DIR,
            query=question,
            embed_model=DEFAULT_EMBED_MODEL,
            k=7,
            rerank_enabled=False,
            ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            # structured
            sentences_per_doc=4,
            group_by_doc=True,
            wrap=100,
            # llm
            use_llm=True,
            gen_model_name=DEFAULT_GEN_MODEL,
            max_context_chars=5000,
            limit_context_hits=6,
            dtype_str="fp16",
            bits="none",
            gpu_mem=None,
            max_new_tokens=500,
        )

        hits = out.get("hits", [])
        structured_ans = out.get("structured_answer")
        llm_ans = out.get("llm_answer")

        retrieved_docs = _extract_retrieved_docs(hits, top_n=5)
        chunk_summaries = _extract_chunk_summaries(hits, max_chunks=10)

        pred_record: Dict[str, Any] = {
            "id": ex_id,
            "question": question,
            # mantemos o gold se existir, ajuda em debug
            "gold_answer": item.get("gold_answer"),
            "gold_docs": item.get("gold_docs"),
            # saídas do seu sistema
            "structured_answer": structured_ans,
            "llm_answer": llm_ans,
            # info de recuperação
            "retrieved_docs": retrieved_docs,
            "retrieved_chunks": chunk_summaries,
        }

        predictions.append(pred_record)

    # Salva tudo em um único JSON
    out_data = {
        "predictions": predictions,
        "num_examples": len(predictions),
    }

    OUT_JSON_PATH.write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n✅ Arquivo de predições salvo em: {OUT_JSON_PATH.resolve()}")
    print(f"Total de entradas: {len(predictions)}")


if __name__ == "__main__":
    main()
