#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_metrics.py

Métricas simples para avaliar um pipeline de RAG usando o arquivo
gerado por build_eval_predictions.py (eval_predictions.json).

Mede:
  - Retrieval:
      * recall@k
      * mrr@k
  - Resposta:
      * token_f1 da resposta structured
      * token_f1 da resposta da LLM

Formato esperado do arquivo de avaliação (JSON ÚNICO, NÃO JSONL):

{
  "num_examples": N,
  "predictions": [
    {
      "id": int,
      "question": str,
      "gold_answer": str,
      "gold_docs": [str, ...],
      "structured_answer": str,
      "llm_answer": str | null,
      "retrieved_docs": [str, ...],
      "retrieved_chunks": [ ... ]
    },
    ...
  ]
}

Uso:
  python rag_metrics.py --file eval_predictions.json --k 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import dataclass
from typing import List, Sequence, Set, Optional
import pprint


# =========================
# Utilidades de texto
# =========================

def _normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, remove pontuação simples, trim."""
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúãõâêôç]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Tokenização bem simples, separando em espaços após normalização."""
    if not text:
        return []
    return _normalize_text(text).split()


def _normalize_doc_id(doc: str) -> str:
    """
    Normaliza um "id" de documento (aqui usamos título/nome do arquivo)
    para comparação em métricas de retrieval.
    """
    if not doc:
        return ""
    return re.sub(r"\s+", " ", doc.strip().lower())


# =========================
# Estrutura de um exemplo
# =========================

@dataclass
class RagExample:
    question: str
    gold_docs: List[str]            # títulos/ids relevantes (gold_docs)
    retrieved_docs: List[str]       # títulos/ids recuperados (retrieved_docs)
    gold_answer: str                # resposta de referência
    structured_answer: str          # resposta structured do RAG
    llm_answer: Optional[str]       # resposta da LLM (pode ser None)


# =========================
# Métricas de retrieval
# =========================

def recall_at_k(
    relevant: Sequence[str],
    retrieved: Sequence[str],
    k: int
) -> float:
    """
    Recall@k = (# relevantes nos top-k) / (# relevantes totais).
    Se não houver relevantes, retorna 0.0.
    """
    if not relevant:
        return 0.0
    rel_set: Set[str] = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in rel_set)
    return hits / float(len(rel_set))


def mrr_at_k(
    relevant: Sequence[str],
    retrieved: Sequence[str],
    k: int
) -> float:
    """
    MRR@k (Mean Reciprocal Rank para UMA query):
      - procura a posição do primeiro documento relevante nos top-k
      - se achar: retorna 1 / rank
      - se não achar: retorna 0.0
    """
    if not relevant:
        return 0.0
    rel_set: Set[str] = set(relevant)
    for idx, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in rel_set:
            return 1.0 / idx
    return 0.0


# =========================
# Métrica de resposta: F1
# =========================

def token_f1(
    reference: str,
    prediction: str
) -> float:
    """
    F1 baseado em tokens:
      - Precisão = overlap / # tokens da predição
      - Recall   = overlap / # tokens da referência
      - F1       = 2 * P * R / (P + R)
    """
    ref_tokens = _tokenize(reference)
    pred_tokens = _tokenize(prediction)

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    overlap = 0
    for t, c in pred_counts.items():
        if t in ref_counts:
            overlap += min(c, ref_counts[t])

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# =========================
# Carregamento do dataset
# =========================

def load_eval_file(path: pathlib.Path) -> List[RagExample]:
    """
    Carrega eval_predictions.json no formato:

    {
      "num_examples": N,
      "predictions": [ ... ]
    }
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    preds = data.get("predictions", [])
    examples: List[RagExample] = []

    for item in preds:
        question = item.get("question", "")
        gold_answer = item.get("gold_answer", "") or ""
        gold_docs = item.get("gold_docs") or []
        retrieved_docs = item.get("retrieved_docs") or []
        structured_answer = item.get("structured_answer", "") or ""
        llm_answer = item.get("llm_answer")  # pode ser None

        ex = RagExample(
            question=question,
            gold_docs=[_normalize_doc_id(d) for d in gold_docs],
            retrieved_docs=[_normalize_doc_id(d) for d in retrieved_docs],
            gold_answer=gold_answer,
            structured_answer=structured_answer,
            llm_answer=llm_answer if llm_answer is not None else "",
        )
        examples.append(ex)

    return examples


# =========================
# Avaliação em lote
# =========================

def evaluate_dataset(
    examples: List[RagExample],
    k: int
) -> dict:
    """Calcula métricas médias no conjunto de avaliação."""
    recalls: List[float] = []
    mrrs: List[float] = []
    f1_structured: List[float] = []
    f1_llm: List[float] = []

    for ex in examples:
        # Retrieval metrics
        recalls.append(recall_at_k(ex.gold_docs, ex.retrieved_docs, k))
        mrrs.append(mrr_at_k(ex.gold_docs, ex.retrieved_docs, k))

        # Answer metrics: structured
        f1_structured.append(token_f1(ex.gold_answer, ex.structured_answer))

        # Answer metrics: LLM
        if ex.llm_answer:
            f1_llm.append(token_f1(ex.gold_answer, ex.llm_answer))

    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "k": k,
        "recall@k": _avg(recalls),
        "mrr@k": _avg(mrrs),
        "answer_token_f1_structured": _avg(f1_structured),
        "answer_token_f1_llm": _avg(f1_llm),
        "num_examples": len(examples),
        "num_llm_examples": len(f1_llm),
    }


# =========================
# CLI simples
# =========================

def main():
    ap = argparse.ArgumentParser(description="Avaliação de RAG (retrieval + resposta) usando eval_predictions.json.")
    ap.add_argument("--file", type=str, required=True, help="Arquivo JSON com predições (eval_predictions.json).")
    ap.add_argument("--k", type=int, default=5, help="Valor de k para recall@k e mrr@k (default: 5)")
    args = ap.parse_args()

    path = pathlib.Path(args.file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de avaliação não encontrado: {path}")

    examples = load_eval_file(path)
    stats = evaluate_dataset(examples, k=args.k)

    pprint.pprint(r"""
            📚📚📚  RESULTADOS DA AVALIAÇÃO RAG  📚📚📚
            
                ________________________________
                /                               /|
                /   ┌──────────────────────┐    / |
            /    │   Livro 1            │   /  |
            /     └──────────────────────┘  /   |
            /________________________________/    |
            |                                |    |
            |   ┌──────────────────────┐     |    |
            |   │   Livro 2            │     |    |
            |   └──────────────────────┘     |    |
            |________________________________|   /
            (___________________________________)

    """)
    pprint.pprint(f"Arquivo: {path}")
    pprint.pprint(f"n = {stats['num_examples']}  |  k = {stats['k']}")
    pprint.pprint(f"Recall@{stats['k']}:                 {stats['recall@k']:.4f}")
    pprint.pprint(f"MRR@{stats['k']}:                    {stats['mrr@k']:.4f}")
    pprint.pprint(f"Answer token F1 (structured):        {stats['answer_token_f1_structured']:.4f}")
    pprint.pprint(f"Answer token F1 (LLM, média):        {stats['answer_token_f1_llm']:.4f}")
    pprint.pprint(f"Exemplos com LLM:                    {stats['num_llm_examples']}")


if __name__ == "__main__":
    main()
