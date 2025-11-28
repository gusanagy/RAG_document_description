# main.py


# def main():
#     if len(sys.argv) < 2:
#         print("Use: python main.py chat | query \"pergunta\"")
#         return

#     mode = sys.argv[1]

#     if mode == "chat":
#         os.system("streamlit run src/chat_ui.py")
#         return

#     if mode == "query":
#         if len(sys.argv) < 3:
#             print("Forneça uma pergunta.")
#             return
#         from src.query_engine import run_query
#         out = run_query(
#             index_dir="/home/pdi_4/Documents/Documentos/rag/books_rag/index",
#             query=sys.argv[2],
#         )
#         print(out["structured"])
#         return

#     print("Modo inválido.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Core CLI do repositório "rag-books": ferramentas para:

  - Extrair livros e construir índice FAISS a partir de uma coleção (ex: Calibre / PDFs / EPUBs)
  - Inspecionar chunks e metadados do índice
  - Rodar um chat RAG com Streamlit sobre os livros indexados
  - Executar pipeline de avaliação (gerar predições + métricas)

Autor: Gustavo Almeida
Repositório: RAG de livros técnicos com FAISS + Sentence-Transformers + LLM local

Uso geral:
    python main.py <subcomando> [opções]

Exemplos:
    python main.py chat
    python main.py query --index-dir /caminho/index --question "O que é programação assertiva?"
    python main.py extract --books-dir /caminho/livros --index-dir /caminho/index
    python main.py check-index --index-dir /caminho/index
    python main.py validate
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
# Importa o motor principal de RAG
from src.query_engine import run_query


# =========================
# Constantes padrão
# =========================

# Diretório padrão do índice de livros
DEFAULT_INDEX_DIR = "/home/pdi_4/Documents/Documentos/rag/books_rag/index"

# Caminhos padrão para avaliação
DEFAULT_EVAL_GOLD = "src/metrics/eval_gold.json"
DEFAULT_EVAL_PRED = "src/metrics/eval_predictions.json"


# =========================
# Subcomando: chat
# =========================

def cmd_chat(args: argparse.Namespace) -> None:
    """
    Abre a interface de chat Streamlit.

    Basicamente um wrapper bonitinho pra:
        streamlit run src/chat_ui.py
    """
    # Se quiser passar o index-dir via env, pode exportar aqui mais tarde.
    cmd = "streamlit run src/chat_ui.py"
    print(f"[chat] Rodando: {cmd}")
    os.system(cmd)


# =========================
# Subcomando: query
# =========================

def cmd_query(args: argparse.Namespace) -> None:
    """
    Executa uma única consulta ao índice e imprime a resposta structured no terminal.
    """
    index_dir = args.index_dir
    question = args.question

    print(f"[query] Usando índice: {index_dir}")
    print(f"[query] Pergunta: {question!r}")

    out = run_query(
        index_dir=index_dir,
        query=question,
        # parâmetros padrão razoáveis
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        k=args.k,
        rerank_enabled=args.rerank,
        ce_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        # structured
        sentences_per_doc=4,
        group_by_doc=True,
        wrap=100,
        # llm (aqui opcional)
        use_llm=args.use_llm,
        gen_model_name="Qwen/Qwen2.5-3B-Instruct",
        max_context_chars=3000,
        limit_context_hits=3,
        dtype_str="fp16",
        bits="4",
        gpu_mem="8GiB",
        max_new_tokens=256,
    )

    print("\n=== QUERY ===")
    print(out["query"])

    print("\n=== STRUCTURED ANSWER ===")
    print(out.get("structured_answer", ""))

    if args.use_llm and out.get("llm_answer"):
        print("\n=== LLM ANSWER ===")
        print(out["llm_answer"])

    print("\n=== PRIMEIROS HITS ===")
    for h in out.get("hits", [])[:args.k]:
        title = h.get("doc_title") or Path(h.get("source_file", "")).stem
        p = h.get("page_start")
        score = h.get("score")
        print(f"- {title} (p.{p}) | score={score:.4f}")


# =========================
# Subcomando: extract
# =========================

def cmd_extract(args: argparse.Namespace) -> None:
    """
    Roda o pipeline de extração + indexação.

    Aqui eu assumo que você tem um script `src/build_faiss_index.py`
    responsável por:
      - ler os livros
      - chunkear
      - gerar embeddings
      - salvar faiss.index + meta.json

    Em vez de reinventar a roda, chamo ele via linha de comando.
    Ajuste o comando conforme a interface real do seu script.
    """

    books_dir = Path(args.books_dir).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    model = args.embed_model

    if not books_dir.exists():
        raise FileNotFoundError(f"Diretório de livros não encontrado: {books_dir}")

    index_dir.mkdir(parents=True, exist_ok=True)

    # Ajuste este comando para bater com o seu build_faiss_index.py
    cmd = (
        f"python src/build_faiss_index.py "
        f"--books-dir {books_dir} "
        f"--index-dir {index_dir} "
        f"--model {model}"
    )

    print(f"[extract] Rodando pipeline de extração + indexação:")
    print(f"         {cmd}")
    os.system(cmd)


# =========================
# Subcomando: check-index
# =========================

def _load_meta(index_dir: str | Path) -> List[Dict[str, Any]]:
    meta_path = Path(index_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json não encontrado em: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def cmd_check_index(args: argparse.Namespace) -> None:
    """
    Inspeciona o índice:
      - número total de chunks
      - número de documentos distintos
      - top-N documentos por número de chunks

    Útil para checar se a extração ficou minimamente razoável.
    """
    index_dir = args.index_dir
    metas = _load_meta(index_dir)

    print(f"[check-index] Inspecionando índice em: {index_dir}")
    print(f"[check-index] Total de chunks: {len(metas)}")

    # Agrupa por doc_title ou source_file
    per_doc: Dict[str, int] = {}
    for m in metas:
        title = m.get("doc_title") or Path(m.get("source_file", "")).stem or "Documento"
        per_doc[title] = per_doc.get(title, 0) + 1

    print(f"[check-index] Total de documentos distintos: {len(per_doc)}")

    # Top-N por número de chunks
    top_n = args.top_n
    print(f"\n[check-index] Top {top_n} documentos por #chunks:")
    for title, n_chunks in sorted(per_doc.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
        print(f"  - {title}: {n_chunks} chunks")


# =========================
# Subcomando: validate
# =========================

def cmd_validate(args: argparse.Namespace) -> None:
    """
    Roda a pipeline de avaliação:

      1) Gera predições do RAG a partir de eval_gold.json
         (script: src/metrics/build_eval_predictions.py)

      2) Calcula métricas no arquivo de predições
         (script: src/metrics/rag_metrics.py ou equivalente)

    Aqui eu assumo que:

      - build_eval_predictions.py lê eval_gold.json e grava eval_predictions.json
      - rag_metrics.py aceita `--file` apontando para eval_predictions.json (ou outro formato que você definiu)

    Ajuste os caminhos/nomes se seu repo tiver variações.
    """
    eval_gold = Path(args.eval_gold).expanduser().resolve()
    eval_pred = Path(args.eval_pred).expanduser().resolve()

    print(f"[validate] Usando gold: {eval_gold}")
    print(f"[validate] Saída de predições: {eval_pred}")

    if not eval_gold.exists():
        raise FileNotFoundError(f"Arquivo de avaliação (gold) não encontrado: {eval_gold}")

    # 1) Gera predições
    print("\n[validate] (1/2) Gerando predições com build_eval_predictions.py ...")
    # Aqui assumo que o script já tem paths fixos internos.
    # Se ele aceitar argumentos, adapte o comando conforme necessário.
    os.system("python src/metrics/build_eval_predictions.py")

    # 2) Calcula métricas (se não foi desativado)
    if not args.skip_metrics:
        print("\n[validate] (2/2) Calculando métricas com rag_metrics.py ...")
        # Ajuste esse comando conforme a interface do seu rag_metrics.py
        cmd = f"python src/metrics/rag_metrics.py --file {eval_pred} --k {args.k}"
        print(f"[validate] Rodando: {cmd}")
        os.system(cmd)
    else:
        print("[validate] skip-metrics=True, pulando cálculo de métricas.")


# =========================
# Parser de linha de comando
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI principal do projeto 'rag-books' (RAG sobre biblioteca de livros)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- chat ----
    p_chat = subparsers.add_parser("chat", help="Abre o chat RAG via Streamlit.")
    p_chat.set_defaults(func=cmd_chat)

    # ---- query ----
    p_query = subparsers.add_parser("query", help="Executa uma consulta rápida ao índice e imprime no terminal.")
    p_query.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Caminho do índice FAISS (default: {DEFAULT_INDEX_DIR})",
    )
    p_query.add_argument(
        "--question",
        type=str,
        required=True,
        help="Pergunta a ser feita ao RAG.",
    )
    p_query.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k documentos/chunks retornados (default: 5).",
    )
    p_query.add_argument(
        "--rerank",
        action="store_true",
        help="Ativa re-ranking com cross-encoder.",
    )
    p_query.add_argument(
        "--use-llm",
        action="store_true",
        help="Se presente, também gera resposta via LLM.",
    )
    p_query.set_defaults(func=cmd_query)

    # ---- extract ----
    p_extract = subparsers.add_parser(
        "extract",
        help="Extrai livros e constrói índice FAISS (wrapper para src/build_faiss_index.py).",
    )
    p_extract.add_argument(
        "--books-dir",
        type=str,
        required=True,
        help="Diretório contendo os livros em formato bruto (PDF, EPUB, etc.).",
    )
    p_extract.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Pasta de saída para o índice (default: {DEFAULT_INDEX_DIR})",
    )
    p_extract.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Modelo de embeddings usado na indexação (deve bater com o da consulta).",
    )
    p_extract.set_defaults(func=cmd_extract)

    # ---- check-index ----
    p_check = subparsers.add_parser(
        "check-index",
        help="Mostra estatísticas básicas sobre os chunks do índice (meta.json).",
    )
    p_check.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Caminho do índice FAISS (default: {DEFAULT_INDEX_DIR})",
    )
    p_check.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Número de documentos a mostrar no ranking por #chunks (default: 10).",
    )
    p_check.set_defaults(func=cmd_check_index)

    # ---- validate ----
    p_val = subparsers.add_parser(
        "validate",
        help="Roda pipeline de avaliação (gera predições + métricas).",
    )
    p_val.add_argument(
        "--eval-gold",
        type=str,
        default=DEFAULT_EVAL_GOLD,
        help=f"Arquivo JSON com qa_pairs de referência (default: {DEFAULT_EVAL_GOLD}).",
    )
    p_val.add_argument(
        "--eval-pred",
        type=str,
        default=DEFAULT_EVAL_PRED,
        help=f"Arquivo JSON de saída com predições (default: {DEFAULT_EVAL_PRED}).",
    )
    p_val.add_argument(
        "--k",
        type=int,
        default=5,
        help="Valor de k para métricas de retrieval (passado para rag_metrics.py).",
    )
    p_val.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Gera predições mas NÃO roda o script de métricas.",
    )
    p_val.set_defaults(func=cmd_validate)

    return parser


# =========================
# Entry point
# =========================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Despacha para a função associada ao subcomando
    args.func(args)


if __name__ == "__main__":
    main()
    """
        python main.py extract --books-dir ...
        python main.py check-index
        python main.py chat
        python main.py validate
    """

