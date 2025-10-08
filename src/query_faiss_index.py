#!/usr/bin/env python3
"""
query_faiss_index.py
Consulta um índice FAISS (criado pelo build_faiss_index.py) e exibe os top-k resultados.
Opcional: gera resposta com LLM local (Transformers) a partir dos trechos recuperados.

Requisitos mínimos:
    pip install faiss-cpu sentence-transformers

Para geração (opcional):
    pip install transformers accelerate

Exemplos:
    # Buscar top-5
    python query_faiss_index.py \
      --index-dir "/caminho/index" \
      --model "sentence-transformers/all-MiniLM-L6-v2" \
      --query "Explique backpropagation" --k 5

    # Gerar resposta com LLM local
    python query_faiss_index.py \
      --index-dir "/caminho/index" \
      --model "sentence-transformers/all-MiniLM-L6-v2" \
      --query "Filtros de Kalman" --k 6 \
      --generate \
      --gen-model "Qwen/Qwen2.5-3B-Instruct" \
      --max-context-chars 4000
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --------- util ----------
def load_index(index_dir: Path):
    """Carrega FAISS e meta.json."""
    faiss_path = index_dir / "faiss.index"
    meta_path  = index_dir / "meta.json"
    if not faiss_path.exists():
        raise FileNotFoundError(f"Índice FAISS não encontrado: {faiss_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadados não encontrados: {meta_path}")

    index = faiss.read_index(str(faiss_path))
    with meta_path.open(encoding="utf-8") as f:
        metas = json.load(f)
    return index, metas

def embed_query(query: str, model_name: str) -> np.ndarray:
    """Gera embedding normalizado para a consulta."""
    model = SentenceTransformer(model_name)
    v = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v

def retrieve(index, metas: List[Dict[str, Any]], query: str, model_name: str, k: int = 5) -> List[Dict[str, Any]]:
    """Busca top-k no FAISS e retorna os registros com score."""
    qv = embed_query(query, model_name)
    D, I = index.search(qv, k)
    hits = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        rec = dict(metas[idx])
        rec["score"] = float(score)
        rec["rank"] = rank
        hits.append(rec)
    return hits

def pretty_print_hits(hits: List[Dict[str, Any]]):
    """Imprime resultados de forma legível com citações."""
    for h in hits:
        cite = f"({h.get('doc_title')}, p.{h.get('page_start')})" if h.get("page_start") else f"({h.get('doc_title')})"
        preview = (h["text"].replace("\n", " "))[:300] + ("…" if len(h["text"]) > 300 else "")
        print(f"\n[{h['rank']}] score={h['score']:.4f}  {cite}")
        print(f"chunk_id: {h.get('chunk_id')}  |  tipo: {h.get('format')}  |  fonte: {h.get('source_file')}")
        print(f"preview: {preview}")

# --------- geração opcional ----------
def generate_answer(question: str, hits: List[Dict[str, Any]], gen_model_name: str, max_context_chars: int = 4000) -> str:
    """
    Usa Transformers para gerar uma resposta com base nos trechos recuperados.
    Monta um contexto com citações curtas para caber no limite.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except Exception as e:
        raise RuntimeError("Transformers não está instalado. Rode: pip install transformers accelerate") from e

    # monta contexto concatenado (trechos + citações)
    ctx_parts = []
    used = 0
    for h in hits:
        cite = f"({h.get('doc_title')}, p.{h.get('page_start')})" if h.get("page_start") else f"({h.get('doc_title')})"
        block = f"{cite}\n{h['text']}\n"
        if used + len(block) > max_context_chars:
            break
        ctx_parts.append(block)
        used += len(block)
    context = "\n---\n".join(ctx_parts)

    # prompt simples e assertivo
    system = ("Você é um assistente técnico. Responda APENAS com base no contexto fornecido. "
              "Sempre cite as fontes entre parênteses no formato (Documento, p.X). "
              "Se não houver informação suficiente, diga isso claramente.")
    user = f"Pergunta: {question}\n\nContexto:\n{context}\n\nResposta:"

    tok = AutoTokenizer.from_pretrained(gen_model_name)
    mdl = AutoModelForCausalLM.from_pretrained(gen_model_name, torch_dtype="auto", device_map="auto")
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=400)

    out = gen(f"{system}\n\n{user}", do_sample=False)[0]["generated_text"]
    # tenta cortar tudo antes de "Resposta:" se o modelo repetir o prompt
    if "Resposta:" in out:
        out = out.split("Resposta:", 1)[-1].strip()
    return out.strip()

def main():
    ap = argparse.ArgumentParser(description="Consulta índice FAISS e opcionalmente gera resposta com LLM.")
    ap.add_argument("--index-dir", type=str, required=True, help="Pasta do índice (onde estão faiss.index e meta.json)")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Modelo de embeddings (o MESMO usado na indexação)")
    ap.add_argument("--query", type=str, required=True, help="Pergunta/consulta")
    ap.add_argument("--k", type=int, default=5, help="Número de resultados retornados (top-k)")
    ap.add_argument("--generate", action="store_true", help="Se presente, chama LLM para gerar resposta")
    ap.add_argument("--gen-model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                    help="Modelo HF para geração (se --generate)")
    ap.add_argument("--max-context-chars", type=int, default=4000, help="Limite de caracteres do contexto")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    index, metas = load_index(index_dir)

    print(f"[1/2] Buscando: {args.query!r}")
    hits = retrieve(index, metas, args.query, args.model, k=args.k)
    pretty_print_hits(hits)

    if args.generate:
        print("\n[2/2] Gerando resposta com LLM …")
        try:
            answer = generate_answer(args.query, hits, args.gen_model, args.max_context_chars)
            print("\n--- RESPOSTA ---\n" + answer)
        except Exception as e:
            print(f"[ERRO] Falha ao gerar resposta: {e}")

if __name__ == "__main__":
    main()
