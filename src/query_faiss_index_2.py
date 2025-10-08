#!/usr/bin/env python3
"""
query_faiss_index.py
Consulta um índice FAISS (criado pelo build_faiss_index.py) e exibe os top-k resultados.
Opcional: gera resposta com LLM local (Transformers), rodando na GPU e com suporte a quantização 4-bit.

Requisitos:
    pip install faiss-cpu sentence-transformers
    pip install transformers accelerate
    (opcional para 4-bit) pip install bitsandbytes

Exemplos:
    # Buscar top-5
    python query_faiss_index.py \
      --index-dir "/caminho/index" \
      --model "sentence-transformers/all-MiniLM-L6-v2" \
      --query "Explique backpropagation" --k 5

    # Gerar resposta com LLM local, em FP16 na GPU
    python query_faiss_index.py \
      --index-dir "/caminho/index" \
      --model "sentence-transformers/all-MiniLM-L6-v2" \
      --query "Filtros de Kalman" --k 6 \
      --generate \
      --gen-model "Qwen/Qwen2.5-3B-Instruct" \
      --dtype fp16

    # Gerar resposta em 4-bit (economiza VRAM)
    python query_faiss_index.py \
      --index-dir "/caminho/index" \
      --model "sentence-transformers/all-MiniLM-L6-v2" \
      --query "Transformers em visão computacional" \
      --k 6 \
      --generate \
      --gen-model "Qwen/Qwen2.5-3B-Instruct" \
      --bits4 --dtype bf16
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ----------------- FAISS UTILS -----------------
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
    """Busca top-k no FAISS e retorna registros + score."""
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


# ----------------- GERAÇÃO COM LLM -----------------
def generate_answer(question: str, hits: List[Dict[str, Any]], gen_model_name: str,
                    max_context_chars: int = 4000, dtype_str: str = "fp16", use_bits4: bool = False) -> str:
    """
    Usa Transformers para gerar resposta com base nos trechos recuperados.
    Roda na GPU com dtype selecionado e, opcionalmente, 4-bit (bitsandbytes).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    # monta contexto concatenado
    ctx_parts, used = [], 0
    for h in hits:
        cite = f"({h.get('doc_title')}, p.{h.get('page_start')})" if h.get('page_start') else f"({h.get('doc_title')})"
        block = f"{cite}\n{h['text']}\n"
        if used + len(block) > max_context_chars:
            break
        ctx_parts.append(block)
        used += len(block)
    context = "\n---\n".join(ctx_parts)

    system = ("Você é um assistente técnico. Responda APENAS com base no contexto fornecido. "
              "Sempre cite as fontes entre parênteses no formato (Documento, p.X). "
              "Se não houver informação suficiente, diga isso claramente.")
    user = f"Pergunta: {question}\n\nContexto:\n{context}\n\nResposta:"

    # dtype
    if dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device_map = "auto"  # distribui para GPU(s) automaticamente

    if use_bits4:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("Para usar --bits4 instale: pip install bitsandbytes") from e

        bnb_conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(torch.bfloat16 if dtype_str=="bf16" else torch.float16),
        )
        tok = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            quantization_config=bnb_conf,
            device_map=device_map
        )
    else:
        tok = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    gen = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=400, do_sample=False)
    out = gen(f"{system}\n\n{user}")[0]["generated_text"]
    if "Resposta:" in out:
        out = out.split("Resposta:", 1)[-1].strip()
    return out.strip()


# ----------------- MAIN -----------------
def main():
    ap = argparse.ArgumentParser(description="Consulta índice FAISS e opcionalmente gera resposta com LLM.")
    ap.add_argument("--index-dir", type=str, required=True, help="Pasta do índice (faiss.index + meta.json)")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Modelo de embeddings (o MESMO usado na indexação)")
    ap.add_argument("--query", type=str, required=True, help="Pergunta/consulta")
    ap.add_argument("--k", type=int, default=5, help="Número de resultados retornados (top-k)")
    ap.add_argument("--generate", action="store_true", help="Se presente, chama LLM para gerar resposta")
    ap.add_argument("--gen-model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                    help="Modelo HuggingFace para geração")
    ap.add_argument("--max-context-chars", type=int, default=4000, help="Limite de caracteres do contexto")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"],
                    help="Precisão da GPU (default: fp16)")
    ap.add_argument("--bits4", action="store_true", help="Ativa quantização 4-bit (bitsandbytes)")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    index, metas = load_index(index_dir)

    print(f"[1/2] Buscando: {args.query!r}")
    hits = retrieve(index, metas, args.query, args.model, k=args.k)
    pretty_print_hits(hits)

    if args.generate:
        print("\n[2/2] Gerando resposta com LLM (GPU se disponível)…")
        try:
            answer = generate_answer(
                args.query, hits,
                gen_model_name=args.gen_model,
                max_context_chars=args.max_context_chars,
                dtype_str=args.dtype,
                use_bits4=args.bits4
            )
            print("\n--- RESPOSTA ---\n" + answer)
        except Exception as e:
            print(f"[ERRO] Falha ao gerar resposta: {e}")

if __name__ == "__main__":
    main()
