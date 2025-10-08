#!/usr/bin/env python3
"""
build_faiss_index.py
Cria um índice FAISS a partir de JSONL de chunks (texto + metadados).

- Lê todos os .jsonl em --jsonl-dir
- Gera embeddings com sentence-transformers (modelo configurável)
- Normaliza embeddings e usa IndexFlatIP (produto interno) ~ coseno
- Salva:
    - FAISS index em <out_dir>/faiss.index
    - Metadados + texto em <out_dir>/meta.json

Requisitos:
    pip install faiss-cpu sentence-transformers numpy tqdm
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(jsonl_dir: Path) -> List[Dict[str, Any]]:
    """
    Lê todos os .jsonl e retorna uma lista de registros
    contendo texto e metadados essenciais.
    Espera linhas com chaves: text, doc_title, format, chunk_id, page_start (opcional).
    """
    files = sorted(jsonl_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Nenhum .jsonl encontrado em {jsonl_dir}")
    rows: List[Dict[str, Any]] = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                rows.append({
                    "text": text,
                    "doc_title": r.get("doc_title"),
                    "format": r.get("format"),
                    "chunk_id": r.get("chunk_id"),
                    "page_start": r.get("page_start"),
                    "source_file": fp.name,  # para rastrear origem
                })
    if not rows:
        raise RuntimeError("Nenhum chunk com texto foi encontrado.")
    return rows

def compute_embeddings(rows: List[Dict[str, Any]], model_name: str, batch_size: int = 64) -> np.ndarray:
    """
    Gera embeddings normalizados (norma L2 ~= 1) para os textos.
    Normalização é fundamental para usar IndexFlatIP como coseno.
    """
    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in rows]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True  # <- chave para IP ≈ coseno
    )
    # sanidade: todas normas ~1
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        print("[AVISO] Nem todos embeddings estão normalizados; normalizando manualmente.")
        emb = emb / np.clip(norms[:, None], 1e-12, None)
    return emb.astype("float32")

def build_faiss(emb: np.ndarray) -> faiss.Index:
    """
    Cria um índice FAISS “denso” simples (IndexFlatIP).
    Com embeddings normalizados, o produto interno é equivalente ao coseno.
    """
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return index

def main():
    ap = argparse.ArgumentParser(description="Cria índice FAISS a partir de JSONL de chunks.")
    ap.add_argument("--jsonl-dir", type=str, required=True, help="Pasta com .jsonl (corpus/jsonl)")
    ap.add_argument("--out-dir", type=str, required=True, help="Pasta de saída do índice (index/)")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Modelo de embeddings (default: all-MiniLM-L6-v2)")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size para encode (default: 64)")
    args = ap.parse_args()

    jsonl_dir = Path(args.jsonl_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Lendo chunks de {jsonl_dir} …")
    rows = load_chunks(jsonl_dir)
    print(f"  - {len(rows)} chunks com texto")

    print(f"[2/4] Gerando embeddings com '{args.model}' …")
    emb = compute_embeddings(rows, args.model, args.batch_size)
    print(f"  - matriz: {emb.shape}")

    print(f"[3/4] Construindo índice FAISS (IndexFlatIP) …")
    index = build_faiss(emb)
    faiss_path = out_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"  - salvo: {faiss_path}")

    print(f"[4/4] Gravando metadados …")
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"  - salvo: {meta_path}")

    print("\n✓ Índice criado com sucesso.")

if __name__ == "__main__":
    main()
