#!/usr/bin/env python3
"""
query_faiss_index.py
Consulta um índice FAISS (criado pelo build_faiss_index.py) e exibe os top-k resultados.
Opcional: gera resposta com LLM local (Transformers), com foco em economia de VRAM:

- Quantização configurável: --bits {none,8,4}
- Limite de VRAM por GPU: --gpu-mem 11GiB (usa max_memory + device_map="auto")
- Contexto controlado: --limit-context-hits e --max-context-chars
- Resposta mais curta: --max-new-tokens

Requisitos:
    pip install faiss-cpu sentence-transformers
    pip install transformers accelerate
    (opcional para 4/8-bit) pip install bitsandbytes
"""

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pprint import pprint


# ----------------- FAISS / IO -----------------
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


# ----------------- CITAÇÃO / IMPRESSÃO -----------------
def _human_title_from_source(source_file: Optional[str]) -> str:
    """Transforma o nome do arquivo em algo legível quando doc_title não existir."""
    if not source_file:
        return "Documento"
    stem = Path(source_file).stem
    title = re.sub(r"[_\-]+", " ", stem)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:1].upper() + title[1:] if title else "Documento"

def format_citation(h: Dict[str, Any]) -> str:
    """Citação rica: '<titulo> — <seção|parte> (p.X / pp.X–Y)' quando aplicável."""
    title = h.get("doc_title") or _human_title_from_source(h.get("source_file"))
    section = h.get("section") or h.get("part")
    fmt = (h.get("format") or "").lower()
    p_start = h.get("page_start")
    p_end = h.get("page_end")
    pages = None
    if isinstance(p_start, int) and p_start > 0:
        if isinstance(p_end, int) and p_end and p_end > p_start:
            pages = f"pp.{p_start}–{p_end}"
        else:
            pages = f"p.{p_start}"
    if fmt == "pdf":
        if section and pages: return f"{title} — {section} ({pages})"
        if pages:               return f"{title} ({pages})"
        if section:             return f"{title} — {section}"
        return title
    else:
        return f"{title} — {section}" if section else title

def colorize(s: str, color: Optional[str] = None, bold: bool = False, enable: bool = True) -> str:
    """Aplica cor ANSI opcional."""
    if not enable:
        return s
    codes = []
    if bold: codes.append("1")
    palette = {"cyan":"36","yellow":"33","green":"32","magenta":"35","red":"31","blue":"34"}
    if color in palette:
        codes.append(palette[color])
    return f"\033[{';'.join(codes)}m{s}\033[0m" if codes else s

def pretty_print_hits(hits: List[Dict[str, Any]], preview_chars: int = 300, color: bool = True):
    """Imprime resultados com citação rica e preview truncado."""
    for h in hits:
        cite = colorize(format_citation(h), "cyan", bold=True, enable=color)
        preview = (h["text"].replace("\n", " "))[:preview_chars] + ("…" if len(h["text"]) > preview_chars else "")
        print(f"\n[{h['rank']}] score={h['score']:.4f}  {cite}")
        print(f"chunk_id: {h.get('chunk_id')}  |  tipo: {h.get('format')}  |  fonte: {h.get('source_file')}")
        print(preview)

def _highlight_terms(text: str, terms: Iterable[str], enable_color: bool = True) -> str:
    """Realça termos (case-insensitive) no texto com ANSI amarelo."""
    if not terms: return text
    toks = sorted({t.strip() for t in terms if len(t.strip()) >= 2}, key=len, reverse=True)
    for t in toks:
        text = re.sub(re.escape(t), colorize(r"\g<0>", "yellow", bold=True, enable=enable_color), text, flags=re.I)
    return text

# def print_full_text(
#     hits: List[Dict[str, Any]],
#     ranks_to_show: Iterable[int],
#     wrap: int = 100,
#     highlight_terms_from_query: Optional[str] = None,
#     color: bool = True,
# ):
#     """Imprime texto completo dos ranks selecionados, com wrap e realce opcional."""
#     ranks_set = set(ranks_to_show)
#     for h in hits:
#         if h["rank"] not in ranks_set:
#             continue
#         header = f"[{h['rank']}] {format_citation(h)}  |  chunk_id: {h.get('chunk_id')}  |  tipo: {h.get('format')}  |  fonte: {h.get('source_file')}"
#         print("\n" + colorize(header, "green", bold=True, enable=color))
#         txt = h["text"]
#         if highlight_terms_from_query:
#             terms = re.split(r"\s+", highlight_terms_from_query.strip())
#             txt = _highlight_terms(txt, terms, enable_color=color)
#         print(textwrap.fill(txt, width=wrap, replace_whitespace=False, drop_whitespace=False))


def print_full_text(
    hits: List[Dict[str, Any]],
    ranks_to_show: Iterable[int],
    wrap: int = 100,
    highlight_terms_from_query: Optional[str] = None,
    color: bool = True,
):
    """Exibe o texto completo dos chunks selecionados com metadados formatados e destaque opcional."""
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
        pprint(colorize("📘  Informações do Documento", "green", bold=True, enable=color))
        pprint(header_info, sort_dicts=False, width=120)
        print("-" * 120)

        txt = h["text"]
        if highlight_terms_from_query:
            terms = re.split(r"\s+", highlight_terms_from_query.strip())
            txt = _highlight_terms(txt, terms, enable_color=color)

        wrapped = textwrap.fill(
            txt.strip(),
            width=wrap,
            replace_whitespace=False,
            drop_whitespace=False,
        )

        pprint(colorize("📝  Texto:", "cyan", bold=True, enable=color))
        pprint(wrapped)
        print("=" * 120 + "\n")


# ----------------- GERAÇÃO COM LLM (economia de VRAM) -----------------
def _parse_gpu_mem(gpu_mem_str: Optional[str], num_gpus: int) -> Optional[dict]:
    """
    Converte '11GiB' em dict p/ max_memory: {0: '11GiB', 1: '11GiB', 'cpu': '30GiB'}
    """
    if not gpu_mem_str:
        return None
    mm = {i: gpu_mem_str for i in range(num_gpus)}
    mm["cpu"] = "30GiB"
    return mm

def _build_bnb_conf(bits: str, dtype_str: str):
    """Cria configuração bitsandbytes para 8-bit ou 4-bit."""
    from transformers import BitsAndBytesConfig
    if bits == "8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif bits == "4":
        import torch
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(torch.bfloat16 if dtype_str=="bf16" else torch.float16),
        )
    return None

def generate_answer(
    question: str,
    hits: List[Dict[str, Any]],
    gen_model_name: str,
    max_context_chars: int = 4000,
    limit_context_hits: Optional[int] = None,
    dtype_str: str = "fp16",
    bits: str = "none",                # 'none' | '8' | '4'
    gpu_mem: Optional[str] = None,     # ex. "11GiB"
    max_new_tokens: int = 300,
) -> str:
    """
    Gera resposta com controle de VRAM:
      - bits: 8-bit/4-bit com bitsandbytes (CUDA)
      - gpu_mem: limita VRAM via max_memory
      - limit_context_hits: limita quantos trechos entram no contexto
      - max_new_tokens: reduz custo na decodificação
    """
    # monta contexto (limitando por hits e por chars)
    ctx_parts, used = [], 0
    selected_hits = hits if not limit_context_hits else hits[:limit_context_hits]
    for h in selected_hits:
        block = f"{format_citation(h)}\n{h['text']}\n"
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

    # device map + orçamento de VRAM
    device_map = "auto"
    num_gpus = torch.cuda.device_count()
    max_memory = _parse_gpu_mem(gpu_mem, num_gpus) if num_gpus > 0 else None

    # quantização (somente CUDA)
    bnb_conf = None
    if bits in {"8","4"}:
        if num_gpus == 0:
            print("[AVISO] Quantização 4/8-bit requer CUDA. Continuando sem quantização.")
        else:
            try:
                bnb_conf = _build_bnb_conf(bits, dtype_str)
            except Exception as e:
                print(f"[AVISO] Falha ao configurar bitsandbytes ({e}). Continuando sem quantização.")
                bnb_conf = None

    tok = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)

    # carrega modelo com as restrições
    try:
        if bnb_conf:
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
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError("CUDA OOM ao carregar o modelo. Tente: --bits 8/4, --gpu-mem menor, ou um modelo menor.")
        raise

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    try:
        out = gen(f"{system}\n\n{user}")[0]["generated_text"]
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise RuntimeError(
                "CUDA OOM durante a geração. Reduza --max-new-tokens, diminua --limit-context-hits, "
                "ou use --bits 8/4 e/ou ajuste --gpu-mem."
            )
        raise

    if "Resposta:" in out:
        out = out.split("Resposta:", 1)[-1].strip()
    return out.strip()

def rerank_hits(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-ranking com cross-encoder (query, passage) -> score."""
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce = CrossEncoder(model_name)
    pairs = [(query, h["text"]) for h in hits]
    scores = ce.predict(pairs)
    for h, s in zip(hits, scores):
        h["score_ce"] = float(s)
    hits = sorted(hits, key=lambda x: x["score_ce"], reverse=True)
    for i, h in enumerate(hits, start=1):
        h["rank"] = i
    return hits


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

    # CONTROLE DE CONTEXTO E SAÍDA (impacto em VRAM)
    ap.add_argument("--limit-context-hits", type=int, default=4,
                    help="Máximo de trechos recuperados a entrar no contexto (default: 4)")
    ap.add_argument("--max-context-chars", type=int, default=4000,
                    help="Limite de caracteres do contexto (default: 4000)")
    ap.add_argument("--max-new-tokens", type=int, default=300,
                    help="Máximo de tokens gerados (default: 300)")

    # PRECISÃO / QUANTIZAÇÃO / MEMÓRIA
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"],
                    help="Precisão em GPU (default: fp16)")
    ap.add_argument("--bits", type=str, default="none", choices=["none","8","4"],
                    help="Quantização com bitsandbytes (8 ou 4). Default: none")
    ap.add_argument("--gpu-mem", type=str, default=None,
                    help='Limite de VRAM por GPU, ex: "11GiB" (usa max_memory + device_map="auto")')

    # RE-RANK
    ap.add_argument("--rerank", action="store_true", help="Ativa re-ranking com cross-encoder")
    ap.add_argument("--m", type=int, default=50, help="Candidatos iniciais do FAISS para re-ranking (default: 50)")

    # VISUALIZAÇÃO
    ap.add_argument("--preview-chars", type=int, default=300, help="Tamanho do preview por hit (default: 300)")
    ap.add_argument("--show-text", type=str, default="", help='Lista de ranks para imprimir inteiro, ex: "1,3,5"')
    ap.add_argument("--wrap", type=int, default=100, help="Largura de quebra de linha no texto completo (default: 100)")
    ap.add_argument("--no-color", action="store_true", help="Desliga cores ANSI no terminal")
    ap.add_argument("--highlight-query", action="store_true", help="Realça termos da consulta no texto completo")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    index, metas = load_index(index_dir)

    print(f"[1/2] Buscando: {args.query!r}")
    m = args.m if args.rerank else args.k
    hits = retrieve(index, metas, args.query, args.model, k=m)
    if args.rerank:
        hits = rerank_hits(args.query, hits)[:args.k]

    pretty_print_hits(hits, preview_chars=args.preview_chars, color=(not args.no_color))

    if args.show_text:
        try:
            ranks = [int(x) for x in re.split(r"[,\s]+", args.show_text.strip()) if x]
        except ValueError:
            ranks = []
        if ranks:
            print_full_text(
                hits,
                ranks_to_show=ranks,
                wrap=args.wrap,
                highlight_terms_from_query=(args.query if args.highlight_query else None),
                color=(not args.no_color),
            )

    if args.generate:
        print("\n[2/2] Gerando resposta (economia de VRAM ativada)…")
        try:
            answer = generate_answer(
                question=args.query,
                hits=hits,
                gen_model_name=args.gen_model,
                max_context_chars=args.max_context_chars,
                limit_context_hits=args.limit_context_hits,
                dtype_str=args.dtype,
                bits=args.bits,
                gpu_mem=args.gpu_mem,
                max_new_tokens=args.max_new_tokens,
            )
            print("\n--- RESPOSTA ---\n" + answer)
        except Exception as e:
            print(f"[ERRO] Falha ao gerar resposta: {e}\n"
                  f"Sugestões: use --bits 8/4, diminua --limit-context-hits, --max-context-chars ou --max-new-tokens, "
                  f"ou passe --gpu-mem (ex: 11GiB).")

if __name__ == "__main__":
    main()
