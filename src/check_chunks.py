#!/usr/bin/env python3
"""
check_chunks.py
Avalia a qualidade dos chunks gerados (JSONL) e produz:
  1) Histograma do tamanho dos chunks (todos os livros)
  2) Barra com taxa de chunks curtos por livro (top-N)
  3) Uma pequena tabela (matplotlib) com resumo por livro: total, curtos, % e tipo (pdf/epub)
  4) Um CSV com o resumo completo

Uso:
  python src/check_chunks.py \
    --jsonl-dir "/home/SEU_USUARIO/books_rag/corpus/jsonl" \
    --min-len 100 \
    --top 15 \
    --out-dir "/home/SEU_USUARIO/books_rag/working/reports"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import csv
import textwrap

def load_jsonl_stats(jsonl_dir: Path, min_len: int) -> Dict[str, Any]:
    """
    Lê todos os .jsonl e computa:
      - por arquivo (doc): total, curtos, formato, título
      - distribuição global de tamanhos (lista com len(text) por chunk)
    Retorna dict com:
      {
        "per_doc": {
          "<filename.jsonl>": {
            "doc_title": str,
            "format": "pdf"|"epub"|None,
            "total": int,
            "short": int,
            "short_pct": float
          },
          ...
        },
        "lengths": [int, int, ...]  # tamanhos de todos os chunks
      }
    """
    per_doc: Dict[str, Dict[str, Any]] = {}
    all_lengths: List[int] = []

    files = sorted(jsonl_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Nenhum .jsonl encontrado em {jsonl_dir}")

    for fp in files:
        total = 0
        short = 0
        fmt = None
        title = fp.stem  # fallback: nome do arquivo sem .jsonl

        with fp.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                txt = (rec.get("text") or "").strip()
                L = len(txt)
                all_lengths.append(L)
                total += 1
                if L < min_len:
                    short += 1

                # tenta descobrir formato/título a partir do JSONL
                if fmt is None:
                    fmt = rec.get("format")
                if title == fp.stem:
                    title = rec.get("doc_title") or title

        per_doc[fp.name] = {
            "doc_title": title,
            "format": fmt or "desconhecido",
            "total": total,
            "short": short,
            "short_pct": (short / total * 100.0) if total else 0.0,
        }

    return {"per_doc": per_doc, "lengths": all_lengths}

def ensure_outdir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def plot_hist_lengths(lengths: List[int], out_dir: Path, bins: int = 40):
    """
    Histograma de tamanhos de chunks (todos os livros).
    Tornado mais legível:
      - figure grande
      - grade leve
      - rótulos e título claros
      - limites automáticos e tight_layout
    """
    if not lengths:
        return

    fig = plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Tamanho do chunk (nº de caracteres)")
    plt.ylabel("Frequência")
    plt.title("Distribuição de tamanhos de chunks (todos os livros)")
    plt.tight_layout()

    png = out_dir / "hist_tamanhos_chunks.png"
    plt.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[OK] Histograma salvo em: {png}")

def plot_short_rate_bar(per_doc: Dict[str, Dict[str, Any]], out_dir: Path, top: int = 15):
    """
    Barra com % de chunks curtos por livro (top-N).
    Mais legível:
      - títulos encurtados
      - rotação de labels
      - grid leve
    """
    if not per_doc:
        return

    # ordena por % de curtos (desc) e pega top-N
    rows = sorted(per_doc.items(), key=lambda kv: kv[1]["short_pct"], reverse=True)[:top]

    labels = []
    values = []
    for fname, data in rows:
        title = data["doc_title"] or fname
        # encurta o título para evitar poluir o eixo x
        title_short = textwrap.shorten(title, width=40, placeholder="…")
        labels.append(title_short)
        values.append(data["short_pct"])

    x = np.arange(len(labels))
    fig = plt.figure(figsize=(12, 6))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("% de chunks curtos")
    plt.title(f"Top {len(labels)} livros por taxa de chunks curtos (< min_len)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    png = out_dir / "top_taxa_chunks_curtos.png"
    plt.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[OK] Barra (top curtos) salva em: {png}")

def plot_small_table(per_doc: Dict[str, Dict[str, Any]], out_dir: Path, max_rows: int = 12):
    """
    Desenha uma pequena tabela (matplotlib) com:
      doc_title | format | total | short | short_pct
    Mostra até max_rows (para caber bem em tela), e salva como PNG.
    """
    # ordena por nome de arquivo para uma ordem estável
    items = sorted(per_doc.items(), key=lambda kv: kv[0])[:max_rows]
    table_data = []
    for fname, data in items:
        table_data.append([
            textwrap.shorten(data["doc_title"] or fname, width=50, placeholder="…"),
            data["format"],
            data["total"],
            data["short"],
            f"{data['short_pct']:.1f}%"
        ])

    # se não houver nada, não plota
    if not table_data:
        return

    col_labels = ["Livro (doc_title)", "Tipo", "Chunks", "Curtos", "% Curtos"]

    fig = plt.figure(figsize=(12, 0.6*len(table_data) + 1.5))
    ax = plt.gca()
    ax.axis("off")

    the_table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left"
    )

    # formatação básica para legibilidade
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.3)  # aumenta altura das linhas

    plt.title("Resumo por livro (amostra)", pad=12)
    plt.tight_layout()

    png = out_dir / "tabela_resumo_amostra.png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Tabela (amostra) salva em: {png}")

def save_summary_csv(per_doc: Dict[str, Dict[str, Any]], out_dir: Path):
    """
    Salva CSV com o resumo completo (todas as linhas).
    """
    out_csv = out_dir / "resumo_chunks_por_livro.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "doc_title", "format", "total_chunks", "short_chunks", "short_pct"])
        for fname, data in sorted(per_doc.items(), key=lambda kv: kv[0]):
            w.writerow([
                fname,
                data["doc_title"],
                data["format"],
                data["total"],
                data["short"],
                f"{data['short_pct']:.4f}"
            ])
    print(f"[OK] CSV salvo em: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Avalia chunks JSONL e gera plots/tabela/CSV.")
    ap.add_argument("--jsonl-dir", type=str, required=True,
                    help="Diretório com .jsonl (ex.: books_rag/corpus/jsonl)")
    ap.add_argument("--min-len", type=int, default=100,
                    help="Tamanho mínimo para considerar chunk 'curto' (default: 100)")
    ap.add_argument("--top", type=int, default=15,
                    help="Quantos livros mostrar no gráfico de barras (default: 15)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Diretório de saída para PNG/CSV (default: <jsonl-dir>/../reports)")
    args = ap.parse_args()

    jsonl_dir = Path(args.jsonl_dir).expanduser().resolve()
    if not jsonl_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {jsonl_dir}")

    # Define pasta de relatórios
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (jsonl_dir.parent / "reports")
    ensure_outdir(out_dir)

    # Carrega e computa estatísticas
    stats = load_jsonl_stats(jsonl_dir, min_len=args.min_len)
    per_doc = stats["per_doc"]
    lengths = stats["lengths"]

    # Plots e tabela
    plot_hist_lengths(lengths, out_dir, bins=40)
    plot_short_rate_bar(per_doc, out_dir, top=args.top)
    plot_small_table(per_doc, out_dir, max_rows=12)

    # CSV completo
    save_summary_csv(per_doc, out_dir)

    print("\n✓ Relatórios gerados em:", out_dir)

if __name__ == "__main__":
    main()
