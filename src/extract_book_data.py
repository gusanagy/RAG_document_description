#!/usr/bin/env python3
# Extrai texto de PDF/EPUB, faz chunking com overlap e grava JSONL por documento.
# Estrutura esperada do projeto:
# books_rag/
#   ├─ corpus/jsonl/
#   ├─ raw/epub/  raw/pdf/
#   ├─ working/manifests/
#   └─ src/extract_book_data.py

import os, re, json, csv, argparse, sys
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
from tqdm import tqdm

# ---- PDF: pdfminer.six ----
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# ---- EPUB: ebooklib + bs4 ----
from ebooklib import epub
from bs4 import BeautifulSoup

# ==========================
# Helpers de caminho/raiz
# ==========================
USER_HOME = Path.home()

def detect_repo_root() -> Path:
    """
    Assume que este arquivo está em books_rag/src/ e define a raiz como .. (books_rag).
    Se o layout estiver diferente, tenta subir diretórios até encontrar 'raw' e 'corpus'.
    """
    here = Path(__file__).resolve()
    cand = here.parents[1]  # .../books_rag
    if (cand / "raw").exists() and (cand / "corpus").exists():
        return cand
    p = here
    for _ in range(5):
        p = p.parent
        if (p / "raw").exists() and (p / "corpus").exists():
            return p
    return here.parent  # fallback

def ensure_writable_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK):
        print(f"[ERRO] Sem permissão de escrita em: {path}")
        sys.exit(1)
    return path

def rel_or_abs(root: Path, p: str) -> Path:
    """
    Converte caminho relativo (em relação ao --root) ou absoluto.
    Garante que o resultado fique DENTRO da HOME do usuário (evita /home/books_rag).
    """
    pth = Path(p).expanduser()
    out = (root / pth).resolve() if not pth.is_absolute() else pth.resolve()
    try:
        out.relative_to(USER_HOME)
    except ValueError:
        print(f"[ERRO] Caminho fora da HOME: {out}\n"
              f"Use um diretório dentro de {USER_HOME} ou passe um --root correto.")
        sys.exit(1)
    return out

# ==========================
# Utils de texto
# ==========================
def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s\-\.]+", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    return s.lower()[:120] if s else "untitled"

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00ad", "")  # soft hyphen
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    lines = [ln.strip() for ln in t.splitlines()]
    rebuilt, buf = [], []
    for ln in lines:
        if not ln:
            if buf:
                rebuilt.append(" ".join(buf).strip()); buf = []
        else:
            buf.append(ln)
    if buf:
        rebuilt.append(" ".join(buf).strip())
    return "\n\n".join([re.sub(r"\s+", " ", p).strip() for p in rebuilt if p.strip()])

def guess_title_from_path(p: Path) -> str:
    return p.stem.replace("_", " ").replace("-", " ").strip()

# ==========================
# Leitura PDF
# ==========================
def iter_pdf_pages(pdf_path: Path) -> Iterator[Tuple[int, str]]:
    """Extrai texto página a página (pageno inicia em 1)."""
    try:
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import TextConverter
        from io import StringIO
        rsrcmgr = PDFResourceManager()
        with open(pdf_path, "rb") as fh:
            for i, page in enumerate(PDFPage.get_pages(fh), start=1):
                retstr = StringIO()
                device = TextConverter(rsrcmgr, retstr, laparams=None)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                interpreter.process_page(page)
                text = retstr.getvalue()
                device.close(); retstr.close()
                yield i, text
    except PDFSyntaxError:
        txt = extract_text(str(pdf_path)) or ""
        yield 1, txt

# ==========================
# Leitura EPUB
# ==========================
def iter_epub_chapters(epub_path: Path) -> Iterator[Tuple[str, str]]:
    book = epub.read_epub(str(epub_path))
    spine = [i[0] for i in book.spine]  # ordem de leitura
    items_doc = {it.get_id(): it for it in book.get_items() if it.get_type() == epub.ITEM_DOCUMENT}
    for idx, sid in enumerate(spine, start=1):
        if sid in items_doc:
            it = items_doc[sid]
            soup = BeautifulSoup(it.get_content(), "lxml")
            h = soup.find(re.compile("^h[1-4]$"))
            title = h.get_text(" ", strip=True) if h else f"chapter-{idx}"
            for br in soup.find_all("br"): br.replace_with("\n")
            for pre in soup.find_all("pre"):
                code_text = pre.get_text("\n", strip=False)
                pre.replace_with(f"\n```code\n{code_text}\n```\n")
            for tag in soup(["script", "style"]): tag.decompose()
            text = soup.get_text("\n", strip=False)
            yield title, text

# ==========================
# Chunking
# ==========================
def sentence_boundaries(text: str) -> List[int]:
    return [m.end() for m in re.finditer(r"([\.!\?;])\s+", text)]

def chunk_text(text: str, max_chars: int = 5000, overlap: int = 700, min_chars: int = 1200) -> List[str]:
    text = clean_text(text)
    code_blocks = []
    def _protect(m):
        token = f"@@CODEBLOCK_{len(code_blocks)}@@"
        code_blocks.append(m.group(0))
        return token
    protected = re.sub(r"```.*?```", _protect, text, flags=re.DOTALL)

    chunks, i, n = [], 0, len(protected)
    sent_cuts = set(sentence_boundaries(protected))
    while i < n:
        j = min(i + max_chars, n)
        candidates = [c for c in sent_cuts if i + int(min_chars*0.6) <= c <= j]
        k = max(candidates) if candidates else j
        chunk = protected[i:k].strip()
        if chunk: chunks.append(chunk)
        if k >= n: break
        i = max(k - overlap, i + 1)

    restored = []
    for ch in chunks:
        ch = re.sub(r"@@CODEBLOCK_(\d+)@@", lambda m: code_blocks[int(m.group(1))], ch)
        ch = re.sub(r"\n{3,}", "\n\n", ch).strip()
        if len(ch) >= min_chars or (len(ch) > 0 and not restored):
            restored.append(ch)
    return restored

# ==========================
# Processadores
# ==========================
def process_pdf(pdf_path: Path, out_dir: Path, max_chars: int, overlap: int, min_chars: int) -> Dict:
    doc_title = guess_title_from_path(pdf_path)
    doc_id = slugify(f"{doc_title}-{pdf_path.stem}")
    jsonl_path = out_dir / f"{doc_id}.jsonl"

    page_buffers = []
    for pageno, raw in iter_pdf_pages(pdf_path):
        txt = clean_text(raw)
        if txt: page_buffers.append((pageno, txt))

    total_chunks = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for pageno, ptxt in tqdm(page_buffers, desc=f"Chunking PDF {pdf_path.name}"):
            parts = chunk_text(ptxt, max_chars=max_chars, overlap=overlap, min_chars=min_chars)
            for idx, part in enumerate(parts, start=1):
                total_chunks += 1
                rec = {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "format": "pdf",
                    "chunk_id": f"{doc_id}_p{pageno:05d}_{idx:02d}",
                    "part": f"page-{pageno}",
                    "section": None,
                    "page_start": pageno,
                    "page_end": pageno,
                    "text": part
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "doc_id": doc_id, "title": doc_title, "format": "pdf",
        "path_raw": str(pdf_path), "jsonl": str(jsonl_path),
        "pages": len(page_buffers), "chunks": total_chunks
    }

def process_epub(epub_path: Path, out_dir: Path, max_chars: int, overlap: int, min_chars: int) -> Dict:
    try:
        book = epub.read_epub(str(epub_path))
        tmeta = book.get_metadata('DC', 'title')
        doc_title = tmeta[0][0] if tmeta else guess_title_from_path(epub_path)
    except Exception:
        doc_title = guess_title_from_path(epub_path)

    doc_id = slugify(f"{doc_title}-{epub_path.stem}")
    jsonl_path = out_dir / f"{doc_id}.jsonl"

    total_chunks = 0
    chapter_count = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for chap_title, chap_text in tqdm(iter_epub_chapters(epub_path), desc=f"Chunking EPUB {epub_path.name}"):
            chapter_count += 1
            parts = chunk_text(chap_text, max_chars=max_chars, overlap=overlap, min_chars=min_chars)
            for idx, part in enumerate(parts, start=1):
                total_chunks += 1
                rec = {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "format": "epub",
                    "chunk_id": f"{doc_id}_c{chapter_count:04d}_{idx:02d}",
                    "part": f"chapter-{chapter_count}",
                    "section": chap_title,
                    "page_start": None,
                    "page_end": None,
                    "text": part
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "doc_id": doc_id, "title": doc_title, "format": "epub",
        "path_raw": str(epub_path), "jsonl": str(jsonl_path),
        "chapters": chapter_count, "chunks": total_chunks
    }

# ==========================
# Main
# ==========================
def main():
    repo_root_default = detect_repo_root()
    ap = argparse.ArgumentParser(description="Extrai texto e cria chunks JSONL de PDFs/EPUBs para RAG.")
    ap.add_argument("--root", type=str, default=str(repo_root_default),
                    help="Raiz do projeto (onde existem raw/, corpus/, working/).")
    ap.add_argument("--pdf_dir", type=str, default="raw/pdf", help="Subpasta de PDFs (relativa a --root ou absoluta)")
    ap.add_argument("--epub_dir", type=str, default="raw/epub", help="Subpasta de EPUBs (relativa a --root ou absoluta)")
    ap.add_argument("--out_jsonl", type=str, default="corpus/jsonl",
                    help="Saída dos JSONL por documento (relativa a --root ou absoluta)")
    ap.add_argument("--manifest_csv", type=str, default="working/manifests/generated_manifest.csv",
                    help="CSV de inventário (relativo a --root ou absoluto)")
    ap.add_argument("--max_chars", type=int, default=5000, help="Tamanho alvo do chunk (caracteres)")
    ap.add_argument("--overlap", type=int, default=700, help="Overlap entre chunks (caracteres)")
    ap.add_argument("--min_chars", type=int, default=1200, help="Tamanho mínimo aceitável de chunk")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    pdf_dir = rel_or_abs(root, args.pdf_dir)
    epub_dir = rel_or_abs(root, args.epub_dir)
    out_dir = rel_or_abs(root, args.out_jsonl)
    manifest_csv = rel_or_abs(root, args.manifest_csv)

    ensure_writable_dir(out_dir)
    ensure_writable_dir(manifest_csv.parent)

    # Listagem informativa
    pdfs = sorted(p for p in pdf_dir.glob("**/*.pdf") if p.is_file()) if pdf_dir.exists() else []
    epubs = sorted(p for p in epub_dir.glob("**/*.epub") if p.is_file()) if epub_dir.exists() else []
    print(f"Encontrados: {len(pdfs)} PDF(s) em {pdf_dir}")
    print(f"Encontrados: {len(epubs)} EPUB(s) em {epub_dir}")

    rows = []
    # PDFs
    for p in pdfs:
        info = process_pdf(p, out_dir, args.max_chars, args.overlap, args.min_chars)
        info["status"] = "chunked"
        rows.append(info)
    # EPUBs
    for p in epubs:
        info = process_epub(p, out_dir, args.max_chars, args.overlap, args.min_chars)
        info["status"] = "chunked"
        rows.append(info)

    # Manifest
    if rows:
        fieldnames = sorted(set(k for r in rows for k in r.keys()).union({"status"}))
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"\n✓ Concluído. JSONL em: {out_dir}")
    if rows:
        print(f"  Manifest: {manifest_csv}")
    else:
        print("  Nenhum arquivo encontrado em {pdf_dir} ou {epub_dir}.")

if __name__ == "__main__":
    main()
