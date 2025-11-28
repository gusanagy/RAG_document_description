# 📚 RAG-Books: Retrieval-Augmented Generation Over Technical Book Libraries

This repository implements a **full RAG (Retrieval-Augmented Generation)** pipeline specialized for **large collections of technical books** (PDF/EPUB/JSONL chunks).
It supports:

* Automated **book extraction** and **FAISS index building**
* A **local retrieval engine** powered by Sentence-Transformers
* **LLM-augmented answers** (GPU-friendly, quantization-ready)
* A clean **Streamlit chat interface**
* A full **evaluation pipeline** with 60 gold-standard questions (40 PT-BR + 20 EN)
* A unified **CLI (`main.py`)** that orchestrates the entire workflow

The goal is to provide an offline, reproducible framework for querying large e-book libraries, validating retrieval quality, and experimenting with RAG architectures.

---

## 🚀 Project Structure

```
RAG/
│
├── books_rag/
│   ├── corpus/
│   │   └── jsonl/
│   ├── index/
│   │   ├── faiss.index
│   │   └── meta.json
│   ├── raw/
│   │   ├── epub/
│   │   └── pdf/
│   ├── working/
│   │   ├── manifests/
│   │   └── reports/
│   └── Books-20251007T165548Z-1-001/
│       └── Books/
│
├── src/
│   ├── __pycache__/
│   ├── chat_ui.py
│   ├── check_chunks.py
│   ├── ext_metric_param.py
│   ├── extract_book_data.py
│   ├── query_engine.py
│   ├──rag_metrics.py
│   ├── metrics/
│   │   ├── eval_gold.json
│   │   └── eval_predictions.json
|   └── old/
|       ├── app_chat_runner.py
|       ├── query_faiss_index_3.py
|       ├── query_faiss_index.py
|       └── build_faiss_index.py
│      
├── .gitignore
├── Books-20251007T165548Z-1-001.zip
├── experiments.ipynb
├── main.py
├── rag_book.yaml
└── README.md
```
## 📚Project Papeline

```
                          ┌───────────────────────────────┐
                          │         Book Library          │
                          │   (PDF / EPUB / JSONL chunks) │
                          └───────────────────────────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────────────┐
                     │       1. Book Extraction Module         │
                     │   - PDF/EPUB text extraction            │
                     │   - Cleaning, normalization             │
                     │   - Chunking (windowed / semantic)      │
                     └─────────────────────────────────────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────────────┐
                     │         2. Embedding Generator          │
                     │  - SentenceTransformers encoder         │
                     │  - Vector normalization                 │
                     └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │            3. FAISS Index                │
                    │   - Stores dense embeddings              │
                    │   - meta.json keeps metadata             │
                    └──────────────────────────────────────────┘
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────────────────────┐
          │                    run_query()  — Core RAG Engine                  │
          │                                                                    │
          │   4. Retrieval:                                                    │
          │      - FAISS top-K search                                          │
          │      - Optional cross-encoder re-ranking                           │
          │                                                                    │
          │   5. Structured Answer Generator                                   │
          │      - Groups hits by doc                                          │
          │      - Extracts key sentences                                      │
          │      - Produces citation-anchored summary                          │
          │                                                                    │
          │   6. LLM Answer (optional)                                         │
          │      - Context assembly (chunks + citations)                       │
          │      - Local LLM inference (GPU→CPU fallback)                      │
          │      - Quantization (4-bit, 8-bit)                                 │
          └────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                   ┌──────────────────────────────────────────────┐
                   │               User Interfaces                │
                   │                                              │
                   │  A) CLI: main.py query "…"                   │
                   │  B) Streamlit Chat UI                        │
                   │  C) Experiments Notebook (experiments.ipynb) │
                   └──────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌──────────────────────────────────────────────────────────────┐
            │                    Evaluation Pipeline                       │
            │                                                              │
            │  - eval_gold.json (40 PT + 20 EN questions)                  │
            │  - build_eval_predictions.py                                 │
            │  - rag_metrics.py → recall@k, MRR@k, token-F1                │
            └──────────────────────────────────────────────────────────────┘
```
---

# 🔧 Installation

This project uses **Conda** for environment management.

```bash
conda env create -f rag_book.yaml
conda activate rag_book
```

Ensure that you have:

* CUDA-capable GPU (recommended)
* PyTorch installed with CUDA support
* bitsandbytes (optional, but useful for 4-bit LLM loading)

---

# 🧠 Core Workflow (Using `main.py`)

`main.py` acts as the **central command-line interface** for the entire RAG system.
It supports multiple subcommands:

---

## 1. **Extract** books and build FAISS index

```bash
python main.py extract \
    --books-dir /path/to/my_books \
    --index-dir /path/to/index
```

This step performs:

* PDF/EPUB text extraction
* Chunk creation
* Embedding generation
* FAISS index construction
* Metadata generation (`meta.json`)

---

## 2. **Inspect Index**

Quick overview of chunk distribution and book coverage:

```bash
python main.py check-index --index-dir /path/to/index
```

Shows:

* total number of chunks
* number of documents
* top-N documents by chunk count

---

## 3. **Chat with your Library**

Launch the Streamlit chat interface:

```bash
python main.py chat
```

You will get a local page with:

* Chat history
* Structured (non-LLM) retrieval answer
* LLM-augmented answer
* Retrieved chunk viewer (metadata + preview)

The UI reads all parameters from `query_engine.py`.

---

## 4. **Run a Single Query (Terminal Mode)**

```bash
python main.py query \
    --question "Explain orthogonality in software design."
```

Options:

* `--use-llm`
* `--rerank`
* `--k <top-k>`
* `--index-dir <path>`

Example:

```bash
python main.py query --question "What is backpropagation?" --use-llm
```

---

## 5. **Full Evaluation Pipeline**

The project contains **60 gold-standard questions**:

* **40 in Portuguese**
* **20 in English**
* Spread across multiple books (Pragmatic Programmer, Deep Learning, Szeliski, Redis, OOP, Domain Adaptation, etc.)

### Run the full validation:

```bash
python main.py validate
```

This performs:

1. **Prediction generation**
   (`build_eval_predictions.py`)
2. **Metric computation**
   (`rag_metrics.py`)

You may skip metrics:

```bash
python main.py validate --skip-metrics
```

You can adjust `k`:

```bash
python main.py validate --k 5
```

---

# 📊 Metrics Used

Under `src/metrics/`, the following metrics are implemented:

### **Retrieval**

* **Recall@k**
* **MRR@k** (Mean Reciprocal Rank)

### **Answer Quality**

* **Token-Level F1** (reference vs generated answer)

These metrics follow the format of:

```json
{
  "query": "...",
  "relevant_doc_ids": ["..."],
  "retrieved_doc_ids": ["...", "..."],
  "reference_answer": "...",
  "generated_answer": "..."
}
```

Ideal for benchmarking retrieval and generation consistency.

---

# 🧪 Jupyter Notebook: `experiments.ipynb`

This notebook allows you to:

* Load and inspect embeddings
* Visualize chunk distribution
* Debug retrieval failures
* Test alternative encoding models
* Benchmark GPU vs CPU inference
* Inspect chunk metadata interactively
* Experiment with prompt formats for LLM answers

It is the recommended place for:

* Research experiments
* Analysis
* Manual validation
* Exploratory metrics

---

# 📚 Dataset Used in Validation

The evaluation dataset includes questions from books such as:

* *The Pragmatic Programmer*
* *Pragmatic Thinking & Learning*
* *Deep Learning — Goodfellow, Bengio & Courville*
* *Computer Vision: Models, Learning, and Inference* (Prince)
* *Build Your Own Redis*
* *Szeliski — Computer Vision: Algorithms and Applications*
* *Probabilistic Approaches to Robotic Perception*
* *Domain Adaptation in Computer Vision*
* *Lambda Calculus and Combinators*
* *Java Programming for Beginners*
* *Functional/OOP/Concurrent Programming*
* *PyTorch Computer Vision Cookbook*

This ensures coverage across:

* Programming
* Computer Vision
* Machine Learning
* Theory of Computation
* Functional Programming
* Robotics
* Algorithms
* Systems Design

---

# 🏗️ Architecture Summary

* **Retrieval:**
  FAISS (flat or HNSW) + Sentence-Transformers

* **Re-ranking (optional):**
  Cross-Encoder / MS-MARCO (MiniLM)

* **Generation:**
  Local LLM (Qwen 2.5 3B)

  * fp16 / fp32 fallback
  * 4-bit quantization
  * Automatic GPU/CPU switching
  * VRAM clearing routines

* **Chunk Metadata:**
  Stored in `meta.json`
  Includes page ranges, section names, doc titles, ranking, etc.

---

# 🧩 Future Improvements (Roadmap)

* Support for **multi-GPU inference**
* Advanced **chunk graph routing** (Haystack style)
* Hybrid embedding (ColBERT, SPECTER2, E5-Mistral)
* Automated chunk evaluation for failure analysis
* API server wrapper with FastAPI

---

# 📝 Author

**Gustavo Almeida**
Federal University of Rio Grande (FURG)
Computer Vision, Robotics & Deep Learning Researcher

If you use or cite this work, please consider referencing the repository.

📚 BibTeX Citation

You can add this to any publication, report, or thesis referencing your RAG-Books toolkit:
```
@software{almeida2025ragbooks,
  author       = {Gustavo Almeida},
  title        = {RAG-Books: A Retrieval-Augmented Generation Toolkit for Large-Scale Technical Book Libraries},
  year         = {2025},
  url          = {https://github.com/your-repo-here},
  version      = {1.0},
  abstract     = {RAG-Books is a modular framework for offline retrieval-augmented generation over large collections of technical books. It integrates automatic chunk extraction, FAISS-based retrieval, structured summarization, LLM-based question answering, GPU-aware inference, and a built-in evaluation suite with multilingual gold-standard questions.}
}
```