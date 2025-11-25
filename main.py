# main.py
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Use: python main.py chat | query \"pergunta\"")
        return

    mode = sys.argv[1]

    if mode == "chat":
        os.system("streamlit run src/chat_ui.py")
        return

    if mode == "query":
        if len(sys.argv) < 3:
            print("Forneça uma pergunta.")
            return
        from src.query_engine import run_query
        out = run_query(
            index_dir="/home/pdi_4/Documents/Documentos/rag/books_rag/index",
            query=sys.argv[2],
        )
        print(out["structured"])
        return

    print("Modo inválido.")

if __name__ == "__main__":
    main()
