# JW-RAG-Bloodless-Surgery
RAG-based assistant for bloodless surgery using JW Medical Library documents.

# 1. Extract PDF
import os
from pypdf import PdfReader

data_dir = r"D:\JW_RAG_Project\data_raw"

for file in os.listdir(data_dir):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(data_dir, file)
        txt_path = pdf_path.replace(".pdf", ".txt")

        try:
            print("Extracting:", pdf_path)
            reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print("Saved:", txt_path)

        except Exception as e:
            print("Error:", file, "->", e)

# 2. Lightweight cleaning
def clean_text(t):
    t = t.replace("\n", " ")
    t = t.replace("\t", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t.strip()

# 3. Chunk text
from pathlib import Path

def chunk_text(text, max_len=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_len):
        chunk = " ".join(words[i:i+max_len])
        chunks.append(chunk)

    return chunks

txt_files = list(Path(data_dir).glob("*.txt"))

documents = []

for f in txt_files:
    content = clean_text(open(f, encoding="utf-8").read())
    chunks = chunk_text(content)
    for c in chunks:
        documents.append(c)

print("Total chunks:", len(documents))

# 4. Load ChromaDB & SentenceTransformer
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=r"D:\JW_RAG_Project\chroma_db")

collection = client.get_or_create_collection(
    name="jw_bloodless",
    metadata={"hnsw:space": "cosine"}
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Insert chunks into ChromaDB
ids = [str(i) for i in range(len(documents))]

collection.add(
    ids=ids,
    documents=documents
)

print("Inserted into ChromaDB:", len(ids))

# 6. Query RAG
def rag_query(question, k=3):
    results = collection.query(
        query_texts=[question],
        n_results=k
    )
    return results["documents"][0]

query = "What are the ethical considerations in bloodless surgery?"

answers = rag_query(query)

for i, ans in enumerate(answers, 1):
    print(f"\n--- Result {i} ---\n{ans[:500]}\n")

# 7. Streamlit Web App
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=r"D:\JW_RAG_Project\chroma_db")

collection = client.get_or_create_collection("jw_bloodless")
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("JW Bloodless Surgery Assistant")

query = st.text_input("Ask a question:")

if query:
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    st.subheader("Top Retrieved Chunks:")
    for doc in results["documents"][0]:
        st.write(doc[:500] + "...")
