# import os
# import fitz
# import nltk
# import numpy as np
# import chromadb
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------
# # 1️⃣ Setup
# # -----------------------------------------
# nltk.download('punkt', quiet=True)
# model = SentenceTransformer("all-MiniLM-L6-v2")  # free, small, fast
# chroma_client = chromadb.Client()
# collection = chroma_client.get_or_create_collection(name="legal_laws")

# # -----------------------------------------
# # 2️⃣ Extract Text from PDFs
# # -----------------------------------------
# def extract_text_from_pdfs(root_folder):
#     pdf_texts = []
#     for subdir, _, files in os.walk(root_folder):
#         for file in files:
#             if file.lower().endswith(".pdf"):
#                 pdf_path = os.path.join(subdir, file)
#                 try:
#                     with fitz.open(pdf_path) as doc:
#                         text = "\n".join(page.get_text() for page in doc)
#                         pdf_texts.append((file, text))
#                         print(f"✅ Extracted text from {file}")
#                 except Exception as e:
#                     print(f"⚠️ Failed to extract {file}: {e}")
#     return pdf_texts

# # -----------------------------------------
# # 3️⃣ Semantic Chunking
# # -----------------------------------------
# def semantic_chunk_text(text, max_chunk_tokens=400, similarity_threshold=0.45):
#     from nltk.tokenize import sent_tokenize

#     sentences = sent_tokenize(text)
#     if not sentences:
#         return []

#     embeddings = model.encode(sentences, convert_to_tensor=False)
#     chunks, current_chunk = [], [sentences[0]]
#     current_tokens = len(sentences[0].split())

#     for i in range(1, len(sentences)):
#         sim = cosine_similarity(
#             [embeddings[i - 1]], [embeddings[i]]
#         )[0][0]

#         token_count = len(sentences[i].split())

#         # decide if new sentence should be added to current chunk
#         if sim >= similarity_threshold and (current_tokens + token_count) <= max_chunk_tokens:
#             current_chunk.append(sentences[i])
#             current_tokens += token_count
#         else:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentences[i]]
#             current_tokens = token_count

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# # -----------------------------------------
# # 4️⃣ Process All PDFs → Chunk → Store in Chroma
# # -----------------------------------------
# def build_knowledge_core(root_dir):
#     pdf_texts = extract_text_from_pdfs(root_dir)

#     for file_name, text in tqdm(pdf_texts, desc="Processing PDFs"):
#         chunks = semantic_chunk_text(text)

#         # Create embeddings for each chunk
#         embeddings = model.encode(chunks).tolist()

#         # Store chunks in Chroma
#         for idx, chunk in enumerate(chunks):
#             collection.add(
#                 documents=[chunk],
#                 metadatas=[{
#                     "source_file": file_name,
#                     "chunk_number": idx + 1
#                 }],
#                 ids=[f"{file_name}_{idx}"]
#             )

#     print(f"✅ Stored all chunks in ChromaDB ({collection.count()} total).")

# # -----------------------------------------
# # 5️⃣ Simple Query Example
# # -----------------------------------------
# def query_knowledge(query_text, n_results=3):
#     results = collection.query(query_texts=[query_text], n_results=n_results)
#     for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
#         print(f"\n📄 From: {meta['source_file']} | Chunk #{meta['chunk_number']}")
#         print(doc[:300], "...\n")

# # -----------------------------------------
# # 🚀 Main
# # -----------------------------------------
# if __name__ == "__main__":
#     ROOT_DIR = "Data/Legal Laws"  # change if needed

#     print("🔍 Building Legal Knowledge Core...")
#     build_knowledge_core(ROOT_DIR)

#     print("\n💬 Example Query:")
#     query_knowledge("What is the punishment for theft?")


"""
01_build_legal_knowledge_core_langchain.py

Description:
------------
RAG pipeline for legal PDFs using LangChain built-in chunking.
Recursively loads PDFs from Data/Legal Laws, extracts text, chunks, embeds,
and stores in a persistent ChromaDB.

No external chunkers (unstructured); uses LangChain-native tools.
"""

import os
from pathlib import Path
from tqdm import tqdm

# LangChain core tools
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings + Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------------------
# 1️⃣ CONFIGURATION
# ---------------------------------------------------------------------

DATA_DIR = Path("Data/Legal Laws")   # Folder with PDFs (may contain subfolders)
CHROMA_DB_DIR = Path("data/legal_ai_db")  # Persistent vector store location
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # good tradeoff

# ---------------------------------------------------------------------
# 2️⃣ FIND ALL PDFs RECURSIVELY
# ---------------------------------------------------------------------

def get_all_pdfs(root_dir):
    pdf_files = []
    for path, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(path, file))
    return pdf_files

pdf_paths = get_all_pdfs(DATA_DIR)
print(f"Found {len(pdf_paths)} PDFs in {DATA_DIR}")

# ---------------------------------------------------------------------
# 3️⃣ INITIALIZE EMBEDDINGS + VECTOR STORE
# ---------------------------------------------------------------------

embedding_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

vectordb = Chroma(
    collection_name="legal_laws_langchain",
    embedding_function=embedding_fn,
    persist_directory=str(CHROMA_DB_DIR)
)

# ---------------------------------------------------------------------
# 4️⃣ DEFINE CHUNKING STRATEGY (LangChain built-in)
# ---------------------------------------------------------------------

# Recursive splitter automatically finds logical breakpoints (e.g., sections, paragraphs)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # fits comfortably within LLM context
    chunk_overlap=200,    # ensures continuity
    length_function=len,  # count by characters
    separators=["\n\n", "\n", ".", " ", ""]
)

# ---------------------------------------------------------------------
# 5️⃣ PARSE, CHUNK, AND STORE ALL PDFs
# ---------------------------------------------------------------------

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    # Split each page into smaller chunks
    docs = text_splitter.split_documents(pages)
    for d in docs:
        d.metadata["source"] = str(pdf_path)
    return docs

all_docs = []
for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
    try:
        docs = process_pdf(pdf_path)
        all_docs.extend(docs)
    except Exception as e:
        print(f"⚠️ Error processing {pdf_path}: {e}")

print(f"Total chunks to embed: {len(all_docs)}")

# ---------------------------------------------------------------------
# 6️⃣ ADD TO CHROMA DB
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 6️⃣ ADD TO CHROMA DB (BATCHED)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 6️⃣ ADD TO CHROMA DB (BATCHED INSERT, AUTO-PERSIST)
# ---------------------------------------------------------------------

if all_docs:
    BATCH_SIZE = 5000  # safe limit to avoid InternalError
    total_batches = (len(all_docs) // BATCH_SIZE) + 1

    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]
        print(f"📦 Inserting batch {i // BATCH_SIZE + 1} / {total_batches} ...")
        vectordb.add_documents(batch)

    print("✅ Successfully embedded and saved all legal PDFs to ChromaDB.")
    print(f"📂 Chroma DB path: {CHROMA_DB_DIR}")

else:
    print("❌ No documents processed. Check your Data/Legal Laws folder.")



# ---------------------------------------------------------------------
# 7️⃣ VERIFY
# ---------------------------------------------------------------------

print(f"ChromaDB path: {CHROMA_DB_DIR}")
print(f"Total stored vectors: {vectordb._collection.count()}")
