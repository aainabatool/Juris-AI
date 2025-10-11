import os
import fitz
import nltk
import numpy as np
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------
# 1️⃣ Setup
# -----------------------------------------
nltk.download('punkt', quiet=True)
model = SentenceTransformer("all-MiniLM-L6-v2")  # free, small, fast
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="legal_laws")

# -----------------------------------------
# 2️⃣ Extract Text from PDFs
# -----------------------------------------
def extract_text_from_pdfs(root_folder):
    pdf_texts = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(subdir, file)
                try:
                    with fitz.open(pdf_path) as doc:
                        text = "\n".join(page.get_text() for page in doc)
                        pdf_texts.append((file, text))
                        print(f"✅ Extracted text from {file}")
                except Exception as e:
                    print(f"⚠️ Failed to extract {file}: {e}")
    return pdf_texts

# -----------------------------------------
# 3️⃣ Semantic Chunking
# -----------------------------------------
def semantic_chunk_text(text, max_chunk_tokens=400, similarity_threshold=0.45):
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences, convert_to_tensor=False)
    chunks, current_chunk = [], [sentences[0]]
    current_tokens = len(sentences[0].split())

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [embeddings[i - 1]], [embeddings[i]]
        )[0][0]

        token_count = len(sentences[i].split())

        # decide if new sentence should be added to current chunk
        if sim >= similarity_threshold and (current_tokens + token_count) <= max_chunk_tokens:
            current_chunk.append(sentences[i])
            current_tokens += token_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_tokens = token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# -----------------------------------------
# 4️⃣ Process All PDFs → Chunk → Store in Chroma
# -----------------------------------------
def build_knowledge_core(root_dir):
    pdf_texts = extract_text_from_pdfs(root_dir)

    for file_name, text in tqdm(pdf_texts, desc="Processing PDFs"):
        chunks = semantic_chunk_text(text)

        # Create embeddings for each chunk
        embeddings = model.encode(chunks).tolist()

        # Store chunks in Chroma
        for idx, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source_file": file_name,
                    "chunk_number": idx + 1
                }],
                ids=[f"{file_name}_{idx}"]
            )

    print(f"✅ Stored all chunks in ChromaDB ({collection.count()} total).")

# -----------------------------------------
# 5️⃣ Simple Query Example
# -----------------------------------------
def query_knowledge(query_text, n_results=3):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"\n📄 From: {meta['source_file']} | Chunk #{meta['chunk_number']}")
        print(doc[:300], "...\n")

# -----------------------------------------
# 🚀 Main
# -----------------------------------------
if __name__ == "__main__":
    ROOT_DIR = "Data/Legal Laws"  # change if needed

    print("🔍 Building Legal Knowledge Core...")
    build_knowledge_core(ROOT_DIR)

    print("\n💬 Example Query:")
    query_knowledge("What is the punishment for theft?")
