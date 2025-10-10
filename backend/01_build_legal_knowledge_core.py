import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from unstructured.partition.auto import partition
from unstructured.chunking import chunk_elements
from unstructured.documents.elements import element_from_dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct


# --------- CONFIG ---------
PDF_ROOT = "Data/Legal Laws"
ENRICHED_CHUNKS_PATH = "enriched_chunks.json"
DB_PATH = "legal_facts.db"
COLLECTION_NAME = "legal_docs_v1"

# --------- LLM METADATA MODEL ---------
class ChunkMetadata(BaseModel):
    summary: str
    keywords: List[str]
    hypothetical_questions: List[str]
    table_summary: Optional[str] = None


# --------- FUNCTIONS ---------
def find_pdfs(root_path: str) -> List[str]:
    pdfs = []
    for root, _, files in os.walk(root_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return pdfs


def parse_pdf_file(file_path: str) -> List[Dict]:
    try:
        elements = partition(filename=file_path, strategy='fast')
        return [el.to_dict() for el in elements]
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def generate_enrichment_prompt(chunk_text: str, is_table: bool) -> str:
    table_instruction = (
        "This chunk is a TABLE. Your summary should describe key insights and trends."
        if is_table else ""
    )
    return f"""
You are a legal expert. Analyze the following document chunk and generate structured metadata.
{table_instruction}
Chunk Content:
---
{chunk_text}
---
"""


def enrich_chunk(chunk, llm) -> Optional[Dict[str, Any]]:
    is_table = 'text_as_html' in chunk.metadata.to_dict()
    content = chunk.metadata.text_as_html if is_table else chunk.text
    truncated_content = content[:3000]

    prompt = generate_enrichment_prompt(truncated_content, is_table)

    try:
        metadata_obj = llm.invoke(prompt)
        return metadata_obj.dict()
    except Exception as e:
        print(f"  - Error enriching chunk: {e}")
        return None


def create_embedding_text(chunk: Dict) -> str:
    return f"Summary: {chunk['summary']}\nKeywords: {', '.join(chunk['keywords'])}\nContent: {chunk['content'][:1000]}"


# --------- MAIN PIPELINE ---------
if __name__ == "__main__":
    print("Finding PDFs...")
    all_pdfs = find_pdfs(PDF_ROOT)
    print(f"Found {len(all_pdfs)} PDFs")

    # Load checkpoint if exists
    if os.path.exists(ENRICHED_CHUNKS_PATH):
        print("Loading existing enriched chunks...")
        with open(ENRICHED_CHUNKS_PATH, 'r') as f:
            all_enriched_chunks = json.load(f)
    else:
        all_enriched_chunks = []

        # Initialize LLM for metadata enrichment
        enrichment_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        ).with_structured_output(ChunkMetadata)

        for pdf_file in tqdm(all_pdfs, desc="Processing PDFs"):
            parsed_elements = parse_pdf_file(pdf_file)
            if not parsed_elements:
                continue

            elements_for_chunking = [element_from_dict(el) for el in parsed_elements]

            chunks = chunk_elements(
                elements_for_chunking,
                strategy="by_title",
                max_characters=2048,
                combine_text_under_n_chars=256,
                new_after_n_chars=1800
            )

            for chunk in tqdm(chunks, desc="Enriching Chunks", leave=False):
                enriched_data = enrich_chunk(chunk, enrichment_llm)
                if enriched_data:
                    is_table = 'text_as_html' in chunk.metadata.to_dict()
                    content = chunk.metadata.text_as_html if is_table else chunk.text
                    final_chunk = {
                        'source': pdf_file,
                        'content': content,
                        'is_table': is_table,
                        **enriched_data
                    }
                    all_enriched_chunks.append(final_chunk)

        # Save checkpoint
        with open(ENRICHED_CHUNKS_PATH, 'w') as f:
            json.dump(all_enriched_chunks, f)

    print(f"Total enriched chunks: {len(all_enriched_chunks)}")

    # --------- VECTOR STORE ---------
    print("Initializing Qdrant...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(":memory:")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_model.get_embedding_dimension(),
            distance=Distance.COSINE
        )
    )

    points_to_upsert = []
    texts_to_embed = []

    for i, chunk in enumerate(all_enriched_chunks):
        texts_to_embed.append(create_embedding_text(chunk))
        points_to_upsert.append(PointStruct(id=i, payload=chunk))

    print("Generating embeddings...")
    embeddings = list(embedding_model.embed(texts_to_embed, batch_size=32))

    for i, emb in enumerate(embeddings):
        points_to_upsert[i].vector = emb.tolist()

    client.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)
    print(f"Points in collection: {client.get_collection(collection_name=COLLECTION_NAME).points_count}")

    # --------- OPTIONAL RELATIONAL DB (example CSV) ---------
    # csv_path = "legal_summary.csv"
    # if os.path.exists(csv_path):
    #     print("Loading structured data into SQLite...")
    #     df = pd.read_csv(csv_path)
    #     conn = sqlite3.connect(DB_PATH)
    #     df.to_sql("legal_summary", conn, if_exists="replace", index=False)
    #     conn.close()
    #     print("Relational DB ready")
