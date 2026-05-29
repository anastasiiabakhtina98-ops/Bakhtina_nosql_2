import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

def chunk_fixed_size(text: str, chunk_size=50, overlap=10) -> list[str]:
    """Розбиває текст на чанки по кількості слів із заданим перекриттям."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def chunk_semantic(text: str, max_words=50) -> list[str]:
    """Розбиває текст на чанки, не розриваючи речення."""
    # Простий поділ на речення (по крапці/знаку питання/оклику + пробіл)
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        # Якщо додавання речення перевищить ліміт і чанк вже не порожній - зберігаємо
        if current_length + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_length = len(words)
        else:
            current_chunk.extend(words)
            current_length += len(words)
            
    if current_chunk: # Зберігаємо залишок
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # 3. Створення індексів
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    for index_name in [INDEX_FIXED, INDEX_SEMANTIC]:
        if index_name not in existing_indexes:
            print(f"Створюємо індекс {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=VECTOR_DIM,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
                
    idx_fixed = pc.Index(INDEX_FIXED)
    idx_semantic = pc.Index(INDEX_SEMANTIC)

    print("Завантажуємо модель...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Читаємо дані...")
    df = pd.read_parquet("data/arxiv_subset.parquet")
    
    # 1. Вибираємо 30 статей із найдовшими анотаціями
    df['abstract_len'] = df['abstract'].apply(lambda x: len(x.split()))
    df_top30 = df.nlargest(30, 'abstract_len').reset_index(drop=True)
    print(f"Вибрано 30 статей. Середня довжина анотації: {df_top30['abstract_len'].mean():.0f} слів.")

    # Списки для підготовки даних
    fixed_chunks_data = []
    semantic_chunks_data = []

    # 2 & 4. Розбиття на чанки та підготовка об'єктів
    print("Розбиваємо тексти на чанки...")
    for doc_idx, row in df_top30.iterrows():
        text = f"{row['title']} [SEP] {row['abstract']}"
        
        # Fixed-size chunking
        f_chunks = chunk_fixed_size(text, chunk_size=50, overlap=10)
        for i, chunk_text in enumerate(f_chunks):
            fixed_chunks_data.append({
                "id": f"{row['id']}_fixed_{i}",
                "text": chunk_text,
                "metadata": {"arxiv_id": row['id'], "title": row['title'], "chunk_index": i, "chunk_text": chunk_text}
            })
            
        # Semantic chunking
        s_chunks = chunk_semantic(text, max_words=50)
        for i, chunk_text in enumerate(s_chunks):
            semantic_chunks_data.append({
                "id": f"{row['id']}_sem_{i}",
                "text": chunk_text,
                "metadata": {"arxiv_id": row['id'], "title": row['title'], "chunk_index": i, "chunk_text": chunk_text}
            })

    def embed_and_upload(chunks_data, index, desc):
        print(f"\nГенерація ембеддингів та завантаження для {desc} ({len(chunks_data)} чанків)...")
        texts = [item["text"] for item in chunks_data]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        
        batch_size = 100
        for i in tqdm(range(0, len(chunks_data), batch_size), desc=f"Upsert {desc}"):
            batch = chunks_data[i:i+batch_size]
            vectors = []
            for j, item in enumerate(batch):
                vectors.append({
                    "id": item["id"],
                    "values": embeddings[i+j].tolist(),
                    "metadata": item["metadata"]
                })
            index.upsert(vectors=vectors)

    # 5. Завантаження чанків
    embed_and_upload(fixed_chunks_data, idx_fixed, "Fixed-size")
    embed_and_upload(semantic_chunks_data, idx_semantic, "Semantic")

    # 6. Пошук по чанках
    time.sleep(3) # Чекаємо оновлення індексів
    query = "performance limitations of neural networks"
    print(f"\n\n---ТЕСТОВИЙ ПОШУК: '{query}' ---")
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    print("\n--- Результати Fixed-size Chunking ---")
    res_fixed = idx_fixed.query(vector=query_vec, top_k=3, include_metadata=True)
    for i, match in enumerate(res_fixed['matches']):
        print(f"{i+1}. [Score: {match['score']:.4f}] {match['metadata']['title']}")
        print(f"   Чанк #{match['metadata']['chunk_index']}: {match['metadata']['chunk_text']}...\n")

    print("--- Результати Semantic Chunking ---")
    res_sem = idx_semantic.query(vector=query_vec, top_k=3, include_metadata=True)
    for i, match in enumerate(res_sem['matches']):
        print(f"{i+1}. [Score: {match['score']:.4f}] {match['metadata']['title']}")
        print(f"   Чанк #{match['metadata']['chunk_index']}: {match['metadata']['chunk_text']}...\n")

if __name__ == "__main__":
    main()