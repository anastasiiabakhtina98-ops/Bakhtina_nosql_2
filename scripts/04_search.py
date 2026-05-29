import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    
    print("Завантаження моделі...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Завантаження локальних даних...")
    df = pd.read_parquet("data/arxiv_subset.parquet")
    local_embeddings = np.load("embeddings/embeddings.npy")

    # 2. Функція кодування запиту
    def encode_query(text: str) -> np.ndarray:
        # Обов'язково нормалізуємо запит, оскільки документи в базі нормалізовані
        embedding = model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu().numpy()
        return np.asarray(embedding)

    def print_results(matches, title="Результати пошуку:"):
        print(f"\n--- {title} ---")
        for i, match in enumerate(matches):
            m = match['metadata']
            abstract_snippet = m.get('abstract', '')[:120].replace('\n', ' ') + "..."
            print(f"{i+1}. [Score: {match['score']:.4f}] {m.get('title')}")
            print(f"   Рік: {m.get('year')} | Категорія: {m.get('category')}")
            print(f"   Abstract: {abstract_snippet}\n")

    # 3. Чистий семантичний пошук
    query_text = "teaching machines to recognize objects in pictures"
    query_vector = encode_query(query_text)
    
    res_pure = index.query(
        vector=query_vector.tolist(), 
        top_k=TOP_K, 
        include_metadata=True
    )
    print_results(res_pure['matches'], f"Чистий семантичний пошук: '{query_text}'")

    # 4. Пошук з фільтрацією
    query_filter = "reinforcement learning in robotics"
    filter_vector = encode_query(query_filter).tolist()

    # Приклад A: останні 5 років (>= 2019) і категорія cs.LG  
    res_filter_a = index.query(
        vector=filter_vector,
        top_k=TOP_K,
        include_metadata=True,
        filter={
            "category": {"$eq": "cs.LG"},
            "year": {"$gte": 2019}
        }
    )
    print_results(res_filter_a['matches'], "Фільтр А (>=2019, cs.LG)")

    # Приклад B: старі статті (< 2015), будь-яка категорія
    res_filter_b = index.query(
        vector=filter_vector,
        top_k=TOP_K,
        include_metadata=True,
        filter={
            "year": {"$lt": 2015}
        }
    )
    print_results(res_filter_b['matches'], "Фільтр B (< 2015, будь-яка категорія)")

    # 5. Порівняння метрик на локальних ембеддингах
    print("\n---Порівняння локальних метрик схожості---")
    
    # Cosine Similarity: (A @ B) / (||A|| * ||B||)
    doc_norms = np.linalg.norm(local_embeddings, axis=1)
    query_norm = np.linalg.norm(query_vector)
    cosine_scores = (local_embeddings @ query_vector) / (doc_norms * query_norm)
    
    # Dot Product: A @ B
    dot_scores = local_embeddings @ query_vector
    
    # L2 Distance: ||A - B|| (чим менше, тим краще)
    l2_scores = np.linalg.norm(local_embeddings - query_vector, axis=1)

    top_cos_idx = np.argsort(cosine_scores)[::-1][:TOP_K]
    top_dot_idx = np.argsort(dot_scores)[::-1][:TOP_K]
    top_l2_idx = np.argsort(l2_scores)[:TOP_K] # Сортування за зростанням

    print(f"Top 5 Cosine (Індекси): {top_cos_idx}")
    print(f"Top 5 Dot Product (Індекси): {top_dot_idx}")
    print(f"Top 5 L2 Distance (Індекси): {top_l2_idx}")
    

if __name__ == "__main__":
    main()