import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    
    print("Завантажуємо модель...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Завантажуємо локальні дані та будуємо BM25 індекс...")
    df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)
    
    # 1. Побудова локального BM25-індексу
    corpus = (df['title'] + " " + df['abstract']).tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Реалізація функцій пошуку
    def search_bm25(query, top_n=TOP_K):
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'id': f"paper_{idx}", # Формат ID 
                'rank': rank + 1,
                'score': scores[idx],
                'title': df.iloc[idx]['title']
            })
        return results

    def search_vector(query, top_n=TOP_K):
        query_vec = model.encode(query, normalize_embeddings=True).tolist()
        res = index.query(vector=query_vec, top_k=top_n, include_metadata=True)
        
        results = []
        for rank, match in enumerate(res['matches']):
            results.append({
                'id': match['id'],
                'rank': rank + 1,
                'score': match['score'],
                'title': match['metadata']['title']
            })
        return results

    def search_hybrid(query, k=60, top_n=5):
        bm25_res = search_bm25(query, top_n=TOP_K)
        vector_res = search_vector(query, top_n=TOP_K)
        
        rrf_scores = {}
        
        def add_to_rrf(results):
            for res in results:
                doc_id = res['id']
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {'score': 0.0, 'title': res['title']}
                # Формула RRF: 1 / (k + rank)
                rrf_scores[doc_id]['score'] += 1.0 / (k + res['rank'])
                
        add_to_rrf(bm25_res)
        add_to_rrf(vector_res)
        
        # Сортування за RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return sorted_docs[:top_n]

    def print_results(method_name, results):
        print(f"\n--- Топ-5 {method_name} ---")
        for i, res in enumerate(results[:5]):
            if isinstance(res, tuple): # Для гібридного пошуку (id, {score, title})
                print(f"{i+1}. [RRF Score: {res[1]['score']:.4f}] {res[1]['title']}")
            else: # Для BM25 та Vector
                print(f"{i+1}. [Score: {res['score']:.4f}] {res['title']}")

    # 5. Тестові запити
    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text"
    ]

    for query in queries:
        print(f"\n\n----------------------------------------------------")
        print(f"ЗАПИТ: '{query}'")
        print(f"----------------------------------------------------")
        
        bm25_results = search_bm25(query)
        print_results("BM25 (Ключові слова)", bm25_results)
        
        vector_results = search_vector(query)
        print_results("Векторний (Семантика Pinecone)", vector_results)
        
        hybrid_results = search_hybrid(query)
        print_results("Гібридний (BM25 + Vector через RRF)", hybrid_results)

if __name__ == "__main__":
    main()