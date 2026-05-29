import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

def main():
    # Ініціалізація клієнта
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY не знайдено! Перевірте файл .env")
        
    pc = Pinecone(api_key=api_key)

    # 1. Створюємо індекс (якщо не існує)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Створюємо індекс '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            # Використовуємо dotproduct, оскільки вектори вже нормалізовані
            metric="dotproduct", 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Дефолтний регіон для Starter
            )
        )
        # Чекаємо, поки індекс буде готовий приймати дані
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
            
    print(f"Підключено до індексу: {INDEX_NAME}")
    index = pc.Index(INDEX_NAME)

    # 2. Завантажити дані
    print(f"Читаємо дані з {INPUT_PARQUET} та {INPUT_EMBEDDINGS}...")
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    # 3. Підготовка та батчеве завантаження
    print("Починаємо завантаження в Pinecone...")
    vectors_to_upsert = []
    
    for i in tqdm(range(len(df)), desc="Завантаження батчів"):
        row = df.iloc[i]
        
        # Обробка метаданих згідно з вимогами (обмеження довжини)
        metadata = {
            "arxiv_id": str(row["id"]),
            "title": str(row["title"]),
            "abstract": str(row["abstract"])[:500],
            "authors": str(row["authors"])[:200],
            "year": int(row["year"]),
            "category": str(row["category"])
        }
        
        # Формування об'єкта вектора для Pinecone
        vectors_to_upsert.append({
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(), # Pinecone приймає звичайні Python списки
            "metadata": metadata
        })
        
        # 4. Відправка батчу, коли назбирали BATCH_SIZE або дійшли до кінця
        if len(vectors_to_upsert) >= BATCH_SIZE or i == len(df) - 1:
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert = []

    # 5. Перевірка результату
    # Даємо Pinecone кілька секунд на оновлення статистики
    time.sleep(3)
    stats = index.describe_index_stats()
    print("\nДані успішно завантажено!")
    print(f"Загальна кількість векторів в індексі: {stats.total_vector_count}")

if __name__ == "__main__":
    main()