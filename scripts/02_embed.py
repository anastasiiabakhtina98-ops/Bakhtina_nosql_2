import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Налаштування шляхів
INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"

def main():
    # 7. Переконатися, що директорія embeddings існує
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Завантажити датасет
    print(f"1. Завантаження даних з {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    # 2. Підготувати тексти для кодування
    print("2. Підготовка текстів (Title + [SEP] + Abstract)...")
    # Об'єднуємо поля за вимогами моделі specter2
    texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

    # 3. Завантажити модель
    print(f"3. Завантаження моделі {MODEL_NAME} з HuggingFace...")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Згенерувати ембеддинги з нормалізацією та батчами
    print("4. Генерація ембеддингів...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # 5. Вивести статистику в консоль
    print("\n--- Статистика ---")
    print(f"Загальна кількість оброблених текстів: {len(texts)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    
    # Обчислюємо норму першого вектора (має бути ~1.0 через normalize_embeddings=True)
    first_emb_norm = np.linalg.norm(embeddings[0])
    print(f"Норма першого ембеддингу: {first_emb_norm:.4f}")

    # 6. Зберегти отримані ембеддинги
    print(f"\n6. Збереження матриці векторів у {OUTPUT_FILE}...")
    np.save(OUTPUT_FILE, embeddings)
    print("Готово!")

if __name__ == "__main__":
    main()