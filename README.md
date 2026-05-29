### 1.2. Вибір інструментів

**1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?**

* **Pinecone** є повністю керованим хмарним рішенням (SaaS) із закритим вихідним кодом. Він не підтримує локальне розгортання (self-hosting). Головна перевага - нульові витрати часу на адміністрування та автоматичне масштабування (serverless architecture), що забезпечує високу продуктивність при роботі з великими обсягами даних "з коробки".
* **Chroma** — це open-source (ліцензія Apache 2.0) векторна база даних, яка підходить для швидкого прототипування та локального розгортання, проте наразі менш оптимізована для величезних production-навантажень порівняно з конкурентами на ринку.
* **Qdrant** — open-source (ліцензія Apache 2.0) рішення, написане на Rust. Пропонує як хмарне (кероване), так і локальне (self-hosted) розгортання. Вирізняється високою продуктивністю та механізмом фільтрації за метаданими.

*Сценарії використання:*
* Оберу **Chroma** для локального пет-проєкту, хакатону або швидкого тестування пайплайну на власному комп'ютері.
* Оберу **Pinecone** для enterprise-проєкту, де бізнес готовий платити за готову інфраструктуру, щоб команда фокусувалася на розробці продукту, а не на підтримці кластерів.
* Оберу **Qdrant**, якщо проєкт вимагає максимального контролю над продуктивністю, складної фільтрації, або якщо через політики безпеки дані повинні зберігатися виключно на власних серверах компанії (on-premise).

---

**2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.**

Універсальна модель `all-MiniLM-L6-v2` навчена на загальновживаних текстах і часто не здатна вловити специфічний контекст та вузькоспеціалізовану термінологію наукової літератури. Натомість **specter2_base** спеціально донавчена на графі цитувань наукових статей. Це дозволяє їй розуміти, які статті є семантично близькими, навіть якщо вони використовують різну термінологію. 

Згідно з карткою моделі на HuggingFace: 
> *"SPECTER2 is a family of models trained on scientific citations and can be used for various document-level tasks including representation learning, classification, and recommendation."*

---

**3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?**

Для моделей родини SPECTER рекомендованою метрикою схожості є **Cosine Similarity** (косинусна відстань). 

Вибір правильної метрики при створенні індексу є критично важливим, оскільки векторна база даних (наприклад, Pinecone) внутрішньо оптимізує свої структури (такі як HNSW графи) саме під вказану функцію відстані. Якщо згенерувати вектори, натреновані для косинусної схожості, а в індексі обрати іншу метрику (наприклад, Euclidean) без їхньої попередньої нормалізації, то результати пошуку будуть спотвореними, і релевантні статті просто не потраплять у топ видачі.

---

### 1.3. Отримання ембеддингів

**1.Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?**

Косинусна схожість обчислюється за формулою: 
`Cosine(A, B) = (A · B) / (||A|| * ||B||)`
де `(A · B)` — це скалярний добуток векторів, а `||A||` та `||B||` — їхні довжини (L2-норми).

Коли ми встановлюємо `normalize_embeddings=True`, модель нормалізує кожен згенерований вектор так, щоб його довжина дорівнювала рівно 1 (`||A|| = 1` і `||B|| = 1`). 

Якщо підставити ці значення у формулу, знаменник перетворюється на 1 (`1 * 1 = 1`). Відповідно, формула спрощується до:
`Cosine(A, B) = (A · B) / 1 = A · B`

Скалярний добуток (Dot Product) обчислюється процесором значно швидше і вимагає менше математичних операцій, ніж повноцінна косинусна схожість (не треба обчислювати квадратні корені та ділити). Нормалізуючи вектори на етапі генерації, ми можемо налаштувати індекс у Pinecone на використання метрики `dotproduct`, отримуючи ту саму якість і точність (ранжирування), що й при `cosine`, але з меншими затримками (latency) та меншим навантаженням на систему.

---

### 3. Пошукові запити

**1. Чи збігаються топ-5 для cosine і dot product і чому?**
Так, вони збігаються абсолютно. Оскільки під час генерації ембеддингів ми використали параметр `normalize_embeddings=True`, довжина (L2-норма) кожного вектора дорівнює 1. У формулі косинусної схожості $Cosine(A,B) = \frac{A \cdot B}{||A|| \times ||B||}$ знаменник перетворюється на $1 \times 1 = 1$. Таким чином, операція зводиться до звичайного скалярного добутку (Dot Product).

---

**2. Чи відрізняються результати для L2 і чому?**
Ні, для нормалізованих векторів результати (рейтинг топ-5) для L2 відстані будуть ідентичними до Cosine/Dot Product. Це пояснюється математичним зв'язком: 
$L2(A, B)^2 = ||A - B||^2 = ||A||^2 + ||B||^2 - 2(A \cdot B)$.
Оскільки вектори нормалізовані ($||A||=1, ||B||=1$), формула спрощується до $2 - 2(A \cdot B)$. Це означає, що L2 відстань обернено пропорційна скалярному добутку: чим більший скалярний добуток, тим менша L2 відстань. Тому сортування видає ті самі документи. (Єдина різниця - L2 ми сортуємо за зростанням, а Dot Product за спаданням).

---

**3. Що сталося б, якби ембеддинги не були нормалізовані?**
Якби вектори не були нормалізовані, результати Dot Product суттєво відрізнялися б від Cosine. Dot Product реагує на магнітуду (абсолютну довжину) векторів: довші вектори отримували б штучно завищені бали, навіть якщо кут між ними і запитом великий (слабка семантична схожість). Косинусна ж відстань міряє виключно кут між векторами, нівелюючи їхню довжину. Тож без нормалізації Dot Product видавав би менш релевантні (але "довші" в сенсі векторної магнітуди) документи, змішуючи семантику з розміром.

---

### 4. Chunking

**1. Яка стратегія дає більш осмислені чанки?**
Більш осмислені чанки дає **Semantic chunking (Семантичне розбиття)**. Ця стратегія має межі речень (або абзаців), гарантуючи, що одна завершена думка не буде розірвана на дві частини. Це дозволяє моделі, яка генерує ембеддинги, краще зрозуміти контекст усього фрагмента.

---

**2. Чи є випадки розрізаних речень і як це впливає на ембеддинги?**
Так, у стратегії **Fixed-size chunking** розрізані речення трапляються дуже часто, оскільки текст рубається строго за лімітом токенів або слів (наприклад, рівно на 50 слові), незалежно від пунктуації. 

Це негативно впливає на якість векторів. Якщо підмет опинився в одному чанку, а присудок і пояснення — в іншому, модель згенерує два "слабких" ембеддинги, жоден з яких повноцінно не відображатиме змісту речення. У результаті релевантний фрагмент може не потрапити у видачу (зниження recall).

---

**3. Як розмір overlap (перекриття) впливає на кількість чанків і покриття тексту?**
* **Overlap (перекриття)** створений спеціально для того, щоб пом'якшити проблему розрізаних речень у стратегії Fixed-size. Він копіює хвіст попереднього чанка в початок наступного.
* **Вплив на кількість:** Чим більший overlap, тим меншим є ефективний крок зсуву, що призводить до генерації *більшої кількості чанків* для того ж тексту. Це збільшує витрати на пам'ять бази даних (збільшує базу векторів) та час генерації ембеддингів.
* **Вплив на покриття:** Збільшення overlap підвищує шанси на те, що важливий концепт або сутність, яка опинилася на межі розрізу, буде цілісно представлена хоча б в одному з двох сусідніх чанків, що покращує якість пошуку (покриття змісту).

--- 

### 5. Гібридний пошук та RRF 

**1. Який метод дав кращий результат і чому?**
Найкращий та найстабільніший результат дав **Гібридний пошук**.
* Запит *"Yann LeCun convolutional networks"* та *"BERT fine-tuning"* краще опрацював BM25, оскільки він знаходить точні власні імена, абревіатури та терміни. Векторний пошук міг трохи "розмити" ці терміни в пошуках загального концепту нейромереж.
* Запит *"making computers understand human emotions from text"* відпрацював векторний пошук. BM25 тут провалився, бо шукав точні слова (computers, emotions), тоді як вектор знайшов статті про NLP та Sentiment Analysis (аналіз тональності), зрозумівши семантику запиту.
Гібрид бере найкраще: піднімає нагору документи з точним збігом ключових слів (від BM25) і доповнює їх документами, які відповідають за загальним змістом (від векторів).

### Порівняльна таблиця результатів пошуку

| Запит | BM25 (Лексичний) | Vector (Семантичний) | Hybrid (RRF) |
| :--- | :--- | :--- | :--- |
| **"BERT fine-tuning"** | 1. Temporal Adaptation of BERT...<br>2. CEFER: A Four Facets Framework...<br>3. LegalRelectra... | 1. Model-Based Diffusion Sampling...<br>2. Computational Social Choice...<br>3. Distributed Area Coverage... | 1. Diffusion Policy Policy...<br>2. Temporal Adaptation of BERT...<br>3. Model-Based Diffusion Sampling... |
| **"Yann LeCun convolutional networks"** | 1. Learning Connectivity with Graph...<br>2. Implementation of Training...<br>3. Dependency-based Convolutional... | 1. Lattice gauge symmetry...<br>2. Implementation of Training...<br>3. A Selective Survey on Versatile... | 1. Implementation of Training...<br>2. Lattice gauge symmetry...<br>3. Learning Connectivity with Graph... |
| **"making computers understand human emotions from text"** | 1. Women in ISIS Propaganda...<br>2. CEFER: A Four Facets Framework...<br>3. Generating Clues for Gender... | 1. CEFER: A Four Facets Framework...<br>2. ERIT Lightweight Multimodal...<br>3. PVG at WASSA 2021... | 1. CEFER: A Four Facets Framework...<br>2. Women in ISIS Propaganda...<br>3. Generating Clues for Gender... |

---

**2. Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?**
Так, це класичний ефект RRF (Reciprocal Rank Fusion). Наприклад, стаття могла посісти 6-те місце у BM25-видачі (тому не потрапила в топ-5) і 7-ме місце у векторній видачі (теж не потрапила в топ-5). Однак, оскільки вона стабільно релевантна для *обох* методів, її RRF score складається з двох сум: $1/(60+6) + 1/(60+7) \approx 0.03$. 
Водночас стаття, яка була на 1-му місці у BM25, але *відсутня* у векторах, отримає лише $1/(60+1) + 0 \approx 0.016$. У підсумку стаття, яка була "середнячком" у двох списках, перемагає вузьконаправленого лідера і потрапляє у гібридний Топ-5.

---

**3. Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?**
Параметр $k$ — це константа згладжування (smoothing constant), яка визначає, наскільки різко ми штрафуємо позицію в рейтингу.
* Якщо **$k=1$**, то стаття на 1-му місці отримує $1/2 = 0.5$, а на 2-му місці $1/3 = 0.33$. Різниця величезна. Це призводить до жорсткої домінації перших місць із кожного списку.
* Якщо **$k=60$** (по стандарт), топ-1 отримує $1/61 \approx 0.0163$, а топ-2 отримує $1/62 \approx 0.0161$. Різниця дуже плавна. Це дозволяє нівелювати шум і винагороджувати ті документи, які стабільно з'являються в *обох* списках, навіть якщо не на перших позиціях.

---

## 6. Аналіз і висновки

**1. Семантичний пошук vs BM25. Наведіть конкретні приклади запитів із вашої роботи, де кожен метод виграв. Сформулюйте загальне правило: для яких типів запитів варто надати перевагу кожному з них?**
Як показала практика, жоден з методів не є абсолютно універсальним. **BM25** беззаперечно виграв у запитах, що містили точні терміни та власні імена (наприклад, *"Yann LeCun convolutional networks"* та *"BERT fine-tuning"*). Він працює як "скальпель", вишукуючи рідкісні токени, тому знаходить статті конкретних авторів або згадки специфічних алгоритмів. Натомість **Векторний (семантичний) пошук** май результат на запитах природною мовою, таких як *"making computers understand human emotions from text"*. Там, де BM25 чіплявся за окремі слова і видавав нерелевантні статті про пропаганду ІДІЛ, семантичний пошук "зрозумів" концепт і знайшов датасети з аналізу тональності тексту (навіть якщо там не було слова "computers"). 
*Загальне правило:* Для запитів із SKU-кодами, іменами, абревіатурами та точними назвами слід надавати перевагу лексичному пошуку (BM25). Для довгих запитів, пошуку за змістом, питань природною мовою або при наявності синонімів та перефразувань — векторному. В ідеалі їх завжди слід об'єднувати через гібридний підхід (RRF).

---

**2. Вплив розміру чанка. Що відбувається з якістю пошуку, якщо чанк занадто маленький (10–15 слів)? Якщо занадто великий (500+ слів)? Чи є оптимальний розмір або він залежить від задачі?**
Розмір чанка — це баланс між "деталізацією" та "контекстом", і він критично впливає на якість векторів. Якщо чанк **занадто маленький (10–15 слів)**, він втрачає зміст. Наприклад, чанк *"The results of our experiment clearly demonstrate that the model achieved state of the art"* не містить інформації про те, *яка* це модель і *який* експеримент. Вектор такого чанка буде "шумовим" і знизить точність пошуку. Якщо чанк **занадто великий (500+ слів)**, виникає проблема "розмиття" (dilution) та "lost in the middle". Вектор стає усередненням багатьох тем, і його важко співставити з коротким, конкретним запитом користувача. Крім того, більшість моделей (як-от Specter чи BERT) мають жорстке обмеження у 512 токенів. 
*Оптимальний розмір:* Він залежить від задачі. Для систем точних відповідей краще брати менші чанки (100–250 слів) з великим перекриттям. Для пошуку документів за загальною тематикою — більші (300–500 слів). Найкращий підхід — семантичний чанкінг (розбиття по логічних абзацах чи реченнях).

---

**3. Невідповідна метрика. Що сталося б, якби ми створили індекс Pinecone з метрикою euclidean (L2), але використовували модель, яка повертає нормалізовані вектори? Обґрунтуйте відповідь математично: виведіть зв’язок між L2 і cosine для одиничних векторів.**
Якби ми створили індекс з метрикою `euclidean` (L2), але завантажили туди нормалізовані вектори (L2-норма = 1), **рейтинг видачі (ранжування) не змінився б, але система працювала б неефективно**. 
*Математичне обґрунтування:* Квадрат евклідової відстані між двома векторами обчислюється як $L2(A, B)^2 = ||A||^2 + ||B||^2 - 2(A \cdot B)$. 
Оскільки наші вектори нормалізовані, їхні довжини дорівнюють 1 ($||A||=1, ||B||=1$). Тоді формула скорочується до: $L2^2 = 2 - 2(A \cdot B)$.
З цього випливає, що L2 відстань обернено пропорційна скалярному добутку (Dot Product). Чим ближчі вектори (більший скалярний добуток), тим менша між ними L2 відстань. Pinecone відсортував би результати за зростанням L2 (від найменшої відстані до найбільшої), і видав би *ті самі* топ-5 документів. Однак розрахунок L2 вимагає зайвих обчислень (квадратів та коренів), що марнує обчислювальні ресурси порівняно з чистим Dot Product, який є просто сумою добутків координат.

---

**4. Обмеження Pinecone Starter. З якими обмеженнями безкоштовного тіру ви зіткнулися (або могли б зіткнутися)? Як би ви вирішили задачу, якби датасет був не 10000, а 10 мільйонів статей?**
У безкоштовному тірі Pinecone (Starter) ми стикаємося з жорсткими лімітами: можна створити лише 1 індекс (довелося б видаляти попередній для нових експериментів) та є обмеження на 100 000 векторів (або до 2 ГБ даних). Для нашого сабсету в 10 000 статей та їхніх чанків цього вистачило.
*Якби датасет складався з 10 мільйонів статей:*
1. **Інфраструктура БД:** Безкоштовного Pinecone не вистачило б. Довелося б переходити на платний тариф (Standard/Enterprise) або розгортати open-source рішення на власних серверах (Qdrant, Milvus), щоб уникнути величезних рахунків за хмару.
2. **Пам'ять:** 10 млн векторів розмірністю 768 у форматі float32 займатимуть близько 30 ГБ оперативної пам'яті (RAM). Довелося б застосовувати методи квантизації (Product Quantization або Scalar Quantization), щоб стиснути вектори з мінімальною втратою якості.
3. **Обчислення:** Генерувати ембеддинги послідовно на одному комп'ютері зайняло б тижні. Потрібно було б підняти кластер з GPU-машинами (наприклад, через AWS EC2) і розпаралелити процес за допомогою фреймворків на кшталт Apache Spark або Ray, відправляючи дані в базу асинхронними батчами (Async Upsert).

## Виводи скриптів (Terminal Outputs)

<details>
  <summary>Вивід 01_prepare_data.py</summary>

  ```text
Крок 1: Випадкова вибірка (Reservoir Sampling) з усього файлу...
Сканування 3.5 ГБ файлу (це займе ~10 секунд): 3050852it [00:03, 888330.67it/s] 

Крок 2: Парсинг вибраних записів...
Обробка JSON: 100%|██████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 163129.49it/s]
\nЗавантажено статей:10000
\nРозподіл за роками (топ-10):
year
1992     21
1993     23
1994     39
1995     47
1996     47
1997     87
1998     82
1999     64
2000     91
2001    100
Name: count, dtype: int64
\nРозподіл за категоріями:
category
cs.CV                468
hep-ph               445
cs.LG                426
quant-ph             413
hep-th               406
astro-ph             299
cs.CL                274
cond-mat.mtrl-sci    252
cond-mat.mes-hall    245
gr-qc                240
Name: count, dtype: int64
\nПриклад запису:
{'id': 'astro-ph/9702010', 'title': 'Attenuation of dipping at low energies in the LMXB source X 1755-338', 'abstract': 'We report spectral fitting results for Rosat PSPC observations of the dipping\nsource X 1755-338. These results are consistent with the two-component model\nthat we previously proposed consisting of a blackbody point source plus an\nextended power law component. Remarkably, the low energy cut-off of the\nspectrum does not change appreciably in dips, and it can be seen that dipping\ntakes place in the higher part of the PSPC band where the blackbody contributes\nto the spectrum. Thus dipping consists primarily of absorption of this\ncomponent, whereas the low energy cut-off is determined by the power law. Thus,\nabove 0.5 keV, the dipping is approximately independent of energy as seen in\nExosat and 10 +/- 0.4% deep. Below 0.5 keV, there may be a small residual\ndipping effect of up to 3%; however dipping is certainly substantially reduced\ncompared with the 20% seen by Exosat, and this is the first time that the\neffective cessation of dipping at low energies in such sources has been seen.', 'authors': 'ChurchM. J., Balucinska-ChurchM.', 'year': 1997, 'category': 'astro-ph'}
\nЗбережено вdata/arxiv_subset.parquet
```

<details>
  <summary>Вивід 02_embed.py</summary>

  ```text
1. Завантаження даних з data/arxiv_subset.parquet...
2. Підготовка текстів (Title + [SEP] + Abstract)...
3. Завантаження моделі allenai/specter2_base з HuggingFace...
config.json: 100%|████████████████████████████████████████████████████████████| 754/754 [00:00<00:00, 1.84MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
pytorch_model.bin: 100%|████████████████████████████████████████████████████| 440M/440M [00:12<00:00, 34.1MB/s]
Loading weights: 100%|████████████████████████████████████████████████████| 199/199 [00:00<00:00, 57983.08it/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████| 453/453 [00:00<00:00, 2.79MB/s]
vocab.txt: 100%|████████████████████████████████████████████████████████████| 228k/228k [00:00<00:00, 8.83MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████| 717k/717k [00:00<00:00, 26.0MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████| 125/125 [00:00<00:00, 231kB/s]
4. Генерація ембеддингів...0%|                                                       | 0.00/125 [00:00<?, ?B/s]
model.safetensors: 100%|████████████████████████████████████████████████████| 440M/440M [00:12<00:00, 35.8MB/s]
Batches: 100%|███████████████████████████████████████████████████████████████| 157/157 [02:31<00:00,  1.04it/s]
Batches: 100%|███████████████████████████████████████████████████████████████| 157/157 [02:31<00:00,  4.75it/s]
--- Статистика ---
Загальна кількість оброблених текстів: 10000
Розмірність ембеддингів: 768
Норма першого ембеддингу: 1.0000

6. Збереження матриці векторів у embeddings/embeddings.npy...
Готово!
```

<details>
  <summary>Вивід 03_load_to_pinecone.py</summary>

  ```text
  Підключено до індексу: arxiv-papers
Читаємо дані з data/arxiv_subset.parquet та embeddings/embeddings.npy...
Починаємо завантаження в Pinecone...
Завантаження батчів: 100%|██████████████████████████████████████████████| 10000/10000 [00:25<00:00, 388.89it/s]

Дані успішно завантажено!
```

<details>
  <summary>Вивід 04_search.py</summary>

  ```text
  Завантаження моделі...
Loading weights: 100%|████████████████████████████████████████████████████| 199/199 [00:00<00:00, 58120.36it/s]
Завантаження локальних даних...

--- Чистий семантичний пошук: 'teaching machines to recognize objects in pictures' ---
1. [Score: 0.8696] Bayesian Brain: Computation with Perception to Recognize 3D Objects
   Рік: 2022 | Категорія: cs.AI
   Abstract: We mimic the cognitive ability of Human perception, based on Bayesian hypothesis, to recognize view-based 3D objects. We...

2. [Score: 0.8533] Hierarchical Abstraction Enables Human-Like 3D Object Recognition in Deep Learning Models
   Рік: 2025 | Категорія: cs.CV
   Abstract: Both humans and deep learning models can recognize objects from 3D shapes depicted with sparse visual information, such ...

3. [Score: 0.8391] Classifying Novel 3D-Printed Objects without Retraining: Towards Post-Production Automation in Additive Manufacturing
   Рік: 2026 | Категорія: cs.CV
   Abstract: Reliable classification of 3D-printed objects is essential for automating post-production workflows in industrial additi...

4. [Score: 0.8379] Towards a Self-Organized Agent-Based Simulation Model for Exploration of
  Human Synaptic Connections
   Рік: 2012 | Категорія: cs.NE
   Abstract: In this paper, the early design of our self-organized agent-based simulation model for exploration of synaptic connectio...

5. [Score: 0.8374] Image-based Detection of Segment Misalignment in Multi-mirror Satellites
  using Transfer Learning
   Рік: 2024 | Категорія: cs.CV
   Abstract: In this paper, we introduce a system based on transfer learning for detecting segment misalignment in multimirror satell...


--- Фільтр А (>=2019, cs.LG) ---
1. [Score: 0.8666] Learning Force Control for Contact-rich Manipulation Tasks with Rigid
  Position-controlled Robots
   Рік: 2020 | Категорія: cs.LG
   Abstract: Reinforcement Learning (RL) methods have been proven successful in solving manipulation tasks autonomously. However, RL ...

2. [Score: 0.8619] Dual Action Policy for Robust Sim-to-Real Reinforcement Learning
   Рік: 2024 | Категорія: cs.LG
   Abstract: This paper presents Dual Action Policy (DAP), a novel approach to address the dynamics mismatch inherent in the sim-to-r...

3. [Score: 0.8547] Improving Performance in Reinforcement Learning by Breaking
  Generalization in Neural Networks
   Рік: 2020 | Категорія: cs.LG
   Abstract: Reinforcement learning systems require good representations to work well. For decades practical success in reinforcement...

4. [Score: 0.8499] Distributed Area Coverage with High Altitude Balloons Using Multi-Agent Reinforcement Learning
   Рік: 2025 | Категорія: cs.LG
   Abstract: High Altitude Balloons (HABs) can leverage stratospheric wind layers for limited horizontal control, enabling applicatio...

5. [Score: 0.8476] Time Adaptive Reinforcement Learning
   Рік: 2020 | Категорія: cs.LG
   Abstract: Reinforcement learning (RL) allows to solve complex tasks such as Go often with a stronger performance than humans. Howe...


--- Фільтр B (< 2015, будь-яка категорія) ---
1. [Score: 0.8295] Towards a Self-Organized Agent-Based Simulation Model for Exploration of
  Human Synaptic Connections
   Рік: 2012 | Категорія: cs.NE
   Abstract: In this paper, the early design of our self-organized agent-based simulation model for exploration of synaptic connectio...

2. [Score: 0.7959] Belief Propagation for Structured Decision Making
   Рік: 2012 | Категорія: cs.AI
   Abstract: Variational inference algorithms such as belief propagation have had tremendous impact on our ability to learn and use g...

3. [Score: 0.7936] Robust multirobot coordination using priority encoded homotopic
  constraints
   Рік: 2013 | Категорія: cs.RO
   Abstract: We study the problem of coordinating multiple robots along fixed geometric paths. Our contribution is threefold. First w...

4. [Score: 0.7930] Evolutionary Hessian Learning: Forced Optimal Covariance Adaptive
  Learning (FOCAL)
   Рік: 2011 | Категорія: cs.NE
   Abstract: The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) has been the most successful Evolution Strategy at exploiti...

5. [Score: 0.7919] Knowledge Representation for Robots through Human-Robot Interaction
   Рік: 2013 | Категорія: cs.AI
   Abstract: The representation of the knowledge needed by a robot to perform complex tasks is restricted by the limitations of perce...


---Порівняння локальних метрик схожості---
Top 5 Cosine (Індекси): [4778 9694 8935 5379  118]
Top 5 Dot Product (Індекси): [4778 9694 8935 5379  118]
Top 5 L2 Distance (Індекси): [4778 9694 8935 5379  118]
```

<details>
  <summary>Вивід 05_chunking.py</summary>

  ```text
Створюємо індекс arxiv-chunks-fixed...
Створюємо індекс arxiv-chunks-semantic...
Завантажуємо модель...
Loading weights: 100%|████████████████████████████████████████████████████| 199/199 [00:00<00:00, 60500.62it/s]
Читаємо дані...
Вибрано 30 статей. Середня довжина анотації: 319 слів.
Розбиваємо тексти на чанки...

Генерація ембеддингів та завантаження для Fixed-size (267 чанків)...
Batches: 100%|███████████████████████████████████████████████████████████████████| 9/9 [00:01<00:00,  4.76it/s]
Upsert Fixed-size: 100%|█████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.19it/s]

Генерація ембеддингів та завантаження для Semantic (254 чанків)...
Batches: 100%|███████████████████████████████████████████████████████████████████| 8/8 [00:01<00:00,  5.81it/s]
Upsert Semantic: 100%|███████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.19it/s]


---ТЕСТОВИЙ ПОШУК: 'performance limitations of neural networks' ---

--- Результати Fixed-size Chunking ---
1. [Score: 0.8422] Sharp Thresholds in Random Simple Temporal Graphs
   Чанк #8: n/n$....

2. [Score: 0.8183] Some Extensions of Probabilistic Logic
   Чанк #11: but also constraint satisfaction among these propositions in the relational network will revise these measures. This mechanism is similar to human reasoning which is an evaluative process converging to the most satisfactory result. The main idea arises from the consistent labeling problem in computer vision. This method is originally applied...

3. [Score: 0.8166] Unposed: Unsupervised Pose Estimation based Product Image
  Recommendations
   Чанк #8: scale....

--- Результати Semantic Chunking ---
1. [Score: 0.8208] The ARCADE Raman Lidar and atmospheric simulations for the Cherenkov
  Telescope Array
   Чанк #7: This contribution includes a description of the ARCADE Lidar and the characterization of the performance of the system....

2. [Score: 0.8139] The Paradox of Talent: how Chance affects Success in Tennis Tournaments
   Чанк #2: Our dataset covers the decade 2010-2019 of main events in the ATP circuit and consists of tourney results and annual rankings for professional male players. After a preliminary data analysis, we introduce an agent-based model able to accurately simulate the tennis players' dynamics along several seasons....

3. [Score: 0.8088] The Paradox of Talent: how Chance affects Success in Tennis Tournaments
   Чанк #5: We find the best agreement between real data and simulation results when talent weights substantially less than luck, i.e. when a is between 0.20 and 0.30. A further comparison between data and simulations, based on the analysis of the direct networks of all the matches, confirms the previous finding....
   ```

<details>
  <summary>Вивід 06_hybrid_search.py</summary>

  ```text
  Завантажуємо модель...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████| 199/199 [00:00<00:00, 56407.82it/s]
Завантажуємо локальні дані та будуємо BM25 індекс...

----------------------------------------------------
ЗАПИТ: 'BERT fine-tuning'
----------------------------------------------------

--- Топ-5 BM25 (Ключові слова) ---
1. [Score: 15.2997] Temporal Adaptation of BERT and Performance on Downstream Document
  Classification: Insights from Social Media
2. [Score: 9.7433] CEFER: A Four Facets Framework based on Context and Emotion embedded
  features for Implicit and Explicit Emotion Recognition
3. [Score: 9.6409] LegalRelectra: Mixed-domain Language Modeling for Long-range Legal Text
  Comprehension
4. [Score: 9.1270] Astro-HEP-BERT: A bidirectional language model for studying the meanings
  of concepts in astrophysics and high energy physics
5. [Score: 9.1188] From Parameter Dynamics to Risk Scoring : Quantifying Sample-Level Safety Degradation in LLM Fine-tuning

--- Топ-5 Векторний (Семантика Pinecone) ---
1. [Score: 0.8529] Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making
2. [Score: 0.8506] Computational Social Choice and Computational Complexity: BFFs?
3. [Score: 0.8504] Distributed Area Coverage with High Altitude Balloons Using Multi-Agent Reinforcement Learning
4. [Score: 0.8501] Branching brownian motion seen from its left-most particule
5. [Score: 0.8486] Robust Taxi Fare Prediction Under Noisy Conditions: A Comparative Study of GAT, TimesNet, and XGBoost

--- Топ-5 Гібридний (BM25 + Vector через RRF) ---
1. [RRF Score: 0.0290] Diffusion Policy Policy Optimization
2. [RRF Score: 0.0164] Temporal Adaptation of BERT and Performance on Downstream Document
  Classification: Insights from Social Media
3. [RRF Score: 0.0164] Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making
4. [RRF Score: 0.0161] CEFER: A Four Facets Framework based on Context and Emotion embedded
  features for Implicit and Explicit Emotion Recognition
5. [RRF Score: 0.0161] Computational Social Choice and Computational Complexity: BFFs?


----------------------------------------------------
ЗАПИТ: 'Yann LeCun convolutional networks'
----------------------------------------------------

--- Топ-5 BM25 (Ключові слова) ---
1. [Score: 13.4623] Learning Connectivity with Graph Convolutional Networks for
  Skeleton-based Action Recognition
2. [Score: 12.5608] Implementation of Training Convolutional Neural Networks
3. [Score: 12.0750] Dependency-based Convolutional Neural Networks for Sentence Embedding
4. [Score: 11.9441] Lattice gauge symmetry in neural networks
5. [Score: 11.5608] Analysis of Convolutional Neural Networks for Document Image
  Classification

--- Топ-5 Векторний (Семантика Pinecone) ---
1. [Score: 0.8815] Lattice gauge symmetry in neural networks
2. [Score: 0.8743] Implementation of Training Convolutional Neural Networks
3. [Score: 0.8732] A Selective Survey on Versatile Knowledge Distillation Paradigm for
  Neural Network Models
4. [Score: 0.8675] Classifying Images with CoLaNET Spiking Neural Network -- the MNIST
  Example
5. [Score: 0.8674] Advances in Electron Microscopy with Deep Learning

--- Топ-5 Гібридний (BM25 + Vector через RRF) ---
1. [RRF Score: 0.0323] Implementation of Training Convolutional Neural Networks
2. [RRF Score: 0.0320] Lattice gauge symmetry in neural networks
3. [RRF Score: 0.0164] Learning Connectivity with Graph Convolutional Networks for
  Skeleton-based Action Recognition
4. [RRF Score: 0.0159] Dependency-based Convolutional Neural Networks for Sentence Embedding
5. [RRF Score: 0.0159] A Selective Survey on Versatile Knowledge Distillation Paradigm for
  Neural Network Models


----------------------------------------------------
ЗАПИТ: 'making computers understand human emotions from text'
----------------------------------------------------

--- Топ-5 BM25 (Ключові слова) ---
1. [Score: 18.5095] Women in ISIS Propaganda: A Natural Language Processing Analysis of
  Topics and Emotions in a Comparison with Mainstream Religious Group
2. [Score: 13.8573] CEFER: A Four Facets Framework based on Context and Emotion embedded
  features for Implicit and Explicit Emotion Recognition
3. [Score: 12.7870] Generating Clues for Gender based Occupation De-biasing in Text
4. [Score: 11.9099] Deep learning models are not robust against noise in clinical text
5. [Score: 11.0806] Harmonization of conflicting medical opinions using argumentation
  protocols and textual entailment - a case study on Parkinson disease

--- Топ-5 Векторний (Семантика Pinecone) ---
1. [Score: 0.8909] CEFER: A Four Facets Framework based on Context and Emotion embedded
  features for Implicit and Explicit Emotion Recognition
2. [Score: 0.8806] ERIT Lightweight Multimodal Dataset for Elderly Emotion Recognition and
  Multimodal Fusion Evaluation
3. [Score: 0.8719] PVG at WASSA 2021: A Multi-Input, Multi-Task, Transformer-Based
  Architecture for Empathy and Distress Prediction
4. [Score: 0.8698] Solution for Emotion Prediction Competition of Workshop on Emotionally
  and Culturally Intelligent AI
5. [Score: 0.8677] EMO-KNOW: A Large Scale Dataset on Emotion and Emotion-cause

--- Топ-5 Гібридний (BM25 + Vector через RRF) ---
1. [RRF Score: 0.0325] CEFER: A Four Facets Framework based on Context and Emotion embedded
  features for Implicit and Explicit Emotion Recognition
2. [RRF Score: 0.0315] Women in ISIS Propaganda: A Natural Language Processing Analysis of
  Topics and Emotions in a Comparison with Mainstream Religious Group
3. [RRF Score: 0.0304] Generating Clues for Gender based Occupation De-biasing in Text
4. [RRF Score: 0.0161] ERIT Lightweight Multimodal Dataset for Elderly Emotion Recognition and
  Multimodal Fusion Evaluation
5. [RRF Score: 0.0159] PVG at WASSA 2021: A Multi-Input, Multi-Task, Transformer-Based
  Architecture for Empathy and Distress Prediction
  ```