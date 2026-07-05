# Vector Database Comparison — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Vector Database Comparison: does GitHub issue #62's feature table still hold in 2026, and how should vector databases be compared? |
| Venue | unknown |
| Year | unknown |
| Authors | unknown |
| Official Code | unknown |
| Venue Kind | survey |

這則筆記不是單篇論文摘要，而是一則跨來源 survey：它以一張流傳甚廣的「vector database 功能對照表」（road-to-master issue #62，資料來源為已失效的 vectorview benchmark，約 2023）為稽核對象，問一個實用問題——**這張把 Pinecone、Weaviate、Milvus、Qdrant、Chroma、Elasticsearch、pgvector 排成一欄的表，在 2026 還能拿來選型嗎？** 表格內的每個數字與勾叉都先當成未驗證的 seed，凡是與一手來源衝突，一手來源勝出。

## 📚 Sources

| # | 來源 | 種類 | 年 | 取用狀態 |
|-|-|-|-|-|
| S1 | ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms（arXiv:1807.05614） | 學術 benchmark 論文 | 2018 | fetched（LaTeX 全文） |
| S2 | Results of the NeurIPS'21 Challenge on Billion-Scale ANN Search（arXiv:2205.03763） | 學術 benchmark 論文 | 2022 | fetched（LaTeX 全文） |
| S3 | Elastic 官方 blog「Elasticsearch is Open Source, Again」 | 廠商公告（primary） | 2024 | fetched |
| S4 | GitHub API repo metadata（milvus / qdrant / weaviate / chroma） | 平台 metadata（primary） | 2026 | fetched |
| S5 | pgvector LICENSE、Pinecone / Weaviate / Qdrant 官方文件 | 廠商文件（primary） | 2024–2026 | fetched |
| S6 | VectorDBBench README、Qdrant benchmarks、TigerData pgvector vs Qdrant | 廠商自跑 benchmark | 2024–2025 | fetched（僅作 context，不單獨支撐 benchmark 結論） |

以下逐一稽核。所有網路來源的擷取日為 2026-07-05；S1/S2 為 arXiv preprint 全文，正式版可能略有差異。

## 為什麼「單一 QPS 數字」在 2026 已不是一個比較

issue #62 的 seed 表把效能寫成單一數字：Milvus 2406、Weaviate 791、pgvector 141（欄名 nyt-256-angular，但**沒有標註 recall**）。問題不在於數字舊，而在於**缺了 recall 就無法重現、無法比較**。中立學術 benchmark 從不給裸 QPS：S1 的標準圖把橫軸畫成 recall、縱軸畫成 QPS，圖說逐字是「Recall-QPS (1/s) tradeoff - up and to the right is better」，也就是每個引擎是一條曲線而非一個點。

S2 把這件事做得更硬：整個 NeurIPS'21 競賽的排名規則是「在固定 QPS 截斷下比 recall，或在固定 recall 下比 QPS」，例如 Track 1 的表頭寫「Recall/AP achieved at 10000 QPS on Azure F32v2 VM with 32 vCPUs」，Track 3 另有一欄直接叫「Querries per second at 90% recall」。三件套（dataset、recall floor、hardware）齊全才有意義；seed 的數字這三件全缺，因此屬於無法回溯的 folklore，應整欄丟棄而非沿用。

還有一個量綱陷阱。S1 明說它量的是「in-memory」的單機 library，每個演算法「is run in an isolated Docker container」，這與廠商 benchmark 那種「100 個並發 client 打 RPS」是不同量綱，不能直接並列；而且 S1 根本不測 Pinecone 這類 hosted service，所以 seed 那個「Pinecone 150 QPS」不可能來自這條 benchmark 線。連 dataset 規模都對不上：seed 稱 nyt-256-angular 有約 29 萬向量，但 S1 的資料表把 NYTimes 記為「234 791 / 10 000」筆、256 維、且相似度是 Euclidean 而非 angular——這種細節錯位正是「數字失去可稽核性」的徵兆。

## 一個做對的對照：固定 recall 下 pgvector vs Qdrant

抽象地講「要綁 recall」不夠，看一個把三件套標齊的實測。TigerData（Timescale，且自承偏袒 Postgres）在 50M 筆 Cohere-768 embeddings、AWS r6id.4xlarge（16 vCPU / 128 GB）、**固定 99% recall** 的條件下對照 pgvector+pgvectorscale 與 Qdrant，得到的不是「誰快」的單一答案，而是一組 Pareto trade-off：

| 指標（99% recall, 50M Cohere-768, r6id.4xlarge） | Postgres（pgvector+pgvectorscale） | Qdrant |
|-|-|-|
| 吞吐 throughput | 471.57 QPS | 41.47 QPS |
| p95 尾延遲 | 60.42 ms | 36.73 ms |

同一組實驗裡，Postgres 拿到約 11.4 倍的吞吐，Qdrant 卻拿到更低的 p95/p99 尾延遲（36.73 ms vs 60.42 ms）。這正是 seed 那種單一數字永遠表達不了的東西：沒有全域贏家，只有「你在乎吞吐還是尾延遲」的取捨，而這個取捨只有在 recall、dataset、hardware 都被釘住時才成立。順帶一提，pgvectorscale 這種 disk-first 的 Postgres 外掛，seed 表格裡根本沒有它的位置。

同樣重要的是：任何單一廠商的自跑 benchmark 都不能當事實。VectorDBBench 的 README 逐字自承「sponsored by Zilliz」（Milvus 母公司），Qdrant 自家 benchmark 的結論句則是教科書級的自家全贏——「highest RPS and lowest latencies in almost all the scenarios」。上面 TigerData 的例子之所以還可用，不是因為它中立（它不中立），而是因為它把方法透明化、把 recall/dataset/hardware 標齊，讓讀者能自己判斷偏誤的方向。

## issue #62 表格：哪些仍成立、哪些已翻盤

把 seed 表逐欄對到 2026 的一手來源，可分成三類：粗分類仍可用、細節已過時、以及被直接推翻。下表挑出最關鍵的翻盤格：

| seed 欄位 | seed 標法 | 2026 一手來源 | 判定 |
|-|-|-|-|
| Elasticsearch 是否開源 | ❌ 閉源 | Elastic 官方：新增 AGPL-3.0（OSI 認可） | 被推翻 |
| Weaviate 授權 | （常被誤記為 Apache） | GitHub API：BSD-3-Clause | 被推翻 |
| pgvector 授權 | （常被誤記為 Apache） | LICENSE 檔：PostgreSQL License | 被推翻 |
| RBAC：Qdrant / Pinecone | ❌ 不支援 | 官方文件皆已支援 RBAC | 被推翻 |
| Pinecone 索引 = HNSW？ | ?（留白） | 官方：從來不用 HNSW | 被推翻 |
| star 數 | 8k–23k | 實測普遍成長 2–3.7 倍 | 過時 |

其中最戲劇性的是 Elasticsearch。seed 在 2023 標 ❌ 當時正確（Elastic 2021 把授權改成 SSPL/ELv2），但 2024-08 官方宣布「adding AGPL as another license option next to ELv2 and SSPL」，且明言這是**新增**不是取代——「adding another option, and not removing anything」——因為 AGPL 是「an OSI approved license」，Elasticsearch 因此重新可稱開源。授權欄還有兩個 seed 常見的標註錯誤：Weaviate 的 GitHub metadata 是 BSD-3-Clause、pgvector 的 LICENSE 是 PostgreSQL License，兩者都不是 Apache。star 數則全數過時，例如 Milvus 從 seed 的 23k 漲到實測 45,073、pgvector 從 6k 漲到實測 22,069；star 只是熱度 proxy，此處僅用來證明 seed 的社群欄已 stale。

RBAC 欄是 seed 最過時的一欄。seed 把 Qdrant、Pinecone 都標成 ❌，但 Qdrant 官方 1.9 版公告以「granular access control using JSON Web Tokens (JWT)」鋪陳出 RBAC，Pinecone 官方安全文件也逐字寫「Pinecone uses role-based access controls (RBAC) to manage access to resources」。也就是說，seed 把 RBAC 當成「purpose-built 向量庫落後」的差異點，這個前提在 2026 已不成立。

## 兩個被 seed 壓成一格的維度：hybrid search 與索引

seed 把 hybrid search 這欄命名為「Hybrid Search (i.e., scalar filtering)」並全標 ✅，這個括號本身就是定義錯誤。2026 的 hybrid search 指的是稠密向量檢索與關鍵字（BM25）檢索兩路並跑、再融合——S5 的權威定義是「vector search and a keyword search in parallel」，交給融合演算法（通常是 RRF）「Reciprocal Rank Fusion (RRF), which combines and ranks the objects into a single list」。這與 scalar / metadata filtering 是兩件事：以 scalar filtering 論幾乎人人 ✅，但以「dense + sparse 融合」論，pgvector 原生並不做 BM25 融合（需靠 ParadeDB / VectorChord 等外掛），把兩軸壓成一格會誤導讀者。

索引欄同理被壓扁。seed 把好幾家標成單一「HNSW」，但 2026 的實況是各家早已分岔：Pinecone 官方甚至逐字強調自己「does not use Hierarchical Navigable Small World (HNSW)」，改用 serverless 下依 slab 大小挑選的專有索引；Milvus、Weaviate、Elasticsearch 則各有 HNSW/IVF/DiskANN/flat 等多種 base index。更關鍵的是 seed 整欄漏掉了 quantization（PQ/SQ/BQ/int8）——這是 2024–2026 記憶體與成本的主戰場。現代的索引比較應該把 base graph 演算法與 quantization 變體分成兩軸，而不是塞進一格「HNSW」。

## seed 漏掉的一整類新品：serverless 與 disk-first

seed 表停在 2023，錯過的不只是幾格數字，而是一整個產品類別的出現：serverless 與 disk-first。Pinecone 2024 起主推的 serverless 架構把儲存與計算分離——官方文件說每個 namespace 的紀錄被組成不可變的檔案（slab），索引主體「stored in distributed object storage that provides virtually unlimited scalability and high availability」，查詢時才把需要的 slab 拉進計算節點。這讓「機器記憶體裝不下的資料量」與「按實際用量計費」變成可能，而 seed 表把每家都預設成「自架一台裝滿記憶體的伺服器」，根本沒有這一欄，也因此它的 Pinecone「150 QPS」這種單點數字，量的是一個已經不存在的部署形態。

另一條被漏掉的線是 disk-first 索引。前面的 pgvectorscale 就是代表：它在 pgvector 原生的 in-memory HNSW 之外，「pgvectorscale adds StreamingDiskANN implemented in Rust」，把圖索引放到 SSD 而非全數塞進記憶體，並為它配上「Statistical Binary Quantization (SBQ)」，用壓縮換取「在更少記憶體下撐更大資料量」。這正好接回索引欄那個 quantization 缺口：2024–2026 真正的成本戰場是「base index 演算法 × 落盤策略 × quantization」三件事一起考量，而 seed 只給了一格「HNSW」。把 serverless 與 disk-first 當成一整個 seed 缺席的維度來談，是本筆記自己的歸納，不是任一來源明講的分類。

## 🧪 Critical Assessment

### 這張表想解決的問題是真的，但它的量綱是錯的
「幫我在一頁內比較主流向量庫」是真需求，issue #62 的粗欄（開源 / 自架 / 是否購買）到今天仍大致可用。但它把效能寫成單一 QPS 數字，這在方法論上就站不住：如同 S1/S2 反覆示範，沒有 recall floor 與 hardware 的 QPS 不可重現，因此 perf 欄不是「舊」，而是從一開始就量錯了維度。

### 一手來源之間彼此不可比，連「做對」的例子也帶偏誤
本稽核用到的量化來源，實驗設定互不相容：S1 量單機 in-memory library、S2 量固定硬體上的競賽 recall、TigerData 量 50M 級 client-server 吞吐與尾延遲。三者不能直接並列成一張表。更麻煩的是廠商 benchmark 幾乎都自家贏（VectorDBBench 由 Zilliz 贊助、Qdrant benchmark 自稱幾乎全場最佳），所以任何「誰最快」的排名都必須存疑；連我引用的 TigerData 對照也自承偏袒 Postgres，只是它把方法標透明而已。

### seed 的價值不在數字，而在提醒「欄位選錯了」
把 seed 當成反例其實有教育意義：它最大的錯不是某格填錯，而是欄位設計——把 hybrid search 等同 scalar filtering、把多樣索引壓成「HNSW」、把 perf 壓成一個數字。這些是分類維度的錯，重填數字並不能修好，得換維度（recall/QPS 曲線、base index × quantization 兩軸、真 hybrid 與 filtering 分列）。

### 對讀者的實際問題，這張表只能給方向、不能給排名
對 RAG / agent-memory 選型，這則稽核能給的是有依據的方向：想少維運一個系統、資料量中小 → pgvector；在乎尾延遲 → Qdrant（見上面固定 recall 的對照）；要最廣索引與 GPU → Milvus。但「誰在你的負載上更快」這種排名，任何現成表格都答不了；唯一能關閉它的方法，是在你自己的 dataset、固定 recall、同一台機器上實測出每個引擎的 recall/QPS 曲線與 p95/p99 尾延遲——這則筆記刻意不代替你做這件事。

## 🔗 Related notes

- [BM25](../../natural_language_processing/information_retrieval/BM25/) — hybrid search 的 sparse/keyword 那一路就是 BM25，本筆記對「hybrid = dense + BM25 融合」的定義即以此為基礎。
- [TF-IDF](../../natural_language_processing/information_retrieval/TFIDF/) — 稀疏檢索的前身，理解 sparse-dense 融合的權重取捨時的背景。
