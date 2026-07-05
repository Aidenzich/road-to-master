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

這則筆記不是單篇論文摘要，而是一則跨來源 survey：它以一張流傳甚廣的「vector database 功能對照表」（road-to-master issue #62，資料來源為已失效的 vectorview benchmark，約 2023）為稽核對象，交出一張 **2026 可直接拿來選型的新對照表**，再說明新表為什麼要換掉舊表的欄位。舊表裡的每個數字與勾叉都先當成未驗證的 seed，凡是與一手來源衝突，一手來源勝出。

## 📚 Sources

| # | 來源 | 種類 | 年 | 取用狀態 |
|-|-|-|-|-|
| S1 | ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms（arXiv:1807.05614） | 學術 benchmark 論文 | 2018 | fetched（LaTeX 全文） |
| S2 | Results of the NeurIPS'21 Challenge on Billion-Scale ANN Search（arXiv:2205.03763） | 學術 benchmark 論文 | 2022 | fetched（LaTeX 全文） |
| S3 | Elastic 官方 blog「Elasticsearch is Open Source, Again」 | 廠商公告（primary） | 2024 | fetched |
| S4 | GitHub API repo metadata（milvus / qdrant / weaviate / chroma / pgvector / pgvectorscale / lancedb / paradedb） | 平台 metadata（primary） | 2026 | fetched |
| S5 | 各家官方文件：Pinecone、Weaviate、Qdrant、Chroma、Elasticsearch、pgvector、pgvectorscale、Turbopuffer、LanceDB、ParadeDB 的 index／quantization／hybrid／RBAC／deploy 頁 | 廠商文件（primary） | 2024–2026 | fetched |
| S6 | VectorDBBench README、Qdrant benchmarks、TigerData pgvector vs Qdrant | 廠商自跑 benchmark | 2024–2025 | fetched（僅作 context，不單獨支撐 benchmark 結論） |

所有網路來源的擷取日為 2026-07-05；S1／S2 為 arXiv preprint 全文，正式版可能略有差異。Milvus 的 milvus.io 文件在自動抓取時一律回 403，因此 Milvus 各格改引其母公司 Zilliz 的官方頁（zilliz.com）；此為同一供應商的一手來源，但非 milvus.io 參考文件本身。

## 2026 對照表：主流向量庫的七個可稽核維度

這是本筆記的主交付物。欄位是照 seed 表的毛病重新設計過的——把被 seed 壓成一格的東西拆開（真 hybrid 與 metadata filtering 分列、base index 與 quantization 分列），並補上 seed 完全沒有的軸（部署形態、落盤／物件儲存架構）。**刻意不設效能欄**：任何跨產品 QPS／latency 數字若沒同時釘住 recall、dataset、hardware 就不可比，理由見下一節。每一格都可回溯到來源與逐字片段；查不到來源的格一律寫 `unknown`，不猜。

部署碼：**自架**＝open-source 可自行部署、**managed**＝廠商託管、**serverless**＝按量計費、儲存與計算分離。標 ‡ 的格是合理推論而非直接來源引用。

| 產品 | 授權 | 部署形態 | Base index families | Quantization | 真 hybrid（dense＋BM25 融合） | RBAC | Disk-first／object-storage |
|-|-|-|-|-|-|-|-|
| Pinecone | proprietary（managed SaaS）‡ | managed＋serverless | 專有（官方明言不用 HNSW） | unknown | ✓（單一 index 存 dense＋sparse） | ✓ | ✓（distributed object storage） |
| Weaviate | BSD-3-Clause | 自架＋managed＋serverless | HNSW／flat／dynamic | PQ／BQ／SQ | ✓（RRF 融合） | ✓ | unknown |
| Milvus | Apache-2.0 | 自架＋managed | FLAT／IVF_FLAT／HNSW／DiskANN／ScaNN | PQ／SQ | ✓（BM25 sparse＋dense） | ✓ | ✓（object storage 段落存儲） |
| Qdrant | Apache-2.0 | 自架＋managed | HNSW | SQ／PQ／BQ | ✓（dense＋sparse，RRF） | ✓ | ✓（on-disk／mmap） |
| Chroma | Apache-2.0 | 自架＋serverless | HNSW | unknown | Cloud 版 ✓（RRF）；OSS 核心僅 full-text filter | unknown | unknown |
| Elasticsearch | AGPL-3.0 | 自架＋managed＋serverless | HNSW（Lucene） | int8／int4／BBQ | ✓（RRF，BM25＋dense） | ✓ | bbq_disk（可落盤變體） |
| pgvector | PostgreSQL License | 自架＋managed（PG 擴充） | HNSW／IVFFlat | halfvec／binary | ✗ 原生無 BM25（靠 PG FTS 或 ParadeDB） | 繼承 Postgres 角色‡ | ✗ 原生（見 pgvectorscale）‡ |
| pgvectorscale | PostgreSQL License | 自架（PG 擴充）‡ | StreamingDiskANN | SBQ（binary） | ✗（同 pgvector）‡ | 繼承 Postgres 角色‡ | ✓（StreamingDiskANN 落盤） |
| Turbopuffer | proprietary（managed）‡ | managed | unknown | unknown | ✓（vector＋full-text） | unknown | ✓（object-storage native） |
| LanceDB | Apache-2.0 | embedded＋managed | unknown | unknown | ✓（vector＋full-text） | unknown | ✓（Lance 列式格式落盤） |

**效能腳註（取代 seed 的 QPS 欄）**：本表不放單一效能數字。要比效能，只能看把 recall、dataset、hardware 都釘住的來源——S1／S2 的 recall-QPS 曲線與 NeurIPS'21 競賽規則、以及 TigerData 在固定 99% recall 下的 pgvector vs Qdrant 實測；細節見〈為什麼表裡沒有效能欄〉。

**ParadeDB 腳註**：pgvector 那格的「靠 ParadeDB」指的是 Postgres 生態另一個擴充 ParadeDB（AGPL-3.0），它「brings Elastic-quality full-text search」進 Postgres、用 Tantivy 實作 BM25，補上 pgvector 缺的關鍵字檢索那一路。但按其 dev 分支 README，ParadeDB 自身的「Vector Search」與「Hybrid Search」仍標為 coming soon，因此它目前是 pgvector 的 BM25 搭檔而非獨立的 hybrid 向量庫，故不單列一列。

## 為什麼表裡沒有效能欄

seed 表把效能寫成單一數字：Milvus 2406、Weaviate 791、pgvector 141（欄名 nyt-256-angular，卻**沒有標註 recall**）。問題不在數字舊，而在**缺了 recall 就無法重現、無法比較**。中立學術 benchmark 從不給裸 QPS：S1 的標準圖把橫軸畫成 recall、縱軸畫成 QPS，圖說逐字是「Recall-QPS (1/s) tradeoff - up and to the right is better」——每個引擎是一條曲線而非一個點。S2 把這件事做得更硬：NeurIPS'21 競賽的排名規則是「在固定 QPS 截斷下比 recall，或在固定 recall 下比 QPS」，Track 1 表頭寫「Recall/AP achieved at 10000 QPS on Azure F32v2 VM with 32 vCPUs」，Track 3 另有一欄直接叫「Querries per second at 90% recall」。三件套（dataset、recall floor、hardware）齊全才有意義；seed 的數字三件全缺，屬於無法回溯的 folklore，應整欄丟棄而非沿用。還有量綱陷阱：S1 明說它量的是「in-memory」的單機 library，每個演算法「is run in an isolated Docker container」，這與廠商 benchmark 那種「100 個並發 client 打 RPS」不同量綱，而且 S1 根本不測 Pinecone 這類 hosted service，所以 seed 那個「Pinecone 150 QPS」不可能來自這條線。

把三件套釘齊會長成什麼樣，看一個具體例子。TigerData（Timescale，且自承偏袒 Postgres）在 50M 筆 Cohere-768 embeddings、AWS r6id.4xlarge（16 vCPU／128 GB）、**固定 99% recall** 的條件下對照 pgvector+pgvectorscale 與 Qdrant：

| 指標（99% recall，50M Cohere-768，r6id.4xlarge） | Postgres（pgvector+pgvectorscale） | Qdrant |
|-|-|-|
| 吞吐 throughput | 471.57 QPS | 41.47 QPS |
| p95 尾延遲 | 60.42 ms | 36.73 ms |

同一組實驗裡，Postgres 拿到約 11.4 倍吞吐，Qdrant 卻拿到更低的 p95 尾延遲（36.73 ms vs 60.42 ms）。這正是 seed 那種單一數字永遠表達不了的：沒有全域贏家，只有「你在乎吞吐還是尾延遲」的取捨，而這個取捨只有在 recall、dataset、hardware 都被釘住時才成立——換一組 recall，兩條線就重排。順帶一提，任何單一廠商的自跑 benchmark 都不能當事實：VectorDBBench 的 README 逐字自承「sponsored by Zilliz」（Milvus 母公司），Qdrant 自家 benchmark 的結論句是教科書級的自家全贏——「highest RPS and lowest latencies in almost all the scenarios」。上面 TigerData 之所以還可用，不是因為它中立，而是因為它把 recall／dataset／hardware 標齊、讓讀者能自己判斷偏誤方向。

## 這張表改了 seed 的哪些欄位

新表不是重填 seed 的格子，而是換掉 seed 選錯的欄位並更新過時的判定。把 seed 表逐欄對到 2026 一手來源，最關鍵的翻盤格如下：

| seed 欄位 | seed 標法 | 2026 一手來源 | 判定 |
|-|-|-|-|
| Elasticsearch 是否開源 | ❌ 閉源 | Elastic 官方：新增 AGPL-3.0（OSI 認可） | 被推翻 |
| Weaviate 授權 | （常被誤記為 Apache） | GitHub API：BSD-3-Clause | 被推翻 |
| pgvector 授權 | （常被誤記為 Apache） | LICENSE 檔：PostgreSQL License | 被推翻 |
| RBAC：Qdrant／Pinecone | ❌ 不支援 | 官方文件皆已支援 RBAC | 被推翻 |
| Pinecone 索引 = HNSW？ | ?（留白） | 官方：從來不用 HNSW | 被推翻 |
| Hybrid Search =「scalar filtering」 | ✅（把 hybrid 等同 metadata 過濾） | 真 hybrid＝dense＋BM25 兩路融合，與過濾是兩件事 | 定義錯，已拆欄 |
| 索引欄壓成單一「HNSW」 | 一格 | 各家分岔（IVF／DiskANN／ScaNN…），且漏掉 quantization | 維度錯，已拆兩軸 |
| serverless／object-storage 部署 | （整類缺席） | Pinecone／Turbopuffer 等已把儲存與計算分離 | 整欄補上 |
| star 數 | 8k–23k | 實測普遍成長 2–3.7 倍 | 過時 |

其中最戲劇性的是 Elasticsearch。seed 在 2023 標 ❌ 當時正確（Elastic 2021 把授權改成 SSPL／ELv2），但 2024-08 官方宣布新增 AGPL，且明言這是新增不是取代——「adding another option, and not removing anything」——因為 AGPL 是「an OSI approved license」，Elasticsearch 因此重新可稱開源。RBAC 欄則是 seed 最過時的一欄：它把 Qdrant、Pinecone 都標 ❌，但 Qdrant 官方以「granular access control using JSON Web Tokens (JWT)」鋪陳 RBAC，Pinecone 官方安全文件也逐字寫「Pinecone uses role-based access controls (RBAC) to manage access to resources」。seed 把 RBAC 當成「purpose-built 向量庫落後」的差異點，這個前提在 2026 已不成立。

至於被 seed 壓成一格的兩個維度，新表各給了一整欄。Hybrid 這欄 seed 命名為「Hybrid Search (i.e., scalar filtering)」，這個括號本身就是定義錯：2026 的 hybrid 指稠密向量與關鍵字（BM25）兩路並跑再融合——Weaviate 的權威定義是「vector search and a keyword search in parallel」交給「Reciprocal Rank Fusion (RRF)」。這與 metadata filtering 是兩件事：以過濾論幾乎人人 ✅，但以「dense＋sparse 融合」論，pgvector 原生並不做 BM25 融合（其 README 只叫你「Use together with Postgres」的全文檢索），要真 hybrid 得靠 ParadeDB 這類擴充。索引欄同理：seed 把好幾家標成單一「HNSW」，但 Pinecone 官方逐字強調自己「does not use Hierarchical Navigable Small World (HNSW)」，Milvus 則同時有 FLAT／IVF_FLAT／HNSW／DiskANN／ScaNN。更關鍵的是 seed 整欄漏掉 quantization——這是 2024–2026 記憶體與成本的主戰場：Weaviate 列出「Binary quantization (BQ) Product quantization (PQ) Scalar quantization (SQ)」，Elasticsearch 提供「int8／int4／bbq」，pgvectorscale 則配上自研的「Statistical Binary Quantization」。base index 與 quantization 本該是兩軸，seed 卻塞進一格。

## 選型對照：需求 → 產品短名單

把新表濃縮成一句話定義配一個例子，方便直接對照。

- **想少維運一個系統、資料量中小、團隊已在 Postgres 上**：選 pgvector；例子——已有 Postgres 的 RAG，直接加 `vector` 欄與 HNSW 索引，授權是寬鬆的 PostgreSQL License，不必多養一個服務。資料長大且記憶體吃緊時，再加 pgvectorscale 把索引換成落盤的 StreamingDiskANN。
- **在乎尾延遲、要 on-disk 也要 managed**：選 Qdrant；例子——見上表固定 99% recall 下 Qdrant 的 p95 是 36.73 ms（低於 Postgres 的 60.42 ms），且支援 mmap on-disk 儲存與託管雲。
- **要最廣的 index 選擇、GPU、十億級並用物件儲存**：選 Milvus；例子——同一套裡可在 IVF_FLAT、HNSW、DiskANN、ScaNN 間換索引，段落資料落在 object storage，適合資料量與索引策略都會變的重負載。
- **要真 hybrid（BM25＋向量）而且不想自己拼**：選 Elasticsearch 或 Weaviate；例子——Elasticsearch 用 RRF 直接融合「a traditional BM25 query and an ELSER query」，Weaviate 同樣以 RRF 併稠密與關鍵字兩路。
- **要 serverless、按量計費、資料超過單機記憶體**：選 Pinecone 或 Turbopuffer；例子——Pinecone 把索引主體放進「distributed object storage」、查詢時才拉需要的片段，Turbopuffer 更是「object-storage native」，兩者都讓「機器裝不下的資料量」與「按用量計費」變可能。
- **要嵌入式、跟資料湖同格式**：選 LanceDB；例子——它「built on top of the Lance columnar format」、可「runs locally or in your cloud」，向量與 full-text 檢索並存，適合把向量檢索直接壓進資料管線。

但這張表只給方向、不給排名：「誰在你的負載上更快」任何現成表格都答不了，只能在你自己的 dataset、固定 recall、同一台機器上實測 recall／QPS 曲線與尾延遲。

## 🧪 Critical Assessment

### 新表能給的是可稽核的方向，不是效能排名
新表每格都綁了來源，但它本質是「能力矩陣」而非「效能榜」。刻意拿掉效能欄不是偷懶，而是因為如同 S1／S2 反覆示範，沒有 recall floor 與 hardware 的 QPS 不可重現；seed 的 perf 欄不是「舊」，而是從一開始就量錯了維度。讀者要排名，得自己做上一節那種固定 recall 的實測。

### 一手來源之間彼此不可比，連「做對」的例子也帶偏誤
本稽核用到的量化來源設定互不相容：S1 量單機 in-memory library、S2 量固定硬體上的競賽 recall、TigerData 量 50M 級 client-server 吞吐與尾延遲，三者不能並列成一張效能表。廠商 benchmark 幾乎都自家贏（VectorDBBench 由 Zilliz 贊助、Qdrant benchmark 自稱幾乎全場最佳），任何「誰最快」都須存疑；連我引用的 TigerData 也自承偏袒 Postgres，只是它把方法標透明而已。

### 新表本身的洞與可信度不均
表裡有不少 `unknown` 與 ‡ 推論格，它們不是能力缺席、而是我在一手來源裡找不到可逐字回溯的證據：Chroma 的 quantization／RBAC、Turbopuffer 與 LanceDB 的 index／RBAC 都留白，寧可空著也不填 folklore。少數格（如 pgvector「繼承 Postgres 角色」的 RBAC、Pinecone 的 proprietary 授權）是合理推論，證據力弱於逐字引用。Milvus 整列改引 Zilliz 官方頁而非 milvus.io，供應商一致但非該產品參考文件本身；ParadeDB 的 hybrid 依其 README 仍是 roadmap，都是讀表時要打折的地方。

## 🔗 Related notes

- [BM25](../../natural_language_processing/information_retrieval/BM25/) — hybrid search 的 sparse／keyword 那一路就是 BM25，本筆記對「hybrid = dense + BM25 融合」的定義即以此為基礎。
- [TF-IDF](../../natural_language_processing/information_retrieval/TFIDF/) — 稀疏檢索的前身，理解 sparse-dense 融合的權重取捨時的背景。
