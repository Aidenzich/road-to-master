# Bigger is not Always Better: Scaling Properties of Latent Diffusion Models — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Bigger is not Always Better: Scaling Properties of Latent Diffusion Models |
| Venue | TMLR (Transactions on Machine Learning Research) |
| Year | 2024 |
| Authors | Kangfu Mei, Zhengzhong Tu, Mauricio Delbracio, Hossein Talebi, Vishal M. Patel, Peyman Milanfar |
| Official Code | unknown |
| Venue Kind | paper |

> 說明：本文為 arXiv 2404.01367 之全文（TMLR 已接受版，LaTeX source 內 `\usepackage[accepted]{tmlr}`、`\def\year{2024}`、OpenReview `forum?id=0u7pWfjri5`）。作者群橫跨 Johns Hopkins、Texas A&M 與 Google，主體實驗於 Google 內部資料與 TPUv5 完成。

## First Principles

### 這篇論文到底在問什麼

擴散模型（diffusion model）的致命傷是**取樣效率（sampling efficiency）**：一張圖要跑很多步去噪才會清晰，總成本 = 取樣步數 × 每步成本。過去加速的兩條路線分別針對這兩個因子——設計更快的網路架構壓低「每步成本」、設計更好的取樣器（sampler）壓低「步數」。這篇論文指出第三個被忽略的變數：**模型大小本身**。它問的是一個很實務的問題：**在固定的推論預算（inference budget）下，我該用大模型還是小模型？**

作者的答案顛覆直覺，也就是標題「Bigger is not Always Better」：在受限的取樣預算下，較小的模型反而常常贏過較大的模型。

### 實驗設定：把 Stable Diffusion 當標尺往下縮

論文以 `866M` 的 Stable Diffusion v1.5 為基準，只改動去噪 U-Net residual block 的基礎通道數 $c$，並讓整體維持 $[c, 2c, 4c, 4c]$ 的比例，其餘架構元件不動，藉此得到一個從 `39M` 到 `5B` 共 12 個模型的家族。所有模型都在約 6 億張經美學過濾（aesthetically-filtered）的內部 text-to-image 資料（WebLI）上，以 batch size 2048、learning rate 1e-4、訓練 500K steps 從頭訓練。這種「只縮通道、其他不變」的設計，讓不同大小之間的比較是乾淨的受控變因（controlled scaling）。

Table 1 是全篇的骨幹，它同時給出架構規格與 50 步 DDIM（CFG=7.5）下於 COCO-2014 驗證集（30k 樣本）量測的 pretraining 品質：

| Params | 39M | 83M | 145M | 223M | 318M | 430M | 558M | 704M | 866M | 2B | 5B |
|-|-|-|-|-|-|-|-|-|-|-|-|
| Filters $c$ | 64 | 96 | 128 | 160 | 192 | 224 | 256 | 288 | 320 | 512 | 768 |
| GFLOPS | 25.3 | 102.7 | 161.5 | 233.5 | 318.5 | 416.6 | 527.8 | 652.0 | 789.3 | 1887.5 | 4082.6 |
| Norm. Cost | 0.07 | 0.13 | 0.20 | 0.30 | 0.40 | 0.53 | 0.67 | 0.83 | 1.00 | 2.39 | 5.17 |
| FID ↓ | 25.30 | 24.30 | 24.18 | 23.76 | 22.83 | 22.35 | 22.15 | 21.82 | 21.55 | 20.98 | 20.14 |
| CLIP ↑ | 0.305 | 0.308 | 0.310 | 0.310 | 0.311 | 0.312 | 0.312 | 0.312 | 0.312 | 0.312 | 0.314 |

有一個容易誤會的地方值得先釐清：這裡的 GFLOPS、成本與參數量**只計算 latent 空間裡的去噪 U-Net**，不含 `1.4B` 的 text encoder 與 `250M` 的 latent encoder/decoder。所以「39M 模型」指的是去噪網路 39M，實際部署時仍背著固定的 encoder/decoder 開銷——這點在解讀「小模型多省」時要放在心上。

從 Table 1 單看 pretraining，結論其實是「越大越好」：50 步滿預算下 FID 從 39M 的 25.30 單調下降到 5B 的 20.14（39M 是唯一的例外離群點），而 CLIP 早在 ~0.312 就飽和。下面三張 50 步 DDIM 的 text-to-image 結果也印證越大細節越好：

| 39M | 866M | 2B |
|-|-|-|
| ![39M 模型 50 步結果](imgs/t2i_39M.jpg) | ![866M 模型 50 步結果](imgs/t2i_866M.jpg) | ![2B 模型 50 步結果](imgs/t2i_2B.jpg) |

那「小模型更好」是怎麼冒出來的？關鍵在於把 x 軸從「參數量」換成「**取樣成本（sampling cost）= normalized cost × sampling steps**」。

### 核心機制：固定預算下的取樣成本換算（worked example）

定義取樣成本 $\text{cost} = (\text{Norm. Cost}) \times (\text{steps})$。給定一個固定預算，每個模型能負擔的步數不同：小模型每步便宜，就能多跑幾步；大模型每步貴，同樣預算下只能跑很少步。

以論文 Fig. 7（`analyze_inference_costs`）明確點名的**固定預算 cost = 3** 為例，用 Table 1 的 Norm. Cost 逐一換算各模型在此預算下可跑的步數：

- `866M`（Norm. Cost 1.00）：$3 / 1.00 = 3$ 步 —— 只有 3 步，去噪嚴重不足，影像糊。
- `318M`（Norm. Cost 0.40）：$3 / 0.40 \approx 7$ 步。
- `145M`（Norm. Cost 0.20）：$3 / 0.20 = 15$ 步。
- `83M`（Norm. Cost 0.13）：$3 / 0.13 \approx 23$ 步 —— 步數充足，能把去噪軌跡跑到接近收斂。

論文的觀察是：在 cost = 3 時，**`83M` 模型拿到所有模型中最好的 FID**。直覺上，取樣品質是「模型容量」與「去噪步數是否足夠」的乘積效應；在小預算區間，步數不足對大模型的傷害，遠大於容量不足對小模型的傷害，於是小模型勝出。反過來，當預算放寬（步數對大模型也夠用了），容量優勢才會浮現，大模型重新領先並在細節生成上超車。這就是標題所謂 bigger is not *always* better——是「在受限預算下」不是。

論文進一步用一系列軸驗證這個 scaling sampling-efficiency 的穩健性：

1. **取樣器無關**：把 DDIM 換成隨機的 DDPM 或高階的 DPM-Solver++，小模型在同一取樣成本下更省的趨勢都成立（DPM-Solver++ 因設計不適合超過 20 步，只測 ≤20 步）。
2. **下游任務（少步數時）成立**：在 4× 真實世界超解析度（real-world super-resolution）上，當步數 ≤ 20 時小模型仍較省；但步數 > 20 後，大模型反而更有效率。
3. **蒸餾後仍成立**：以 conditional consistency distillation 把各模型蒸餾成 4 步取樣，蒸餾對每個模型都帶來約 $5\times$ 的一致加速並全面改善 FID；但在 sampling cost ≈ 8 時，**未蒸餾的 83M 小模型仍能追平已蒸餾的 866M 大模型**——顯示蒸餾並未推翻 scaling 趨勢。

同時，有一條與效率結論方向相反、但同樣重要的發現：**下游品質由 pretraining 決定**。在超解析度上，FID 主要受模型大小驅動而非 finetuning 的訓練量，小模型即使多訓練也補不齊大模型 pretraining 帶來的品質差距（Fig. 4 顯示大 SR 模型即使短暫 finetune 也勝過小模型）。所以本文並非鼓吹「一律用小模型」，而是把選擇條件化在「你的推論預算落在哪一段」以及「你要的是低步數快取樣還是最終極致品質」。

## 🧪 Critical Assessment

### 取樣延遲是真痛點，12 個受控模型的稀缺資料本身即是貢獻

取樣效率確實是擴散模型落地的真痛點，論文動機（行動裝置上 50 步 DDIM 延遲過高）站得住腳。而「模型大小 × 取樣預算」這個交互作用，過去的加速研究（改架構、改取樣器、蒸餾）確實較少正面處理，切入點是實在的。更難得的是，作者從頭訓練了 12 個 39M–5B 的模型，這種規模的受控實驗一般團隊做不起（需要 TPU 叢集、數週與數十萬美元成本），資料點的稀缺性本身就是貢獻。

### FID 是唯一裁判，成本又只算 U-Net 而略過 1.4B text encoder 與 250M autoencoder

受控 scaling（只改通道數）是乾淨的設計，取樣器（DDIM/DDPM/DPM-Solver++）與蒸餾兩條 robustness 軸也補得誠實。但有三個結構性弱點需要點名。其一，**指標幾乎全靠 FID**：作者自己在 limitations 承認因為變體超過 1000 種而放棄人評（human evaluation），並坦言 FID 與視覺品質可能背離。整篇「小模型更省」的結論本質上是「小模型在 FID 上更省」，而 FID 對取樣多樣性/模糊度的敏感方式，恰好可能系統性地偏袒步數足、較平滑的小模型輸出，這個潛在偏差沒有被獨立指標交叉檢驗。其二，**成本定義只算去噪 U-Net**，把固定的 1.4B text encoder 與 250M autoencoder 開銷排除在 normalized cost 之外；在真實端到端延遲裡，這塊固定開銷會稀釋小模型「便宜」的相對優勢，論文的 cost 軸因此對小模型偏樂觀。其三，資料與模型皆為 Google 內部、未釋出程式碼，外部無法重現這 12 個模型或驗證 Table 1 的數字。

### 「越大越好」是既有的預訓練結論，成本軸上的反轉才是真正的新意

需要保持清醒：Nichol et al. 早已指出擴散模型「越大越好」，而 LLM 的 scaling law 文獻（Kaplan、Hoffmann 等）也早就在講 compute-optimal 的取捨。本文的 pretraining 部分（Table 1、Fig. 3）基本是在重述「越大 FID 越低」。真正的新意集中在把 x 軸換成「取樣成本」後浮現的反轉現象，以及它在取樣器/下游/蒸餾三軸的一致性。這是一個有價值但相對侷限的觀察——它更接近「一個穩健的實證規律」，而非可外推的定量 scaling law（論文並未給出可預測最佳模型大小的閉式關係）。

### 反轉點只在自訂的 normalized-cost 與 FID-only 座標下才成立

有一個需要警惕的地方：最亮眼的結論（cost=3 時 83M 最佳、cost≈8 時 83M 追平蒸餾 866M）都是在作者自訂的 normalized-cost 座標、且以 FID 為唯一裁判下成立的。換個成本定義（納入 encoder/decoder）、換個品質指標（人評或多樣性指標），反轉點很可能位移甚至消失。論文也誠實地把普適性收斂到「僅對本文研究的 SD v1.5 U-Net 家族成立」，明言 transformer backbone（DiT/SiT/MM-DiT）與 cascaded 模型（Imagen3、Stable Cascade）未驗證——這是恰當的自我設限，但也意味著結論的適用面比標題的普遍口吻要窄。

### 作為低預算部署的選型指引可用，但缺乏端到端延遲與定量選型的驗證

作為「決策指引」它是有用的：若你的部署卡在低步數/低預算區間，優先選較小模型是有實證支撐的合理策略，且此結論對取樣器與蒸餾都穩健。但它沒有解決「如何在給定硬體與品質門檻下自動選出最佳模型大小」的定量問題，也未在端到端真實延遲、真實指標下驗證優勢的量級。因此我的判斷是：這是一份紮實、誠實、但結論條件化且指標單一的實證研究，價值在於提出並穩健化一個反直覺規律，而非提供可外推的 scaling 定律。它沒有無病呻吟，但也不宜被讀成「小模型全面更好」。

## 🔗 Related notes

- [DDPM, Denoising Diffusion Probabilistic Models](../diffusion/) — 本文取樣效率討論所依賴的去噪擴散基礎與 DDPM/DDIM 取樣器。
