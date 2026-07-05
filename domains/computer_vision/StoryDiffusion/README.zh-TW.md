# StoryDiffusion — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation |
| Venue | NeurIPS |
| Year | 2024 |
| Authors | Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, Qibin Hou |
| Official Code | https://github.com/HVision-NKU/StoryDiffusion |
| Venue Kind | paper |

> 本篇筆記基於 arXiv 預印本 `2405.01434v1` 的全文與作者釋出的官方程式碼撰寫；論文正式發表於 NeurIPS 2024（Spotlight），camera-ready 版本的細節可能與預印本略有出入。

## First Principles

StoryDiffusion 想解決的是一個很具體的痛點：用擴散模型（diffusion model）畫一則「故事」時，同一個角色跨越好幾張圖必須長得一樣——臉、髮型、衣著都要一致，否則就不成故事。論文把這個需求拆成兩件事來做：第一階段用一個免訓練（training-free）的注意力改造，讓「一批」圖之間互相看得到彼此，藉此收斂出一致的角色；第二階段再用一個在語意空間做運動預測的模組，把這些一致的圖串成過場影片。整體方法分為這兩個階段，第一階段以 Consistent Self-Attention 生成主體一致的圖像，第二階段則把這些圖像轉為過場影片。

![StoryDiffusion 第一階段：把 Consistent Self-Attention 插入預訓練 T2I 擴散模型，對一個 batch 內的多張圖建立連結](imgs/pipeline_csa.png)

### 為什麼從 self-attention 下手

作者的核心動機來自一個觀察：self-attention 是決定生成內容整體結構最關鍵的模組之一，而且注意力權重是「輸入相依」（input-dependent）的——既然權重由輸入的 token 決定，只要在計算注意力時「餵進」參考圖的 token，理論上就能把參考圖的一致性帶進來，而不必重新訓練或微調模型。這是整個方法之所以能 zero-shot 的立論基礎：如果能用一張參考圖去引導 self-attention 的計算，兩張圖之間的一致性就會顯著提升。

相對地，論文把既有做法當作反例來對照。IP-Adapter 這類以整張參考圖為條件的方法，因為引導太強，會反過來壓縮文字 prompt 對生成內容的可控性；而 InstantID、PhotoMaker 這類 ID 保存方法雖然守得住身份，卻不保證衣著與場景一致。StoryDiffusion 的目標是「同時」守住身份與衣著的一致，又盡量保留文字可控性。

### Consistent Self-Attention 的形式化

給定一個 batch 的影像特徵 $\mathcal{I} \in \mathbb{R}^{B \times N \times C}$，其中 $B$、$N$、$C$ 分別是 batch size、每張圖的 token 數與通道數。定義注意力函數 $\operatorname{Attention}(X_q, X_k, X_v)$。原始 self-attention 是在每張圖自己的特徵 $I_i$ 內獨立計算的：

$$
O_i = \operatorname{Attention}\left(Q_i, K_i, V_i\right).
$$

Consistent Self-Attention 的做法是：對第 $i$ 張圖，先從 batch 內「其他」圖的特徵隨機抽樣一批 token $S_i$：

$$
S_i = \operatorname{RandSample}\left(I_1, I_2, ..., I_{i-1}, I_{i+1}, ..., I_{B-1}, I_{B}\right),
$$

再把抽到的 $S_i$ 與本圖特徵 $I_i$ 拼成一組新的 token $P_i$，對 $P_i$ 做線性投影得到新的 key $K_{Pi}$ 與 value $V_{Pi}$，而 query 仍維持本圖原本的 $Q_i$ 不變。最後計算：

$$
O_i = \operatorname{Attention}\left(Q_i, K_{Pi}, V_{Pi}\right).
$$

三條式子擺在一起，關鍵差異只有一處：key/value 的來源池從「本圖 $N$ 個 token」擴大成「本圖 + 抽樣自其他圖的 token」，而 query 完全沒動。因為 $Q$-$K$-$V$ 投影權重是沿用原模型的，沒有新增任何參數，所以整套操作免訓練、可熱插拔（hot-pluggable）。直覺上，這樣的跨圖互動會推動模型在生成過程中讓角色的臉、衣著等特徵彼此收斂。

### 用 tile 與 sliding window 撐住長故事

論文附上 Algorithm 1 的偽碼，補上了工程上兩個現實考量：一是用 `tile_size` 把 token 切塊處理，避免一次算全部造成 GPU 記憶體爆掉；二是沿時間維度滑動 tile，讓峰值記憶體不再隨故事文字長度線性成長，因此可以生成長故事。

```python
def ConsistentSelfAttention(images_features, sampling_rate, tile_size):
  output = zeros(B, N, C), count = zeros(B, N, C), W = tile_size
  for t in range(0, N - tile_size + 1):
    # 用 tile 分塊，避免超出 GPU 記憶體
    tile_features = images_tokens[t:t + W, :, :]
    reshape_featrue = tile_feature.reshape(1, W*N, C).repeat(W, 1, 1)
    sampled_tokens = RandSample(reshape_featrue, rate=sampling_rate, dim=1)
    # 把其他圖抽到的 token 與原 token 串接
    token_KV = concat([sampled_tokens, tile_features], dim=1)
    token_Q = tile_features
    X_q, X_k, X_v = Linear_q(token_Q), Linear_k(token_KV), Linear_v(token_KV)
    output[t:t+w, :, :] += Attention(X_q, X_k, X_v)
    count[t:t+w, :, :]  += 1
  output = output/count
  return output
```

值得對照的是官方程式碼怎麼落實這個「RandSample」。在 `utils/gradio_utils.py` 的 `cal_attn_mask_xl` 中，抽樣其實是用一個布林遮罩實現的：`bool_matrix1024 = torch.rand((1, total_length * 1024)) < sa32`，也就是對「整個 batch 攤平後」的 token 逐一以機率 `sa32` 做 Bernoulli 取樣，被留下的 token 才會參與該解析度的注意力。換句話說，論文式 (2) 的隨機抽樣，在實作上等價於一張隨機的注意力遮罩，`sa32` / `sa64` 就是不同解析度下的取樣率。`app.py` 的 `SpatialAttnProcessor2_0` 還加了一個時間表排程：`cur_step < 5` 的前幾步完全走原始 self-attention（`__call2__`），之後才有機率切換到一致化路徑（`__call1__`），且套用一致化的機率門檻在第 20 步前後由 `1-0.3` 收緊到 `1-0.1`，讓一致性在去噪後期更強地介入。

### 一個具體的前向例子

把數字帶進去會更清楚。假設要畫一則 5 格漫畫（官方程式碼裡對應 `id_length=4`、`total_length=id_length+1=5`）。在 SDXL 生成 1024×1024 圖時，U-Net 某個下採樣階段的特徵圖是 32×32，也就是每張圖 $N = 32\times32 = 1024$ 個 token（程式碼以 `(height//32)*(width//32)` 判斷這個解析度）。

- **標準 self-attention**：第 $i$ 格的 query（1024 個 token）只能對自己的 1024 個 key/value 做注意力，注意力矩陣是 $1024\times1024$，五格之間毫無資訊往來，於是各畫各的、角色飄移。
- **Consistent Self-Attention**：把 5 格攤平成 $5\times1024 = 5120$ 個 token 的池子，以取樣率 0.5 的 Bernoulli 遮罩留下約 $0.5\times5120 \approx 2560$ 個可見 token 當作 key/value。於是第 $i$ 格的 query 除了看自己的 1024 個 token，還看得到另外四格抽樣出來的 token——別格的臉部與衣著 token 被引入計算，把本格往共同的角色外觀拉。因為只擴大了 key/value 池、query 與投影權重都沒變，這一切都不需要訓練。

論文的消融顯示這個取樣率不能太低：取樣率 0.3 時第三欄的圖已守不住主體一致性，較高的取樣率才守得住；實務上預設把取樣率設為 0.5，以對擴散過程造成最小干擾同時維持一致性。

![消融實驗：(a) 不同取樣率對一致性的影響；(b) 引入外部 ID 控制](imgs/ablation.png)

### Semantic Motion Predictor：在語意空間預測運動

第二階段要把相鄰兩張一致圖之間補出中間幀，變成過場影片。論文先指出既有方法（SEINE、SparseCtrl）的問題：它們只靠時間模組在影像 latent 空間逐一空間位置獨立地預測中間內容，缺乏對空間資訊的整體考量，因此當首尾兩幀差異很大（例如角色大幅移動）時就接不穩、產生崩壞的中間幀。

![StoryDiffusion 第二階段：把條件圖編碼進語意空間預測過場 embedding，再由影片擴散模型當解碼器](imgs/pipeline_smp.png)

Semantic Motion Predictor 的對策是把預測搬到「影像語意空間」做。給定起始幀 $F_s$ 與結束幀 $F_e$，先用一個編碼器 $E$（論文用預訓練的 CLIP image encoder，取其 zero-shot 能力）壓成語意向量：

$$
K_s, K_e = E\left(F_s, F_e\right).
$$

接著在語意空間先對 $K_s$、$K_e$ 做線性插值，展開成長度為 $L$ 的序列 $K_1, K_2, ..., K_L$，再送進一串 transformer 區塊 $B$ 預測過場幀：

$$
P_1, P_2, ..., P_l = B\left(K_1, K_2, ..., K_L\right).
$$

最後把這些語意 embedding 當成控制訊號、以影片擴散模型當解碼器：對每個影片幀特徵 $V_i$，把文字 embedding $T$ 與預測的語意 embedding $P_i$ 串接後投影成 key/value 餵進 cross-attention：

$$
V_i = \mathrm{CrossAttention}\left(V_i, \operatorname{concat}(T, P_i), \operatorname{concat}(T, P_i)\right),
$$

並以預測影片與 ground truth 之間的 MSE 為訓練損失 $Loss = \mathrm{MSE}(G, O)$。具體規格上，這個預測器用 OpenCLIP ViT-H-14 當編碼器、以 AnimateDiff V2 的 motion module 為時間模組初始權重，含 8 層 transformer、hidden 維度 1024、12 個注意力頭，學習率 1e-4，在 Webvid10M 上以 8 張 A100 訓練 100k 次迭代。

### 實驗數據

一致圖像生成上，論文用 GPT-4 生成 20 個角色 prompt 與 100 個活動 prompt 交叉組合成測試集，在 Stable Diffusion XL 上與 IP-Adapter、PhotoMaker 對比，全部用 50 步 DDIM 取樣、classifier-free guidance 5.0。以 CLIP 分數量測文字對齊與角色一致：

| Metric | IP-Adapter | Photo Maker | StoryDiffusion (ours) |
|-|-|-|-|
| Text-Image Similarity | 0.6129 | 0.6541 | **0.6586** |
| Character Similarity | 0.8802 | 0.8924 | **0.8950** |

過場影片生成上，隨機抽約 1000 支影片為測試集，與 SEINE、SparseCtrl 對比四個指標：

| Methods | LPIPS-*first* (↓) | LPIPS-*frames* (↓) | CLIPSIM-*first* (↑) | CLIPSIM-*frames* (↑) |
|-|-|-|-|-|
| SEINE | 0.4332 | 0.2220 | 0.9259 | 0.9736 |
| SparseCtrl | 0.4913 | 0.1768 | 0.9032 | 0.9756 |
| Ours | **0.3794** | **0.1635** | **0.9606** | **0.9870** |

此外還有 30 人、每人 50 題的 user study：一致圖像生成上使用者偏好 StoryDiffusion 的比例為 72.8%（IP-Adapter 10.4%、PhotoMaker 16.8%）；過場影片生成上為 82%（SEINE 11.6%、SparseCtrl 6.4%）。

## 🧪 Critical Assessment

### 跨圖角色一致是把 T2I 推向漫畫與分鏡時的真痛點

「跨圖角色一致」確實是把 T2I 模型推向漫畫、繪本、分鏡等實際應用時的真痛點，這一點無須刻意抬舉也站得住：既有的 IP-Adapter 會犧牲文字可控性、PhotoMaker 會漏掉衣著一致，都是使用者實際會踩到的坑。更值得肯定的是，方法選了一條「免訓練、可插拔」的低成本路線，對只有推論資源的人特別友善。這個定位是這篇工作最扎實的價值所在。

### 用 CLIP 當一致性裁判的偏差與 0.003 量級的差距

這是我認為最需要打折的部分。一致圖像生成的量化只比了 IP-Adapter 與 PhotoMaker 兩個基線，而且兩個核心指標 Text-Image Similarity 與 Character Similarity **都是用 CLIP 分數算的**——注意 Character Similarity 量的是「角色圖之間的 CLIP 相似度」，而 CLIP embedding 本來就偏向抓語意/風格而非細粒度身份，用它當「一致性」的裁判，天生對「讓多張圖看起來相似」的方法有利。更關鍵的是，StoryDiffusion 在 Character Similarity 上只贏 PhotoMaker 0.8950 對 0.8924（差 0.0026）、Text-Image Similarity 贏 0.6586 對 0.6541（差 0.0045），差距落在幾乎可忽略的量級，論文卻沒有給任何顯著性檢定或多次執行的變異數。單看這張表，很難說是壓倒性勝出。

### 取樣率消融只掃兩點，排程與 tile_size 無對應消融

消融也偏薄：取樣率只掃了 0.3 與「較高值」兩檔，最後直接宣稱 0.5 是最佳，但沒有 0.4、0.6、0.7 的曲線，「0.5 為最佳」比較像經驗選點而非掃描結論；`tile_size`、以及程式碼裡那個「前 5 步走原始注意力、後期收緊門檻」的排程都沒有對應消融，讀者無從判斷這些設計各自貢獻多少。

### CSA 與既有跨圖/跨幀 KV 共享的關係

要誠實地問：Consistent Self-Attention 到底新在哪裡。它本質上是把 self-attention 的 key/value 從單圖擴展到 batch 內多圖——這種「跨圖/跨幀共享 KV」的想法在影片與參考生成領域（例如各種 reference/extended attention）並不算全新，論文的貢獻更多在於「用最省的方式（隨機抽樣 + 沿用權重 + 免訓練）把它用在 storytelling 這個場景並做通」。這是紮實的工程整合與場景落地，但把它敘述成一種全新的注意力機制，稍微高估了機制層面的新穎性。Semantic Motion Predictor 同理，是「CLIP 語意編碼 + transformer 內插 + IP-Adapter 式 cross-attention 解碼」三個現成積木的組合。

### 自建 prompt 集與 user study 壓倒性偏好的落差

測試 prompt 由作者用 GPT-4 自行生成、測試影片自行抽樣，沒有採用社群公認的公開基準，評測集實質上是圍繞方法自身情境所定義的，缺少第三方可複現的固定基準來對齊。加上唯一的「人類判準」是 user study，而 72.8% / 82% 這種壓倒性偏好與量化指標上僅 0.003 量級的差距形成強烈反差——這反差本身就提示：CLIP 指標可能量不出人眼真正在意的一致性，或者 user study 的呈現方式（例如挑選展示樣本）放大了差距。兩者都指向「評測設計偏向有利於本方法」的疑慮，論文並未正面處理。

### 落地價值扎實，但一致性只守到臉與大件衣著

務實地看，方法確實在「可插拔、對既有 SD1.5/SDXL 生態零門檻」這點上有很強的真實世界關聯，官方程式碼與 Gradio demo 也降低了落地門檻，這是它能被社群廣泛採用的實際原因。但「問題被解決」要打個折：論文自陳的限制已點出兩個硬傷——細節（如領帶等小配件）仍會不一致、需要更細的 prompt 補救；而長影片因為缺乏全域資訊交換，sliding window 只是權宜，並非為長影片而設計。一致性守在「臉與大件衣著」層級是可信的，但守到「逐一細節」則尚未被證成。

## 🔗 Related notes

- [DDPM](../diffusion/)
- [Wan 2.2](../wan/)
