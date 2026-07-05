# shapiq — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | shapiq: Shapley Interactions for Machine Learning |
| Venue | NeurIPS 2024 (Datasets & Benchmarks Track) |
| Year | 2024 |
| Authors | Maximilian Muschalik, Hubert Baniecki, Fabian Fumagalli, Patrick Kolpaczki, Barbara Hammer, Eyke Hüllermeier |
| Official Code | https://github.com/mmschlk/shapiq |
| Venue Kind | paper |

## First Principles

### 從 Shapley 值到 Shapley 交互作用

合作賽局理論把一個「賽局」定義為冪集上的價值函數 $\nu: \mathcal{P}(N) \rightarrow \mathbb{R}$（其中 $\nu(\emptyset)=0$），描述 $n$ 個實體（玩家）在所有可能聯盟下的產出。Shapley 值（SV）是唯一同時滿足 linearity、dummy、symmetry 與 efficiency 的 semivalue：它把整體產出 $\nu(N)$ 公平地分配到每個個別玩家。在機器學習裡，這個框架被反覆用來做特徵歸因（feature attribution）、全域特徵重要性與資料估值（data valuation）。然而 SV 只落在單一玩家身上，它 does not give insights on synergies or redundancies —— 例如「緯度」與「經度」兩個特徵各自看似獨立，唯有把它們一起考慮才顯露出「精確位置」這個綜效。shapiq 這篇論文的核心，就是把 SV 延伸到能描述「一群實體」聯合貢獻的 Shapley 交互作用（SI），並提供一個開源 Python 套件把相關演算法統一起來。

![shapiq 將賽局理論、近似演算法與預先計算的基準連成一體，並提供解釋任意階特徵交互作用的介面](imgs/diagram_shapiq.png)

SV 與 Banzhaf 值（BV）都可寫成 marginal contribution $\Delta_i(T) := \nu(T \cup \{i\}) - \nu(T)$ 的加權平均，差別只在權重：

$$
\phi^{\text{SV}}(i) := \sum_{T \subseteq N \setminus \{i\}} \frac{1}{n \binom{n-1}{\vert T \vert}} \Delta_i(T)
\quad\text{以及}\quad
\phi^{\text{BV}}(i) := \sum_{T \subseteq N \setminus \{i\}} \frac{1}{2^{n-1}} \Delta_i(T)
$$

要把「價值」推廣到一群實體 $S$，論文採用 interaction index（II）路線：它以 discrete derivative 為基礎，並且會扣掉 $S$ 的所有子集所貢獻的低階效果。以一對玩家 $i,j$ 為例，$\Delta_{\{i,j\}}(T)$ 等於聯合 marginal contribution $\nu(T\cup\{i,j\})-\nu(T)$ 再減去各自的 $\Delta_i(T)$ 與 $\Delta_j(T)$。一般化後，A positive value indicates synergy, whereas a negative value indicates redundancy of $S$；為零則代表可加性獨立。其定義為：

$$
\Delta_S(T) := \sum_{L \subseteq S} (-1)^{\vert S \vert-\vert L \vert} \nu(T \cup L)
$$

### 交互作用階數與效率公理

SI 的關鍵設計，是引入一個 explanation order $k$，只對大小 $\vert S\vert \le k$ 的聯盟分配聯合貢獻 $\Phi_k(S)$，並要求它們滿足廣義的 efficiency，即所有階數的貢獻總和仍回到整體產出：

$$
\nu(N) = \sum_{S\subseteq N, \vert S \vert \leq k}\Phi_k(S)
$$

在這個框架下有數個具體指標。The $k$-SII are the unique SI that coincide with SII for the highest order（即 $k$-Shapley Values 與 SII 在最高階一致）；STII 把重心放在最高階交互作用；FSII 則直接最佳化 Shapley-weighted faithfulness。$k=1$ 時所有 SI 都退化為 SV，$k=n$ 時則變成 Möbius 交互作用（MI），對所有聯盟的價值完全忠實（faithfulness loss 為 0）。換言之，SI 提供一條從「最簡單的 SV」到「最完整的 MI」的複雜度—精確度光譜。

### shapiq 套件的組成

shapiq 把上述理論落成三類可組合的元件。在近似器方面，We implement 7 algorithms for approximating SI across 4 different interaction indices, and another 7 algorithms for approximating SV，並用共用的 `shapiq.CoalitionSampler` 介面統一 border-trick、pairing-trick 等取樣加速。在精確計算方面，`shapiq.ExactComputer` 提供 computing 18 interaction indices and game-theoretic concepts（含 MI）的能力，可作為評估近似器的 ground truth。在賽局方面，`shapiq.Game` 定義了 11 個基準賽局、共 100 個獨立賽局實例（applications × dataset–model 配對），並預先計算與分享了 $2\,042$ 個賽局設定的精確 SI 值。

下表整理套件的三大元件與規模（數字取自論文正文與 Table 1–3）：

| 元件類別 | 代表實作 | 規模 |
|-|-|-|
| Approximator | KernelSHAP-IQ、SVARM-IQ、Permutation Sampling… | 7 個 SI 近似器 + 7 個 SV 近似器（4 種交互作用指標） |
| ExactComputer | Möbius Converter、Exact Computer | 精確計算 18 個交互作用指標與賽局概念 |
| Game（基準） | Local/Global/Tree Explanation、Data Valuation… | 11 個基準賽局、100 個賽局實例、2 042 個預算好的設定 |

### 用 shapiq 解釋一筆預測（程式介面）

論文示範了 `shapiq.Explainer` 的最小用法：建立解釋器時指定最高階 `max_order=3`，再對單一樣本以固定的評估預算做近似：

```python
import shapiq
# 建立一個最高到 3 階交互作用的解釋器
explainer = shapiq.Explainer(model=model, data=X, max_order=3)
x = X[0]
# 在 budget=1024 次價值函數呼叫的預算下近似特徵交互作用
interaction_values = explainer.explain(x=x, budget=1024)
# 取出 3 階交互作用並以網路圖視覺化
interaction_values.get_n_order_values(3)
interaction_values.plot_network(feature_names=...)
```

### 一個具體的前向例子（Vision Transformer，16 個 patch）

以論文基準中的影像分類賽局 `vit_16_patches` 為例：把一張影像切成 16 個 patch，每個 patch 當成一個玩家，因此 $n=16$。要精確算出所有聯盟的價值，需要評估 $2^{16} = 65\,536$ 個聯盟——這正是論文對 $n \le 16$ 的賽局選擇「預先計算並存檔」的原因。

順著這個例子把階數展開（以下組合數為本文推導）：若只取 $k=2$，需要輸出 $16$ 個單體加上 $\binom{16}{2}=120$ 個配對，共 $136$ 個交互作用值；取到範例程式的 `max_order=3` 則再加 $\binom{16}{3}=560$ 個三元組，達 $696$ 個值，且它們必須依 efficiency 公理加總回模型輸出。而範例只給 `budget=1024` 次評估，約為完整 $65\,536$ 次的 $1.6\%$——這就是為什麼即使小到 16 個玩家，實務上仍要依賴 shapiq 的近似器而非窮舉。

### 跨領域基準測試的發現

論文用這 100 個賽局比較各近似器，最醒目的結論是 the ranking of approximators varies strongly between the different applications domains：沒有單一演算法全面稱王。作者進一步歸納出兩個家族——stratification-based estimators perform superior in settings where the size of a coalition naturally impacts its worth（例如 Dataset Valuation 的訓練集大小），而 kernel-based estimators achieve state-of-the-art in settings where the dependency between size and worth of a coalition is less pronounced（例如 Local Explanation 的預測突跳）。對實務者，論文建議 $k$-SII is a good default choice for shapiq，因為它與 SII 一致且在 $k=2$ 時就對 SV 有明顯的忠實度提升。

## 🧪 Critical Assessment

### 問題是否真實且重要

「SV 無法表達綜效」這個出發點是站得住腳的：論文引述大量文獻指出 the SV is limited when explaining complex decision systems，而緯度／經度這類例子直覺上也確實需要聯合考量。把 SV 推廣到任意階交互作用、並用統一 API 讓實務者取用，是一個有真實需求的工程與研究缺口——既有的 `shap` 只提供 2 階樹模型交互作用，缺乏跨指標、跨領域的統一實作。就「降低使用門檻」而言，問題的真實性沒有明顯疑慮。

### 基線、度量與 ground truth 的充分性

基準設計相對嚴謹：以 MSE 與 Precision@5 兩種度量、橫跨多個領域比較所有近似器，並盡量提供精確 ground truth。但有兩點值得保留。其一，精確 ground truth 只在 $n \le 16$ 時可窮舉，For $n > 16$, where pre-computing a game and ground truth values becomes computationally prohibitive，此時改用 TreeSHAP-IQ 或 SOUM 的解析解——也就是說「大玩家數」情境的評估其實綁定在特定模型族上，泛用性不如小賽局。其二，整套評估都是數值近似誤差，並沒有人因（human-centric）實驗證明高階交互作用真的幫助人理解模型；論文自己也把這點列為未來工作。

### 是統整既有演算法，還是新方法

必須誠實看待：shapiq 的核心貢獻是「統整」而非提出全新估計量——7+7 個近似器、18 個指標幾乎都來自既有文獻，論文屬於 Datasets & Benchmarks 類別而非方法論突破。這在該軌道是合理定位，但也意味著若期待演算法層面的新穎性會落空。另一個需要警惕的是基準由作者自行定義：100 個賽局的組成、領域切分與「哪個家族在哪個領域勝出」的敘事，都在作者掌控之下；「不同領域排名不同」這個賣點雖有 MI 階數結構的解釋佐證，但本質上仍是作者自選場景下的觀察，換一組賽局是否維持同樣結論並無外部驗證。

### 宣稱的問題是否真的被解決，以及現實關聯

就「提供一個可重現、可延伸、涵蓋任意階 SI 的工具」這個目標而言，論文大致達成，且預先計算 2 042 個設定對可重現性與能耗都有實際好處。但「讓人真正理解模型」這個更高目標尚未被解決：論文坦承 the TreeSHAP-IQ algorithm is currently implemented in Python（效能受限），也承認高階交互作用的視覺化本身困難、甚至可能造成 information overload 而被誤讀。因此我的判斷是——它成功降低了「計算與取用 SI」的門檻，但「SI 是否改善真實決策」仍是開放問題，現實效益偏向研究基礎設施而非終端可解釋性的定論。

## 🔗 Related notes
