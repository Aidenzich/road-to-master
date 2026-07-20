# AutomaticPromptOptimization — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Automatic Prompt Optimization with "Gradient Descent" and Beam Search |
| Venue | EMNLP 2023 |
| Year | 2023 |
| Authors | Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, Michael Zeng (Microsoft Azure AI) |
| Official Code | https://github.com/microsoft/LMOps/tree/main/prompt_optimization |
| Venue Kind | paper |

> 本筆記基於 arXiv 預印本 `2305.03495v2`（2023-10-19 修訂版）的全文與 LaTeX 原始碼撰寫；正式出處為 EMNLP 2023，camera-ready 版本細節可能與預印本略有差異。

## Introduction

大型語言模型（LLM）的能力高度依賴 prompt，而 prompt 至今仍靠人手以反覆試錯的方式寫成。這篇論文要解決的問題很具體：在**只能透過 API 存取黑箱 LLM**（拿不到模型內部梯度、logits 或權重）的前提下，如何自動把一段隨手寫的、模糊的任務描述，改寫成一段精準、能提高任務表現的 prompt。既有做法要嘛需要模型內部狀態（soft prompt、AutoPrompt 這類 token 層級調參），要嘛在 prompt 的語意空間裡做「沒有方向」的蒙地卡羅或強化學習搜尋，前者不適用於 API 使用者，後者常常低效或產出人看不懂的字串。

論文提出的解法叫 ProTeGi（Prompt Optimization with Textual Gradients）。核心點子是把數值梯度下降「翻譯」成一場文字的蘇格拉底式對話：用一小批訓練資料跑當前 prompt、蒐集它答錯的例子，再請一個 LLM 用自然語言描述「這個 prompt 為什麼會錯」——這段批評就是「文字梯度」（textual gradient），方向上等同於數值梯度指向的「使表現變差」的方向；接著再請 LLM 把 prompt 往梯度的**相反語意方向**改寫，等同於沿負梯度更新一步。這些改寫步驟被包進一個 beam search 的外層迴圈，並用 best-arm identification（bandit）的方式決定哪些候選 prompt 值得留到下一輪，藉此壓低昂貴的 API 評估次數。

論文如何衡量成功？它在 4 個二元分類任務上做初步案例研究：Jailbreak（自建的 452 筆多語 jailbreak 偵測）、Ethos（997 筆英文仇恨言論）、Liar（4000 筆英文假新聞）、Sarcasm（10000 筆阿拉伯文諷刺偵測）。每個任務隨機取 50 筆做開發、150 筆做測試，回報 3 次試驗平均的 binary F1，預設用 2023 年 1 月版的 `gpt-3.5-turbo`。比較對象包括 Monte-Carlo（即 APE 的無方向改寫）、RL（GrIPS/TEMPERA 式的片語層級操作）、AutoGPT，以及作為 bandit 選擇基準的 uniform 均分預算法。論文的主張是：ProTeGi 在四個任務上都勝過這些 state-of-the-art 基準，平均超出 MC 與 RL 各 3.9% 與 8.2%，並在最好的情況下把初始 prompt 的表現提升多達 31%，同時用更少的 API 呼叫。

![ProTeGi 的一次最佳化步驟總覽：初始 jailbreak prompt 在一個 minibatch 上答錯（Label True 卻 Prediction False），LLM 產生的「梯度」指出 prompt 假設攻擊會明講、忽略了間接或隱晦的手法，據此改寫出「不論多隱晦都要偵測」的新 prompt，再經 bandit 選擇。](imgs/overview.png)

## First Principles

### 離散最佳化的障礙，以及「文字梯度」如何繞過它

把 prompt 最佳化寫成最佳化問題後，目標是找到

$$p^{*} = \arg\max_{p \in \mathcal{L}} \; m(p, \mathcal{D}_{te})$$

其中 $\mathcal{L}$ 是連貫自然語言的空間、$m(\cdot)$ 是任意度量函數（本文用 F1）、$\mathcal{D}_{te}$ 是測試/開發資料。困難在於 $\mathcal{L}$ 是離散且組合爆炸的，無法對它直接微分做梯度下降。ProTeGi 的關鍵操作是用兩個固定的 LLM 提示詞把「微分」與「反向傳播」各換成一個 LLM 呼叫：一個叫 $\nabla$ 的提示詞負責生成梯度，一個叫 $\delta$ 的提示詞負責施加梯度。$\nabla$ 永遠拿到當前 prompt $p_0$ 與它在 minibatch 上的行為（特別是錯誤），輸出一段描述 $p_0$ 缺陷的自然語言，這段文字就是梯度 $g$；$\delta$ 則拿梯度 $g$ 與 $p_0$，把 $p_0$ 往 $g$ 的相反語意方向改一步、修掉 $g$ 指出的問題。

![模仿梯度下降的文字對話樹：左上 $\nabla$（"What is wrong with p0?"）從 $p_0$ 與預測 $\hat{y}$、真值 $y$ 生成梯度 $g_1..g_m$；右上 $\delta$（"Use g to fix p0"）把每個梯度套回 $p_0$ 產生改寫候選 $p'$，再由 mc 提示詞產生語意相近的釋義候選 $p''$，最後全部丟進 bandit selection 選出下一輪的 beam。](imgs/gradient-dialogue-tree.png)

$\nabla$ 與 $\delta$ 的實際內容是任務無關的固定字串（所有任務共用同一組）。以下是論文附錄給的生成梯度提示詞 $\nabla$：

```text
I'm trying to write a zero-shot classifier prompt.

My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

give {num_feedbacks} reasons why the prompt could
have gotten these examples wrong.
Wrap each reason with <START> and <END>
```

以及施加梯度、產生改寫 prompt 的提示詞 $\delta$：

```text
I'm trying to write a zero-shot classifier.

My current prompt is:
"{prompt}"

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this
prompt is that {gradient}

Based on the above information, I wrote
{steps_per_gradient} different improved prompts.
Each prompt is wrapped with <START> and <END>.
```

值得注意的是，這裡沒有明確的學習率或步長：論文選擇讓 LLM 自己決定改寫幅度，等於採用一種「自適應步長」，把步長控制留給未來工作。

### Beam search 外層迴圈與 expansion 步驟

一次梯度只是一個方向；ProTeGi 不只走一步，而是把這些梯度步驟當成一個 beam search 的展開來源，反覆做「展開—選擇」。外層演算法如下：

```text
Algorithm 1: ProTeGi
Require: p0 初始 prompt, b beam 寬度, r 搜尋深度, m 度量函數
  B0 <- {p0}
  for i = 1 to r-1:
      C <- {}
      for all p in B_i:
          C <- C ∪ Expand(p)
      B_{i+1} <- Select_b(C, m)
  返回 argmax_{p in B_r} m(p)
```

`Expand(p)` 才是把「文字梯度下降」落地的地方：先從訓練資料抽一個 minibatch，用當前 prompt $p$ 跑一遍、蒐集答錯的例子 $e$；把 $e$ 餵給 $\nabla$ 得到梯度 $\{g_1,...,g_m\}$；把每個 $g_i$ 餵給 $\delta$ 改寫出候選 $\{p'_{i1},...,p'_{iq}\}$；最後把每個改寫候選再丟給一個釋義提示詞 $LLM_{mc}$，產生語意相近但措辭不同的蒙地卡羅後繼候選 $p''$，用來探索新候選附近的局部空間。這一步把「有方向的梯度改寫」與「無方向的局部釋義探索」結合在同一個展開裡。

### 把 beam 選擇當成 best-arm identification

展開後候選 prompt 會爆量，而在整個訓練集上評估每個候選極貴，所以選擇步驟的目標是：用盡量少的資料評估次數，挑出 $b$ 個最好的候選。論文把這件事對應到 bandit 理論裡的 best-arm identification 問題——$n$ 個候選就是 $n$ 支手臂，候選在資料上的真實表現是手臂的隱藏價值，「拉一次手臂」就是在一個隨機資料點上評估一次 prompt。論文試了四種選擇器。UCB／UCB-E 用一個 acquisition 分數在「利用」與「探索」間權衡，時間步 $t$ 挑分數最高的候選評估：

$$Q_t(p) + c \sqrt{\frac{\log t}{N_t(p)}}$$

其中 $Q_t(p)$ 是候選 $p$ 目前的估計表現、$N_t(p)$ 是它到目前為止被評估的總次數、$c$ 是探索係數（實驗全設 2.0）。另一族是 Successive Rejects 與更激進的 Successive Halving，它們不需調超參數：Successive Rejects 分 $n-1$ 個階段，每階段評估存活候選、淘汰分數最低者，第 $t$ 階段給每個候選的評估點數 $n_t$ 隨階段遞增：

$$n_t = \left\lceil \frac{1}{0.5 + \sum_{i=2}^{T} 1/i} \cdot \frac{B - T}{T + 1 - t} \right\rceil$$

其中 $B$ 是總查詢預算；原文在公式附近只明確定義了 $B$，公式裡的 $T$ 並未在該處給出定義（演算法本身另以 $n$ 表示候選 prompt 的數量，故 $T$ 的確切含義需回到原文脈絡推敲）。理論上 Successive Rejects 對 best-arm identification 是可證明最優的，但論文的實測結果卻相反（見下）。

### 一次最佳化步驟的具體走查

用 Ethos 的真實例子把上面的機制走一遍。初始 prompt $p_0$ 是工程師隨手寫的一句「Is the following text hate speech?」。在 minibatch 裡它答錯了一則帶反諷、間接提到 Muslims 的留言：真值是「No（非仇恨言論）」，但 $p_0$ 預測「Yes」。把這個錯誤餵給 $\nabla$，它生成的梯度 $g$ 是：「這個 prompt 假設仇恨言論一定含有明確直接的用語；但這則文字是對 Muslims 的反諷、間接評論，模型較難辨識它其實不是仇恨言論。」再把 $g$ 餵給 $\delta$，改寫出的 ProTeGi 候選 $p'$ 是：「Does the following text contain language that targets a group of people based on their religion, gender, or other personal characteristics?」——可以看到 prompt 從一句模糊問句，被資料驅動地改寫成一條更精確的標註指令。

候選數量的算術也值得攤開，才知道 API 呼叫從哪來。論文設定：minibatch 大小 $|\mathcal{D}_{mini}|=64$、beam 寬度 $b=4$、跑 6 個最佳化步驟；每次抽 4 個錯誤成一組，每組生 $m=4$ 個梯度、每個梯度改寫一次（得到 4 個新候選），每個新候選再產 $p=2$ 個蒙地卡羅釋義，於是一組錯誤約產生 $4 + 4\times2 = 12$ 個候選。為避免計算爆炸，論文在 bandit 選擇前，對每個父 prompt 隨機抽樣到 8 個後繼候選才進選擇。整個過程沒有做任何超參數搜尋，所有任務共用同一組預設值。

### 主要證據

![四個任務上 F1 對「每個候選的評估預算」的曲線：ProTeGi（藍）在 Jailbreak、Ethos、Liar、Sarcasm 四格都位於最上方，勝過 MC（橘）、RL（綠）、AutoGPT（紅虛線）與初始 prompt p0（紫點線）；隨評估預算從約 12 增到 50，各法大致上升。](imgs/main-result.png)

主結果如上圖。平均而言 ProTeGi 超出 MC 與 RL 各 3.9% 與 8.2%，超出原始 prompt $p_0$ 15.3%、超出 AutoGPT 15.2%。基準之間也有值得注意的落差：RL 的片語層級操作在 Ethos 與 Sarcasm 上幾乎沒把 prompt 拉離起點，而 AutoGPT 跑 6 輪回饋在 Jailbreak 與 Sarcasm 上反而讓起始 prompt 變差。針對 beam search 本身的消融顯示，用 beam 取代「攤平枚舉」（No iteration）與「貪婪 DFS」（Greedy）在三個任務都更好：Jailbreak 0.85 對 0.80/0.82、Liar 0.67 對 0.63/0.63、Sarcasm 0.88 對 0.87/0.85。至於選擇器，所有近似 best-arm 法都勝過 uniform 均分基準，但和理論預期相反的是，UCB 式反而穩定勝過 Successive Rejects 式（例如 50-per-prompt 的 Jailbreak，UCB 0.85 對 SR 0.82、SH 0.80）。換底層模型時，RLHF 調過的模型大幅超越 GPT-3：GPT-4 在 Jailbreak 上最高（0.88）；但 Sarcasm 上 GPT-4 與 ChatGPT 並列最高（同為 0.86），並非 GPT-4 獨勝，對比之下 GPT-3 僅有 0.73／0.55。

## 🧪 Critical Assessment

### 問題是真的，但「31%」這個標題數字要小心讀

「prompt 靠人手試錯、又常常只有 API 存取」這個痛點是真實且普遍的，把黑箱、非參數、可用任意度量當作設計約束，也切中實務。但摘要主打的「提升多達 31%」是**單一最佳情況**的數字，而全文正文報告的、跨任務對原始 $p_0$ 的**平均**提升只有 15.3%；對最強基準 MC 的平均領先更只有 3.9%。把最好的個案數字放進摘要標題、把平均值藏在正文，是常見的成績呈現方式，讀者應以 15.3%／3.9% 這類平均值來理解實際效益，而非 31%。還要留意論文把這些數字統稱為「margin」，並未在該段說明是絕對百分點還是相對百分比，因此宜當作論文自報的改善幅度來讀，而非精確的 F1 點數差。

### 評估規模小、且四個任務全是二元分類

論文自己就把這項工作定位為「limited and preliminary case study」。證據面的弱點相當實在：每個任務只用 150 筆測試、3 次試驗平均，樣本量小到 F1 的小數點後第二位差異（例如 Sarcasm 0.88 對 0.87）很難說有統計意義，論文也沒有回報主結果的顯著性檢定或信賴區間。更根本的是，四個任務全是二元分類，而方法宣稱可套用到 parsing、chatbot、摘要等任意任務——這個一般性宣稱完全沒有被實驗支持，屬於尚未驗證的外推。

### 自建的 Jailbreak benchmark 與自訂目標的膨脹風險

四個任務裡最亮眼的改進（beam 消融、換模型的最大增益）多半落在 Jailbreak，而 Jailbreak 正是作者自己提出工作定義、並自行建立的任務——論文明說「We define jailbreak attack as...」，這個 452 筆多語資料集帶有人工標註（human-annotated）的標籤（論文未交代標註者身分）。在自建、未經第三方驗證的資料集上取得最大改進，存在拿方法去對齊自訂目標的風險：我們無法排除「初始 prompt 特別差、因此改進空間特別大」造成的膨脹，論文也沒有提供這個資料集的標註一致性或難度校準資訊。相對地，Ethos、Liar、Sarcasm 是既有公開資料集，其改進幅度（例如 Ethos 全程只在 0.93–0.965 之間移動）就溫和得多。

### 新意在於「有方向」，但組件多為既有零件的重組

方法的真正新意是把「LLM 生成的批評」當成有方向的梯度來引導離散搜尋，這相對於 APE 的無方向蒙地卡羅、或 RLPrompt 的不可讀輸出確實是有意義的一步。但要誠實地說，beam search、UCB／Successive Rejects 這些 bandit 選擇器、以及蒙地卡羅釋義，都是現成零件；把梯度下降類比到文字對話在敘事上很漂亮，實作上則是「用固定提示詞產生批評、再用固定提示詞改寫」的兩次 LLM 呼叫，本質接近以自我批評驅動的迭代改寫。它與 Self-Refine、Reflexion 這類自回饋迭代方法共享同一個大家族，差別在於這裡的回饋被明確錨定在 minibatch 錯誤上、並被包進 beam+bandit 的搜尋預算控制裡。

### 過擬合與變異：方法會「進步後退步」

![四個任務上 F1 對最佳化步數（0–7）的學習曲線：Ethos 與 Sarcasm 在約第 3 步達到高點後明顯下滑，Jailbreak 與 Liar 則在約第 3–4 步後大致持平。讀圖近似值：Ethos 從第 3 步約 0.965 一路降到第 7 步約 0.942、Sarcasm 從第 3 步約 0.877 降到第 6 步約 0.857（皆為肉眼讀圖的近似，非論文明列數字）。](imgs/learning-curves.png)

一個削弱「問題已解決」敘事的觀察是學習動態本身：如上圖，所有任務大約在第 3 步就達到高點，之後 Ethos、Sarcasm 甚至反轉下滑，顯示這個過程會在訓練資料上過擬合或卡進局部極小值。附錄的變異數實驗也指出，ProTeGi 雖然平均較好，變異卻可能更高（例如 Ethos 的 SE 0.003 對 MC 的 0.001），論文推測正是梯度更新的語意方向性帶來的。要注意這張變異數表用的是與主結果不同的評估協定——每個候選只跑 6 次 query、每個變體重複 12 次試驗（刻意壓低 query 數以放大變異），回報的是 Accuracy 與 SE 而非主結果的 F1，因此不宜直接拿它的數字和前面主表相比。加上作者在 Limitations 明說，即使查詢預算不大，一次最佳化因大量 API 呼叫（含每輪 beam 候選的完整評估）常會跑超過 1 小時——這對「即時或大規模」應用是實在的門檻。綜合來看，這是一個把方向性引入 prompt 搜尋、且結果可解讀的有價值原型，但它的效益幅度、統計穩健度與跨任務一般性都還停留在初步階段，不宜過度外推。

## 🔗 Related notes

- [SelfRefine](../SelfRefine/) — 同屬 LLM 自回饋迭代改寫家族，以自身回饋反覆精修輸出。
- [Reflexion](../Reflexion/) — 用語言化的自我反思當作回饋訊號來改進後續嘗試，與「文字梯度」的回饋機制概念相近。

