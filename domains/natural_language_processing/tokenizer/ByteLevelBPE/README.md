| Property | Data |
|-|-|
| Created | 2026-06-23 |
| Updated | 2026-06-23 |
| Author | @Aiden |
| Tags | #study #tokenizer |

# Byte-Level BPE (BBPE)

## 核心設計目標與解決痛點

傳統 BPE 以「字元 (Character)」為單位，會遇到未登錄詞 (OOV / `[UNK]`) 以及詞表因多語言膨脹的問題。BBPE 透過將底層切換為「位元組 (Byte)」解決此問題。其三大核心目標為：

- 避免出現 `[UNK]` / OOV：任何沒見過的字或符號都能保底解碼。
- 全語言與符號支持：所有數位文字在電腦底層都是 byte，皆可被拆解。
- 動態詞表結構：256 個初始 byte 詞表 + 經 BPE 合併的高頻 token。

## 詞表雙層結構

現代大模型（如 GPT-4o、LLaMA 3）的詞表主要由兩大區塊組成：

- 基礎保底區：256 個 byte token，從 `<0x00>` 到 `<0xFF>`。當遇到罕見字、火星文、新 emoji（如 🚀）或特殊控制碼時，會退化到此區域，確保 100% 不崩潰。
- 高效壓縮區：合併 token，例如 `the`、`apple`、`app`、`你好`。由 BPE 演算法根據海量文本統計高頻出現的組合，用來提升模型的編碼與讀寫效率。

## Tokenizer 動態拆解機制

運作時遵循「最長、最匹配（壓縮率最高）」原則：

- 高頻詞彙走捷徑：輸入 `apple`，詞表有對應 token，拆成 `['apple']`（1 token）。
- 組合詞彙次之：輸入 `applet`，拆解為 `['app', 'let']`（2 tokens）。
- 罕見未知走底層：輸入完全沒見過的極罕見古字，依 UTF-8 編碼退化拆解為 4 個 byte，例如 `['<0xF0>', '<0xAA>', '<0x9A>', '<0xA5>']`（4 tokens）。

## 現代主流模型應用現況

Byte-level 概念已成為現代 LLM 的業界標準，主要分為兩大實作流派：

- Tiktoken 陣營：OpenAI GPT-4o/o1、LLaMA 3。純粹的 BBPE。先透過精心設計的正則表達式預分詞，再轉 byte 進行 BPE 合併。近期趨勢為擴大詞表（如 12.8 萬至 20 萬），大幅提升中文等非英文字元的壓縮效率。
- SentencePiece 陣營：Google Gemma 2、Mistral。將空格視為普通字元處理，並在遇到未知字元時觸發 byte fallback 機制，本質上同樣達到了 byte-level 零 OOV 的效果。
