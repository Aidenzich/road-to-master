# BERT4Rec
| Title | Venue | Year | Code |
|-|-|-|-|
| [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690) | CIKM | ['19](https://dl.acm.org/doi/proceedings/10.1145/3357384) | [code](https://github.com/Aidenzich/BERT4Rec-VAE-Pytorch) |

## Quick Note
- 跟 NLP 基本無關
- 結構與訓練方式與 BERT 相同，訓練任務目標類似 Masked Language Model，但把 word 換成 item，讓模型去預測 [MASK] 可能是哪個 item 
- Inference 階段，是從隨機或熱門的 item 中採樣用戶沒有買過的 item ， 用 Bert 去預測這些 items 中哪些商品最有可能出現在行為序列後。 
- 在預測用戶下一個購買的商品上，擁有高於 GNN 或 VAE 等的準確率
