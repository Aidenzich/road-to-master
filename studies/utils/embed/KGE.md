# Knowledge Graph Embedding (KGE)
> Note: The note is not made by our collaborators, may have many error, must be rewrite
## Introduction
- [KGE-wiki](https://en.wikipedia.org/wiki/Knowledge_graph_embedding)
- [Graph-Embedding](https://en.wikipedia.org/wiki/Graph_embedding)

## Survey
- [A Survey of Knowledge Graph Embedding and Their Applications, arXiv'21](https://arxiv.org/abs/2107.07842)
- [(SOTA)Hyperbolic Knowledge Graph Embedding](https://arxiv.org/pdf/2005.00545.pdf)
- [INK: knowledge graph embeddings for node classification](https://link.springer.com/content/pdf/10.1007/s10618-021-00806-z.pdf)
- [Knowledge Graph Embeddings and Explainable AI](https://arxiv.org/pdf/2004.14843.pdf)
  - [github](https://github.com/uma-pi1/kge)
- [ExCut: Explainable Embedding-Based Clustering over Knowledge Graphs, ISWC'20](https://link.springer.com/chapter/10.1007/978-3-030-62419-4_13)
## Algorithms of KGE
- 現有算法在 
![](https://i.imgur.com/drvrxP0.png)
以上表格是現有的演算法能學到relation pattern與否

## Type of Relation Patterns
![](https://i.imgur.com/A9X4Vgb.png)
- Symmetry
  - e.g. A--朋友-->B  ， 可推 B --朋友--> A 成立
- Antisymmetry
  - e.g. A--在上面-->B ，可推  B--在上面--> A 不成立。
- Inversion 
  - e.g. A---借錢-->銀行 ，可推 銀行--放貸-->A
- Composition
  - e.g. A--is爸爸-->B ， B--is爸爸-->C， A--is爺爺 --> C

## Algorithm Targets
- 目的是透過這些演算法，根據輸入 Graph 投射於高維空間中，有relation 的 node 會聚在一起
- toy example
    ![](https://i.imgur.com/1FU6pho.png)
    ![](https://i.imgur.com/m63CHcV.png)
- 空間中的embedding叢集會具有相似特徵，可用於可解釋性

### Methods of Node Classifications
![](https://i.imgur.com/GO9dBKQ.png)
- 目前survey的 node classification 有以上三種，
  - Steam paper uses `(c)`
  - KGE uses `(b)`
  - INK ....

## Application
### 方法敘述－node classfication
目前我們要將銀行客戶分四類
![](https://i.imgur.com/N0oDzU4.png)

而目前想到解決問題的步驟(可參考上圖)：
1. 建立銀行的知識圖譜(knowledge graph)，點(entity)可以是任意物品或是客戶，而邊(relation)可代表任意行為，比如購買，屬性...等等關係，可將現實的資訊直接搬進來學。
2. 轉成embedding，Embedding演算法paper附在下面。，
3. 將客戶entities(客戶node)提取出來，並訓練一個傳統ML classifier分4類。
- 想法起源於 [Graph to embedding and do classification](https://github.com/Accenture/AmpliGraph/blob/master/docs/tutorials/ClusteringAndClassificationWithEmbeddings.ipynb)
### Implement
- [Graph to embedding library](https://colab.research.google.com/drive/1IrLoWS3y5oGPZu1povAUqCxXVp35JZXN?usp=sharing)
- [KGE](https://colab.research.google.com/drive/1D0SQ2kw_yZ_SAPmIZjs9pYsfLJgEMaFK#scrollTo=xITDxEG_sp3o)
  - [dataset](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation)
  - workflow
    - 1. 點選上傳
        ![](https://i.imgur.com/0G0hZjS.png)
    - 2. 上傳從dataset下載的Test.csv與Train.csv
        ![](https://i.imgur.com/vtnE8DR.png)
    - 3. 新增一個train資料夾放寫入資料
        ![](https://i.imgur.com/Iiy3cXE.png)
    - 4. 選擇GPU執行階段，run all。
  - 實作使用到的KGE文檔
    - https://aws-dglke.readthedocs.io/en/latest/