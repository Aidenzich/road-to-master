| Property  | Data |
|-|-|
| Created | 2022-12-19 |
| Updated | 2022-12-19 |
| Author | @Aiden |
| Tags | #study |

# Bert Summary
> **English** | [繁體中文](./README.zh-TW.md)
This note starts by running some related experiments with [bert-extractive-summarizer](https://github.com/dmmiller612/bert-extractive-summarizer).
When there is an opportunity, study the papers related to the summarizer to understand what the SOP of the whole architecture looks like.

## Experiment Example
- [My Playground](https://github.com/Aidenzich/helloBertSummary)

- We use the summarization model to summarize this [article]((https://bc165870081.medium.com/%E6%99%82%E9%96%93%E5%BA%8F%E5%88%97%E7%9A%84ai%E4%BB%8B%E7%B4%B9-ff250cfc2ff9)); the summary result is as follows

  >When we input a time series:
  >If we want to know how the values of this series will change, then this is a regression problem
  If we want to know whether this series represents A or B, then it is a classification problem. First, we need to create many short time series
  Using the Sliding window technique, we can turn one long piece of time-series data into many short pieces of time-series data. Then we keep repeating, taking two weeks of data every one-week interval; this method is called Sliding Windows. Model-1: Zero-Rule model
  We want to know what score the most naive model can achieve. This model is called the Zero-Rule Model, and it always assumes that the value of y equals the average of y in the training data. ( We use the Zero-Rule model together with an LSTM model of the same architecture to train and evaluate their scores; the results are as follows:
  Basically, the R2 on the test data is lower than before, but the mse has shrunk. Based on the experimental results, we infer that Conv1D performs well because it can find the features where inflection points appear, while LSTM performs poorly because its data type is not quite suitable. We present the correlation coefficients required for the LSTM's input data in the form of a matrix and a heatmap respectively (Fig. For example, if you just guess every time that the closing price on day 14 = the closing price on day 13, you will also get a fairly high score (Fig. A very simple test, without tuning penalty terms, etc.)
  The most practical approach is still to reprocess the data
  (ex: EDA, feature engineering)
  Currently we only use the closing price; simply using the closing price to predict the closing price actually carries very little information. Deep learning can not only project the input data into a very high-dimensional Space to find the regression line, but can also simultaneously find features to process the data at the same time.

- When actually reading it, we did Get some of the key information, but it has the following drawbacks:
  - Part of the information is lost
  - The semantics are not smooth
  - Paragraphs are inserted into one another

- Use another piece of [news](https://www.bbc.com/zhongwen/trad/science-59993121) to see the effect
    >The virtual world offers all kinds of immersive experiences
    As the metaverse concept becomes popular, buying and selling real estate in the virtual world has become increasingly popular, and has even repeatedly set new highs in virtual real estate transaction amounts. Why is metaverse property speculation so attractive, and what mysteries does it hide? Recently, a piece of virtual real estate on the virtual reality platform "Decentraland" was sold for cryptocurrency worth 2.4 million US dollars, breaking the record for the amount and once again sparking intense attention to metaverse property speculation. However, experts warn that spending money to buy fictional land or property risks losing everything. "Decentraland" is an online virtual space, that is, one of the so-called metaverses. Users can buy virtual land on this platform, build houses, decorate their own houses, open storefronts, and also buy, sell, and trade real estate, roaming through it as a virtual avatar. The famous singer JJ Lin also stated on Twitter that he owns digital land on the "Decentraland" virtual platform. On the non-fungible token trading marketplace "OpenSea," there are now also virtual land and virtual houses for sale. The value of virtual real estate has already risen. In June 2021, the digital real estate investment fund "Republic Realm" reportedly spent an amount roughly equivalent to 900,000 US dollars to buy a batch of land in "Decentraland." The metaverse property speculation boom has driven up prices; at the current stage, it seems that only enterprises and investment funds have the ability to get involved, but not all metaverse real estate necessarily costs millions of dollars—you can also build your own little home in some virtual games.
    


## How to application ?
1. You can use this summary to quickly extract information; although some information is inevitably lost, it can effectively increase reading volume
2. It is necessary to add different ratios and length limits for different article lengths, so that the summary can be more concise