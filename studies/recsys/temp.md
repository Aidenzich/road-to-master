### Is News Recommendation a Sequential Recommendation Task?
News recommendation is often modeled as a sequential recommendation task, which assumes that there are rich short-term dependencies over historical clicked news. 
However, in news recommendation scenarios users usually have strong preferences on the temporal diversity of news information and may not tend to click similar news successively, which is very different from many sequential recommendation scenarios such as e-commerce recommendation. 
In this paper, we study whether news recommendation can be regarded as a
standard sequential recommendation problem. Through extensive experiments on two realworld datasets, we find that modeling news recommendation as a sequential recommendation problem is **suboptimal**. 
To handle this challenge, we further propose a temporal diversityaware news recommendation method that can promote candidate news that are diverse from
recently clicked news, which can help predict future clicks more accurately. 
Experiments show that our approach can consistently improve various news recommendation methods.

### Improving Conversational Recommender Systems via Transformer-based Sequential Modelling
In `Conversational Recommender Systems (CRSs)`, conversations usually involve a set of related items and entities e.g., attributes of items. 
These items and entities are mentioned in order following the development of a dialogue.
In other words, potential sequential dependencies exist in conversations. 
However, most of the existing CRSs neglect(忽視) these potential sequential dependencies. 
In this paper, we propose a **Transformer-based sequential conversational recommendation method (TSCR)**, which models the sequential dependencies in the conversations to improve CRS. 
We represent conversations by items and entities, and construct user sequences to discover user preferences by considering both mentioned items and entities. Based on the constructed sequences, we deploy a `Cloze task` to predict the recommended items along a sequence. 

### Bias Mitigation for Toxicity Detection via Sequential Decisions
Increased social media use has contributed to the greater prevalence of `abusive`, `rude`, and `offensive` textual comments. Machine learning models have been developed to detect toxic comments online, yet these models tend to show biases against users with marginalized or minority identities (e.g., females and African Americans). 
Established research in debiasing toxicity classifiers often:
- (1) Takes a static or batch approach, assuming that all information is available and then making a one-time decision
- (2) Uses a generic strategy to mitigate different biases (e.g., gender and racial biases) that assumes the biases are independent of one another.

However, in real scenarios :
- the input typically arrives as a sequence of comments/words over time instead of all at once. Thus, decisions based on partial information must be made while additional input is arriving. 
- Moreover, social bias is complex by nature. Each type of bias is defined within its unique context, which consistent with `intersectionality theory` within the social sciences, might be correlated with the contexts of other forms of bias. 

In this work, we consider `debiasing toxicity detection` as a sequential decision-making process where different biases can be interdependent. 
In particular, we study debiasing toxicity detection with two aims: 
- (1) to examine whether different biases tend to correlate with each other; 
- (2) to investigate how to jointly mitigate these correlated biases in an interactive manner to minimize the total amount of bias. 

At the core of our approach is a framework built upon theories of sequential `Markov Decision Processes` that seeks to maximize the prediction accuracy and minimize the bias measures tailored to individual biases. 
Evaluations on two benchmark datasets empirically validate the hypothesis that biases tend to be correlated and corroborate the effectiveness of the proposed sequential `debiasing strategy`


### Who to follow
- [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/)
- [Jiacheng Li](https://jiachengli1995.github.io/)
