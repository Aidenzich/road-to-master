| Property  | Data |
|-|-|
| Created | 2022-12-19 |
| Updated | 2022-12-19 |
| Author | @Aiden |
| Tags | #study |

# ChatGPT
## Training language models to follow instructions with human feedback
| Title | Venue | Year | Code |
|-|-|-|-|
| [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf) | - | '22 | - |

### Abstract
- **Cause:** Making language models bigger does not <font color='red'>inherently</font> make them better at following a user’s <font color='red'>intent</font>. 
    - For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show <font color='red'>an avenue(一種途徑)</font> for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. 
- **Process:** Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. 
    - We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. 
- **Effect:** We call the resulting models **InstructGPT**. 
    - In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters.
    - Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. 
    - Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.