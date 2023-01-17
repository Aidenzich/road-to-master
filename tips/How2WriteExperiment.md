# How to write experiments?
Writing experiments involves several key steps, including selecting appropriate datasets and features, defining evaluation metrics, establishing a baseline, preprocessing data, and setting up the experiment. Here is a brief overview of each step:

| Component | Description |
|-|-|
| **Datasets and Features** | Select the datasets and features that are most relevant to the problem you are trying to solve. Make sure the datasets are representative and large enough to support your conclusions. |
| **Evaluation Metrics** | Choose metrics that are appropriate for the task and the type of data. For example, `accuracy` is a common metric for classification tasks, while `mean squared error` is a common metric for regression tasks. |
| **Basline** | Establish a baseline by training a simple model on your dataset and using your chosen evaluation metrics to measure its performance. This will provide a point of `comparison` for your more complex models. |
| **Preprocessing and Experiment Setup** | Perform any necessary preprocessing on your data, such as `normalizing` or `scaling` the features. Then, set up the experiment, including splitting the data into training and testing sets, and defining the parameters for your model. |
| **Comparative Results** | Train your model(s) and compare their performance to the baseline. Include a thorough analysis of the results, discussing any trends or patterns that you observe. |
| **Ablation Study** | A type of experiment that is used to understand the **impact** of `individual components or features` on the overall performance of a model. It involves training the model multiple times, each time removing or **"ablating"** a different component or feature, and comparing the results. |
| **Discussion** | A `discussion section` is an important part of any experiment, as it allows you to **interpret** the results and provide insights into the broader context of your findings.  |

<details>
    <summary><strong>Details of Discussion</strong></summary>

1. Summarize your main findings: Provide a brief summary of the main results of your experiment, including any key trends or patterns that you observed.
2. Interpret the results: Interpret the results in the context of your research question or hypothesis. Explain what the results mean and why they are important.
3. Discuss the limitations: Acknowledge any limitations of your experiment, such as the sample size, the data, the model or the evaluation metrics.
4. Compare with previous works: Compare your results with those of previous studies in the literature, highlighting any similarities or differences.
5. Implications and Future Work: Discuss the implications of your results for future research and practice. Suggest areas for future research that would help to build on your findings.
6. Conclusion: Summarize the main findings, the limitations and the implications of your study. Provide a brief statement of the overall conclusion.
</details>

<details>
    <summary><strong>Details of Ablation Study</strong></summary>
    
Here are a few steps to conduct an ablation study:
1. Select the features or components of the model that you want to investigate.
2. Train the model multiple times, each time removing or "ablating" a different feature or component.
3. Compare the performance of the model using the same evaluation metrics for each ablation run.
4. Analyze the results to understand the impact of each feature or component on the overall performance of the model.
5. Draw the conclusion and provide insights on which feature or component is important and which is not.
</details>

## Vocabulary
`interpret` 解釋
`interrupt` 中斷