#  🛠 Utils
This folder contains some general AI training techniques.
## Keywords
| Keyword | Definition |
|-|-|
|Pretrained Model | A machine learning model that has already been trained on a large dataset and can be used for transfer learning on a new task with similar characteristics. |
| Fine Tuning | The process of adapting a pretrained model to a new task by fine-tuning its parameters. This often involves training the model on a smaller dataset specific to the new task while keeping the pretrained parameters frozen to a certain extent.
| Transfer Learning | A machine learning technique where knowledge gained from solving one task is used to solve a related task. Pretrained models are often used in transfer learning.
| Convolutional Neural Network (CNN) | A type of deep neural network used for image and video recognition, object detection and segmentation. CNNs are designed to process input data with a grid-like topology, such as an image.
| Recurrent Neural Network (RNN) | A type of deep neural network used for processing sequential data, such as time series or text. RNNs are designed to handle sequences of variable length and maintain an internal state to preserve information from previous time steps.
| Generative Adversarial Network (GAN) | A type of deep neural network used for generative modeling. GANs consist of two networks, a generator and a discriminator, that are trained together in a two-player game-like manner to produce new data that is indistinguishable from the training data.
| Autoencoder | A type of deep neural network used for dimensionality reduction and feature learning. Autoencoders consist of two parts, an encoder that maps the input data to a lower-dimensional representation and a decoder that maps the lower-dimensional representation back to the original space.
| Reinforcement Learning | A type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties based on its actions. Reinforcement learning is often used for decision-making tasks, such as robotics and game playing.
| Artificial Neural Network (ANN) | A type of machine learning algorithm inspired by the structure and function of the human brain. ANNs consist of layers of interconnected nodes that process input data and generate output predictions.| 
| Long Short-Term Memory (LSTM) | A type of RNN designed to handle long-term dependencies in sequential data. LSTMs have a memory cell that can store information over long periods of time and gates that control the flow of information into and out of the cell. |
| Deep Learning | A subset of machine learning that focuses on using deep neural networks with multiple hidden layers to learn complex patterns in data.
| Support Vector Machine (SVM) | A type of supervised learning algorithm used for classification and regression tasks. SVM finds a boundary that separates the data into classes in a way that maximizes the margin between the classes. |
| Decision Tree | A type of supervised learning algorithm used for classification and regression tasks. Decision trees represent decisions and their possible consequences in a tree-like structure, where each node represents a test on a feature and each branch represents the outcome of the test. |
| K-Nearest Neighbors (KNN) | A type of non-parametric supervised learning algorithm used for classification and regression tasks. KNN predicts the target value for a new data point by finding the K nearest neighbors in the training data and combining their target values in a certain way. |
| Naive Bayes | A type of probabilistic algorithm used for classification tasks. Naive Bayes models the class-conditional probability of each feature given a class and the prior probability of each class, and combines these probabilities to make predictions. |
| Random Forest | A type of ensemble learning algorithm used for classification and regression tasks. Random forest consists of multiple decision trees trained on different random subsets of the data and features, and makes predictions by combining the predictions of the individual trees. |

## Papers
| Title | Venue | Year | Code | Review |
|-|-|-|-|-|
| [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://proceedings.mlr.press/v37/ioffe15.html) | ICML | '15 | [✓](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) | [✓](./batch_norm/) |
| [Dropout: A Simple Way to Prevent Neural Networks from Overfitting]() | jmlr | '14 | - | [✓](./dropout/) |
| [A Literature Survey on Domain Adaptation of Statistical Classifiers](http://www.mysmu.edu/faculty/jingjiang/papers/da_survey.pdf)| ? | '08 | x | [✓](./domain_adaptation_survey/)|

## Notes
| Title | Review |
|-|-|
| Embedding Related | [✓](./embed/) |

## Other questions
### What is the meaning of logits?
In machine learning and deep learning, "logits" refer to the raw, pre-activated outputs of a model before the activation function is applied. 
Logits are the inputs to the activation function, which is typically the final layer of a neural network, and the output of this activation function gives the final prediction of the model. Logits represent the un-normalized outputs of the model, which are then passed through an activation function (such as a sigmoid or softmax function) to produce the final predictions. Logits are used to preserve the model's ability to make confident predictions, whereas the final outputs of the activation function are typically normalized to produce probabilities or scores that can be compared to make predictions.

### Use dot product to measure the similarity
The `dot product` is used in the computation of logits to measure the similarity between two vectors results in a scalar, which can be interpreted as a measure of similarity. 

In essence, the `dot product` measures the alignment between two vectors, with **larger values indicating a stronger similarity** between the two vectors. By computing the dot product between the positive item embeddings and the output sequence embeddings, the model is able to determine how similar the positive item is to the generated output sequence, which can then be used in the computation of the binary cross entropy loss.

However, that the `dot product` is just one way to measure similarity between two vectors. Other methods for measuring similarity, such as `cosine similarity`, can also be used. The choice of similarity measure will depend on the specific problem being solved and the requirements of the model.