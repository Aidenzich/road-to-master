# Singular value decomposition
![svd](./assets/svds.png)

`Singular value decomposition (SVD)` is a mathematical technique used to break down a matrix into its constituent parts. 

It represents a matrix as the product of **3 matrices**, which can be used to analyze and manipulate the original data. 

The 3 matrices are the `left singular vectors`, the `singular values`, and `the right singular vectors`. 
- The `left singular vectors` represent the structure of the input data
- The `singular values` represent the strengths of each component of the data
- The `right singular vectors` represent the structure of the output data. 

SVD is commonly used in data analysis and machine learning for tasks such as data compression, noise reduction, and feature extraction.
## Formula

$$
A = U \times \Sigma \times V^{T}
$$