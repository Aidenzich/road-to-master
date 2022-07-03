# Evaluating RecSys by RMSE and MAE
## MAE
$$
\text{MAE} = \frac{1}{|\hat{R}|} \sum_{\hat{r}_{ui}\in \hat{R}} |r_{ui} - \hat{r}_{ui}|
$$
- MAE doesn't give any bias to extrema in error terms, means:
  -  If there are *outliers* or *large error terms*, it will weigh those equally to the other predictions. 
- Should be preferred when 
  - You're not really looking toward the importance of outliers.
  - To get a holistic(整體的) view or representation of the Recommender System.

## RMSE
$$
\text{RMSE} = \sqrt{\frac{1}{|\hat{R}|} \sum_{\hat{r}_{ui} \in \hat{R}}(r_{ui} - \hat{r}_{ui})^2}
$$
- RMSE tends to `disproportionately penalize` large errors as the residual (error term) is squared, means it is more prone to being affected by outliers or bad predictions.

## Conclusion
- RMSE will never be as small as MAE. 
- However, if the error terms follow a normal distribution, `T. Chai and R. R. Draxler` showed that using RSME allows for a reconstruction of the error set, given enough data. 
- MAE can only accurately recreate 0.8 of the data set. 
- RSME does not use Absolute Values, which is a lot more **mathematically convenient** whenever calculating distance, gradient, or other metrics.

That’s why most cost functions in Machine Learning avoid using MAE and rather use sum of squared errors or Root Means Squared Error.