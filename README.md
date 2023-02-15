# Introduction

The goal of this project is to develop a model for [MoneyLion](https://www.moneylion.com/) that can predict whether a given loan application will default, based on the outcomes of previously funded loans. 

The process for a prospective borrower (hereafter: _client_) is as follows:
1. Client submits loan application.
2. MoneyLion sends client details to underwriters for fraud detection (can be automated, manual, or both).
3. If there are no clear & obvious signs of fraud, application is approved.
4. Once a loan deposit is made, application gets funded.

Ideally, we'd like our model to operate between steps 2 & 3, assessing the risk of each application & acting as a final guardrail before a loan ultimately gets approved. Naturally, we can assume that step 2 filters out a large majority of fraudulent applications, allowing us to focus our attention solely on risk assessment. 

We hope that this will provide MoneyLion a straightforward way of evaluating the risk of each application against their own risk tolerance, opening up the channel for more well-informed business decisions. 

# Feature Engineering

After sufficiently cleaning the dataset, we're left with the following features:

- `loan_id` & `anon_ssn` can be used to identify and track the client.
- `applicationDate` & `originatedDate` provide information on the timing of the loan.
- `loanAmount` & `apr` provide information on the cost of borrowing, which can be used to assess the affordability of the loan.
- `payFrequency` & `originallyScheduledPaymentAmount` provide information on the terms of the loan, which can be used to assess the client's ability to make regular payments.
- `loanStatus` is used to identify the status of the loan (default, not default, ongoing, etc).
- `nPaidOff` provides information on the client's loan repayment history, which is relevant to determining the client's trustworthiness. 
- `leadType` & `leadCost` can provide information on the source & cost of obtaining the loan, which may be relevant to assessing the quality of the loan.
- `clearfraudscore` can be used to identify and flag any potential fraud. 

From there, we create 3 additional features:

### Target Variable

First, a binary target variable to represent whether a given loan application has defaulted or not. We do so by transforming the `loanStatus` column, such that a value of `target == 1` indicates a favorable outcome (i.e. did not default), whereas a value of `target == 0` indicates an unfavorable outcome (i.e. default).

### Temporal Features

Additionally, we transform the `applicationDate` & `originatedDate` columns into `time_to_originate` & `time_since_last`; the former is calculated from the difference between `application_date` & `originated_date`, in hours, providing information on the time it took for the loan to be originated, which may be relevant to assessing the quality of the application; the latter represents the time between consecutive loan applications by a given client, in days, where -1 is used inplace of non-returning clients.

# Exploratory Data Analysis (EDA)

We then generate effective visualizations to better understand the data. Our findings are summarized as follows:

1. General business insights: 

<p align="center">
  <img src="images/application_trends.png" alt="Temporal feature analysis: application/originated date">
  <br>
  <em>(a) MoneyLion funded roughly 1.75x more loans from 2016 vs 2015; (b) the number of loan applications are concentrated near December & January, most likely as a result of the holiday season. Naturally, MoneyLion funds the most loans during Q1 & Q4; (c) MoneyLion funds less loans whose applications were received on weekends compared to weekdays (most likely explanation: less people submit loan applications on weekends).</em>
</p>

<p align="center">
  <img src="images/turnover_rates.png" alt="Temporal feature analysis: custom features.">
  <br>
  <em>(d) Most loan applications are originated within a few hours; (e) most returning clients submit another successful application roughly 200 days after their last.</em>
</p>

2. There is a significant class imbalance against defaulted loans, which we will address using oversampling.

<p align="center">
  <img src="images/class_imbalance.png" alt="Class imbalance.">
  <br>
  <em>Before oversampling: 97.38% of our data represents non-defaulted loans, whereas only 2.62% of the data are defaulted.</em>
</p>

3. The numeric features are heavily skewed, implying that we should either pick a model that's robust to skewness, and/or normalize these data during preprocessing.

<p align="center">
  <img src="images/unnormalized_feature_distributions.png" alt="Distributions of unnormalized features.">
  <br>
  <em>Distribution of unnormalized features: notice the skewness, outliers, and bimodality in certain histograms.</em>
</p>

<p align="center">
  <img src="images/normalized_feature_distributions.png" alt="Distributions of normalized features.">
  <br>
  <em>Distribution of scaled/normalized features: notice the relative scaling, reduced skewness, and improved Gaussianity in certain histograms.</em>
</p>

<p align="center">
  <img src="images/normalized_feature_distributions_correction.png" alt="Corrected distribution of normalized features.">
  <br>
  <em>Since `time_since_last` uses -1 inplace of missing values, the distribution of positive values paint a clearer picture of the improvements as a result of normalization.</em>
</p>

4. There are a few potential sources of multicolinearity to keep in mind, in the event that our model ends up generating unreliable/unstable predictions.

<p align="center">
  <img src="images/correlation_heatmap.png" alt="Correlation heatmap.">
  <br>
  <em>Correlation heatmap: the target variable is not significantly correlated with any of the features; further, the only (relevant) highly correlated features are `loanAmount` & `originallyScheduledPaymentAmount` at 0.94.</em>
</p>

# Model Development

Once our data is in good shape, we train a series of popular binary classification models, and evaluate their performance using ROC-AUC score.

<p align="center">
  <img src="images/roc_auc.png" alt="ROC-AUC scores.">
  <br>
  <em>Incredibly, the Random Forest Ensemble vastly outperforms the other 3 models, with an ROC-AUC score of 99.90%! Since the model performed so well at identifying loans at risk of default, we'll skip the fine-tuning phase altogether, as any incremental improvement in performance would not be worth the additional time spent + computational costs (not to mention the potential for overfitting).</em>
</p>

Digging deeper, we uncover the features that were most useful for the Random Forest Ensemble to arrive at the right answer. 

<p align="center">
  <img src="images/feature_importances.png" alt="Feature importances.">
  <br>
  <em>Notice, the various lead types don't seem to be of much importance to our model, while features that describe the financial obligations of the loan (like `loanAmount`, `apr`, and `originallyScheduledPaymentAmount`) play a bigger part in its decision making process. Further, `clearfraudscore` and `time_to_originate` also play significant roles from the perspective of the model.</em>
</p>

Since `time_to_originate` is the most significant feature, yet it's unclear how it affects the probability of a loan defaulting, we use a Partial Dependence Plot (PDP) to visualize the relationship between the feature and the model's predictions for that feature, while holding all other features constant.

<p align="center">
  <img src="images/pdp_time_to_originate.png" alt="PDP time_to_originate.">
  <br>
  <em>We find that the longer it takes for an application to originate, the higher the probability of default; loans that are originated soon after the application date are more likely to be paid in full. We suspect that this may be because clients with their "ducks in a row" simply pass through an automated origination filter, leading to quicker turnarounds, whereas clients that don't have to wait for their applications to be manually inspected. Though, without more knowledge of the origination process, it's hard to draw any concrete conclusions. </em>
</p>
