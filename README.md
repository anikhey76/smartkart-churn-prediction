# smartkart-churn-prediction
SmartKart — Customer Churn Prediction

An end-to-end, fully-commented machine learning pipeline that predicts which SmartKart customers are likely to churn (leave), so the retention team can act before they do.

Built as a complete no-code-concept, full-code walkthrough using Logistic Regression — from a deliberately messy raw dataset all the way to a business-ready, ranked risk report.

Project Context
Course: Introduction to AI & ML | BBA AI/ML | Chitkara Business School
CLO: CLO02 — Apply data preprocessing, feature selection & ML models to business scenarios; evaluate performance using appropriate metrics
Dataset: SmartKart_dirty_100_rows.csv — 100 customer records, intentionally messy (duplicates, missing values, invalid entries, outliers)
Algorithm: Logistic Regression (binary classification)
Business Problem

SmartKart wants to know which customers are at risk of leaving and why, so the retention team can intervene early with targeted offers or support — instead of reacting only after a customer has already churned.

Pipeline (15 Steps)
Step	Stage	What Happens
1	Data Collection	Load the raw CSV export
2	Data Understanding	Inspect shape, dtypes, missing values, duplicates, outliers
3	Data Cleaning	Strip whitespace, fix types, remove duplicates & invalid values, impute with median
4	Outlier Detection & Treatment	IQR method to cap (winsorize) extreme values
5	Feature Selection	Keep Age, Monthly_Spend, Complaints; drop Customer_ID
6	Define Target Variable	Churn (1 = left, 0 = stayed)
7	Encode Target Variable	Verify target is already numeric
8	Train-Test Split	80/20 stratified split
9	Feature Standardisation	StandardScaler, fit on train only
10	Model Building	Instantiate LogisticRegression
11	Model Training	Fit on scaled training data
12	Prediction	Predict class + churn probability on test set
13	Model Evaluation	Confusion matrix, accuracy, precision, recall, F1
14	Model Interpretation	Read coefficients to explain why customers churn
15	Final Output	Ranked, business-ready churn risk report
Key Results
Recall ≈ 100% on the test set — the model catches nearly every actual churner, which matters most for retention (a missed churner is a silently lost customer).
Precision ≈ 83–91% — a small number of loyal customers get flagged as "at risk" (an acceptable trade-off: an unnecessary discount is cheaper than a lost customer).
Accuracy ≈ 89–95% overall.
What Drives Churn
Feature	Effect	Business Takeaway
Monthly_Spend	Strong negative effect	Higher spenders churn less — your most loyal, highest-value customers
Complaints	Strong positive effect	Each additional complaint sharply raises churn risk — the clearest actionable lever
Age	Weak positive effect	Mild tendency for churn risk to rise slightly with age

Takeaway: Reducing customer complaints and protecting high-spend customer relationships are SmartKart's two most effective retention levers.

Repository Contents
├── SmartKart_Churn_Prediction_ML_Pipeline.ipynb   # Full 15-step pipeline (Colab notebook)
├── smartkart_churn_risk_report.csv                # Final output: ranked churn risk per test customer
└── README.md
Output: Churn Risk Report

smartkart_churn_risk_report.csv contains the test-set predictions, ranked by churn probability (highest risk first):

Column	Description
Customer_ID	Customer identifier
Age, Monthly_Spend, Complaints	Model input features
Actual_Churn	Ground-truth label
Predicted_Churn	Model's predicted class (0/1)
Churn_Probability	Model's confidence the customer will churn
Risk_Label	Human-readable flag: "Likely to Churn" / "Not Likely to Churn"
How to Run
Open SmartKart_Churn_Prediction_ML_Pipeline.ipynb in Google Colab (or Jupyter).
Run all cells top to bottom (Runtime → Run all).
When prompted in Step 1, upload SmartKart_dirty_100_rows.csv.
Each code cell is commented; each markdown cell explains what the step does and how to read its output.
Requirements
pandas, numpy, matplotlib, seaborn, scikit-learn
Methodology Notes
Median imputation was used instead of mean, since it's robust to the outliers present in the raw data.
IQR-based winsorization (capping, not deleting) was used for outlier treatment, to preserve each customer's other information.
Standardization was fit only on the training set and applied to the test set, to avoid data leakage.
Stratified train-test split preserves the churn/no-churn ratio in both sets for a fair evaluation.
