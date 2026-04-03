# Dynamic Pricing and Purchase Prediction
## Predicting Order Behaviour in a Mail-Order Pharmacy

### Analytics Project
Authors: Anna Geiser, Armir Lecaj, Ray Pinzon, Jan Suter Lecturer: Prof. Andreas Reber City, Date: Brugg, 18 May 2026

# 1. Introduction
Dynamic pricing strategies have become an important factor of modern e-commerce. Online retailers operate in a competitive environment where automated, data-driven price adjustments can influence consumer purchasing behaviour and revenue outcomes. This project investigates whether product-level purchase decisions can be accurately predicted using transactional and product-attribute data collected during a dynamic pricing experiment.
The dataset used in this study originates from the Data Mining Cup (DMC) 2017 competition (adapted) and captures user interactions (e.g., clicks, basket additions, and purchases) across a range of pharmaceutical products sold through an online mail-order pharmacy. The central prediction task is binary classification: given a set of observable features at the time of a user action, will the product be ordered (order = 1) or not (order = 0)?
This paper applies the Cross Industry Standard Process for Data Mining (CRISP-DM) methodology, which structures the analytical workflow into six iterative phases: Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, and Deployment. As this project focuses on model development rather than production deployment, the Deployment phase is discussed conceptually but not implemented.
[Add 1–2 sentences describing the structure of the remainder of the paper.]

# 2 Business Understanding

## 2.1 Background Information
Dynamic pricing refers to the practice of adjusting product prices in real time or near real time in response to market conditions, competitor behaviour, and demand signals. In online retail, this practice is facilitated by automated pricing systems that leverage historical transaction data to optimise prices towards defined business objectives such as revenue maximisation, gross margin improvement, or market share growth.
The scenario under examination involves a mail-order pharmacy that applies daily automatic price adjustments across its product catalogue. Products are characterised by static attributes (e.g., dosage form, product group, generic status) and dynamic, attributes which change over time (e.g., current price, competitor price, promotional flag, availability). User interactions are captured at the event level, where each row in the dataset represents one user action on one product on one specific day.
[Optionally expand with 2–3 sentences on the regulatory context of pharmaceutical pricing or the specific competitive dynamics of the online pharmacy market.]

## 2.2 Business Objective, Risks, and Benefits
The primary business objective is to build a predictive model that accurately identifies whether a given product will be purchased (order = 1) for a given user action event. The direct business benefit of such a model is the ability to prioritise promotional and pricing actions toward products most likely to convert, thereby increasing revenue efficiency.
Objective: Predict purchase probability (order = 1) at the product-event level to support dynamic pricing decisions.
Benefit: Enables targeted promotion scheduling, stock pre-allocation, and competitor-price response strategies.
Risk: Class imbalance may produce a model that achieves high accuracy by predicting all non-orders. Model must be evaluated on F1 and AUC-ROC, not accuracy alone.
Constraint: Revenue data must not be used as a predictor — it is only observable after a purchase occurs (data leakage).

## 2.3 Success Criteria and Data Mining Goals
Business success is defined as a model that reliably distinguishes purchasers from non-purchasers with sufficient recall to be actionable in a real-time pricing context. The following quantitative thresholds are proposed:
1. AUC-ROC ≥ 0.80 on the held-out test set
2. F1-score (for order = 1 class) ≥ 0.35
3. Recall (for order = 1 class) ≥ 0.50
The data mining goal is to train and compare at least three classification algorithms — Logistic Regression, Random Forest, and Gradient Boosting — on a merged, preprocessed dataset derived from train.csv and items.csv, and to identify the best-performing model according to the criteria above.

## 2.4 Project Phases and Tools
The project follows the six-phase CRISP-DM process model.
All analytical work is conducted in Python 3.10+ using the following libraries:
1. pandas, numpy: data manipulation and numerical computation
2. matplotlib, seaborn: visualisation
3. scikit-learn: model training, evaluation, and hyperparameter tuning
4. imbalanced-learn: SMOTE oversampling for class imbalance
5. xgboost: Gradient Boosting implementation

# 3 Data Understanding
## 3.1 Description of the Data
Two data sources are provided for this project. The static product catalogue is contained in items.csv, while the time-varying transactional log is stored in train.csv. Both files are pipe-delimited (separator: "|"). The key linking the two sources is the product identifier pid.

### 3.1.1 items.csv - Static Product Attributes
| Variable | Description | Value Range |
| --- | --- | --- |
| pid | Product number (join key) | Natural number |
| manufacturer | Manufacturer ID (anonymised) | Positive integer |
| group | Product group | String (caps + digits) |
| content | Package content | Positive float or NxN string |
| unit | Unit of measurement | String (caps) |
| pharmForm | Dosage form | 3-char string (caps) |
| genericProduct | Generic drug flag | {0, 1} |
| salesIndex | Dispensing regulation code | Natural number |
| category | Main shop category | Natural number |
| campainIndex | Campaign label | {A, B, C} |
| rrp | Reference retail price | Positive decimal |

### 3.1.2 train.csv - Time-Varying Transactional Data
| Variable | Description | Value Range |
| --- | --- | --- |
| lineID | Unique action identifier | Natural number |
| day | Day in the observation period | Natural number |
| pid | Product number (join key) | Natural number |
| adFlag | In marketing campaign | {0, 1} |
| availability | Availability status | {1, 2, 3, 4} |
| competitorPrice | Lowest competitor price | Positive decimal (may be missing) |
| click | Click flag | {0, 1} |
| basket | Basket flag | {0, 1} |
| order | Target variable: Purchase flag | {0, 1} |
| price | Product price | Positive decimal |
| revenue | Revenue (leakage risk) | Positive decimal (0 if no order) |

Note: Per the project specification, the columns click, basket, and order are mutually exclusive, exactly one can equal 1 per row. The train.csv file contains approximately 2.5 million records.

## 3.2 Further Exploration of the Data

### 3.2.1 Class Distribution (Target Variable)
Figure - Class distribution bar chart (click / basket / order)
Expected finding: order=1 accounts for 25.6% of all records, yielding a 2.9:1 imbalance. A mutual exclusivity was verified with zero violations. This justifies the use of SMOTE in Data Preparation.

[Figure 4: Distributions of price, competitorPrice, revenue. Insert Figure 5: Price vs. competitorPrice scatter coloured by order status. Describe the price range, typical discount relative to rrp, and whether ordered products tend to be priced lower than non-ordered ones.]

### 3.2.3 Price-to-RRP Ratio
[Figure 6: Distribution of price/rrp ratio. State the percentage of products priced below their reference price. Discuss what this implies for the pharmacy's pricing strategy.]

### 3.2.4 Categorical Feature Distributions
[Figures 7–8: Availability, adFlag, campainIndex, genericProduct distributions. Comment on the dominant availability code and advertising campaign coverage.]

### 3.2.5 Temporal Patterns
[Figure 9: Orders per day and daily order rate time series. Describe any trends, seasonality, or anomalies observed across the observation period.]

[Figures 10–13: Order rate by categorical variables (adFlag, availability, campainIndex, genericProduct) and price boxplots by order status. Identify which features show the strongest association with order=1.]

[Figure 9: Orders per day and daily order rate time series. Describe any trends, seasonality, or anomalies observed across the observation period.]

### 3.2.6 Feature–Target Relationships

## 3.3 Verification of Data Quality
The following data quality issues were identified during the exploratory analysis:
[Figure 1: Missing value heatmap. Insert Figure 2: competitorPrice missingness by availability status.]
1. Missing values: competitorPrice contains missing values. The missingness is not random — it varies systematically by availability status, suggesting a structured reason (e.g., no competitor data available for out-of-stock products). See Figure 2.
2. Data leakage: The revenue column is non-zero exclusively when order = 1, confirming that it cannot be used as a predictor without introducing severe data leakage. It will be dropped in Data Preparation.
3. Mixed data types: The content column in items.csv contains both numeric values and strings of the format NxN (e.g., "5X10"), requiring parsing logic.
4. PID join gaps: A subset of PIDs in train.csv have no corresponding entry in items.csv. These records will result in NaN values for all item-level attributes after the join.
5. Scale heterogeneity: Features span multiple measurement scales — nominal (pharmForm, group), ordinal (availability, salesIndex), and proportional (price, rrp). This is relevant for algorithm selection and preprocessing.

# 4 Data Preparation
CRISP-DM Phase 3: Clean, transform, and construct the final modelling dataset. Expected to constitute 50–80% of total project effort.

## 4.1 Handling Duplicates and Missing Values
No duplicate rows were detected in either dataset. Missing values were identified exclusively in the competitorPrice column of train.csv. Two complementary strategies are applied:
1. Imputation: Missing values are replaced with the median competitorPrice computed per product group (group from items.csv). Median imputation is preferred over mean imputation due to the right-skewed distribution of prices.
2. Indicator flag: A binary feature competitorPrice_missing is added to preserve the information that the value was absent, as missingness may itself be predictive.
[State the exact number of imputed records and the resulting missing value count after imputation (should be 0).]

## 4.2 Feature Engineering
The project brief explicitly requires that new attributes be derived from the existing variables. The following engineered features are constructed:
| Engineered Feature | Construction | Purpose / Interpretation |
| --- | --- | --- |
| price_ratio | price / rrp | Relative position of current price to reference retail price (RRP). Values < 1 indicate a discount. |
| price_vs_competitor | price / competitorPrice | Competitive positioning versus the lowest competitor price. Values < 1 indicate the pharmacy undercuts competitors. |
| price_discount_abs | rrp - price | Absolute monetary discount from the reference price. Higher values indicate stronger discounting. |
| content_numeric | Parsed numeric version of content; for NxN strings, multiply both numeric parts (e.g., 5X10 -> 50). | Converts mixed-format package content into a model-ready numeric variable. |

[Description of any additional engineered features, e.g., aggregated click-to-basket ratios per product, or day-of-week indicators if the observation period spans identifiable weekly cycles.]

## 4.3 Balancing the Dataset
The class imbalance identified in Section 3.2.1 presents a significant modelling risk: a naive classifier that predicts order = 0 for all records would achieve high accuracy while providing no business value. To address this, Synthetic Minority Oversampling Technique (SMOTE) is applied to the training set only, generating synthetic samples of the minority class (order = 1) by interpolating between existing positive examples in feature space. The test set is not oversampled to preserve an unbiased evaluation.
[State the class ratio before and after SMOTE, and the resulting training set size.]

## 4.4 Feature Selection
The following features are excluded from the model prior to training:
1. revenue — confirmed data leakage source (see Section 3.3)
2. lineID — unique row identifier with no predictive value
3. click, basket — contemporaneous action flags that are consequences of the same event, not predictors of order
4. pid — used only as a join key; retained product-level information is encoded in item attributes
The remaining features are assessed for relevance using Pearson correlation with the target variable and feature importance scores from a preliminary Random Forest fit.

[Figure 15: Ranked bar chart of feature correlations with order. Discuss the top 5 most important features as identified by the Random Forest feature importance plot.]

## 4.5 Encoding and Scaling
Categorical variables are encoded as follows:
1. campainIndex (A/B/C): one-hot encoded into two binary dummy variables (drop first to avoid multicollinearity)
2. pharmForm, group: one-hot encoded; high-cardinality groups with fewer than 50 occurrences are collapsed into an "Other" category
3. availability: treated as ordinal and label-encoded (1–4)
Feature scaling is applied selectively: StandardScaler (zero mean, unit variance) is applied for Logistic Regression. Tree-based models (Random Forest, Gradient Boosting) do not require scaling and receive unscaled features.

## 4.6 Sampling
Given that train.csv contains approximately 2.5 million records, a stratified random sample is drawn for computational efficiency during the modelling phase. The sample preserves the original order/no-order ratio and is drawn using a fixed random seed (42) for reproducibility.

[State the final sample size used, the justification for this size, and confirm that class proportions are preserved.]

## 4.7 Final Dataset
After all preparation steps, the final modelling dataset has the following properties:
[Insert a summary table: number of rows, number of features, number of positive labels (order=1), class ratio after SMOTE.]

# 5 Modelling
CRISP-DM Phase 4 — Build, configure, and assess multiple classification models.

## 5.1 Selection of Modelling Techniques
Three classification algorithms are selected to represent a range of model families, enabling a comprehensive comparison of performance and interpretability:
1. Logistic Regression — a linear probabilistic baseline model. Selected for its interpretability, fast training time, and its sensitivity to the direction and magnitude of individual features. Serves as the reference against which more complex models are benchmarked.
2. Random Forest — an ensemble of decision trees trained via bootstrap aggregation (bagging). Selected for its robustness to outliers, its native handling of feature interactions, and its ability to provide feature importance rankings without additional computation.
3. Gradient Boosting (XGBoost) — a sequential ensemble method that builds trees to correct the residual errors of previous trees. Selected for its state-of-the-art performance on tabular classification tasks and its built-in regularisation parameters.

## 5.2 Generation of the Test Design
The preprocessed dataset is partitioned into training and test sets using a stratified 80/20 split, ensuring that both subsets reflect the overall class distribution. A further 20% of the training set is reserved as a validation set for hyperparameter tuning via 5-fold stratified cross-validation.
[State the exact split sizes (n rows) for train, validation, and test sets. Confirm that SMOTE is applied only to the training portion.]
No data from the test set is used at any point during feature selection, transformation fitting, or model training. All preprocessing transformers (e.g., StandardScaler, SMOTE) are fit on the training set only and applied to the test set.

## 5.3 Building the Models

### 5.3.1 Logistic Regression
Logistic Regression is implemented using scikit-learn's LogisticRegression class with the lbfgs solver and a maximum of 1000 iterations. The regularisation strength (C) is tuned via 5-fold cross-validation over the range {0.01, 0.1, 1, 10, 100}. The class_weight="balanced" parameter is set as a secondary strategy to account for class imbalance in addition to SMOTE.
[State the optimal C value found. Report training time.]

### 5.3.2 Random Forest
Random Forest is implemented using scikit-learn's RandomForestClassifier. Key hyperparameters tuned include: n_estimators ∈ {100, 200, 500}, max_depth ∈ {10, 20, None}, min_samples_leaf ∈ {1, 5, 10}, and class_weight="balanced_subsample".
[State the optimal hyperparameter combination. Report training time and number of trees in the final model.]

### 5.3.3 Gradient Boosting (XGBoost)
XGBoost is implemented using the XGBClassifier from the xgboost library. The scale_pos_weight parameter is set to the negative-to-positive class ratio to handle imbalance. Key hyperparameters tuned include: n_estimators ∈ {100, 300}, max_depth ∈ {4, 6, 8}, learning_rate ∈ {0.05, 0.1, 0.3}, and subsample ∈ {0.8, 1.0}.
[State the optimal hyperparameter configuration. Report training time.]

## 5.4 Assessing the Models
Each model is evaluated on the held-out test set using the following metrics:
1. Accuracy — proportion of correctly classified instances (reported for completeness, not the primary criterion)
2. Precision (order = 1) — proportion of predicted purchases that are actual purchases
3. Recall (order = 1) — proportion of actual purchases that are correctly predicted
4. F1-score (order = 1) — harmonic mean of precision and recall; primary evaluation criterion
5. AUC-ROC — area under the Receiver Operating Characteristic curve; measures discriminative ability across all thresholds

### 5.4.1 Logistic Regression
[Insert confusion matrix. Report: Accuracy, Precision, Recall, F1, AUC-ROC. Describe the model's strengths and weaknesses based on the confusion matrix.]

### 5.4.2 Random Forest
[Insert confusion matrix and feature importance plot. Report: Accuracy, Precision, Recall, F1, AUC-ROC. Identify the top 5 most important features.]

### 5.4.3 Gradient Boosting (XGBoost)
[Insert confusion matrix. Report: Accuracy, Precision, Recall, F1, AUC-ROC.]

### 5.4.4 Model Performance Comparison
[Insert comparative table: all three models side-by-side across all five metrics. Insert Figure: ROC curves for all three models on one plot. State which model performs best and why.]

# 6 Evaluation
CRISP-DM Phase 5: Assess results against business criteria. Decide whether to deploy or iterate.

## 6.1 Evaluating Results
The evaluation phase examines whether the best-performing model meets the business success criteria defined in Section 2.3.
[State whether each criterion (AUC-ROC ≥ 0.80, F1 ≥ 0.35, Recall ≥ 0.50) is met. If not met, explain why and what would be needed to close the gap.]
From a business perspective, the model's utility is best assessed by estimating the revenue impact of acting on its predictions. A model with high recall but moderate precision will surface the majority of purchasers for targeted pricing interventions, at the cost of also including some non-purchasers. The pharmacy's marketing team must weigh the incremental revenue from correctly identified buyers against the cost of price concessions offered to non-buyers who would have purchased anyway.
[Optionally include a simple revenue impact calculation: estimated uplift in orders captured vs. baseline (no model) given the model's recall and precision.]

## 6.2 Process Review
This section reflects on the analytical process and identifies what could have been done differently or improved:
[Discuss at least 3 of the following: (1) data limitations encountered, (2) alternative features considered but not implemented, (3) model configurations not explored due to time constraints, (4) assumptions made during preprocessing that could be challenged, (5) potential sources of bias in the dataset.]

## 6.3 Determining Next Steps
Based on the evaluation outcomes, the following next steps are recommended:
1. Model refinement: Conduct additional hyperparameter search using Bayesian optimisation (e.g., Optuna) to explore a broader configuration space.
2. Feature expansion: Incorporate aggregated behavioural features at the product level (e.g., historical click-to-order conversion rate per PID) and temporal features derived from the day variable.
3. Deployment pathway: Integrate the trained model into the pharmacy's daily pricing pipeline as a scoring service. Each morning, the model scores the product catalogue and flags high-probability items for dynamic discount adjustment.
4. Monitoring: Implement concept drift detection (e.g., Population Stability Index on key features) to alert when model performance degrades due to market shifts.

# 7 Findings
This project applied the CRISP-DM methodology to build and evaluate three classification models for predicting purchase behaviour in a dynamic pricing context. The key findings are as follows:

## 7.1 Data Insights
[Summarise 3–5 concrete, data-driven insights discovered during EDA. Example: "Products in campaign category B exhibited an order rate 2.3× higher than category A, suggesting campaign design is a stronger predictor of purchase than price alone."]

## 7.2 Model Results
[State the winning model, its test-set metrics, and explain in one paragraph why it outperformed the alternatives.]

## 7.3 Business Implications
[Translate the model findings into 2–3 actionable business recommendations for the pharmacy's pricing team. Write this in non-technical language suitable for a business audience.]

# References
[Add references]