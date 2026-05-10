# Load libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import hstack
from sklearn.metrics import classification_report, roc_auc_score


# ================================================================================================
# ===================================== Preliminaries ============================================
# ================================================================================================

# Load data
train = pd.read_csv('/Users/annageiser/Documents/GITHUB/analytics-project/data/raw/train.csv', sep='|')
items = pd.read_csv('/Users/annageiser/Documents/GITHUB/analytics-project/data/raw/items.csv', sep='|')

df = train.merge(items, 
                   on='pid', 
                   how='left',
                   validate="m:1")


# ================================================================================================
# ====================================== Feature Engineering =====================================
# ================================================================================================


# We will perform the following feature engineering steps:
# 1. Sort the data by 'linedID and 'day' to maintain the temporal order for TimeSeriesSplit.
# 2. Take 10% of the whole dataset for model selection (The earliest 10%).
# 3. Create train/test temporal split
# 4. Apply TimeSeriesSplit
# 5. Drop highly correlated features
# 6. Drop irrelevant features
# 7. Handle missing values
# 8. Create new features
# 9. Balance the classes for 'order' prediction using undersampling.
# 10. Encode categorical variables and scale numerical features.
 

# ================================================================================================
# Sort Dataset
# ================================================================================================

# Sort the data by 'linedID and 'day' to maintain the temporal order for TimeSeriesSplit.
df = df.sort_values(["day", "lineID"])

# ================================================================================================
# Data Sampling
# ================================================================================================


# Take 10% of the whole dataset for model selection (The earliest 10%).
df_sample = df.iloc[:int(len(df) * 0.1)]


# ================================================================================================
# Feature Creation
# ================================================================================================


# Create a new column 'priceRatio'
df['priceRatio'] = df['price'] / df['rrp'].replace(0, np.nan)
df['priceRatio'] = df['priceRatio'].fillna(1.0)

# Create a new column 'unitsBought'
df_sample['unitsBought'] = (df_sample['revenue'] / df_sample['price']).round().fillna(0) # We fill any potential NaN values (e.g., from 0/0) with 0 and handle division by zero
df_sample['unitsBought'] = df_sample['unitsBought'].astype(int) # Convert the column to an integer data type for cleanliness

# Create a binary flag for missing 'competitorPrice'
df_sample['missingCompetitorPrice'] = df_sample['competitorPrice'].isnull().astype(int)

# Create weekDay feature from 'day' (assuming 'day' is a sequential day number)
df_sample['weekDay'] = (df_sample['day'] % 7).replace({0: 7})  # Map 0 to 7 for better interpretability


# ================================================================================================
# Train Test split
# ================================================================================================


# Let's define the targets. First task: predict 'order'.
X = df_sample.drop(columns=['order'])
y = df_sample['order']


# Split data (stratified, to preserve order/non-order ratio) ---
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]


X_train.columns
X_test.columns


# ================================================================================================
# TimeSeriesSplit
# ================================================================================================

unique_days = sorted(X_train['day'].unique())

fold_results = []

for i in range(14, len(unique_days) - 7):

    # =============================================================================
    # Create rolling weekly split
    # =============================================================================

    train_days = unique_days[:i]
    val_days = unique_days[i:i+7]

    train_idx = X_train[X_train['day'].isin(train_days)].index
    val_idx = X_train[X_train['day'].isin(val_days)].index

    X_fold_train = X_train.loc[train_idx].copy()
    X_fold_val = X_train.loc[val_idx].copy()

    y_fold_train = y_train.loc[train_idx].copy()
    y_fold_val = y_train.loc[val_idx].copy()


# ================================================================================================
# Drop highly correlated features
# ================================================================================================


num_features = ['competitorPrice',
                'price',
                'revenue',
                'rrp',
                'priceRatio',
                'unitsBought']

cat_features = ['lineID',
                'day',
                'weekDay',
                'pid',
                'adFlag',
                'availability',
                'click',
                'basket',
                'manufacturer',
                'group',
                'content',
                'unit',
                'pharmForm',
                'genericProduct',
                'salesIndex',
                'category',
                'campaignIndex']

# Remove highly correlated features
correlation_matrix = X_fold_train[num_features].corr().abs()

upper = correlation_matrix.where(
    np.tri(correlation_matrix.shape[0], k=-1).astype(bool)
)

to_drop = [
    column for column in upper.columns
    if any(upper[column] > 0.95)
]

X_fold_train.drop(columns=to_drop, inplace=True)
X_fold_val.drop(columns=to_drop, inplace=True)

# ================================================================================================
# Drop irrelevant columns
# ================================================================================================


drop_cols = [
    'click',
    'basket',
    'revenue',
    'lineID',
    'unitsBought',
    'day'
    ]

X_fold_train.drop(columns=drop_cols, inplace=True)
X_fold_val.drop(columns=drop_cols, inplace=True)


# ================================================================================================
# Handle missing values
# ================================================================================================


X_fold_train.fillna({
    'pharmForm': 'Missing',
    'category': 'Missing'
}, inplace=True)

X_fold_val.fillna({
    'pharmForm': 'Missing',
    'category': 'Missing'
}, inplace=True)


# ================================================================================================
# Balance the classes for 'order' prediction using undersampling.
# ================================================================================================


train_data = pd.concat([X_fold_train, y_fold_train], axis=1)

majority = train_data[train_data['order'] == 0]
minority = train_data[train_data['order'] == 1]

majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

train_balanced = pd.concat([
    majority_downsampled,
    minority
])

X_fold_train = train_balanced.drop(columns=['order'])
y_fold_train = train_balanced['order']


# ================================================================================================
# Encode categorical variables and scale numerical features.
# ================================================================================================


# Define remaining numerical and categorical features
cat_features = [
        'weekDay',
        'pid',
        'adFlag',
        'availability',
        'manufacturer',
        'group',
        'content',
        'unit',
        'pharmForm',
        'genericProduct',
        'salesIndex',
        'category',
        'campaignIndex'
    ]

remaining_num_features = [
    col for col in X_fold_train.columns
    if col not in cat_features
]

remaining_cat_features = [
    col for col in X_fold_train.columns
    if col in cat_features
]


# =============================================================================
# Convert categorical features to string
# =============================================================================


X_fold_train[remaining_cat_features] = (
    X_fold_train[remaining_cat_features].astype(str)
)

X_fold_val[remaining_cat_features] = (
    X_fold_val[remaining_cat_features].astype(str)
)


# =============================================================================
# Scale numerical features
# =============================================================================


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_fold_train[remaining_num_features]
)

X_val_scaled = scaler.transform(
    X_fold_val[remaining_num_features]
)


# =============================================================================
# Encode categorical features
# =============================================================================

encoder = OneHotEncoder(handle_unknown='ignore')

X_train_cat = encoder.fit_transform(
    X_fold_train[remaining_cat_features]
)

X_val_cat = encoder.transform(
    X_fold_val[remaining_cat_features]
)


# =============================================================================
# Combine features
# =============================================================================

X_train_final = hstack([
    X_train_scaled,
    X_train_cat
])

X_val_final = hstack([
    X_val_scaled,
    X_val_cat
])


# ================================================================================================
# Verify the shapes and contents of the datasets before modeling
# ================================================================================================


print("Training set shape:", X_train_final.shape)
print("Validation set shape:", X_val_final.shape)


# ================================================================================================


# Sort by day to keep chronological order valid for TimeSeriesSplit
# if 'day' in df_balanced.columns:
#    df_balanced.sort_values('day', inplace=True)

# Let's define the targets. First task: predict 'order'.
# X = df_balanced.drop(columns=['order', 'interaction_type', 'revenue'])  # We drop 'interaction_type' and 'revenue' to avoid data leakage since they are derived from 'order'
# y = df_balanced['order']

# We won't use train_test_split here, we use TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=5)

# For the explicit train/test split, we can just take the last split or a chronological split.
# split_idx = int(len(X) * 0.8)
# X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
# y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

# Encoding Categorical Variables
# categorical_cols = ['pid', 'manufacturer', 'content', 'group', 'pharmForm', 'category', 'adFlag', 'availability', 'genericProduct', 'unit', 'interaction_type']
# categorical_cols = [c for c in categorical_cols if c in X_train.columns]

# We will use Label Encoding for simplicity, but we need to handle unseen categories in the test set.
# We will create a mapping dictionary for each categorical column to ensure that unseen categories in the test set
# encoders = {}
# for col in categorical_cols:
#     X_train[col] = X_train[col].astype(str)
#     X_test[col] = X_test[col].astype(str)
#     le = LabelEncoder()
#     X_train[col] = le.fit_transform(X_train[col])

# Create a mapping dictionary for the test set to handle unseen categories
#     d = dict(zip(le.classes_, le.transform(le.classes_)))
#     X_test[col] = X_test[col].apply(lambda x: d.get(x, -1)) 
#     encoders[col] = le

# Scaling Numerical Variables
# numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

# scaler = StandardScaler()
# X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])

# feature_names = X_train.columns
# X_train_scaled = X_train.values
# X_test_scaled = X_test.values

# print("Training set shape:", X_train_scaled.shape)
# print("Testing set shape:", X_test_scaled.shape)

# We merge the features click, basket, order into one new feature 'interaction_type' to capture the type of interaction in a single column
# def determine_interaction_type(row):
#     if row['order'] == 1:
#         return 'order'
#     elif row['basket'] == 1:
#         return 'basket'
#     elif row['click'] == 1:
#         return 'click'
#     else:
#         return 'none'

# df_sample['interaction_type'] = df_sample.apply(determine_interaction_type, axis=1)


# ================================================================================================
# ============================= Classification Task: Predicting 'order' ==========================
# ================================================================================================


def get_importance_with_tscv(model, X, y, model_name, cv_splits=5):
    # Use TimeSeriesSplit instead of standard CV
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Get cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')

    # Get feature importance
    model.fit(X, y)
    if model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    return pd.Series(importances, index=feature_names, name=model_name), cv_scores.mean()

models_classification = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Note: since we balanced the classes with undersampling, we do not need class_weight='balanced'
top_features_by_model = {}
cv_scores = {}
all_importances = []

for name, model in models_classification.items():
    print(f"Training Classification Model: {name}...")
    imp, cv_score = get_importance_with_tscv(model, X_train_scaled, y_train, name)
    top_features = imp.sort_values(ascending=False).head(10)
    top_features_by_model[name] = top_features
    cv_scores[name] = cv_score
    all_importances.append(imp)

print("\nTimeSeriesSplit CV F1-Scores for 'order':")
for name, score in cv_scores.items():
    print(f"{name}: {score:.4f}")

print("\nTop 10 Features by Model:")
for model_name, features in top_features_by_model.items():
    print(f"\n{model_name}:")
    print(features)



# ================================================================================================
# ============================= Regression Task: Predicting 'revenue' ============================
# ================================================================================================


# Let's predict revenue for the cases where an order actually occurred (or we can predict for everything)
# Typically, predicting revenue makes sense for positive orders.
df_revenue = df[df['order'] == 1].copy()

# Sort by day
if 'day' in df_revenue.columns:
    df_revenue.sort_values('day', inplace=True)

# Target
y_reg = df_revenue['revenue']
# Drop target-related metrics to prevent leakage in regression
cols_to_drop_reg = ['revenue', 'units_bought'] 
# We keep order, basket, click, funnel_status inside as requested, order is just 1.
X_reg = df_revenue.drop(columns=[col for col in cols_to_drop_reg if col in df_revenue.columns])

split_idx_reg = int(len(X_reg) * 0.8)
X_train_reg, X_test_reg = X_reg.iloc[:split_idx_reg].copy(), X_reg.iloc[split_idx_reg:].copy()
y_train_reg, y_test_reg = y_reg.iloc[:split_idx_reg].copy(), y_reg.iloc[split_idx_reg:].copy()

# 4. Encoding Categorical Variables
categorical_cols_reg = [c for c in categorical_cols if c in X_train_reg.columns]

encoders_reg = {}
for col in categorical_cols_reg:
    X_train_reg[col] = X_train_reg[col].astype(str)
    X_test_reg[col] = X_test_reg[col].astype(str)
    le = LabelEncoder()
    X_train_reg[col] = le.fit_transform(X_train_reg[col])
    d = dict(zip(le.classes_, le.transform(le.classes_)))
    X_test_reg[col] = X_test_reg[col].apply(lambda x: d.get(x, -1)) 
    encoders_reg[col] = le

# 5. Scaling Numerical Variables
numerical_cols_reg = [c for c in X_train_reg.columns if c not in categorical_cols_reg]

scaler_reg = StandardScaler()
X_train_scaled_reg = X_train_reg.copy()
X_test_scaled_reg = X_test_reg.copy()

X_train_scaled_reg[numerical_cols_reg] = scaler_reg.fit_transform(X_train_reg[numerical_cols_reg])
X_test_scaled_reg[numerical_cols_reg]  = scaler_reg.transform(X_test_reg[numerical_cols_reg])

models_regression = {
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost Regressor': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

print("Training Regression Models for 'revenue'...")
for name, model in models_regression.items():
    model.fit(X_train_scaled_reg, y_train_reg)
    preds = model.predict(X_test_scaled_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
    mae = mean_absolute_error(y_test_reg, preds)
    print(f"\n{name} Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")