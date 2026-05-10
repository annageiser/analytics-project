# Load libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# ================================================================================================
# ============================= Preliminaries ============================
# ================================================================================================

# Load data
train = pd.read_csv('/Users/annageiser/Documents/GITHUB/analytics-project/data/raw/train.csv', sep='|')
items = pd.read_csv('/Users/annageiser/Documents/GITHUB/analytics-project/data/raw/items.csv', sep='|')

df = train.merge(items, 
                   on='pid', 
                   how='left',
                   validate="m:1")

# Handle Missing Values

num_features = ['competitorPrice',
                'price',
                'revenue',
                'rrp']

cat_features = ['lineId',
                'day',
                'pid',
                'adFlag',
                'availability',
                'click',
                'basket',
                'order',
                'manufacturer',
                'group',
                'content',
                'unit',
                'pharmForm',
                'genericProduct',
                'salesIndex',
                'category',
                'campaignIndex']

num_transformer = StandardScaler()  # Standardization for numerical features
cat_transformer = OneHotEncoder(handle_unknown='ignore')  # One-hot encoding for categorical features

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

selected_features = ['lineID',
                     'day',
                     'pid',
                     'adFlag',
                     'availability',
                     'competitorPrice',
                     'click',
                     'basket',
                     'order',
                     'price',
                     'revenue',
                     'manufacturer',
                     'group',
                     'content',
                     'unit',
                     'pharmForm',
                     'genericProduct',
                     'salesIndex',
                     'category',
                     'campaignIndex',
                     'rrp']


X = df[selected_features]
y = df['order']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Data transformation
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # ML model
])

# Train the model
pipeline.fit(X_train, y_train)
print("Model training complete!")


# Make predictions
y_pred = pipeline.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Save the trained pipeline
joblib.dump(pipeline, 'ml_pipeline.pkl')

# Load the model
loaded_pipeline = joblib.load('ml_pipeline.pkl')

# Predict using the loaded model
sample_data = pd.DataFrame([{'Pclass': 3, 'Sex': 'male', 'Age': 25, 'SibSp': 0, 'Parch': 0, 'Fare': 7.5, 'Embarked': 'S'}])
prediction = loaded_pipeline.predict(sample_data)
print(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did not Survive'}")