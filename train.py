from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

# Load data
df = pd.read_csv("bank.csv")
df['y'] = df['y'].map({'yes':1,'no':0})

# Preprocessing
one_hot_cols = ['job', 'marital', 'default', 'housing','loan','contact','poutcome','month']
ordinal_col = ['education']
numeric_cols = [col for col in df.columns if col not in one_hot_cols+ordinal_col+['y']]

preprocess = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), one_hot_cols),
        ('ordinal', OrdinalEncoder(), ordinal_col),
        ('scale', MinMaxScaler(), numeric_cols)
    ],
    remainder='drop'
)

# Split
X = df.drop('y', axis=1)
Y = df['y']

# SMOTE only for training
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_sm, Y_sm = smote.fit_resample(preprocess.fit_transform(X), Y)

# Train model
model = RandomForestClassifier()
model.fit(X_sm, Y_sm)

# -------------------------------
# Deployment pipeline (without SMOTE!)
# -------------------------------
# Combine preprocessing + model for inference
deployment_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

# Save deployment-ready pickle
joblib.dump(deployment_pipeline, "bank_marketing_model.pkl")
