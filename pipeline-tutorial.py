# Import packages
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import data

df = pd.read_csv("/Data/pipeline_data.csv")

# Split into input (X) and output (y)

X = df.drop(["purchase"], axis=1)
y = df["purchase"]

# Split into training data and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Specify numeric vs categorical variables

numeric_features = ["age", "credit_score"]
categorical_features = ["gender"]


###########################
# Set up Pipelines
###########################


# Numeric feature transformer

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
)

# Categorical feature transformer

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="U")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Preprocessing Pipeline

preprocessing_pipeline = ColumnTransformer(
    transformer=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)


###################################
# Apply Pipeline
##
#################################


# Logistic regression

clf = Pipeline(
    steps=[
        ("preprocessing_pipeline", preprocessing_pipeline),
        ("classifier", LogisticRegression(random_state=42)),
    ]
)

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)


###################################
# Save the pipeline
###################################


joblib.dump(clf, "/Data/Model/classification_model.joblib")
