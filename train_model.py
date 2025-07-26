# train_model.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

from custom_transformers import TotalVisitingAdder  # ✅ important

# Sample training data
df = pd.read_csv(r"D:\travel_prediction\Travel.csv")  # Replace with your actual dataset
X = df.drop(columns=['ProductPitched'])  # Replace with your actual target column
y = df['ProductPitched']

# Define pipeline
numeric_features = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                    'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome']

categorical_features = ['TypeofContact', 'Occupation', 'Gender', 
                        'MaritalStatus', 'Designation']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('total_visiting_adder', TotalVisitingAdder()),
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

pipeline.fit(X, y)

# ✅ Save pipeline
dump(pipeline, 'decision_tree_pipeline.joblib')
print("Model saved successfully.")
