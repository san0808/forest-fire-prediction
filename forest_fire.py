import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the synthetic dataset
data = pd.read_csv("Synthetic_Forest_fire_dataset.csv")

# Separate the target variable
X = data.drop("Fire Occurrence", axis=1)
y = data["Fire Occurrence"]

# Define categorical and numerical features
categorical_features = ["Area", "Proximity to Water", "Soil Type"]
numerical_features = ["Oxygen", "Temperature", "Humidity", "Wind Speed", "Vegetation Density"]

# Create transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False, drop='first'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Define the model (Random Forest Classifier)
rf_classifier = RandomForestClassifier(random_state=0)

# Create a pipeline that includes preprocessing and modeling
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', rf_classifier)])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_

# Save the best model to a file
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

print("Best model saved")


