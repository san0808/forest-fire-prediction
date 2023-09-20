# Forest Fire Detection Project

![download](https://github.com/san0808/forest-fire-prediction/assets/72181610/92c63ae2-452c-40c2-9321-4a28070fc6fd)

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Flask Web Application](#flask-web-application)
- [Running the Application](#running-the-application)
- [Random Forest Classifier](#random-forest-classifier)
- [Conclusion](#conclusion)

## Project Overview

The Forest Fire Detection project aims to predict the probability of forest fire occurrence based on various environmental factors. This README provides a detailed explanation of the project, including data preprocessing, model training, and how to use the Flask web application for predictions.

### Project Goals

- Predict the probability of forest fires based on environmental parameters.
- Build and deploy a user-friendly web application for predictions.

## Data Preprocessing

### Data Source

The project uses a synthetic dataset named "Synthetic_Forest_fire_dataset.csv" to train and test the model. The dataset contains the following columns:

- **Area**: Geographic area description.
- **Oxygen**: Oxygen content in parts per million (ppm).
- **Temperature**: Temperature in degrees Celsius (°C).
- **Humidity**: Humidity percentage (%).
- **Wind Speed**: Wind speed.
- **Vegetation Density**: Density of vegetation in the area.
- **Proximity to Water**: Categorical feature indicating the proximity to water bodies (e.g., "Near" or "Far").
- **Soil Type**: Categorical feature representing the soil type (e.g., "Sandy," "Loamy," "Clayey").

### Data Preprocessing Steps

1. **Loading Data**: The dataset is loaded into a Pandas DataFrame for further processing.
2. **Feature Engineering**: The target variable "Fire Occurrence" is separated from the feature matrix.
3. **Feature Types**: Categorical and numerical features are identified.
4. **Categorical Encoding**: Categorical features are one-hot encoded using a **`OneHotEncoder`**.
5. **Numerical Scaling**: Numerical features are standardized using **`StandardScaler`**.
6. **Column Transformation**: All preprocessing steps are combined into a **`ColumnTransformer`** named **`preprocessor`** to handle both categorical and numerical features.

## Model Training

### Model Selection

The selected model for this project is a Random Forest Classifier. This classifier is well-suited for binary classification tasks like forest fire prediction.

### Model Pipeline

A scikit-learn **`Pipeline`** is used to create a unified workflow that includes both data preprocessing (**`preprocessor`**) and the Random Forest Classifier (**`rf_classifier`**).

### Hyperparameter Tuning

Hyperparameter tuning is performed using **`GridSearchCV`** to find the best combination of hyperparameters for the Random Forest Classifier. The hyperparameters considered include:

- Number of Estimators (**`n_estimators`**)
- Maximum Depth (**`max_depth`**)
- Minimum Samples Split (**`min_samples_split`**)

### Saving the Best Model

The best-trained model obtained from hyperparameter tuning is saved to a file named "best_model.pkl" using the **`pickle`** library for later use.

## Flask Web Application

A Flask web application (**`app.py`**) is developed to allow users to interact with the trained model and predict forest fire probabilities.

- The application serves an HTML form for users to input environmental parameters.
- When the user submits the form, the Flask route **`/predict`** processes the input and returns the prediction results.
- The prediction includes the probability of forest fire occurrence and a prediction text ("Your Forest is in Danger" or "Your Forest is Safe").

## Running the Application

To run the application locally, execute **`app.py`**. The application can be accessed through a web browser.

## Random Forest Classifier

### Introduction

The Random Forest Classifier is a popular machine learning algorithm used for classification tasks. It belongs to the ensemble learning family, which means it combines the predictions of multiple individual models (decision trees) to make more accurate and robust predictions. Random Forests are particularly well-suited for both classification and regression tasks.

### Key Characteristics

1. **Ensemble of Decision Trees**: The Random Forest Classifier is an ensemble of decision trees. Each tree in the forest is trained independently on a random subset of the data and features.
2. **Random Subsampling**: During the training process, a random subset of the training data is selected with replacement. This process is known as bootstrapping. Additionally, a random subset of features is considered for each split in the tree. These randomization techniques reduce overfitting and increase model diversity.
3. **Voting Mechanism**: In classification tasks, each tree in the forest makes a prediction, and the final class prediction is determined by a majority vote (mode) among the individual tree predictions. In regression tasks, the final prediction is typically the average (mean) of the individual tree predictions.
4. **Highly Parallelizable**: Random Forests are naturally parallelizable since each tree can be trained independently. This makes them suitable for parallel or distributed computing environments.

### Advantages

1. **Robust to Overfitting**: Random Forests are less prone to overfitting compared to individual decision trees, thanks to the randomness introduced during training.
2. **High Accuracy**: They generally provide high accuracy and are considered one of the top-performing algorithms in many classification tasks.
3. **Feature Importance**: Random Forests can measure the importance of each feature in the classification process. This is valuable for feature selection and understanding which factors contribute most to predictions.
4. **Handles Both Categorical and Numerical Features**: Random Forests can handle a mixture of categorical and numerical features without requiring extensive preprocessing.
5. **Non-Linear Relationships**: They can capture complex non-linear relationships in the data.

### Hyperparameters

Random Forests have various hyperparameters that can be tuned to optimize model performance. Some key hyperparameters include:

- **n_estimators**: The number of decision trees in the forest. A higher number typically leads to better performance but increases computation time.
- **max_depth**: The maximum depth of each decision tree. Controlling tree depth helps prevent overfitting.
- **min_samples_split**: The minimum number of samples required to split a node in a tree. It controls tree growth and can prevent small branches.
- **max_features**: The maximum number of features considered for each split. It introduces randomness into the feature selection process.

### Applications

Random Forest Classifiers are widely used in various applications, including:

- **Medical Diagnosis**: Predicting diseases based on patient data.
- **Credit Scoring**: Determining creditworthiness of individuals.
- **Image Classification**: Classifying objects in images.
- **Environmental Monitoring**: Detecting forest fires, as in this project.
- **Anomaly Detection**: Identifying outliers or unusual events in datasets.

In the Forest Fire Detection project, the Random Forest Classifier is used to learn patterns in environmental data and make predictions about the occurrence of forest fires. Its robustness, accuracy, and ability to handle both categorical and numerical features make it a suitable choice for this task. The hyperparameter tuning process using **`GridSearchCV`** helps find the best configuration for the Random Forest model, ensuring optimal performance.

### Conclusion

In conclusion, the Forest Fire Detection project is a comprehensive initiative to predict the probability of forest fire occurrence using a Random Forest Classifier. This method has several advantages, including its ability to handle mixed data types, robustness against overfitting, and high accuracy in classification tasks. Additionally, the use of hyperparameter tuning ensures that the model is optimized for performance.

However, while the Random Forest Classifier is a robust choice, it is essential to be aware of potential challenges such as data quality, class imbalance, and the need for continuous model evaluation and adaptation to changing environmental conditions. Additionally, model interpretability and computational resource constraints should be considered when deploying the model in real-world applications.

Overall, this project combines data preprocessing, machine learning modeling, and a user-friendly web application to address a critical environmental concern—forest fire detection. It provides a foundation for further research and development in the field of environmental monitoring and prediction.


