
# House Price Prediction Using Multiple Regression Models

This project aims to predict house prices using various machine learning regression techniques. The dataset used in this project contains information on different features of houses such as size, condition, and location, and the goal is to predict the sale price based on these features.

## Overview

In this project, we perform several tasks:
- Data preprocessing and feature selection
- Handling missing data
- Visualizing relationships between features and target variable
- Model training and evaluation using various regression models
- Comparing the performance of these models

The models included are:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet**
- **Support Vector Regression (SVR)**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Polynomial Regression (degree=2)**

The goal is to assess which model performs the best in terms of predicting house prices.

## Steps Performed in the Code

### 1. **Importing Libraries and Setting Up**
The required libraries are imported, including:
- `numpy`, `pandas`, `seaborn`, `matplotlib` for data manipulation and visualization.
- Various scikit-learn models (`LinearRegression`, `Ridge`, `Lasso`, etc.) for model training.
- The `xgboost` library is used for the XGBoost regressor model.
- `StandardScaler` is used to scale the data for better model performance.

### 2. **Loading and Exploring the Dataset**
The dataset is loaded using `pandas.read_csv()`, which reads the data from a CSV file. The first few rows of the dataset are displayed using `df.head()`, and the size and structure of the data are checked with `df.shape` and `df.info()`.

The dataset contains several columns, and we aim to predict the `SalePrice` based on other features.

### 3. **Feature Selection**
To reduce the complexity of the model and focus on relevant features, the code selects features based on their correlation with the target variable `SalePrice`. Numerical columns with a correlation greater than 0.50 or less than -0.50 with `SalePrice` are selected, along with a few categorical variables that are expected to have significant impact, such as `MSZoning` and `Heating`.

### 4. **Handling Missing Data**
The code checks for any missing values in the dataset using `df.isna().sum()`. It calculates the total number of missing values, though the code doesn't explicitly handle them in the snippet, suggesting that the dataset is either clean or that handling missing data is done elsewhere.

### 5. **Data Visualization**
Several visualizations are created to understand the relationships between different features and the target variable (`SalePrice`):
- A heatmap visualizes the correlations between numerical variables.
- Pairplots help visualize the pairwise relationships between the important numerical features.
- Jointplots are used to visualize the relationship between `SalePrice` and other features such as `OverallQual`, `TotalBsmtSF`, etc.

### 6. **Data Preprocessing**
The dataset is split into features (`X`) and the target variable (`y`), and the categorical features are one-hot encoded using `pd.get_dummies()`. This converts categorical variables into numerical format.

Additionally, numerical features are standardized using `StandardScaler()` to ensure they have a mean of 0 and a standard deviation of 1. Standardization is important as it improves the performance of models that are sensitive to feature scaling.

### 7. **Train-Test Split**
The dataset is split into training and testing subsets using `train_test_split()`, where 80% of the data is used for training and 20% for testing.

### 8. **Model Training and Evaluation**
The following regression models are trained and evaluated:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet**
- **Support Vector Regression (SVR)**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Polynomial Regression (degree=2)**

For each model:
- The model is trained on the training set (`X_train`, `y_train`).
- Predictions are made on the test set (`X_test`).
- The model’s performance is evaluated using common metrics: **MAE (Mean Absolute Error)**, **MSE (Mean Squared Error)**, **RMSE (Root Mean Squared Error)**, and **R2 Score**.
- Cross-validation is performed using `cross_val_score()` to evaluate the model's stability by splitting the data into multiple folds.

Here are the results from each model:

### Model Performance Results

| Model                          | MAE       | MSE        | RMSE       | R2 Score   | RMSE (Cross-Validation) |
|---------------------------------|-----------|------------|------------|------------|-------------------------|
| **Linear Regression**           | 24593.67  | 1.06e+09   | 32551.45   | 0.871      | 33412.67               |
| **Ridge Regression**            | 24593.77  | 1.06e+09   | 32551.75   | 0.870      | 33413.80               |
| **Lasso Regression**            | 25910.63  | 1.11e+09   | 33349.88   | 0.860      | 33560.43               |
| **ElasticNet**                  | 25919.99  | 1.11e+09   | 33351.29   | 0.859      | 33561.54               |
| **Support Vector Regression (SVR)** | 44323.11  | 2.42e+09   | 49223.97   | 0.644      | 55012.23               |
| **Random Forest Regressor**     | 22674.36  | 9.78e+08   | 31374.87   | 0.889      | 31923.13               |
| **XGBoost Regressor**           | 21546.98  | 8.82e+08   | 29704.01   | 0.899      | 30814.74               |
| **Polynomial Regression (degree=2)** | 27087.39  | 1.16e+09   | 34007.12   | 0.849      | 34734.20               |

### 9. **Model Comparison**
After training and evaluating each model, the results are stored in a DataFrame. The models are then compared based on their RMSE (Root Mean Squared Error) score, with a lower RMSE indicating better model performance.

A bar plot is generated to visualize the performance of each model, showing the RMSE for each model. The model with the lowest RMSE (cross-validation) is considered the best.

### 10. **Conclusion**
Based on the results, the **XGBoost Regressor** achieved the lowest RMSE (29704.01), followed by the **Random Forest Regressor** (31374.87). These two models performed better than the other models in terms of RMSE and are considered the best for predicting house prices.

## Requirements

Before running the code, you need to install the following Python libraries:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
```

- **Numpy**: Used for numerical operations.
- **Pandas**: For data manipulation and handling.
- **Seaborn**: For data visualization.
- **Matplotlib**: For plotting graphs.
- **Scikit-learn**: For machine learning models and evaluation.
- **XGBoost**: For the XGBoost model.

## Dataset

The dataset used in this project can be downloaded from the Kaggle House Prices dataset. Make sure to place the dataset file (`train.csv`) in the appropriate directory.

## Usage

1. Clone the repository or download the Python script.
2. Run the script to load, preprocess, train, and evaluate multiple regression models.
3. Check the printed evaluation metrics and the final bar plot comparing model performances.

---

This project provides a solid foundation for anyone looking to explore machine learning for regression tasks. By comparing multiple models, you can easily find the best-performing model for predicting house prices based on the available features.

---

Let me know if you'd like further adjustments or have any other questions!
