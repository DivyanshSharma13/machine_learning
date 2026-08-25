# Medical Insurance Charges Prediction Using Linear Regression

## Project Overview

This project focuses on predicting medical insurance charges using machine learning. A Linear Regression model is developed to understand how factors such as age, BMI, smoking status, number of children, gender, and other processed features are related to medical insurance expenses.

The project follows a complete basic machine learning workflow, starting from understanding and cleaning the dataset and continuing through preprocessing, feature engineering, model training, prediction, and evaluation.

The main objective is to build an interpretable regression model that can estimate insurance charges for a new individual based on the available input features.

---

## Problem Statement

Medical insurance charges can vary considerably from one person to another. Factors such as age, BMI, smoking habits, number of children, and other demographic characteristics can influence the final insurance cost.

The goal of this project is to use historical insurance data to build a regression model that learns these relationships and predicts the expected insurance charges for new observations.

Since the target variable, `charges`, is a continuous numerical value, this problem is treated as a regression problem.

---

## Dataset

The project uses an insurance dataset containing information about individuals and their corresponding medical insurance charges.

The dataset includes variables related to:

* Age
* Gender
* BMI
* Number of children
* Smoking status
* Region
* Insurance charges

The target variable is:

```text
charges
```

The remaining processed variables are used as input features for the machine learning model.

---

## Features Used

After preprocessing and feature engineering, the model uses features such as:

* `age`
* `is_female`
* `bmi`
* `children`
* `is_smoker`
* `region_southeast`
* `bmi_category_Obese`

The exact feature representation is based on the transformations performed in the notebook.

Categorical variables were converted into numerical representations so that they could be used by the regression algorithm.

---

## Project Workflow

The project follows these major stages:

```text
Dataset
   |
   v
Data Understanding
   |
   v
Exploratory Data Analysis
   |
   v
Data Cleaning
   |
   v
Handling Invalid Values
   |
   v
Categorical Encoding
   |
   v
Feature Engineering
   |
   v
Feature Selection
   |
   v
Train-Test Split
   |
   v
Linear Regression
   |
   v
Predictions
   |
   v
Model Evaluation
   |
   v
Feature/Coefficient Analysis
```

---

## 1. Data Understanding

The dataset was initially inspected to understand its structure, columns, data types, and numerical characteristics.

Basic operations such as checking the first few records, dataset dimensions, data types, descriptive statistics, and unique values were used to understand the data before applying transformations.

This step helps identify:

* Numerical and categorical columns
* Potentially invalid values
* Missing values
* Feature distributions
* The target variable
* Possible relationships between variables

---

## 2. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution of the variables and identify unusual observations.

Different visualizations were used to inspect numerical features and identify potential outliers.

Boxplots were particularly useful for observing the spread of numerical variables and identifying values that may require further investigation.

The relationship between the available features and insurance charges was also considered before selecting the variables for the model.

---

## 3. Data Cleaning

The dataset contains values that may not represent meaningful measurements for certain variables.

For example, zero values in variables such as `Cholesterol` and `RestingBP` in the earlier preprocessing workflow were treated as invalid values and replaced using the mean of the valid observations.

The general approach was:

1. Identify invalid zero values.
2. Exclude those values while calculating the mean.
3. Calculate the mean using valid observations.
4. Replace invalid zero values with the calculated mean.
5. Round the resulting values where appropriate.

This helps prevent invalid measurements from directly affecting the machine learning model.

---

## 4. Categorical Data Processing

Machine learning algorithms such as Linear Regression require numerical input.

Categorical variables were therefore converted into numerical representations.

For example, binary categories were represented using values such as:

```text
0 = False / No
1 = True / Yes
```

Additional categorical features were represented using encoded columns such as:

```text
is_female
is_smoker
region_southeast
```

This allows the categorical information to be incorporated into the regression model.

---

## 5. Feature Engineering

Feature engineering was performed to create useful representations of the original data.

One example is the creation of BMI-related categories, including:

```text
bmi_category_Obese
```

These engineered features provide the model with additional information that may help explain variations in insurance charges.

Feature engineering is useful because the quality and representation of the input variables can have a significant effect on the performance and interpretability of a machine learning model.

---

## 6. Feature and Target Separation

Before training the model, the dataset was divided into:

### Input Features

The independent variables used to make predictions.

```python
X = final_df.drop('charges', axis=1)
```

### Target Variable

The value that the model needs to predict.

```python
y = final_df['charges']
```

In this project:

```text
X → Individual characteristics and processed features
y → Insurance charges
```

---

## 7. Train-Test Split

The dataset is divided into training and testing portions.

A typical 80-20 split is used:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The training data is used to learn the model parameters.

The testing data is kept separate and is used later to evaluate how well the model performs on previously unseen observations.

The `random_state` makes the split reproducible.

---

## 8. Linear Regression Model

The machine learning algorithm used in this project is Linear Regression.

The basic mathematical form for multiple linear regression is:

```text
y = b0 + b1x1 + b2x2 + ... + bnxn
```

where:

* `y` is the predicted insurance charge
* `b0` is the intercept
* `b1 ... bn` are model coefficients
* `x1 ... xn` are the input features

The model learns the coefficients from the training data.

For this project, the model attempts to learn how the processed demographic and health-related features contribute to the predicted insurance charges.

---

## 9. Model Training

The Linear Regression model is created using scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

The model is then trained:

```python
model.fit(X_train, y_train)
```

During training, the model learns coefficients that minimize the squared difference between the actual insurance charges and the predicted charges.

Conceptually, the model tries to minimize the Mean Squared Error:

```text
MSE = average((actual - predicted)^2)
```

The resulting coefficients define the fitted regression equation.

---

## 10. Making Predictions

After training, predictions are generated using the test features:

```python
y_pred = model.predict(X_test)
```

The model has not used the actual `y_test` values to make these predictions.

Instead, it receives the features from `X_test` and estimates the corresponding insurance charges.

The predictions can then be compared with the actual values stored in `y_test`.

---

## 11. Model Evaluation

Several regression metrics can be used to evaluate the model.

### Mean Absolute Error

MAE measures the average absolute difference between the actual and predicted values.

```text
MAE = average(|actual - predicted|)
```

A lower MAE indicates that predictions are, on average, closer to the actual values.

---

### Mean Squared Error

MSE calculates the average squared prediction error.

```text
MSE = average((actual - predicted)^2)
```

Because the errors are squared, larger errors have a greater effect on the final value.

---

### Root Mean Squared Error

RMSE is the square root of MSE:

```text
RMSE = sqrt(MSE)
```

RMSE is easier to interpret than MSE because it is expressed in the same units as the target variable.

---

### R² Score

R² measures the proportion of variation in the target that is explained by the model.

The formula is:

```text
R² = 1 - (sum of squared residuals / total sum of squares)
```

An R² value closer to 1 generally indicates that the model explains a larger proportion of the variation in the target.

For this project, the model achieved an R² of approximately `0.80` on the evaluated test split.

This should be interpreted as approximately 80% of the variation in the test-set insurance charges being explained by the model, rather than as 80% prediction accuracy.

---

## 12. Coefficient Analysis

The learned coefficients can be inspected to understand how the model uses each feature.

```python
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coefficients)
```

The intercept can be obtained using:

```python
model.intercept_
```

A coefficient represents the change in the model's predicted target associated with a one-unit change in that feature, while keeping the other included features constant.

For binary features, moving from `0` to `1` changes the prediction according to that feature's coefficient, with the other features held constant.

This makes Linear Regression useful when model interpretability is important.

---

## 13. Actual vs Predicted Visualization

An actual-versus-predicted plot can be used to visually evaluate the regression model.

The actual insurance charges are placed on one axis and the predicted charges on the other.

A good model should generally produce points that follow the diagonal relationship:

```text
Predicted = Actual
```

The closer the predictions are to this ideal relationship, the closer the model's estimates are to the actual values.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Machine Learning Concepts Covered

This project demonstrates the following concepts:

* Supervised learning
* Regression
* Exploratory Data Analysis
* Data cleaning
* Handling invalid values
* Categorical encoding
* Feature engineering
* Feature selection
* Train-test splitting
* Linear Regression
* Model fitting
* Prediction
* Mean Absolute Error
* Mean Squared Error
* Root Mean Squared Error
* R² Score
* Regression coefficients
* Model interpretation
* Actual vs predicted analysis

---

## Project Structure

A simple repository structure can be:

```text
medical-insurance-linear-regression/
│
├── insurance(1).ipynb
├── insurance(1).csv
├── README.md
└── requirements.txt
```

If the dataset is not included in the repository, the CSV file can be omitted and the notebook can instead document where the dataset was obtained.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-url>
```

### 2. Move into the project directory

```bash
cd medical-insurance-linear-regression
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open:

```text
insurance(1).ipynb
```

and execute the cells sequentially.

---

## Results

The Linear Regression model provides a baseline solution for predicting medical insurance charges.

On the evaluated test split, the model achieved approximately:

```text
R² Score  : 0.80
MAE       : 4,300
RMSE      : 6,000
```

The exact values can vary depending on the preprocessing and train-test split used.

The results indicate that the selected features capture a substantial portion of the variation in insurance charges, while the remaining prediction error suggests that additional factors and nonlinear relationships may also influence insurance costs.

---

## Limitations

Linear Regression makes assumptions about the relationship between the input variables and the target.

The model may not completely capture complex nonlinear relationships in medical insurance costs.

Other limitations include:

* The model depends on the quality of the available features.
* Extreme observations can have a significant effect on ordinary least squares regression.
* Strong multicollinearity can make coefficient interpretation unstable.
* A linear model may not capture all interactions between variables.
* Performance depends on how representative the dataset is of the population being predicted.

---

## Possible Improvements

The project can be extended by:

1. Performing residual analysis.
2. Checking multicollinearity using correlation analysis or VIF.
3. Using cross-validation for a more reliable performance estimate.
4. Trying regularized models such as Ridge and Lasso Regression.
5. Exploring Polynomial Regression for nonlinear relationships.
6. Comparing Linear Regression with tree-based models.
7. Performing additional feature engineering.
8. Tuning the preprocessing and feature selection process.

---

## Conclusion

This project demonstrates a complete introductory machine learning workflow for a regression problem.

The model uses processed demographic and health-related information to estimate medical insurance charges. Linear Regression provides a useful baseline because it is relatively simple, efficient, and easy to interpret.

The project also demonstrates an important machine learning principle: model development does not end after training. The model needs to be evaluated on unseen data, its errors should be analyzed, and its assumptions and limitations should be considered before drawing conclusions.

The current implementation provides a foundation for experimenting with more advanced regression techniques and comparing their performance against the Linear Regression baseline.

---

## Author

**Divyansh Sharma**

Machine Learning | Data Science | C++ | SQL | Python
