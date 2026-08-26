# Ford Car Price Prediction

## Overview

This project predicts the selling price of used Ford cars using machine learning. The goal is to understand how different car features such as model, year, mileage, engine size, transmission, and fuel type affect price and to build a regression model that can estimate a car's value.

The project covers the complete machine learning workflow, including data exploration, preprocessing, feature engineering, model training, and evaluation.

## Dataset

The dataset contains information about used Ford cars with the following features:

| Feature      | Description            |
| ------------ | ---------------------- |
| model        | Car model              |
| year         | Manufacturing year     |
| mileage      | Distance driven        |
| transmission | Transmission type      |
| fuelType     | Fuel type              |
| tax          | Road tax               |
| mpg          | Fuel efficiency        |
| engineSize   | Engine size in litres  |
| price        | Selling price (Target) |

## Project Workflow

### 1. Data Exploration

The dataset was first analyzed to understand its structure and identify important patterns.

The following analyses were performed:

* Dataset information and summary statistics
* Distribution of numerical features
* Outlier detection using boxplots
* Relationship between features and price using scatter plots
* Correlation analysis with a heatmap

### 2. Data Preprocessing

Before training the model, the data was cleaned and prepared.

Steps included:

* Separating features and target variable
* One-hot encoding for categorical columns
* Label encoding for comparison
* Standard scaling of numerical features
* Train-test split for model evaluation

### 3. Model Building

A Linear Regression model was used to predict car prices.

The model was trained on the processed dataset and evaluated using unseen test data.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Model Evaluation

The model was evaluated using multiple regression metrics:

* R² Score
* Adjusted R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)

These metrics help measure how accurately the model predicts car prices.

## Project Structure

```text
Ford-Car-Price-Prediction/
│
├── ford.ipynb          # Complete project notebook
├── ford.csv            # Dataset
├── README.md           # Project documentation
```

## How to Run

1. Clone this repository.
2. Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open `ford.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells in order.

## Results

The project shows that features such as model, manufacturing year, mileage, engine size, and transmission have a significant impact on the selling price of a used Ford car. The trained regression model is able to estimate prices based on these characteristics and provides a simple baseline solution for used car price prediction.

## Future Improvements

* Train advanced models such as Random Forest and XGBoost
* Perform hyperparameter tuning
* Build a web application for price prediction
* Improve feature engineering for better accuracy

## Author

Divyansh Sharma
