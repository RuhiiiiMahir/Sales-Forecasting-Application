# Predict Future Sales with XGBoost

This repository contains a machine learning project aimed at predicting future sales using the XGBoost model. The project involves data preprocessing, model training, and deployment using Streamlit for real-time predictions.

## Project Structure

- **.gitattributes**: This file is used to manage text file normalization across different platforms.
- **app.py**: A Streamlit application file that allows users to input feature values and get sales predictions in real-time.
- **predict-future-sales-with-xg-boost.ipynb**: A Jupyter Notebook that contains the complete process of data exploration, feature engineering, model training, and evaluation.
- **xgboost.pickle.dat**: A serialized XGBoost model saved as a pickle file.
- **xgboost_model.json**: The XGBoost model saved in JSON format.

## Getting Started

### Prerequisites

To run this project, you will need the following libraries:

- Python 3.6+
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn
- Streamlit

## Running the Project

### Run the Jupyter Notebook:

- Open `predict-future-sales-with-xg-boost.ipynb` in Jupyter Notebook or Jupyter Lab.
- Execute the cells sequentially to understand the entire process of model training and evaluation.

### Deploy the Streamlit App:

- To start the Streamlit application and make predictions using the trained model, run the following command:

    ```bash
    streamlit run app.py
    ```

- The app will open in your default web browser where you can input feature values and get real-time predictions.

## Model Explanation

The model uses the XGBoost algorithm, which is a robust and efficient implementation of gradient boosting for supervised learning tasks. This model is specifically tuned for the sales prediction problem, taking into account multiple features such as:

- Shop ID
- Item ID
- Shop Category
- Shop City
- Item Category ID
- Category Type
- Category Subtype
- Monthly Sales Data Lags

The model was trained using hyperparameter tuning to optimize its performance, which results in accurate and reliable sales predictions.
