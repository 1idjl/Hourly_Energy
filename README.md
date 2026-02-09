# Energy Consumption Forecasting with LSTM

This project focuses on building and evaluating a Long Short-Term Memory (LSTM) neural network model to forecast hourly energy consumption based on the `PJME_hourly.csv` dataset.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Methodology](#methodology)
    *   [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
    *   [2. Feature Engineering](#2-feature-engineering)
    *   [3. Data Splitting](#3-data-splitting)
    *   [4. Model Building and Training (LSTM)](#4-model-building-and-training-lstm)
    *   [5. Model Evaluation and Saving](#5-model-evaluation-and-saving)
3.  [Results](#results)
4.  [Technologies Used](#technologies-used)

## Project Overview

The goal was to develop a deep learning model capable of predicting future energy consumption values (`PJME_MW`) using historical hourly data. The project follows a standard machine learning pipeline, emphasizing the temporal nature of the data.

## Methodology

### 1. Exploratory Data Analysis (EDA)

The initial step involved understanding the structure and characteristics of the dataset.

*   **Dataset Shape:** 104,855 entries, 2 columns (`Datetime`, `PJME_MW`).
*   **Data Quality:** No missing values were found.
*   **Statistical Summary:** Key metrics like mean, standard deviation, and percentiles were calculated for the target variable.
*   **Visualizations (using Plotly):**
    *   **Time Series Plot:** Revealed clear seasonal patterns (higher consumption in summer/winter) and daily fluctuations.
    *   **Box Plots (by Hour/Month):** Highlighted variations in consumption based on time of day and month.
    *   **Histogram:** Showed the distribution of consumption values, revealing a right-skewed distribution.

### 2. Feature Engineering

New features were created to help the model learn temporal dependencies and patterns.

*   **Time-based Features:** `Year`, `Month`, `Day`, `Hour`, `DayOfWeek`, `WeekOfYear`.
*   **Categorical Feature:** `IsWeekend` (binary indicator).
*   **Lagged Features:** `PJME_Lag_1`, `PJME_Lag_2`, `PJME_Lag_3`, `PJME_Lag_24`, `PJME_Lag_48`, `PJME_Lag_168` (values from 1 hour, 2 hours, ..., 1 week ago).
*   **Rolling Statistics:** `PJME_RollingMean_24`, `PJME_RollingMean_168`, `PJME_RollingStd_24`, `PJME_RollingStd_168` (mean and std over 24h and 168h windows).
*   **Cyclical Encoding:** `Hour_sin`, `Hour_cos`, `DayOfWeek_sin`, `DayOfWeek_cos` (to capture the cyclical nature of time features).
*   **Handling NaNs:** Rows resulting from lag/rolling feature creation were dropped to ensure clean data for training.

### 3. Data Splitting

The engineered dataset was split temporally to prevent data leakage:

*   **Training Set:** 70% of the data (earliest).
*   **Validation Set:** 15% of the data (middle).
*   **Test Set:** 15% of the data (most recent).
*   **Separation:** Features (`X`) and target (`y`) were separated for each split.

### 4. Model Building and Training (LSTM)

An LSTM model was constructed and trained to predict the next hour's consumption.

*   **Data Preparation:**
    *   Features and target were scaled using `MinMaxScaler`.
    *   Data was reshaped into sequences using a `lookback_steps` of 24 (meaning the model uses the last 24 hours of data to predict the next hour).
*   **Model Architecture:**
    *   Input Layer accepting sequences of shape `(timesteps=24, features=N)`.
    *   LSTM Layer (50 units, ReLU activation, `return_sequences=True`).
    *   LSTM Layer (50 units, ReLU activation).
    *   Dense Layer (1 unit, linear activation for regression).
*   **Compilation:** Optimizer: `Adam`, Loss Function: `Mean Squared Error (MSE)`, Metrics: `Mean Absolute Error (MAE)`.
*   **Training:** The model was trained for 20 epochs with a batch size of 32, using the validation set for monitoring.

### 5. Model Evaluation and Saving

*   **Prediction:** The trained model made predictions on the test set.
*   **Inverse Scaling:** Predictions and actual values were transformed back to their original scale.
*   **Metrics:** Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) were calculated on the test set.
*   **Visualization:** Actual vs. predicted values were plotted for a sample period. Training/validation loss and MAE curves were also plotted.
*   **Model Saving:** The trained LSTM model was saved in `.h5` format for later use.

## Results

The LSTM model demonstrated its ability to learn from the temporal patterns in the energy consumption data. While specific numerical results (RMSE, MAE) depend on the exact parameters and training runs, the visualizations typically showed the model capturing the general trend and fluctuations reasonably well, especially compared to a simple baseline. The training history plots indicated that the model learned effectively without obvious signs of severe overfitting within the limited epochs.

## Technologies Used

*   **Python:** Programming language.
*   **Pandas:** Data manipulation and analysis.
*   **NumPy:** Numerical operations.
*   **Plotly:** Interactive data visualization.
*   **Scikit-learn (sklearn):** Metrics calculation and preprocessing (scaling).
*   **TensorFlow/Keras:** Building, training, and evaluating the LSTM model.