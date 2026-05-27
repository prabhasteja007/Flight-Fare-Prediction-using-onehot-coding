
# Flight Fare Prediction Using Linear and Logistic Regression

## Overview

This project implements a hybrid machine learning system to predict flight ticket prices using a combination of **logistic regression** (for price category classification) and **linear regression** (for exact price prediction). The system also includes a fully functional **Graphical User Interface (GUI)** built with `Tkinter`, enabling user-friendly interaction with the model.

The project is based on a dataset of Indian domestic flights sourced from Kaggle and includes preprocessing, feature engineering, model training, evaluation, and deployment via GUI.

---

## Features

- Predict whether a flight price is **low** or **high** using logistic regression.
- Predict the **actual price** using separate linear regression models for low- and high-cost flights.
- Clean, interactive GUI for real-time predictions.
- Data visualizations and performance metrics built-in.

---

## Folder Structure

```
Flight_Fare_Prediction/
├── 601_main.py       # Complete pipeline: preprocessing, models, and GUI
├── README.md         # Project description and instructions
└── requirements.txt  # Python dependencies
```

---

## Dataset

- Source: [Flight Price Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- Size: ~300,000 records
- Features:
  - Airline, Source City, Destination City
  - Arrival and Departure Time
  - Number of Stops
  - Travel Class (Economy/Business)
  - Duration (hours)
  - Days Left (until departure)
  - Price (target variable)

---

## How It Works

1. **Preprocessing**:
   - Drops irrelevant columns (`Unnamed: 0`, `flight`)
   - Encodes categorical variables with one-hot encoding
   - Maps stops and travel class into numerical values
   - Adds a `log_price` feature for smoother regression

2. **Model Training**:
   - Logistic Regression to classify high/low price tiers
   - Two Linear Regression models trained separately for low and high tiers

3. **GUI Interface**:
   - Allows user to input flight details through dropdowns and text fields
   - Displays predicted price and model error metrics (MAE, RMSE)
   - Built using `Tkinter`

---

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
pandas
numpy
scikit-learn
seaborn
matplotlib
tk
```

---

## Dataset

Download `Clean_Dataset.csv` from the [Kaggle link above](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) and place it in the same folder as `601_main.py` before running.

> The dataset is not included in this repo due to file size (~300K records).

---

## How to Run

```bash
pip install -r requirements.txt
python 601_main.py
```

---

## Visualizations

- Confusion matrix for classification
- Residual plots and error histograms
- Feature importance bar charts
- Actual vs Predicted scatter plots
- Price distribution by airline

---

## Authors

- **Prabhas Teja Penugonda** – Metric analysis, documentation  
- **Tharun Kumar Molapally** – GUI development  
- **Tilakraj Manickam Kumar** – Data preprocessing, feature engineering  
- **Venu Bandi** – Model training, hybrid logic design  

---

## License

This project is for academic use under the University of Maryland Baltimore County’s DATA 601 coursework. Not licensed for commercial use.

---

## Acknowledgments

- Kaggle community for providing the dataset
- scikit-learn and Tkinter documentation for modeling and GUI support
