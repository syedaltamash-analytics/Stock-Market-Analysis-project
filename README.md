# ⚡ Energy Production Prediction — Regression Project

A machine learning project that predicts **power plant energy output** based on environmental sensor readings. Multiple regression models are trained, compared, and deployed as an interactive web app using Streamlit.

---
## 👥 Group Members

- Syed Mohd Altamash

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Streamlit App](#streamlit-app)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)

---

## Overview

This project applies supervised regression techniques to predict the hourly electrical energy output of a Combined Cycle Power Plant (CCPP). The workflow covers:

- Exploratory Data Analysis (EDA)
- Data cleaning (duplicates, outlier removal)
- Training and evaluating 11 regression models
- Saving the best model and deploying it with a Streamlit web app

---

## Dataset

The dataset contains sensor readings collected from a power plant and includes the following features:

| Feature | Description |
|---|---|
| `temperature` | Ambient temperature (°C) |
| `exhaust_vacuum` | Exhaust vacuum pressure (cm Hg) |
| `amb_pressure` | Ambient pressure (atm) |
| `r_humidity` | Relative humidity (%) |
| `energy_production` | Electrical energy output in MW *(target)* |

**Preprocessing steps:**
- Removed 41 duplicate rows
- No missing values found
- Outliers removed using the IQR method
- 80/20 train-test split

---

## Project Structure

```
├── Regression_Project.ipynb   # Main notebook with full analysis
├── app.py                     # Streamlit web application
├── energy_model.pkl           # Saved Random Forest model
├── data/
│   └── energy_production_data.csv
└── README.md
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/energy-production-prediction.git
   cd energy-production-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib
   ```

---

## Usage

### Run the Jupyter Notebook

Open `Regression_Project.ipynb` to explore the full analysis, visualizations, and model comparisons.

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501` and enter sensor values to get a predicted energy output.

---

## Models & Results

Eleven regression models were trained and evaluated using R², MAE, and RMSE:

| Model | Train R² | Test R² | Test MAE |
|---|---|---|---|
| Linear Regression | 0.9284 | 0.9283 | 3.64 |
| Ridge Regression | — | — | — |
| Lasso Regression | — | — | — |
| Elastic Net | — | — | — |
| **Random Forest** | **0.9945** | **0.9617** | **2.34** |
| Decision Tree | 1.0000 | 0.9327 | 2.97 |
| Bagging Regressor | 0.9923 | 0.9588 | 2.44 |
| **XGBoost** | **0.9878** | **0.9647** | **2.23** |
| AdaBoost | 0.8977 | 0.8892 | 4.59 |
| Gradient Boosting | 0.9539 | 0.9461 | 2.98 |
| K-Nearest Neighbors | — | — | — |

> **Best models:** XGBoost and Random Forest achieve the highest test R² with the lowest error, striking the best balance between accuracy and generalization.

---

## Streamlit App

The app loads the saved `energy_model.pkl` (Random Forest) and lets users input live sensor readings to get instant predictions.

**Input fields:**
- Temperature (°C)
- Exhaust Vacuum (cm Hg)
- Ambient Pressure (atm)
- Relative Humidity (%)

**Output:** Predicted energy output in MW.

---

## Conclusion

- **XGBoost** and **Random Forest** are the top performers overall.
- **Decision Tree** overfits severely (perfect train score, lower test score).
- **Linear Regression** is simple, consistent, and a solid baseline.
- **AdaBoost** shows the weakest generalization on this dataset.

For production use, **XGBoost** is recommended for its balance of performance and generalization.

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)

---


## 👤 Author

<div align="center">

### Syed Mohd Altamash

*Data Science & Machine Learning Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-syedaltamash--analytics-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/syedaltamash-analytics)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Syed%20Mohd%20Altamash-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syedaltamash-analytics)

</div>

---



