# ⚡ DeepCSAT — Customer Satisfaction Score Prediction Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Regression-yellow?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A deep-learning-powered analytical dashboard that predicts CSAT scores from 85,907 e-commerce customer support interaction records — enabling real-time service quality insight.**

🔗 **[Live Demo → prasanthkumars777-csat-prediction-app-ig5nuc.streamlit.app](https://prasanthkumars777-csat-prediction-app-ig5nuc.streamlit.app/)**

</div>

---

## 📌 Project Overview

**DeepCSAT** is an end-to-end machine learning project built on a real-world eCommerce customer support dataset. It combines exploratory data analysis, NLP preprocessing, feature engineering, PCA dimensionality reduction, and three regression models (CatBoost, Random Forest, XGBoost) into a fully interactive Streamlit dashboard.

The goal: **predict customer satisfaction (CSAT) scores** from interaction metadata, enabling businesses to proactively identify and fix service quality issues before they escalate.

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Overview** | KPI cards, satisfaction gauge, score distribution, ML pipeline steps |
| 🔍 **Data Explorer** | Raw data preview, column info, missing value analysis, statistics |
| 📊 **EDA** | 14 interactive charts — univariate, bivariate, multivariate analysis |
| 🧪 **Hypothesis Testing** | ANOVA, Welch t-test, Chi-Square with violin plots and heatmaps |
| ⚙️ **Feature Engineering** | ExtraTrees importance, PCA variance explained, NLP pipeline steps |
| 🤖 **Models** | Live training of CatBoost, Random Forest, XGBoost with residual plots |
| 🏆 **Comparison** | Side-by-side MSE/R² bar charts, performance radar chart |
| 🔮 **Predictor** | Real-time CSAT prediction from user-input interaction details |
| 📥 **Export & Report** | Download cleaned CSV, model metrics, auto-generated summary report |

---

## 🧠 ML Pipeline

```
Raw CSV (85,907 rows)
        │
        ▼
┌─────────────────────┐
│  Data Cleaning      │  lowercase, strip, null handling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│  datetime features, response_time_hrs
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Label Encoding     │  channel, category, shift, tenure, etc.
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  NLP Pipeline       │  contractions → lowercase → punctuation
│                     │  → URL strip → stopwords → TF-IDF (100)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ExtraTrees         │  top-8 feature selection
│  Feature Selection  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  StandardScaler     │  zero mean / unit variance
│  + PCA (10 comps)   │  captures ~variance of dataset
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Train/Test Split   │  80% train / 20% test, random_state=42
└────────┬────────────┘
         │
    ┌────┴────┐────────────┐
    ▼         ▼            ▼
CatBoost  Random Forest  XGBoost
    │         │            │
    └────┬────┘────────────┘
         │
         ▼
  MSE · R² · RMSE · Radar Chart
```

---

## 📊 Hypothesis Tests

| Test | Variables | Result |
|------|-----------|--------|
| **ANOVA** | Channel Name vs CSAT | ✅ Channel significantly affects CSAT |
| **Welch t-test** | Item Price (High/Low) vs CSAT | ✅ Price range influences satisfaction |
| **Chi-Square** | Agent Shift vs CSAT | ✅ Shift timing affects CSAT |

---

## 🗂️ Project Structure

```
CSAT_Prediction/
│
├── app.py                          # Streamlit dashboard (main app)
├── file.py                         # Standalone ML pipeline script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   └── eCommerce_Customer_support_data.csv   # Raw dataset (85,907 rows)
│
├── outputs/                        # EDA charts (generated by file.py)
│   ├── 01_csat_distribution.png
│   ├── 02_channel_distribution.png
│   └── ... (22 charts total)
│
├── models/                         # Saved model artifacts
│
└── catboost_info/                  # CatBoost training logs
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PrasanthKumarS777/CSAT_Prediction.git
cd CSAT_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place the CSV file inside the `data/` folder:

```
data/eCommerce_Customer_support_data.csv
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app.py
```

### 5. (Optional) Run the standalone ML pipeline

```bash
python file.py
```
This generates all 22 EDA + model comparison charts in `outputs/`.

---

## 📦 Requirements

```
streamlit>=1.53.1
pandas
numpy
plotly
scikit-learn
scipy
catboost
xgboost
```

Install everything at once:
```bash
pip install streamlit pandas numpy plotly scikit-learn scipy catboost xgboost
```

---

## 📈 Dataset

| Property | Value |
|----------|-------|
| Source | eCommerce Customer Support Interactions |
| Rows | 85,907 |
| Columns | 20 |
| Target | CSAT Score (1–5) |
| Missing Data | ~25.4% overall |
| Duplicates | 0 |

**Key columns:** `channel_name`, `category`, `Sub-category`, `Agent Shift`, `Tenure Bucket`, `Agent_name`, `Issue_reported at`, `issue_responded`, `Item_price`, `CSAT Score`, `Customer Remarks`

---

## 🔮 Live Predictor

The **Predictor** page lets you input real interaction details and get an instant predicted CSAT score:

- Select channel, shift, tenure, category
- Set reported/responded hour and day of week
- Set response time in hours
- Click **🔮 Predict CSAT Score**

The best-performing model (by R²) is automatically selected for prediction.

---

## 👤 Author

**Prasanth Kumar Sahu**

[![GitHub](https://img.shields.io/badge/GitHub-PrasanthKumarS777-181717?style=flat-square&logo=github)](https://github.com/PrasanthKumarS777)

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ using Streamlit · Plotly · Scikit-Learn · CatBoost · XGBoost</sub>
</div>