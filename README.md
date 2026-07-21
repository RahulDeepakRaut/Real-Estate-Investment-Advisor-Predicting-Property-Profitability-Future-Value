# 🏘️ Real Estate Investment Advisor
### Predicting Property Profitability & Future Value

An end-to-end machine learning application that helps investors, buyers, and real estate platforms make data-backed property decisions — classifying properties as "Good Investments" and forecasting their estimated value 5 years into the future.

---

## 📌 Project Goals

- Empower real estate investors with intelligent tools to assess long-term returns
- Support buyers in choosing high-return properties in developing areas
- Help real estate companies automate investment analysis for listings
- Improve customer trust in real estate platforms with data-backed predictions

## 🧩 Problem Statement

Develop a machine learning application to assist potential investors in making real estate decisions. The system:
1. **Classifies** whether a property is a "Good Investment" (Classification)
2. **Predicts** the estimated property price after 5 years (Regression)

The pipeline covers preprocessing, feature engineering, exploratory analysis, model training with MLflow tracking, and deployment via an interactive Streamlit application backed by MySQL.

---

## 🏗️ Architecture

Raw CSV Data → Preprocessing & Feature Engineering → EDA
↓
Classification Model + Regression Model
↓
MLflow Tracking & Model Registry
↓
MySQL Database ←→ Streamlit Application

---

## 📊 Dataset

- **Source**: `india_housing_prices.csv` — 250,000 property listings across India
- **Columns**: State, City, Locality, Property_Type, BHK, Size_in_SqFt, Price_in_Lakhs, Price_per_SqFt, Year_Built, Furnished_Status, Floor_No, Total_Floors, Age_of_Property, Nearby_Schools, Nearby_Hospitals, Public_Transport_Accessibility, Parking_Space, Security, Amenities, Facing, Owner_Type, Availability_Status

### ⚠️ Target Variable Methodology (Important — Read Before Interpreting Results)

The raw dataset has **no historical price data or existing labels** for "Good Investment" or "Future Price." Both targets were engineered transparently using a documented, statistically-grounded approach — nothing was fabricated arbitrarily:

**Appreciation Potential Score** (0–1, drives both targets) is a weighted composite of:
| Component | Weight | Derived From |
|---|---|---|
| Location Index | 30% | Min-max normalized city-average Price/SqFt |
| Infrastructure Index | 30% | Normalized Nearby_Schools, Nearby_Hospitals, Public_Transport_Accessibility, Security (proxy for infrastructure — dataset has no dedicated crime/infra column) |
| Amenity Index | 20% | Count of amenities per listing |
| Age Index | 20% | Inverse of property age (newer = higher potential) |

- **CAGR range (4%–9%)** is grounded in reported Indian residential real estate appreciation rates, not invented
- **Realistic noise** (~±8% on score, ±4% on price) was added to avoid a deterministic/leaking label and produce realistic (non-perfect) model metrics
- `Good_Investment` = binary label from a **median split** of the score (data-driven threshold, not arbitrary)
- `Future_Price_5Y = Price_in_Lakhs × (1 + CAGR)^5 × noise_factor`

This means: **classification metrics reflect a genuinely learnable but engineered pattern**, and **regression R² is naturally high** because current price is the dominant, legitimate driver of 5-year future price (as in real-world forecasting).

---

## 🔬 Exploratory Data Analysis

20 questions answered across four themes:
- **Price & Size Analysis** — distributions, outliers, property-type comparisons
- **Location-Based Analysis** — state/city/locality price trends, BHK mix
- **Feature Relationships** — correlation matrix, schools/hospitals/furnishing/facing vs price
- **Investment & Amenities** — ownership types, parking, amenities, transport accessibility

*(See `/notebooks/eda.ipynb` or `eda.py` for all 20 individual charts.)*

---

## 🤖 Model Development

**Classification — Target: `Good_Investment`**
| Model | Accuracy | F1 Score | ROC AUC |
|---|---|---|---|
| Logistic Regression | 0.753 | 0.759 | 0.834 |
| Decision Tree | 0.699 | 0.698 | 0.745 |
| Random Forest | 0.756 | 0.762 | 0.842 |
| Gradient Boosting | 0.793 | 0.799 | 0.877 |
| **XGBoost (Best)** | **0.803** | **0.807** | **0.889** |

**Regression — Target: `Future_Price_5Y`**
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 19.00 | 13.67 | 0.991 |
| Decision Tree | 23.69 | 16.56 | 0.985 |
| Random Forest | 19.65 | 13.81 | 0.990 |
| **Gradient Boosting (Best)** | **18.93** | **13.27** | **0.991** |
| XGBoost | 19.19 | 13.40 | 0.990 |

Evaluation metrics: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix (classification); RMSE, MAE, R² (regression).

---

## 🧪 MLflow Integration

- Experiment tracking for all 10 model runs (params, metrics, artifacts)
- Model registry with `production` alias for both `GoodInvestmentClassifier` and `FuturePriceRegressor`
- Backend store: MySQL (`mysql+mysqlconnector://`) via SQLAlchemy

```bash
mlflow ui --backend-store-uri mysql+mysqlconnector://<user>:<password>@localhost:3306/mlflow_tracking
```

---

## 🗄️ Database Schema (MySQL)

Single denormalized `properties` table (indexed on `city`, `state`, `price_lakhs`, `bhk`, `property_type`) storing both raw and engineered columns — see `schema.sql` for full DDL.

---

## 💻 Streamlit Application

Three-page interactive app:
1. **🏠 Predict Investment** — form-based prediction with classification result, 5-year price estimate, model confidence, and feature importance
2. **🔍 Property Explorer** — filter listings by city, price range, and BHK (queries MySQL directly)
3. **📊 Market Insights** — city-wise price rankings, investment-rate rankings, and a State × Property Type price heatmap

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Database | MySQL (`mysql-connector-python`) |
| Web App | Streamlit |
| Model Persistence | Joblib (encoders/scalers), MLflow Model Registry |

---

## 📁 Project Structure

real-estate-investment-advisor/
│
├── data/
│ ├── india_housing_prices.csv
│ └── india_housing_prices_cleaned.csv
│
├── notebooks/
│ └── eda.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── eda.py
│ ├── train_models.py
│ ├── mlflow_training.py
│ └── db_setup.py
│
├── models/
│ ├── label_encoders.pkl
│ ├── scaler_clf.pkl
│ └── scaler_reg.pkl
│
├── app.py
├── schema.sql
├── requirements.txt
└── README.md

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd real-estate-investment-advisor

# Install dependencies
pip install -r requirements.txt

# Set up MySQL database
python src/db_setup.py

# Run preprocessing + feature engineering
python src/preprocessing.py

# Run EDA
python src/eda.py

# Train models + log to MLflow
python src/mlflow_training.py

# Launch the Streamlit app
streamlit run app.py
```

### requirements.txt

pandas
numpy
scikit-learn
xgboost
mlflow
streamlit
mysql-connector-python
joblib
matplotlib
seaborn
sqlalchemy

---

## 📈 Key Findings

- Location, infrastructure proximity, and amenities are the strongest engineered drivers of investment potential in this dataset
- Current property price is the dominant predictor of 5-year future value — consistent with real-world real estate forecasting
- XGBoost consistently outperforms other models on classification; Gradient Boosting and XGBoost lead on regression

## 🚀 Future Scope

- Incorporate real historical price/appreciation data to replace the engineered target formula
- Add geocoding for true geographic heatmaps (current dataset has no lat/long)
- Extend the Streamlit app with user authentication and saved searches
- Add SHAP-based explainability for individual predictions

---

## 📄 License

This project is for educational and portfolio purposes.
