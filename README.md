# KKBox Churn Intelligence Platform

> Production-grade machine learning platform predicting music streaming subscription churn across 970,960 users.

## Results

| Metric | Score |
|--------|-------|
| AUC-ROC (test) | 0.9941 |
| AUC-PR (test) | 0.9537 |
| CV AUC-ROC | 0.9942 ± 0.0001 |
| Precision @ threshold 0.86 | 84.96% |
| Recall @ threshold 0.86 | 92.85% |

## Architecture

Raw Data (30GB logs + 3 CSV files)
→ Data Pipeline — chunked streaming, memory-efficient accumulator
→ Feature Engineering — 55 features: rolling windows, decay signals, interaction terms
→ Model Training — LightGBM, 5-fold StratifiedKFold, scale_pos_weight
→ SHAP Analysis — global beeswarm + individual waterfall explanations
→ Business Impact — risk segmentation, revenue at risk, retention ROI
→ Streamlit Dashboard — live what-if simulator, SHAP explorer


## Dataset

KKBox Music Streaming Churn (WSDM 2018 Kaggle Competition)
- 970,960 labelled users
- 400M+ raw behavioural log records (30GB)
- 9:91 class imbalance handled via scale_pos_weight

## Key Features Engineered

- 7/30/90-day rolling listening windows (active days, total songs, listening seconds)
- Activity decay trend signals (trend_7d_30d, trend_30d_90d)
- Transaction risk signals (cancel_ratio, auto_renew_ratio, days_to_expiry)
- Behavioural engagement score (completion rate × listening intensity)
- Binary risk flags (silent_march, dropout_risk, expiry_imminent)

## Why AUC-PR over AUC-ROC?

With a 91:9 class imbalance, AUC-ROC is misleading — a model that predicts everything as non-churn still scores ~0.5. AUC-PR (0.9537 vs random baseline of 0.09) is the correct primary metric for this problem.

## Project Structure

├── src/
│ ├── data_loader.py # Streaming pipeline for 30GB logs
│ ├── feature_engineering.py # 55 feature transformations
│ ├── model_training.py # LightGBM + CV + imbalance handling
│ ├── model_evaluation.py # PR curve, ROC, threshold analysis
│ ├── shap_analysis.py # Global + local SHAP explanations
│ └── business_impact.py # Revenue at risk + retention ROI
├── app.py # Streamlit dashboard
├── notebooks/
│ └── 01_exploration.ipynb # EDA and hypothesis formation
└── data/
├── raw/ # KKBox competition files (not tracked)
└── processed/ # Engineered features (not tracked)


## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data must be downloaded from the KKBox Churn Prediction Challenge on Kaggle and placed in `data/raw/`.

## Technical Decisions

**Why LightGBM over Random Forest?**
LightGBM handles large datasets faster, supports native class imbalance weighting via scale_pos_weight, and produces SHAP values natively. Random Forest on 970k rows would be significantly slower with no accuracy benefit.

**Why streaming accumulator for log processing?**
The user_logs.csv file is 30GB. Standard pandas read_csv on 16GB RAM fails. A pure Python csv.DictReader accumulator processes the file row by row, updating per-user statistics in place, keeping peak RAM under 2GB.

**Why AUC-PR as primary metric?**
On a 91:9 imbalanced dataset, a naive model achieves 91% accuracy and 0.5 AUC-ROC by predicting all non-churn. AUC-PR penalises this correctly — random performance is 0.09, our model achieves 0.9537.
