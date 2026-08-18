# Bank Marketing Classification — ML Assignment 2

## a. Problem Statement

This project aims to predict whether a client will subscribe to a term deposit with a Portuguese bank. The target variable is binary (`yes`/`no`), and the prediction is based on client demographic, account, and campaign-related features.

**Prediction Goal:** Given a client's information, determine the likelihood of their subscription to a term deposit.

The bank wants to optimize marketing campaigns by identifying clients most likely to subscribe, thereby reducing unnecessary outreach and improving ROI.

---

## b. Dataset Description

### Overview

| Property                     | Details                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| **Dataset Name**       | UCI Bank Marketing Dataset                                                               |
| **Source**             | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| **Problem Type**       | Binary Classification                                                                    |
| **Total Instances**    | 45,211 rows                                                                              |
| **Number of Features** | 16 predictors + 1 target column                                                          |
| **Target Variable**    | `y` (Subscription: yes/no)                                                             |
| **Class Distribution** | No: 39,922 (88.3%), Yes: 5,289 (11.7%) —**Imbalanced Dataset**                    |

### Feature Descriptions

**Client Demographics (8 features):**

- `age` — Client's age in years
- `job` — Type of job (categorical)
- `marital` — Marital status (categorical)
- `education` — Education level (categorical)
- `default` — Has credit in default? (categorical: yes/no)
- `balance` — Annual balance in euros (numeric)
- `housing` — Has housing loan? (categorical: yes/no)
- `loan` — Has personal loan? (categorical: yes/no)

**Campaign Information (8 features):**

- `contact` — Contact communication type (categorical)
- `day` — Day of month for last contact (numeric)
- `month` — Month of year for last contact (categorical)
- `duration` — Duration of last contact in seconds (numeric)
- `campaign` — Number of contacts during campaign (numeric)
- `pdays` — Days since previous contact (-1 if never contacted) (numeric)
- `previous` — Number of previous contacts (numeric)
- `poutcome` — Outcome of previous campaign (categorical)

### Data Preparation

1. **Categorical Encoding:** Label-encoded 9 categorical features (job, marital, education, default, housing, loan, contact, month, poutcome) to numeric format
2. **Train/Test Split:** Stratified 80/20 split with `random_state=42`
   - Training set: 36,168 samples (80%)
   - Test set: 9,043 samples (20%)
3. **Feature Scaling:** StandardScaler fitted on training data only, then applied to both train and test sets to prevent data leakage

---

## c. GitHub Repository Link

**Public GitHub Repository:**

🔗 **[https://github.com/vinoth-champ/bits_ml_assignment](https://github.com/vinoth-champ/bits_ml_assignment)**

### Repository Contents

The GitHub repository contains all required files for model training, evaluation, and deployment:

- ✅ `app.py` — Streamlit web application for interactive model evaluation
- ✅ `requirements.txt` — Python dependencies (with version pinning)
- ✅ `README.md` — Complete project documentation
- ✅ `test_data.csv` — Test dataset (9,043 samples for evaluation)
- ✅ `model/` — Directory containing:
  - `train_models.py` — Training workflow script
  - `logistic_regression.pkl` — Trained Logistic Regression model
  - `decision_tree.pkl` — Trained Decision Tree model
  - `knn.pkl` — Trained K-Nearest Neighbor model
  - `naive_bayes.pkl` — Trained Naive Bayes model
  - `random_forest.pkl` — Trained Random Forest model (recommended)
  - `encoders.pkl` — Label encoders for categorical features
  - `scaler.pkl` — StandardScaler for feature normalization
  - `comparison_metrics.csv` — Evaluation metrics for all models

---

## d. Models Used

### Overview

Five classification models were trained on the preprocessed Bank Marketing dataset using an 80/20 train-test split with stratification. Each model was evaluated using six standardized metrics.

| Model # | Model Name               | Algorithm Type | Key Hyperparameters  |
| ------- | ------------------------ | -------------- | -------------------- |
| 1       | Logistic Regression      | Linear         | `max_iter=1000`    |
| 2       | Decision Tree            | Tree-based     | `max_depth=10`     |
| 3       | K-Nearest Neighbor (kNN) | Distance-based | `n_neighbors=5`    |
| 4       | Naive Bayes              | Probabilistic  | Default (GaussianNB) |
| 5       | Random Forest (Ensemble) | Ensemble       | `n_estimators=100` |

---

### Evaluation Metrics Comparison Table

**All 6 models evaluated on the same test set (9,043 samples):**

| ML Model Name            | Accuracy         | AUC              | Precision        | Recall           | F1               | MCC              |
| ------------------------ | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Logistic Regression      | 0.8914           | 0.8726           | 0.5945           | 0.2259           | 0.3274           | 0.3205           |
| Decision Tree            | 0.8994           | 0.8534           | 0.5847           | 0.4830           | 0.5290           | 0.4759           |
| kNN                      | 0.8923           | 0.8089           | 0.5717           | 0.3166           | 0.4075           | 0.3724           |
| Naive Bayes              | 0.8380           | 0.8127           | 0.3554           | 0.4726           | 0.4068           | 0.3183           |
| Random Forest (Ensemble) | **0.9067** | **0.9247** | **0.6574** | **0.4225** | **0.5144** | **0.4794** |

#### Metric Definitions

- **Accuracy:** Proportion of correct predictions (TP + TN) / Total = Useful for balanced datasets
- **AUC (Area Under ROC Curve):** Probability of correctly ranking a random positive instance higher than a random negative instance (0-1 scale)
- **Precision:** True Positives / (True Positives + False Positives) = Reduces false alarms
- **Recall:** True Positives / (True Positives + False Negatives) = Captures actual positives
- **F1 Score:** Harmonic mean of Precision and Recall = Balanced metric for imbalanced data
- **MCC (Matthews Correlation Coefficient):** Correlation coefficient between predicted and actual labels (-1 to +1 scale) = Best for imbalanced data

---

### Model Performance Observations Table

| ML Model Name                      | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Achieves highest precision (0.5945) and strong accuracy (0.8914), making it effective at minimizing false alarms. However, the lowest recall (0.2259) and F1 score (0.3274) indicate it misses many actual subscribers. Linear model struggles with imbalanced data; not suitable for practical deployment without class weight balancing.                                                                                      |
| **Decision Tree**            | Provides good balance with recall (0.4830) and precision (0.5847), achieving the highest F1 score (0.5290). Captures non-linear patterns and is interpretable. Moderate AUC (0.8534) suggests reasonable ranking ability. Risk of overfitting with depth=10; could benefit from post-pruning or increased depth constraints.                                                                                                    |
| **kNN**                      | Maintains good accuracy (0.8923) with moderate precision (0.5717). However, lowest AUC score (0.8089) indicates poor probability ranking. Moderate recall (0.3166) and F1 (0.4075) show limited balance. Computationally expensive; distance calculations scale poorly with high-dimensional data.                                                                                                                              |
| **Naive Bayes**              | Highest recall (0.4726) captures most actual subscribers, beneficial for maximizing marketing reach. However, lowest precision (0.3554) and accuracy (0.8380) produce many false positives. Feature independence assumption violated in real data; probabilistic outputs may not align with true probabilities. Trade-off favors coverage over efficiency.                                                                      |
| **Random Forest (Ensemble)** | **BEST OVERALL PERFORMER** — Dominates all key metrics: highest accuracy (0.9067), AUC (0.9247), precision (0.6574), F1 (0.5144), and MCC (0.4794). Ensemble averaging prevents overfitting; handles feature interactions and non-linearity effectively. Slightly lower recall (0.4225) than Decision Tree/Naive Bayes is acceptable given superior performance on other metrics. Recommended for production deployment. |

---

### Overall Winner for your dataset?

**🏆 Random Forest (Ensemble)**

**Justification:**

- **Highest Accuracy (90.67%):** Best overall predictive performance across the dataset
- **Highest AUC (0.9247):** Superior ability to discriminate and rank classes; optimal for threshold tuning
- **Highest Precision (0.6574):** Minimizes costly false positives in marketing campaigns; reduces wasted outreach
- **Balanced Performance:** Strong F1 score (0.5144) and MCC (0.4794) indicate good balance across all metrics
- **Robustness:** Ensemble method naturally mitigates overfitting through bootstrap aggregating
- **Feature Importance:** Handles feature interactions and non-linear relationships inherent in the Bank Marketing dataset
- **Practical Suitability:** For marketing optimization, precision (avoiding false positives) is more critical than recall; Random Forest excels here

Random Forest is the **recommended production model** for predicting term deposit subscriptions.

---

## Running the Streamlit Application

The interactive web application `app.py` demonstrates the trained models in action:

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Features

- Upload test data CSV files
- Select any of the 5 trained models
- View predictions and evaluation metrics
- Visualize confusion matrices
- Compare all models side-by-side

---

## Live Deployment

**Streamlit Community Cloud Deployment:** *(Link to be added after deployment)*
