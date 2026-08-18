"""Staged data preparation workflow for the Bank Marketing assignment."""

from pathlib import Path
import logging
import pickle

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Configure logging before running the first workflow stage.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bank-marketing" / "bank_marketing.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "y"
RANDOM_STATE = 42


# Load the selected dataset.
def load_dataset(path):
    logger.info("Loading dataset from %s", path)
    dataset = pd.read_csv(path, sep=";")
    logger.info("Dataset loaded with shape %s", dataset.shape)
    return dataset


dataset = load_dataset(DATA_PATH)


# Inspect the dataset structure and target distribution.
def inspect_dataset(dataframe):
    logger.info("Inspecting columns: %s", list(dataframe.columns))
    logger.info("Data types:\n%s", dataframe.dtypes.to_string())
    logger.info("Missing values:\n%s", dataframe.isna().sum().to_string())
    logger.info("Target distribution:\n%s", dataframe[TARGET_COLUMN].value_counts().to_string())
    if dataframe.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    logger.info("Dataset inspection completed successfully")


inspect_dataset(dataset)


# Encode categorical features and the binary target.
def encode_dataset(dataframe):
    encoded = dataframe.copy()
    feature_encoders = {}

    categorical_columns = encoded.drop(columns=TARGET_COLUMN).select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in categorical_columns:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        feature_encoders[column] = encoder

    target_encoder = LabelEncoder()
    encoded[TARGET_COLUMN] = target_encoder.fit_transform(
        encoded[TARGET_COLUMN].astype(str)
    )
    encoders = {"features": feature_encoders, "target": target_encoder}
    logger.info("Encoded %d categorical feature columns and target", len(feature_encoders))
    return encoded, encoders


encoded_dataset, encoders = encode_dataset(dataset)
with (MODEL_DIR / "encoders.pkl").open("wb") as file:
    pickle.dump(encoders, file)
logger.info("Saved encoders to %s", MODEL_DIR / "encoders.pkl")


# Define predictors and the binary target, then create a stratified split.
def split_dataset(original_data, encoded_data):
    features = encoded_data.drop(columns=TARGET_COLUMN)
    target = encoded_data[TARGET_COLUMN]
    original_features = original_data.drop(columns=TARGET_COLUMN)
    original_target = original_data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test, raw_x_train, raw_x_test, raw_y_train, raw_y_test = (
        train_test_split(
            features,
            target,
            original_features,
            original_target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=target,
        )
    )
    logger.info("Created stratified train/test split: %s / %s", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test, raw_x_test, raw_y_test


x_train, x_test, y_train, y_test, raw_x_test, raw_y_test = split_dataset(
    dataset, encoded_dataset
)


# Fit the scaler only on training predictors and transform both splits.
def scale_features(train_features, test_features):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)
    logger.info("Fitted StandardScaler on training data only")
    return scaled_train, scaled_test, scaler


x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)
with (MODEL_DIR / "scaler.pkl").open("wb") as file:
    pickle.dump(scaler, file)
logger.info("Saved scaler to %s", MODEL_DIR / "scaler.pkl")


# Save the original test rows so the application can apply saved preprocessing.
def save_test_data(features, target, path):
    test_data = features.copy()
    test_data[TARGET_COLUMN] = target
    test_data.to_csv(path, index=False)
    logger.info("Saved test split with target to %s; shape=%s", path, test_data.shape)


save_test_data(raw_x_test, raw_y_test, TEST_DATA_PATH)
logger.info(
    "Train predictors=%s, Test predictors=%s",
    x_train_scaled.shape,
    x_test_scaled.shape,
)


# Create the required classifiers with their assignment hyperparameters.
def create_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
    }
    logger.info("Created %d model configurations", len(models))
    return models


models = create_models()


# Fit every classifier on the same scaled training split and save it for inference.
def train_and_save_models(model_registry, train_features, train_target, output_dir):
    trained_models = {}
    output_names = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "K-Nearest Neighbor": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
    }
    for name, model in model_registry.items():
        logger.info("Training %s", name)
        model.fit(train_features, train_target)
        output_path = output_dir / output_names[name]
        joblib.dump(model, output_path)
        trained_models[name] = model
        logger.info("Saved %s to %s", name, output_path)
    logger.info("All models trained and saved successfully")
    return trained_models


trained_models = train_and_save_models(models, x_train_scaled, y_train, MODEL_DIR)


# Evaluate each fitted classifier using the six required assignment metrics.
def evaluate_models(model_registry, test_features, test_target):
    metric_rows = []
    for name, model in model_registry.items():
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)[:, 1]
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(test_target, predictions),
            "AUC": roc_auc_score(test_target, probabilities),
            "Precision": precision_score(test_target, predictions, zero_division=0),
            "Recall": recall_score(test_target, predictions, zero_division=0),
            "F1": f1_score(test_target, predictions, zero_division=0),
            "MCC": matthews_corrcoef(test_target, predictions),
        }
        metric_rows.append(metrics)
        logger.info("Evaluated %s: %s", name, metrics)
    logger.info("All models evaluated successfully")
    return pd.DataFrame(metric_rows).set_index("Model")


comparison_metrics = evaluate_models(trained_models, x_test_scaled, y_test)
comparison_metrics_path = MODEL_DIR / "comparison_metrics.csv"
comparison_metrics.to_csv(comparison_metrics_path)
logger.info("Saved comparison metrics to %s", comparison_metrics_path)
