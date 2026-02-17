"""
training_pipeline.py
---------------------
Fetches historical (features, targets) from MongoDB,
trains and evaluates multiple ML models on REAL US EPA AQI (0-500),
generates SHAP feature importance, and saves the best model to the registry.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import joblib
import logging
import certifi
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed — run: pip install shap")

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")


# ─────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────

def connect_to_mongodb():
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client["aqi_database"]
        db.command("ping")
        logger.info("✓ Connected to MongoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


def fetch_training_data(db) -> pd.DataFrame:
    """Fetch all historical feature records from the feature store."""
    logger.info("Fetching training data from MongoDB feature store ...")
    features_col = db["merged_features"]
    df = pd.DataFrame(list(features_col.find({})))
    if "_id" in df.columns:
        df = df.drop("_id", axis=1)
    logger.info(f"✓ Fetched {len(df)} records")

    # Quick sanity check on AQI range
    if not df.empty and "aqi" in df.columns:
        logger.info(
            f"  AQI range: {df['aqi'].min()} – {df['aqi'].max()} "
            f"(mean: {df['aqi'].mean():.1f}) — should be 0-500 scale"
        )
        if df["aqi"].max() <= 5:
            logger.error(
                "⚠️  AQI values are in 1-5 range! "
                "Your stored data still uses the old European index. "
                "Re-run backfill_data.py after fixing fetch_pollution.py."
            )

    return df


# ─────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """Sort chronologically, split X / y, drop non-feature columns."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Columns that are NOT input features
    exclude_cols = ["timestamp", "aqi", "aqi_category"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].fillna(0)
    y = df["aqi"].fillna(df["aqi"].median())

    logger.info(f"✓ Prepared data: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, feature_cols


def chronological_split(X, y, test_size=0.2):
    """Time-series safe split — no shuffling."""
    idx = int(len(X) * (1 - test_size))
    X_tr, X_te = X.iloc[:idx], X.iloc[idx:]
    y_tr, y_te = y.iloc[:idx], y.iloc[idx:]
    logger.info(f"✓ Train: {len(X_tr)} | Test: {len(X_te)}")
    return X_tr, X_te, y_tr, y_te


# ─────────────────────────────────────────────────────────────
# Model training & evaluation
# ─────────────────────────────────────────────────────────────

def train_models(X_tr, y_tr, X_te, y_te):
    """
    Train five sklearn/XGBoost models, evaluate with RMSE / MAE / R²,
    and return the best model by test RMSE.
    """
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_split=10, min_samples_leaf=4,
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=42,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=7,
            learning_rate=0.05, random_state=42, n_jobs=-1,
        ),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=5000),
    }

    results = {}
    best_model, best_name, best_rmse = None, None, float("inf")

    logger.info("\n" + "=" * 60)
    logger.info("Training & Evaluating Models")
    logger.info("=" * 60)

    for name, model in models.items():
        logger.info(f"\nTraining {name} ...")
        try:
            model.fit(X_tr, y_tr)
            y_pred_tr = model.predict(X_tr)
            y_pred_te = model.predict(X_te)

            metrics = {
                "model": model,
                "train_rmse": float(np.sqrt(mean_squared_error(y_tr, y_pred_tr))),
                "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_pred_te))),
                "train_mae":  float(mean_absolute_error(y_tr, y_pred_tr)),
                "test_mae":   float(mean_absolute_error(y_te, y_pred_te)),
                "train_r2":   float(r2_score(y_tr, y_pred_tr)),
                "test_r2":    float(r2_score(y_te, y_pred_te)),
            }
            results[name] = metrics

            logger.info(f"  Train RMSE: {metrics['train_rmse']:.2f}  |  Test RMSE: {metrics['test_rmse']:.2f}")
            logger.info(f"  Train MAE:  {metrics['train_mae']:.2f}  |  Test MAE:  {metrics['test_mae']:.2f}")
            logger.info(f"  Train R²:   {metrics['train_r2']:.4f}  |  Test R²:   {metrics['test_r2']:.4f}")

            if metrics["test_rmse"] < best_rmse:
                best_rmse  = metrics["test_rmse"]
                best_model = model
                best_name  = name

        except Exception as e:
            logger.error(f"  ✗ Error training {name}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Best Model: {best_name}  (Test RMSE: {best_rmse:.2f})")
    logger.info("=" * 60)
    return best_model, best_name, results


# ─────────────────────────────────────────────────────────────
# SHAP feature importance
# ─────────────────────────────────────────────────────────────

def generate_shap_analysis(model, X_te: pd.DataFrame, model_name: str) -> str | None:
    """
    Compute SHAP values for the best model and save a summary bar plot.
    Works with tree-based models (RF, GB, XGBoost).
    Returns the saved plot path, or None on failure.
    """
    if not SHAP_AVAILABLE:
        logger.warning("Skipping SHAP — library not installed (pip install shap)")
        return None

    try:
        logger.info("\nGenerating SHAP feature importance ...")
        os.makedirs("models", exist_ok=True)

        # Use a sample for speed if dataset is large
        sample = X_te.sample(min(500, len(X_te)), random_state=42)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, sample,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13)
        plt.tight_layout()

        plot_path = "models/shap_feature_importance.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ SHAP plot saved → {plot_path}")
        return plot_path

    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────
def save_all_models_to_mongodb(db, results, feature_cols, scaler=None):
    """
    Save all trained models (Random Forest, GB, XGBoost, Ridge, Lasso)
    to MongoDB. Only the best model is marked is_active=True.
    """
    os.makedirs("models", exist_ok=True)

    # Determine best model by test RMSE
    best_name = min(results, key=lambda x: results[x]["test_rmse"])

    for name, metrics in results.items():
        model = metrics["model"]
        safe_name = name.replace(" ", "_").lower()
        model_path = f"models/{safe_name}_model.pkl"
        joblib.dump(model, model_path)

        scaler_path = None
        if scaler is not None:
            scaler_path = "models/scaler.pkl"
            joblib.dump(scaler, scaler_path)

        db["model_registry"].insert_one({
            "model_name":  name,
            "model_path":  model_path,
            "scaler_path": scaler_path,
            "shap_plot":   None,  # optional: only for best model
            "features":    feature_cols,
            "metrics":     {
                "train_rmse": metrics["train_rmse"],
                "train_mae":  metrics["train_mae"],
                "train_r2":   metrics["train_r2"],
                "test_rmse":  metrics["test_rmse"],
                "test_mae":   metrics["test_mae"],
                "test_r2":    metrics["test_r2"],
                "model_type": name,
                "training_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "n_train": len(metrics.get("X_tr", [])),
                "n_test":  len(metrics.get("X_te", [])),
                "n_features": len(feature_cols),
            },
            "aqi_scale":   "US EPA 0-500",
            "created_at":  datetime.utcnow(),
            "is_active":   name == best_name,
        })


# ─────────────────────────────────────────────────────────────
# Hazardous AQI alert helper (used by app.py)
# ─────────────────────────────────────────────────────────────

def check_aqi_alerts(predictions: list[int]) -> list[dict]:
    """
    Return alert messages for any predicted AQI above 150.
    Intended for use in the Streamlit dashboard.
    """
    thresholds = [
        (300, "Hazardous",        "🚨 HAZARDOUS: Avoid all outdoor activity."),
        (200, "Very Unhealthy",   "⛔ VERY UNHEALTHY: Everyone should stay indoors."),
        (150, "Unhealthy",        "⚠️ UNHEALTHY: Sensitive groups must stay indoors."),
    ]
    alerts = []
    for aqi_val in predictions:
        for threshold, label, msg in thresholds:
            if aqi_val > threshold:
                alerts.append({"aqi": aqi_val, "level": label, "message": msg})
                break
    return alerts


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_training_pipeline():
    logger.info("\n" + "=" * 60)
    logger.info("AQI TRAINING PIPELINE")
    logger.info("=" * 60)

    db = connect_to_mongodb()
    df = fetch_training_data(db)

    if len(df) < 100:
        logger.error("Insufficient data for training (need ≥ 100 samples). Run backfill first.")
        return

    X, y, feature_cols = prepare_data(df)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y)

    # Scale features
    scaler       = StandardScaler()
    X_tr_scaled  = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols)
    X_te_scaled  = pd.DataFrame(scaler.transform(X_te),     columns=feature_cols)

    best_model, best_name, results = train_models(X_tr_scaled, y_tr, X_te_scaled, y_te)

    # SHAP analysis on best model
    shap_plot = generate_shap_analysis(best_model, X_te_scaled, best_name)

    bm = results[best_name]
    metrics = {
        "test_rmse":      bm["test_rmse"],
        "test_mae":       bm["test_mae"],
        "test_r2":        bm["test_r2"],
        "train_rmse":     bm["train_rmse"],
        "train_mae":      bm["train_mae"],
        "train_r2":       bm["train_r2"],
        "model_type":     best_name,
        "training_date":  datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "n_train":        len(X_tr),
        "n_test":         len(X_te),
        "n_features":     len(feature_cols),
        "aqi_scale":      "US EPA 0-500",
    }

    # Add SHAP only to best model
if shap_plot:
    results[best_name]["shap_plot"] = shap_plot

save_all_models_to_mongodb(db, results, feature_cols, scaler)


    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY (all models)")
    logger.info("=" * 60)
    for name, res in results.items():
        marker = " ← BEST" if name == best_name else ""
        logger.info(
            f"  {name:25s}  "
            f"RMSE: {res['test_rmse']:6.2f}  "
            f"MAE: {res['test_mae']:6.2f}  "
            f"R²: {res['test_r2']:.4f}{marker}"
        )

    logger.info("\n✓ Training pipeline completed successfully!")
    return best_model, results


if __name__ == "__main__":
    run_training_pipeline()
