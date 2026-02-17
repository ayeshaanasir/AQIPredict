import pandas as pd
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")


def connect_to_mongodb():
    """Connect to MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_database"]
        logger.info("✓ Connected to MongoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


def fetch_training_data(db):
    """Fetch historical features from MongoDB"""
    try:
        logger.info("Fetching training data from MongoDB...")
        features_col = db["merged_features"]
        cursor = features_col.find({})
        df = pd.DataFrame(list(cursor))

        if '_id' in df.columns:
            df = df.drop('_id', axis=1)

        logger.info(f"✓ Fetched {len(df)} records from MongoDB")
        return df

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise


def prepare_data(df):
    """Prepare features and target for training"""
    try:
        df = df.sort_values('timestamp').reset_index(drop=True)

        exclude_cols = ['timestamp', 'aqi']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df['aqi']

        X = X.fillna(0)
        y = y.fillna(y.median())

        logger.info(f"✓ Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"  Feature columns: {feature_cols[:5]}... (showing first 5)")

        return X, y, feature_cols

    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise


def create_train_test_split(X, y, test_size=0.2):
    """Split data chronologically (time series split)"""
    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    logger.info(f"✓ Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return the best one"""

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1, max_iter=5000)
    }

    results = {}
    best_model = None
    best_score = float('inf')
    best_model_name = None

    logger.info("\n" + "=" * 60)
    logger.info("Training and Evaluating Models")
    logger.info("=" * 60)

    for name, model in models.items():
        logger.info(f"\nTraining {name}...")

        try:
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }

            logger.info(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
            logger.info(f"  Train MAE:  {train_mae:.4f} | Test MAE:  {test_mae:.4f}")
            logger.info(f"  Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")

            if test_rmse < best_score:
                best_score = test_rmse
                best_model = model
                best_model_name = name

        except Exception as e:
            logger.error(f"  Error training {name}: {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info(f"Best Model: {best_model_name} (Test RMSE: {best_score:.4f})")
    logger.info("=" * 60)

    return best_model, best_model_name, results


def save_model_to_mongodb(db, model, model_name, metrics, feature_cols, scaler=None):
    """Save trained model to MongoDB"""
    try:
        logger.info(f"\nSaving {model_name} to MongoDB Model Registry...")

        os.makedirs("models", exist_ok=True)

        model_path = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)

        scaler_path = None
        if scaler:
            scaler_path = "models/scaler.pkl"
            joblib.dump(scaler, scaler_path)

        models_col = db["model_registry"]

        model_doc = {
            "model_name": model_name,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "features": feature_cols,
            "metrics": metrics,
            "created_at": datetime.utcnow(),
            "is_active": True
        }

        models_col.update_many(
            {"is_active": True},
            {"$set": {"is_active": False}}
        )

        models_col.insert_one(model_doc)

        logger.info(f"✓ Model metadata saved to MongoDB")
        logger.info(f"✓ Model file saved to {model_path}")

        return model_path

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def run_training_pipeline():
    """Main training pipeline"""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("AQI TRAINING PIPELINE (MongoDB)")
        logger.info("=" * 60)

        db = connect_to_mongodb()
        df = fetch_training_data(db)

        if len(df) < 100:
            logger.error("Insufficient data for training (need at least 100 samples)")
            return

        X, y, feature_cols = prepare_data(df)
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_model, best_model_name, results = train_models(
            pd.DataFrame(X_train_scaled, columns=feature_cols),
            y_train,
            pd.DataFrame(X_test_scaled, columns=feature_cols),
            y_test
        )

        best_metrics = results[best_model_name]
        metrics = {
            "test_rmse": float(best_metrics['test_rmse']),
            "test_mae": float(best_metrics['test_mae']),
            "test_r2": float(best_metrics['test_r2']),
            "train_rmse": float(best_metrics['train_rmse']),
            "train_mae": float(best_metrics['train_mae']),
            "train_r2": float(best_metrics['train_r2']),
            "model_type": best_model_name,
            "training_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(feature_cols)
        }

        save_model_to_mongodb(db, best_model, best_model_name, metrics, feature_cols, scaler)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        for name, res in results.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Test RMSE: {res['test_rmse']:.4f}")
            logger.info(f"  Test MAE:  {res['test_mae']:.4f}")
            logger.info(f"  Test R²:   {res['test_r2']:.4f}")

        logger.info("\n✓ Training pipeline completed successfully!")

        return best_model, results

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_training_pipeline()