# backend/scripts/retrain_model.py - COMPLETE RETRAINING SCRIPT
import pandas as pd
import sqlite3
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)

# ================= PATHS =================
DB_PATH = Path("../data/predictions.db")
ORIGINAL_DATA_PATH = Path("../data/diabetes_prediction_dataset.csv")
MODEL_DIR = Path("../models")
MODEL_PATH = MODEL_DIR / "best_diabetes_pipeline.joblib"

RANDOM_STATE = 42
MIN_NEW_SAMPLES = 50  # Minimum samples needed for retraining

# ================= LOAD DATA =================
def load_original_data():
    """Load original training data"""
    if not ORIGINAL_DATA_PATH.exists():
        print(f"❌ Original data not found at {ORIGINAL_DATA_PATH}")
        return None
    
    df = pd.read_csv(ORIGINAL_DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Ensure target is binary
    df['diabetes'] = df['diabetes'].map({0: 0, 1: 1}).astype(int)
    
    print(f"✅ Loaded original data: {df.shape}")
    return df

def load_new_feedback_data():
    """Load new labeled data from production database"""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return None
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load only samples with actual diagnosis
    query = """
        SELECT 
            age, bmi, HbA1c_level, blood_glucose_level,
            gender, smoking_history, hypertension, heart_disease,
            actual_diabetes as diabetes
        FROM predictions
        WHERE actual_diabetes IS NOT NULL
    """
    
    df_new = pd.read_sql(query, conn)
    conn.close()
    
    if len(df_new) == 0:
        print("ℹ️ No new labeled samples found in database")
        return None
    
    print(f"✅ Loaded {len(df_new)} new labeled samples from production")
    return df_new

# ================= RETRAIN MODEL =================
def retrain_model():
    """Perform incremental retraining with new data"""
    print("\n" + "="*60)
    print("🔄 DIABETES MODEL RETRAINING")
    print("="*60)
    
    # Load data
    df_original = load_original_data()
    df_new = load_new_feedback_data()
    
    if df_original is None:
        print("❌ Cannot proceed without original training data")
        return False
    
    if df_new is None or len(df_new) < MIN_NEW_SAMPLES:
        print(f"⚠️ Not enough new samples (need {MIN_NEW_SAMPLES}, have {len(df_new) if df_new is not None else 0})")
        print("   Skipping retraining...")
        return False
    
    # Combine datasets
    df_combined = pd.concat([df_original, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates().reset_index(drop=True)
    
    print(f"\n📊 Combined Dataset:")
    print(f"   Original: {len(df_original)} samples")
    print(f"   New:      {len(df_new)} samples")
    print(f"   Total:    {len(df_combined)} samples")
    print(f"   Diabetes rate: {df_combined['diabetes'].mean():.1%}")
    
    # Prepare features
    X = df_combined.drop(columns=['diabetes'])
    y = df_combined['diabetes']
    
    # Add dataset_source if not present
    if 'dataset_source' not in X.columns:
        X['dataset_source'] = 'main'
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"\n🔀 Train/Test Split:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # ================= BUILD PIPELINE =================
    numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_cols = ['gender', 'smoking_history', 'hypertension', 'heart_disease', 'dataset_source']
    
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
    
    # Train both models
    print("\n🚀 Training models...")
    
    # Random Forest
    rf_pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    print("   ✅ Random Forest trained")
    
    # XGBoost
    xgb_pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ])
    xgb_pipeline.fit(X_train, y_train)
    print("   ✅ XGBoost trained")
    
    # ================= EVALUATE =================
    print("\n📈 Model Performance:")
    
    models = {
        "Random Forest": rf_pipeline,
        "XGBoost": xgb_pipeline
    }
    
    best_model = None
    best_mcc = -1
    
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        
        print(f"\n   {name}:")
        for metric, value in metrics.items():
            print(f"      {metric}: {value:.4f}")
        
        if metrics["MCC"] > best_mcc:
            best_mcc = metrics["MCC"]
            best_model = (name, pipeline, metrics)
    
    # ================= SAVE MODELS =================
    print(f"\n💾 Saving models...")
    
    # Create backup of current model
    if MODEL_PATH.exists():
        backup_path = MODEL_PATH.with_stem(f"best_diabetes_pipeline_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        joblib.dump(joblib.load(MODEL_PATH), backup_path)
        print(f"   📦 Backup saved: {backup_path.name}")
    
    # Save new models
    joblib.dump(rf_pipeline, MODEL_DIR / "rf_diabetes_pipeline_retrained.joblib")
    joblib.dump(xgb_pipeline, MODEL_DIR / "xgb_diabetes_pipeline_retrained.joblib")
    joblib.dump(best_model[1], MODEL_PATH)
    
    print(f"   ✅ New models saved")
    print(f"\n🎉 RETRAINING COMPLETE!")
    print(f"   Best Model: {best_model[0]}")
    print(f"   MCC: {best_model[2]['MCC']:.4f}")
    print(f"   ROC-AUC: {best_model[2]['ROC-AUC']:.4f}")
    
    # Save retraining log
    log_path = MODEL_DIR / "retraining_log.txt"
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Retraining: {datetime.now().isoformat()}\n")
        f.write(f"New samples: {len(df_new)}\n")
        f.write(f"Total samples: {len(df_combined)}\n")
        f.write(f"Best model: {best_model[0]}\n")
        f.write(f"MCC: {best_model[2]['MCC']:.4f}\n")
        f.write(f"ROC-AUC: {best_model[2]['ROC-AUC']:.4f}\n")
    
    return True

# ================= MAIN =================
if __name__ == "__main__":
    success = retrain_model()
    
    if success:
        print("\n✅ Restart your API server to use the new model!")
    else:
        print("\n⚠️ Retraining skipped. Continue using current model.")
