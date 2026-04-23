# 2_dataset_diabetes_training_FINAL.py
# SMART CODE: Uses main (100k) + risk (520) datasets intelligently

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, matthews_corrcoef,
    cohen_kappa_score, precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ================= CONFIG =================
DATASETS = {
    "main": "diabetes_prediction_dataset.csv",
    "risk": "diabetes_risk_prediction.csv",
}
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODELS_DIR = Path("models")
DOCS_DIR = Path("documentation")
MODELS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

plt.style.use('default')
sns.set_palette("husl")

# ================= SMART LOAD + HARMONIZE =================
def load_and_harmonize():
    all_dfs = []
    
    # ===== DATASET 1: Main (iammustafatz - 100k rows) =====
    try:
        print("Loading main dataset...")
        df_main = pd.read_csv(DATASETS["main"])
        print(f"  Raw: {df_main.shape}")
        
        # Standard target mapping
        df_main['diabetes'] = df_main['diabetes'].map({0: 0, 1: 1}).astype(int)
        
        # Keep core features
        main_cols = ['age', 'gender', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                     'hypertension', 'heart_disease', 'smoking_history', 'diabetes']
        df_main = df_main[main_cols].dropna()
        df_main['dataset_source'] = 'main'
        
        # Clean categoricals
        df_main['gender'] = df_main['gender'].astype(str).str.lower().map({
            'male': 'Male', 'female': 'Female'
        }).fillna('Other')
        df_main['smoking_history'] = df_main['smoking_history'].fillna('No Info')
        
        all_dfs.append(df_main)
        print(f"  ✅ Clean: {df_main.shape}, diabetes rate: {df_main['diabetes'].mean():.1%}")
    except Exception as e:
        print(f"  ❌ Main dataset error: {e}")
    
    # ===== DATASET 2: Risk (rcratos - 520 rows, symptom-based) =====
    try:
        print("Loading risk dataset...")
        df_risk = pd.read_csv(DATASETS["risk"])
        print(f"  Raw: {df_risk.shape}")
        
        # Find target column (could be 'class', 'Diabetes', etc.)
        if 'class' in df_risk.columns:
            df_risk['diabetes'] = df_risk['class'].map({'Positive': 1, 'Negative': 0}).astype(int)
        elif 'Diabetes' in df_risk.columns:
            df_risk['diabetes'] = df_risk['Diabetes'].map({'Yes': 1, 'No': 0}).astype(int)
        else:
            print(f"  ⚠️ Risk dataset columns: {df_risk.columns.tolist()}")
            print(f"  ⚠️ Cannot find target - skipping")
            return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        
        # Map symptoms to binary
        symptom_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                       'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
                       'Irritability', 'delayed healing', 'partial paresis',
                       'muscle stiffness', 'Alopecia', 'Obesity']
        
        for col in symptom_cols:
            if col in df_risk.columns:
                df_risk[col] = df_risk[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        
        # Harmonize to match main dataset structure
        df_risk_clean = pd.DataFrame()
        
        # Age
        if 'Age' in df_risk.columns:
            df_risk_clean['age'] = df_risk['Age']
        elif 'age' in df_risk.columns:
            df_risk_clean['age'] = df_risk['age']
        
        # Gender
        if 'Gender' in df_risk.columns:
            df_risk_clean['gender'] = df_risk['Gender'].str.lower().map({
                'male': 'Male', 'female': 'Female'
            }).fillna('Other')
        elif 'gender' in df_risk.columns:
            df_risk_clean['gender'] = df_risk['gender'].str.lower().map({
                'male': 'Male', 'female': 'Female'
            }).fillna('Other')
        
        # Create proxy BMI from obesity flag
        if 'Obesity' in df_risk.columns:
            df_risk_clean['bmi'] = df_risk['Obesity'].map({1: 32.0, 0: 24.0})
        else:
            df_risk_clean['bmi'] = 26.0  # average BMI
        
        # Create synthetic HbA1c and glucose (rough estimates based on symptoms)
        # More symptoms → higher risk → higher proxy values
        symptom_count = df_risk[symptom_cols].sum(axis=1) if symptom_cols[0] in df_risk.columns else 0
        df_risk_clean['HbA1c_level'] = 5.5 + (symptom_count * 0.3)  # range 5.5-10
        df_risk_clean['blood_glucose_level'] = 100 + (symptom_count * 15)  # range 100-300
        
        # Binary features (set to 0 if not available)
        df_risk_clean['hypertension'] = 0
        df_risk_clean['heart_disease'] = 0
        df_risk_clean['smoking_history'] = 'No Info'
        
        df_risk_clean['diabetes'] = df_risk['diabetes']
        df_risk_clean['dataset_source'] = 'risk'
        df_risk_clean = df_risk_clean.dropna()
        
        all_dfs.append(df_risk_clean)
        print(f"  ✅ Clean: {df_risk_clean.shape}, diabetes rate: {df_risk_clean['diabetes'].mean():.1%}")
        
    except Exception as e:
        print(f"  ❌ Risk dataset error: {e}")
    
    if not all_dfs:
        raise ValueError("No datasets loaded successfully!")
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n🎉 COMBINED: {df_combined.shape} rows from {len(all_dfs)} dataset(s)!")
    print(f"   Overall diabetes rate: {df_combined['diabetes'].mean():.1%}")
    return df_combined

# ================= LOAD DATA =================
df = load_and_harmonize()
X = df.drop(columns=['diabetes'])
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ================= SMART PREPROCESSOR =================
numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history', 'hypertension', 'heart_disease', 'dataset_source']

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# ================= MODELS =================
rf_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1))
])

xgb_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False,
        eval_metric='logloss'))
])

print("\n🚀 Training 2-dataset models...")
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

# ================= EVALUATION FUNCTION =================
def evaluate_and_document(name, pipeline, X_test, y_test, docs_dir):
    print(f"\n===== {name} (2-DATASET) EVALUATION =====")
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # FIXED PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    sorted_idx = np.argsort(recall_vals)
    recall_sorted = recall_vals[sorted_idx]
    precision_sorted = precision_vals[sorted_idx]
    pr_auc = auc(recall_sorted, precision_sorted)
    
    # ALL METRICS
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": pr_auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Cohen-Kappa": cohen_kappa_score(y_test, y_pred),
    }
    
    for metric, value in metrics.items():
        print(f"  {metric:12}: {value:.4f}")
    
    # Save CSV
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", name])
    metrics_df.to_csv(docs_dir / f"{name.lower().replace(' ', '_')}_2dataset_metrics.csv", index=False)
    
    # ===== CONFUSION MATRIX PNG =====
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                cbar_kws={'label': 'Count'})
    plt.title(f'{name} - 2 Datasets Combined\n'
              f'Accuracy: {metrics["Accuracy"]:.3f} | MCC: {metrics["MCC"]:.3f}',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    cm_png = docs_dir / f"{name.lower().replace(' ', '_')}_2dataset_confusion.png"
    plt.savefig(cm_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  🔥 Confusion matrix: {cm_png.name}")
    
    # ===== ROC + PR CURVES =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, linewidth=3, label=f'AUC = {metrics["ROC-AUC"]:.3f}', color='blue')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    ax2.plot(recall_sorted, precision_sorted, linewidth=3,
             label=f'PR-AUC = {pr_auc:.3f}', color='green')
    ax2.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} - 2 Dataset Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    curves_png = docs_dir / f"{name.lower().replace(' ', '_')}_2dataset_curves.png"
    plt.savefig(curves_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  📈 ROC/PR curves: {curves_png.name}")
    
    return metrics

# ================= EVALUATE BOTH =================
rf_metrics = evaluate_and_document("Random Forest", rf_pipeline, X_test, y_test, DOCS_DIR)
xgb_metrics = evaluate_and_document("XGBoost", xgb_pipeline, X_test, y_test, DOCS_DIR)

# ================= COMPARISON =================
comparison_data = []
for model_name, metrics in [("Random Forest", rf_metrics), ("XGBoost", xgb_metrics)]:
    comparison_data.append([
        model_name, metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
        metrics["F1-Score"], metrics["ROC-AUC"], metrics["PR-AUC"],
        metrics["MCC"], metrics["Specificity"]
    ])

comparison_df = pd.DataFrame(comparison_data, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC", "Specificity"
])
comparison_csv = DOCS_DIR / "2dataset_model_comparison.csv"
comparison_df.to_csv(comparison_csv, index=False)

print("\n📊 FINAL COMPARISON:")
print(comparison_df.round(4).to_string(index=False))

# ================= SAVE BEST =================
best = max([rf_metrics, xgb_metrics], key=lambda m: (m["MCC"], m["PR-AUC"]))
best_name = "Random Forest" if best["MCC"] == rf_metrics["MCC"] else "XGBoost"
best_pipeline = rf_pipeline if best_name == "Random Forest" else xgb_pipeline

joblib.dump(rf_pipeline, MODELS_DIR / "rf_2dataset_pipeline.joblib")
joblib.dump(xgb_pipeline, MODELS_DIR / "xgb_2dataset_pipeline.joblib")
joblib.dump(best_pipeline, MODELS_DIR / "best_diabetes_pipeline.joblib")

print(f"\n🎉 BEST MODEL: {best_name}")
print(f"   MCC: {best['MCC']:.4f}, PR-AUC: {best['PR-AUC']:.4f}")
print(f"\n💾 Models saved in: {MODELS_DIR.resolve()}")
print(f"📁 All documentation in: {DOCS_DIR.resolve()}")
print(f"\n✅ READY FOR WEB APP: best_diabetes_pipeline.joblib")
