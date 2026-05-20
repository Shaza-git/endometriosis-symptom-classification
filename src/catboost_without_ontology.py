import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, 
    precision_score, f1_score, roc_curve
)
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA 
# ==========================================
file_path = 'data/final3_training_data.csv'

if not os.path.exists(file_path):
    file_path = 'final3_training_data.csv'

print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path).fillna(0)

groups = df['post_id']
y = df['Y']

# ==========================================
# 2. FEATURE SELECTION (SUBSET)
# ==========================================
# Resolving potential temp_Cyclic vs temp_Cyclical name mismatch
temp_cyclic_col = 'temp_Cyclical' if 'temp_Cyclical' in df.columns else 'temp_Cyclic'

features = ['intensity', 'radiation', 'temp_Chronic', temp_cyclic_col]

# Ensure all selected features exist in the dataframe
features = [f for f in features if f in df.columns]
print(f"Using features: {features}")

# Extract features
X_cb = df[features]

# ==========================================
# 3. 5-FOLD GROUP CV WITH DEFAULT PARAMS
# ==========================================
gkf = GroupKFold(n_splits=5)
results = []

print("\nStarting Validation Run (Default CatBoost Settings)...")
print("-" * 60)

fold = 1
for train_idx, val_idx in gkf.split(df, y, groups):
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train_cb, X_val_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
    
    train_pool = Pool(X_train_cb, y_train)
    val_pool = Pool(X_val_cb, y_val)
    
    # Initialize default CatBoost (No hyperparameter tuning)
    cb = CatBoostClassifier(verbose=0, random_seed=42, allow_writing_files=False)
    cb.fit(train_pool)
    y_probs_cb = cb.predict_proba(val_pool)[:, 1]
    
    # --- METRICS & THRESHOLD OPTIMIZATION ---
    fpr, tpr, thresholds = roc_curve(y_val, y_probs_cb)
    
    # Youden's J Statistic
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    y_pred = (y_probs_cb >= best_thresh).astype(int)
    
    results.append({
        'AUC': roc_auc_score(y_val, y_probs_cb),
        'Precision': precision_score(y_val, y_pred, zero_division=0),
        'Recall': recall_score(y_val, y_pred, zero_division=0),
        'F1': f1_score(y_val, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_val, y_pred)
    })

    print(f"Fold {fold} Complete.")
    fold += 1

# ==========================================
# 4. FINAL RESULTS TABLE
# ==========================================
print("\n" + "="*100)
print("FINAL RESULTS TABLE (Subset Features | Default CatBoost)")
print("="*100)

metrics_order = ['AUC', 'AUC_Std', 'Precision', 'Recall', 'F1', 'Accuracy']

# Calculate means and std deviation for AUC
df_m = pd.DataFrame(results)
means = df_m.mean()
means['AUC_Std'] = df_m['AUC'].std()

row = {'Model': 'CatBoost (Default)'}
for m in metrics_order:
    row[m] = means[m]

res_df = pd.DataFrame([row])
print(res_df.round(4).to_string(index=False))
print("="*100)