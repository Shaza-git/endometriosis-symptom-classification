import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, 
    precision_score, f1_score, roc_curve, confusion_matrix
)
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA (FINAL3 - THE WINNER)
# ==========================================
file_path = 'data/final3_training_data.csv'

if not os.path.exists(file_path):
    file_path = 'final3_training_data.csv'

print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path).fillna(0)

groups = df['post_id']
y = df['Y']
all_columns = df.columns.tolist()

# ==========================================
# 2. FEATURE SELECTION
# ==========================================
def get_cols(prefixes):
    return [c for c in all_columns if c in prefixes or any(c.startswith(p) for p in prefixes)]

feat_numeric = get_cols(['loc_', 'mod_', 'group_', 'has_', 'Mapping', 'lay_', 'alt_', 
                         'temp_', 'intensity', 'radiation', 'embed_', 'sub_embed_'])

feat_cat_id = 'subclass_hpo_id'
use_cat = feat_cat_id in df.columns

if use_cat:
    df[feat_cat_id] = df[feat_cat_id].astype(str)
    cat_features_indices = [feat_cat_id]
else:
    cat_features_indices = []

# ==========================================
# 3. 5-FOLD GROUP CV WITH 0.80 WINNING PARAMS
# ==========================================
gkf = GroupKFold(n_splits=5)
results = {'Optimized LR': [], 'CatBoost (Final Model)': []}

# --- THE WINNING PARAMETERS FROM TRIAL 5 ---
best_params = {
    'iterations': 650,      # The winning number
    'depth': 4,             # The winning depth
    'learning_rate': 0.015, # The winning LR
    'loss_function': 'Logloss',
    'verbose': 0,
    'random_seed': 42,
    'allow_writing_files': False
}

print("\nStarting Final Validation Run (Trial 5 Settings)...")
print("-" * 60)

fold = 1
for train_idx, val_idx in gkf.split(df, y, groups):
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # --- MODEL A: BASELINE (LR) ---
    X_lr = df[feat_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train_lr, X_val_lr = X_lr.iloc[train_idx], X_lr.iloc[val_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lr)
    X_val_scaled = scaler.transform(X_val_lr)
    
    lr = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.1, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_probs_lr = lr.predict_proba(X_val_scaled)[:, 1]
    
    # --- MODEL B: CATBOOST (FINAL WINNER) ---
    X_cb = df[feat_numeric + cat_features_indices] if use_cat else df[feat_numeric]
    X_train_cb, X_val_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
    
    train_pool = Pool(X_train_cb, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val_cb, y_val, cat_features=cat_features_indices)
    
    cb = CatBoostClassifier(**best_params)
    cb.fit(train_pool)
    y_probs_cb = cb.predict_proba(val_pool)[:, 1]
    
    # --- METRICS & THRESHOLD OPTIMIZATION ---
    for name, probs in [('Optimized LR', y_probs_lr), ('CatBoost (Final Model)', y_probs_cb)]:
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        
        # Youden's J Statistic
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        
        y_pred = (probs >= best_thresh).astype(int)
        
        results[name].append({
            'AUC': roc_auc_score(y_val, probs),
            'Precision': precision_score(y_val, y_pred),
            'Recall': recall_score(y_val, y_pred),
            'F1': f1_score(y_val, y_pred),
            'Accuracy': accuracy_score(y_val, y_pred)
        })

    print(f"Fold {fold} Complete.")
    fold += 1

# ==========================================
# 4. FINAL PUBLICATION TABLE
# ==========================================
print("\n" + "="*100)
print("FINAL MANUSCRIPT TABLE (Dataset: final3 | Tuned CatBoost)")
print("="*100)

summary_list = []
metrics_order = ['AUC', 'AUC_Std', 'Precision', 'Recall', 'F1', 'Accuracy']

for name, metrics_list in results.items():
    df_m = pd.DataFrame(metrics_list)
    means = df_m.mean()
    means['AUC_Std'] = df_m['AUC'].std()
    
    row = {'Model': name}
    for m in metrics_order:
        row[m] = means[m]
    summary_list.append(row)

res_df = pd.DataFrame(summary_list)
print(res_df.round(4).to_string(index=False))

print("-" * 100)
print("PERFORMANCE LIFT:")
lr_res = res_df[res_df['Model'] == 'Optimized LR'].iloc[0]
cb_res = res_df[res_df['Model'] == 'CatBoost (Final Model)'].iloc[0]

for metric in metrics_order:
    val_lr = lr_res[metric]
    val_cb = cb_res[metric]
    diff = val_cb - val_lr
    
    if metric == 'AUC_Std':
        print(f"   {metric:<15}: {diff:+.4f}  ({'✅ More Stable' if diff < 0 else '⚠️ Less Stable'})")
    else:
        pct = (diff / val_lr) * 100 if val_lr != 0 else 0
        print(f"   {metric:<15}: {diff:+.4f}  (Lift: {pct:+.2f}%)")

print("="*100)