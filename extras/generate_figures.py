"""
Generate all figures for Water Potability ML Project Report
Author: Sohaib Farooq
Email: sohaib.farooq@bigacademy.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import os
import warnings
warnings.filterwarnings('ignore')

# Create figures directory if it doesn't exist
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("WATER POTABILITY ML PROJECT - FIGURE GENERATION")
print("Author: Sohaib Farooq")
print("Email: sohaib.farooq@bigacademy.com")
print("=" * 60)

# Load and prepare data
print("\n[1/10] Loading data...")
df = pd.read_csv('water_potability.csv')

# Handle missing values
for column in df.columns:
    if df[column].isnull().sum() > 0:
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

# Prepare features and target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("[2/10] Training models...")

# Logistic Regression with balanced class weights
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# XGBoost
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, 
                          eval_metric='logloss', verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# LightGBM
lgbm_model = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
lgbm_model.fit(X_train_scaled, y_train)
lgbm_pred = lgbm_model.predict(X_test_scaled)
lgbm_proba = lgbm_model.predict_proba(X_test_scaled)[:, 1]

algorithms = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
predictions = [lr_pred, rf_pred, xgb_pred, lgbm_pred]
probas = [lr_proba, rf_proba, xgb_proba, lgbm_proba]

# Calculate metrics
results = {}
for algo, pred, proba in zip(algorithms, predictions, probas):
    results[algo] = {
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred),
        'AUC': auc(*roc_curve(y_test, proba)[:2])
    }

# FIGURE 1: Missing Values
print("[3/10] Generating: Missing values chart...")
df_original = pd.read_csv('water_potability.csv')
missing_values = df_original.isnull().sum()

plt.figure(figsize=(10, 5))
missing_values.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Missing Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/01_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 2: Target Distribution
print("[4/10] Generating: Target distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#e74c3c', '#2ecc71']
labels = ['Not Potable (0)', 'Potable (1)']
df['Potability'].value_counts().plot(kind='pie', ax=axes[0], colors=colors, 
                                      autopct='%1.1f%%', labels=labels,
                                      explode=(0.02, 0.02))
axes[0].set_title('Target Distribution (Pie Chart)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('')

df['Potability'].value_counts().plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
axes[1].set_title('Target Distribution (Bar Chart)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Potability')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(labels, rotation=0)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/02_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 3: Correlation Heatmap
print("[5/10] Generating: Correlation heatmap...")
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 4: Feature Distributions
print("[6/10] Generating: Feature distributions...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = df.columns[:-1]

for idx, feature in enumerate(features):
    row = idx // 3
    col = idx % 3
    
    sns.histplot(data=df, x=feature, hue='Potability', kde=True, 
                 ax=axes[row, col], palette=['#e74c3c', '#2ecc71'])
    axes[row, col].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/04_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 5: Box Plots
print("[7/10] Generating: Box plots...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = df.columns[:-1]

for idx, feature in enumerate(features):
    row = idx // 3
    col = idx % 3
    
    sns.boxplot(data=df, x='Potability', y=feature, ax=axes[row, col],
                palette=['#e74c3c', '#2ecc71'])
    axes[row, col].set_title(f'{feature} by Potability')
    axes[row, col].set_xticklabels(['Not Potable', 'Potable'])

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/05_box_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 6: Feature Importance
print("[8/10] Generating: Feature importance chart...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 7: Model Comparison
print("[9/10] Generating: Model comparison charts...")
comparison_df = pd.DataFrame(results).T

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison bar chart
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = axes[0].bar(algorithms, comparison_df['Accuracy'] * 100, color=colors, edgecolor='black')
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

axes[0].tick_params(axis='x', rotation=45)

# All metrics comparison
x = np.arange(len(algorithms))
width = 0.2

axes[1].bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy', color='#3498db')
axes[1].bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision', color='#2ecc71')
axes[1].bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall', color='#e74c3c')
axes[1].bar(x + 1.5*width, comparison_df['F1-Score'], width, label='F1-Score', color='#9b59b6')

axes[1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score')
axes[1].set_xticks(x)
axes[1].set_xticklabels(algorithms, rotation=45)
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/07_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 8: Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (algo, pred) in enumerate(zip(algorithms, predictions)):
    row = idx // 2
    col = idx % 2
    
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                xticklabels=['Not Potable', 'Potable'],
                yticklabels=['Not Potable', 'Potable'])
    axes[row, col].set_title(f'{algo}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('Predicted')
    axes[row, col].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/08_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 9: ROC Curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_proba)

lr_auc = auc(lr_fpr, lr_tpr)
rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)
lgbm_auc = auc(lgbm_fpr, lgbm_tpr)

plt.figure(figsize=(10, 8))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', linewidth=2)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', linewidth=2)
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})', linewidth=2)
plt.plot(lgbm_fpr, lgbm_tpr, label=f'LightGBM (AUC = {lgbm_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/09_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# FIGURE 10: Precision-Recall Curves
print("[10/10] Generating: Precision-Recall curves...")
lr_precision_curve, lr_recall_curve, _ = precision_recall_curve(y_test, lr_proba)
rf_precision_curve, rf_recall_curve, _ = precision_recall_curve(y_test, rf_proba)
xgb_precision_curve, xgb_recall_curve, _ = precision_recall_curve(y_test, xgb_proba)
lgbm_precision_curve, lgbm_recall_curve, _ = precision_recall_curve(y_test, lgbm_proba)

lr_ap = average_precision_score(y_test, lr_proba)
rf_ap = average_precision_score(y_test, rf_proba)
xgb_ap = average_precision_score(y_test, xgb_proba)
lgbm_ap = average_precision_score(y_test, lgbm_proba)

plt.figure(figsize=(10, 8))
plt.plot(lr_recall_curve, lr_precision_curve, label=f'Logistic Regression (AP = {lr_ap:.3f})', linewidth=2)
plt.plot(rf_recall_curve, rf_precision_curve, label=f'Random Forest (AP = {rf_ap:.3f})', linewidth=2)
plt.plot(xgb_recall_curve, xgb_precision_curve, label=f'XGBoost (AP = {xgb_ap:.3f})', linewidth=2)
plt.plot(lgbm_recall_curve, lgbm_precision_curve, label=f'LightGBM (AP = {lgbm_ap:.3f})', linewidth=2)

baseline = y_test.sum() / len(y_test)
plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})', linewidth=1)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/10_precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("All figures generated successfully!")
print(f"Saved to: {os.path.abspath(FIGURES_DIR)}")
print("=" * 60)

# Print summary of metrics for the report
print("\nMODEL PERFORMANCE SUMMARY:")
print("-" * 60)
for algo in algorithms:
    print(f"\n{algo}:")
    for metric, value in results[algo].items():
        print(f"  {metric}: {value:.4f}")
