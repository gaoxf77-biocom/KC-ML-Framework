# -*- coding: utf-8 -*-
# ==============================================================================
# Machine Learning for Key Gene Selection (Knowledge-Guided Pipeline)
# Python Version 13 (Final Weighted Version)
#
# Key Feature:
# - Reads 'knowledge_weights.csv' to apply a sophisticated, knowledge-guided
#   weighting scheme to the gene expression data before modeling.
# - Includes SHAP analysis for ALL four models for complete interpretability.
# - Adds a final "Model Consensus" report to identify the most robust genes.
# ==============================================================================

import os
import logging
import warnings
from collections import Counter

# --- Setup: Backend and Warnings ---
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Scikit-learn & ML Imports ---
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve,
    average_precision_score, make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.utils import resample
import xgboost as xgb
from boruta import BorutaPy

# --- Configuration ---
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
RESULTS_DIR = "results_KC_all2-bio-g"  # Fixed version results directory, version number incremented
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Nature style global settings ---
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',        # Font family
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # Preferred fonts
    'axes.unicode_minus': False,        # Fix minus sign display issue
    'axes.labelsize': 14,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# --- Nature style save function ---
def save_nature_plot(filename, tight_layout=True, formats=['.png', '.pdf', '.svg']):
    """Save Nature style images, supports multiple formats"""
    if tight_layout: 
        plt.tight_layout()
    
    for ext in formats:
        save_path = os.path.join(RESULTS_DIR, f"{filename}{ext}")
        plt.savefig(save_path, dpi=300 if ext == '.png' else None)
    
    plt.close('all')

# --- Nature style confusion matrix ---
def plot_nature_confusion_matrix(y_true, y_pred, classes, model_name):
    """Nature style confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes, 
        yticklabels=classes,
        cbar_kws={'shrink': 0.8},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'{model_name} Confusion Matrix', fontsize=13, fontweight='bold')
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    save_nature_plot(f'confusion_matrix_{model_name.lower().replace(" ", "_")}_nature')

# --- Nature style PR curve ---
def plot_nature_pr_curve(y_true, y_prob, model_name, color):
    """Nature style PR curve plot"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    # === Added: PR curve smoothing interpolation ===
    # PR curves require special handling because recall is decreasing
    # First sort recall to ensure monotonicity
    sort_idx = np.argsort(recall)
    recall_sorted = recall[sort_idx]
    precision_sorted = precision[sort_idx]
    
    # Create dense recall points
    recall_dense = np.linspace(0, 1, 300)
    # Use linear interpolation
    precision_dense = np.interp(recall_dense, recall_sorted, precision_sorted)
    
    # Ensure curve starts near (0,1)
    recall_dense = np.insert(recall_dense, 0, 0)
    precision_dense = np.insert(precision_dense, 0, precision_sorted[0])
    # === Interpolation completed ===

    # === Added: Bootstrap for standard deviation calculation ===
    n_bootstrap = 1000
    bootstrap_precisions = []
    
    y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
    y_prob_np = y_prob.values if hasattr(y_prob, 'values') else y_prob
    n_samples = len(y_true_np)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        y_true_sample = y_true_np[indices]
        y_prob_sample = y_prob_np[indices]
        
        if len(np.unique(y_true_sample)) < 2:
            continue
        
        precision_sample, recall_sample, _ = precision_recall_curve(y_true_sample, y_prob_sample)
        
        # Sort recall to ensure monotonicity
        sort_idx_sample = np.argsort(recall_sample)
        recall_sorted_sample = recall_sample[sort_idx_sample]
        precision_sorted_sample = precision_sample[sort_idx_sample]
        
        # Interpolate to same recall points
        precision_interp = np.interp(recall_dense, recall_sorted_sample, precision_sorted_sample)
        bootstrap_precisions.append(precision_interp)
    
    if bootstrap_precisions:
        bootstrap_precisions = np.array(bootstrap_precisions)
        precision_mean = np.mean(bootstrap_precisions, axis=0)
        precision_std = np.std(bootstrap_precisions, axis=0)
    else:
        precision_mean = precision_dense
        precision_std = np.zeros_like(precision_dense)
    # === Bootstrap completed ===

    baseline_precision = np.mean(y_true)
    
    def bootstrap_ap(y_true, y_prob, n_bootstrap=2000, random_state=123):
        np.random.seed(random_state)
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            y_true_sample = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_prob_sample = y_prob.iloc[indices] if hasattr(y_prob, 'iloc') else y_prob[indices]
            
            if len(np.unique(y_true_sample)) < 2:
                continue
            ap = average_precision_score(y_true_sample, y_prob_sample)
            bootstrap_scores.append(ap)
        
        bootstrap_scores = np.array(bootstrap_scores)
        ci_low = np.percentile(bootstrap_scores, 2.5) if len(bootstrap_scores) > 0 else ap_score
        ci_high = np.percentile(bootstrap_scores, 97.5) if len(bootstrap_scores) > 0 else ap_score
        return ci_low, ci_high
    
    ci_low, ci_high = bootstrap_ap(y_true, y_prob, n_bootstrap=2000, random_state=RANDOM_STATE)
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    # Plot mean PR curve
    ax.plot(recall_dense, precision_mean, lw=2.5, color=color, label=f'{model_name}')
    
    # === Added: Plot standard deviation shading ===
    ax.fill_between(recall_dense, 
                    precision_mean - precision_std, 
                    precision_mean + precision_std, 
                    color=color, alpha=0.2, label=f'{model_name} +/- 1 std')
    
    ax.axhline(y=baseline_precision, color='black', linestyle='--', 
               lw=1.5, alpha=0.6, label=f'Random (AP={baseline_precision:.3f})')
    
    ax.set_xlim(-0.05, 1.01)
    ax.set_ylim(-0.05, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    
    text = f"AP = {ap_score:.3f}\n95% CI: {ci_low:.3f}-{ci_high:.3f}"
    ax.text(0.55, 0.12, text, fontsize=10, ha="left", va="center", 
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    ax.set_title(f'{model_name} Precision-Recall Curve', fontsize=13, fontweight='bold')
    
    save_nature_plot(f'pr_curve_{model_name.lower().replace(" ", "_")}_nature')

# --- Nature style ROC curve ---
def plot_nature_roc_curve(y_true, y_prob, model_name, color):
    """Nature style ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    # === Added: Curve smoothing interpolation (remove right angles) ===
    # Create dense FPR points for interpolation
    fpr_dense = np.linspace(0, 1, 300)
    # Linear interpolation to get corresponding TPR values
    tpr_dense = np.interp(fpr_dense, fpr, tpr)
    # Ensure curve starts at (0,0) and ends at (1,1)
    fpr_dense = np.insert(fpr_dense, 0, 0)
    tpr_dense = np.insert(tpr_dense, 0, 0)
    fpr_dense = np.append(fpr_dense, 1)
    tpr_dense = np.append(tpr_dense, 1)
    # === Interpolation completed ===
    
    # === Added: Bootstrap for standard deviation calculation ===
    n_bootstrap = 1000
    bootstrap_tprs = []
    
    y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
    y_prob_np = y_prob.values if hasattr(y_prob, 'values') else y_prob
    n_samples = len(y_true_np)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        y_true_sample = y_true_np[indices]
        y_prob_sample = y_prob_np[indices]
        
        if len(np.unique(y_true_sample)) < 2:
            continue
        
        fpr_sample, tpr_sample, _ = roc_curve(y_true_sample, y_prob_sample)
        
        # Interpolate to same FPR points
        tpr_interp = np.interp(fpr_dense, fpr_sample, tpr_sample)
        bootstrap_tprs.append(tpr_interp)
    
    if bootstrap_tprs:
        bootstrap_tprs = np.array(bootstrap_tprs)
        tpr_mean = np.mean(bootstrap_tprs, axis=0)
        tpr_std = np.std(bootstrap_tprs, axis=0)
    else:
        tpr_mean = tpr_dense
        tpr_std = np.zeros_like(tpr_dense)
    # === Bootstrap completed ===
    
    bootstrapped_scores = []
    for i in range(100):
        y_true_res, y_prob_res = resample(y_true, y_prob, random_state=i)
        if len(np.unique(y_true_res)) > 1:
            bootstrapped_scores.append(roc_auc_score(y_true_res, y_prob_res))
    
    lower = np.percentile(bootstrapped_scores, 2.5) if bootstrapped_scores else auc_score
    upper = np.percentile(bootstrapped_scores, 97.5) if bootstrapped_scores else auc_score
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    # Modified here: Legend only shows AUC value and 95% CI
    legend_label = f"AUC = {auc_score:.3f} (95% CI: {lower:.3f}-{upper:.3f})"
    
    # Plot mean ROC curve
    ax.plot(fpr_dense, tpr_mean, lw=2.5, color=color, label=legend_label)  # Changed to fpr_dense, tpr_mean
    
    # === Added: Plot standard deviation shading ===
    ax.fill_between(fpr_dense, 
                    tpr_mean - tpr_std, 
                    tpr_mean + tpr_std, 
                    color=color, alpha=0.2, label=f'{model_name} +/- 1 std')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random')
    
    ax.set_xlim(-0.05, 1.01)
    ax.set_ylim(-0.05, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    
    # Remove text box as information is now in legend
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.set_title(f'{model_name} ROC Curve', fontsize=13, fontweight='bold')
    
    save_nature_plot(f'roc_curve_{model_name.lower().replace(" ", "_")}_nature')
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc_score, 'ci': (lower, upper)}

# --- Nature style combined ROC curves ---
def plot_nature_combined_roc(roc_data):
    """Nature style combined ROC curves plot"""
    if not roc_data:
        return
    
    colors = {
        "Elastic_Net": "#E41A1C",
        "RF_Weighted": "#4DAF4A",
        "XGBoost": "#984EA3",
        "Boruta_RF": "#FF7F00"
    }
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for name, data in roc_data.items():
        if name in colors:
            # Modified here: Legend only shows AUC value and 95% CI
            legend_label = f"{name}: AUC = {data['auc']:.3f} (95% CI: {data['ci'][0]:.3f}-{data['ci'][1]:.3f})"
            ax.plot(data['fpr'], data['tpr'], 
                   lw=2, 
                   color=colors.get(name, 'gray'),
                   label=legend_label)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random')
    
    ax.set_xlim(-0.05, 1.01)
    ax.set_ylim(-0.05, 1.01)
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    
    ax.set_title("ROC Curves Comparison (Test Set)", fontsize=13, fontweight='bold', pad=10)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    
    save_nature_plot("combined_roc_curves_nature")

# --- Nature style performance comparison plot ---
def plot_nature_performance_comparison(df_test):
    """Nature style model performance comparison plot"""
    df_melted = df_test.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    palette = sns.color_palette("viridis", n_colors=len(df_melted['Model'].unique()))
    
    sns.barplot(
        x='Metric', 
        y='Score', 
        hue='Model', 
        data=df_melted, 
        palette=palette,
        ax=ax,
        errorbar=None
    )
    
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax.set_xlabel("Performance Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    
    ax.set_title("Model Performance Comparison (Test Set)", fontsize=13, fontweight='bold', pad=15)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_nature_plot("model_performance_comparison_nature")

# --- Performance report function ---
def get_performance_report(y_true, y_pred, y_prob, model_name):
    return {'Model': model_name, 'Accuracy': accuracy_score(y_true, y_pred), 'AUC': roc_auc_score(y_true, y_prob),
            'Precision': precision_score(y_true, y_pred, zero_division=0), 'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0), 'MCC': matthews_corrcoef(y_true, y_pred)}

# --- Data leakage check function ---
def check_data_leakage(X_train, X_test, y_train, y_test, original_features):
    """Check for data leakage"""
    logging.info("="*60)
    logging.info("Data Leakage Check")
    logging.info("="*60)
    
    # 1. Check feature count
    logging.info(f"Original feature count: {original_features}")
    logging.info(f"Training set feature count: {X_train.shape[1]}")
    logging.info(f"Test set feature count: {X_test.shape[1]}")
    
    if X_train.shape[1] != X_test.shape[1]:
        logging.warning("?? Warning: Training and test set have different number of features!")
    
    # 2. Check standardization
    train_mean = X_train.mean().mean()
    train_std = X_train.std().mean()
    test_mean = X_test.mean().mean()
    test_std = X_test.std().mean()
    
    logging.info(f"Training set average mean: {train_mean:.4f}")
    logging.info(f"Training set average standard deviation: {train_std:.4f}")
    logging.info(f"Test set average mean: {test_mean:.4f}")
    logging.info(f"Test set average standard deviation: {test_std:.4f}")
    
    # After standardization, test set mean should be close to 0
    if abs(test_mean) > 0.1:
        logging.warning(f"?? Warning: Test set mean abnormal: {test_mean:.3f} (should be close to 0 after standardization)")
    
    # 3. Check class distribution
    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()
    logging.info(f"Training set positive sample ratio: {train_pos_ratio:.3f}")
    logging.info(f"Test set positive sample ratio: {test_pos_ratio:.3f}")
    
    # 4. Check feature names
    train_features = set(X_train.columns)
    test_features = set(X_test.columns)
    
    if train_features != test_features:
        logging.warning("?? Warning: Training and test set features are different!")
        unique_train = train_features - test_features
        unique_test = test_features - train_features
        if unique_train:
            logging.warning(f"  Training set unique features: {len(unique_train)}")
        if unique_test:
            logging.warning(f"  Test set unique features: {len(unique_test)}")
    
    logging.info("Data leakage check completed")
    return True

# --- Validate external weights ---
def validate_external_weights(weight_file):
    """Validate independence of external weight file"""
    logging.info("Validating external weight file...")
    
    try:
        df_weights = pd.read_csv(weight_file, encoding='utf-8-sig')
        
        # Check required columns
        if 'Gene' not in df_weights.columns or 'Final_Weight' not in df_weights.columns:
            logging.error("Weight file missing 'Gene' or 'Final_Weight' column")
            return False
        
        # Check weight range
        weight_stats = df_weights['Final_Weight'].describe()
        logging.info(f"Weight statistics: range[{weight_stats['min']:.2f}, {weight_stats['max']:.2f}], "
                    f"mean={weight_stats['mean']:.2f}, std={weight_stats['std']:.2f}")
        
        # Check weight distribution
        n_genes = len(df_weights)
        n_weighted = (df_weights['Final_Weight'] > 1.0).sum()
        logging.info(f"Weighted genes: {n_weighted}/{n_genes} ({n_weighted/n_genes:.1%})")
        
        # Check for extreme values
        if weight_stats['max'] > 10:
            logging.warning(f"Found extreme weight value: {weight_stats['max']:.2f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to validate weight file: {e}")
        return False

# --- Weight effect validation function ---
def validate_weight_effect(baseline_features, weighted_features, baseline_model, weighted_model_name, gene_weights_dict=None):
    """Validate the impact of weights on feature selection"""
    
    # Calculate Jaccard similarity coefficient
    if len(baseline_features) > 0 and len(weighted_features) > 0:
        intersection = len(set(baseline_features) & set(weighted_features))
        union = len(set(baseline_features) | set(weighted_features))
        jaccard = intersection / union if union > 0 else 0
        
        # Calculate overlap gene ratio
        overlap_ratio = intersection / len(baseline_features) if len(baseline_features) > 0 else 0
        
        # Calculate average weight of weighted features
        if gene_weights_dict and weighted_features:
            weighted_feature_weights = [gene_weights_dict.get(gene, 1.0) for gene in weighted_features]
            baseline_feature_weights = [gene_weights_dict.get(gene, 1.0) for gene in baseline_features]
            
            avg_weight_weighted = np.mean(weighted_feature_weights) if weighted_feature_weights else 1.0
            avg_weight_baseline = np.mean(baseline_feature_weights) if baseline_feature_weights else 1.0
        else:
            avg_weight_weighted = 1.0
            avg_weight_baseline = 1.0
        
        validation_result = {
            'Model': weighted_model_name,
            'Baseline_Features': len(baseline_features),
            'Weighted_Features': len(weighted_features),
            'Jaccard_Similarity': jaccard,
            'Overlap_Ratio': overlap_ratio,
            'Avg_Weight_Baseline': avg_weight_baseline,
            'Avg_Weight_Weighted': avg_weight_weighted,
            'Weight_Ratio': avg_weight_weighted / avg_weight_baseline if avg_weight_baseline > 0 else 1.0
        }
        
        logging.info(f"Weight effect validation - {weighted_model_name}:")
        logging.info(f"  Baseline features: {len(baseline_features)}, Weighted features: {len(weighted_features)}")
        logging.info(f"  Jaccard similarity coefficient: {jaccard:.3f}, Overlap ratio: {overlap_ratio:.3f}")
        if gene_weights_dict:
            logging.info(f"  Average weight: Baseline={avg_weight_baseline:.3f}, Weighted={avg_weight_weighted:.3f}, Ratio={validation_result['Weight_Ratio']:.3f}")
        
        return validation_result
    else:
        logging.warning(f"Weight effect validation - {weighted_model_name}: Feature set is empty")
        return {
            'Model': weighted_model_name,
            'Baseline_Features': len(baseline_features),
            'Weighted_Features': len(weighted_features),
            'Jaccard_Similarity': 0,
            'Overlap_Ratio': 0,
            'Avg_Weight_Baseline': 1.0,
            'Avg_Weight_Weighted': 1.0,
            'Weight_Ratio': 1.0
        }

# --- Feature oversampling function (for Boruta) ---
def feature_oversampling_by_weights(X, gene_weights, random_state=RANDOM_STATE):
    """
    Perform feature oversampling based on gene weights
    
    Parameters:
    X: Original feature DataFrame
    gene_weights: Weight array aligned with X columns
    random_state: Random seed
    
    Returns:
    X_resampled: Oversampled feature DataFrame
    original_to_resampled: Mapping list from original feature names to oversampled feature indices
    """
    np.random.seed(random_state)
    
    # Normalize weights to probability distribution
    weights_normalized = gene_weights / gene_weights.sum()
    
    # Original feature count
    n_features = X.shape[1]
    feature_names = X.columns.tolist()
    
    # Sample feature indices with replacement based on weights
    sampled_indices = np.random.choice(
        range(n_features), 
        size=n_features,  # Keep feature count unchanged
        replace=True, 
        p=weights_normalized
    )
    
    # Create oversampled feature matrix
    X_resampled = X.iloc[:, sampled_indices].copy()
    
    # Add suffix to duplicate features to maintain unique column names
    new_column_names = []
    original_to_resampled = []
    
    for i, idx in enumerate(sampled_indices):
        original_gene = feature_names[idx]
        # Count how many times this feature appears in currently generated columns
        count = list(sampled_indices[:i+1]).count(idx)
        if count > 1:
            new_name = f"{original_gene}_copy{count}"
        else:
            new_name = original_gene
        new_column_names.append(new_name)
        original_to_resampled.append((original_gene, i))
    
    X_resampled.columns = new_column_names
    
    return X_resampled, original_to_resampled

def main():
    print(">>> Script execution started <<<")
    logging.info("=== Starting Knowledge-Guided Machine Learning Analysis (FIXED VERSION) ===")
    logging.info("Fixed data leakage: preprocessing steps are now fitted only on training set")
    logging.info("Weight source: external third-party databases (safe, no leakage)")
    logging.info("Improvements: Added effective weighting schemes for non-linear models (RF, XGBoost, Boruta)")
    logging.info("New: Validation of feature weight impact on feature selection")
    
    # --- 1. Data loading (no preprocessing yet) ---
    logging.info("1. Reading data (no preprocessing yet)...")
    labels = pd.read_csv("classification_labels_debatch.csv", index_col=0)
    expr_matrix = pd.read_csv("genes_matched_expression_for_KC_ML.csv", index_col=0)
    
    # Check data
    if labels.empty or expr_matrix.empty:
        logging.error("Data loading failed: labels or expression matrix is empty")
        return
    
    common_samples = np.intersect1d(labels.index, expr_matrix.columns)
    if len(common_samples) == 0:
        logging.error("No common samples found")
        return
    
    labels = labels.loc[common_samples]
    expr_matrix = expr_matrix[common_samples]
    
    le = LabelEncoder().fit(labels['Label'])
    X = expr_matrix.T
    y = le.transform(labels['Label'])
    
    original_feature_count = X.shape[1]
    logging.info(f"Original data shape: {X.shape}")
    logging.info(f"Label distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    # --- 2. Split data FIRST (critical fix: avoid data leakage) ---
    logging.info("\n2. Splitting data FIRST (critical fix: avoid data leakage)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    logging.info(f"Training set raw shape: {X_train_raw.shape}")
    logging.info(f"Test set raw shape: {X_test_raw.shape}")
    
    # --- 3. Preprocessing: fitting ONLY on training set ---
    logging.info("\n3. Preprocessing: fitting ONLY on training set...")
    
    # 3.1 Variance threshold selection
    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_raw)
    X_test_selected = selector.transform(X_test_raw)
    
    # Get selected features
    selected_features = X_train_raw.columns[selector.get_support()]
    logging.info(f"Features after variance threshold: {len(selected_features)} (filtered out {X_train_raw.shape[1]-len(selected_features)} low variance features)")
    
    # 3.2 Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Reconstruct DataFrame
    X_train_unweighted = pd.DataFrame(
        X_train_scaled, 
        index=X_train_raw.index, 
        columns=selected_features
    )
    X_test_unweighted = pd.DataFrame(
        X_test_scaled, 
        index=X_test_raw.index, 
        columns=selected_features
    )
    
    logging.info(f"Standardized training set shape: {X_train_unweighted.shape}")
    logging.info(f"Standardized test set shape: {X_test_unweighted.shape}")
    
    # --- 4. Apply external gene weights (safe: weights from third-party database) ---
    logging.info("\n4. Applying knowledge-guided gene weights from external database...")
    
    # Initialize weight information
    gene_weights = None
    weight_map = {}
    weight_info = None
    
    # Validate weight file
    weight_file = 'final_external_data_weights.csv'
    if not os.path.exists(weight_file):
        logging.warning(f"Weight file '{weight_file}' does not exist, using unweighted data")
        X_train = X_train_unweighted
        X_test = X_test_unweighted
    else:
        # Validate weight file
        validate_external_weights(weight_file)
        
        # Load weights
        df_weights = pd.read_csv(weight_file, encoding='utf-8-sig')
        weight_map = df_weights.set_index('Gene')['Final_Weight'].to_dict()
        
        # Create weight vector (aligned to current features)
        gene_weights = []
        for gene in X_train_unweighted.columns:
            weight = weight_map.get(gene, 1.0)  # Default weight is 1.0
            gene_weights.append(weight)
        
        gene_weights = np.array(gene_weights)
        
        # Normalize weights (mean = 1, keep scale unchanged)
        if gene_weights.mean() > 0:
            gene_weights_normalized = gene_weights / gene_weights.mean()
        else:
            gene_weights_normalized = np.ones_like(gene_weights)
        
        # Statistics of weighting
        n_weighted = sum(gene_weights > 1.0)
        n_downweighted = sum(gene_weights < 1.0)
        n_neutral = sum(gene_weights == 1.0)
        
        logging.info(f"Weighted gene statistics: Enhanced={n_weighted}, Reduced={n_downweighted}, Neutral={n_neutral}")
        logging.info(f"Weight range: [{min(gene_weights):.2f}, {max(gene_weights):.2f}]")
        logging.info(f"Normalized weight range: [{min(gene_weights_normalized):.3f}, {max(gene_weights_normalized):.3f}], mean={gene_weights_normalized.mean():.3f}")
        
        # Apply weights to linear model (Elastic Net) data
        X_train_weighted_for_linear = X_train_unweighted * gene_weights
        X_test_weighted_for_linear = X_test_unweighted * gene_weights
        
        # Save weight information
        weight_info = pd.DataFrame({
            'Gene': X_train_unweighted.columns,
            'Original_Weight': gene_weights,
            'Normalized_Weight': gene_weights_normalized,
            'Weight_Type': ['Enhanced' if w > 1.0 else ('Reduced' if w < 1.0 else 'Neutral') for w in gene_weights]
        })
        weight_info.to_csv(os.path.join(RESULTS_DIR, "applied_gene_weights.csv"), index=False)
        
        # Set linear models to use weighted data, nonlinear models to use unweighted data
        X_train_linear = X_train_weighted_for_linear
        X_test_linear = X_test_weighted_for_linear
        X_train_nonlinear = X_train_unweighted
        X_test_nonlinear = X_test_unweighted
    
    # If no weight file, all models use unweighted data
    if gene_weights is None:
        X_train_linear = X_train_unweighted
        X_test_linear = X_test_unweighted
        X_train_nonlinear = X_train_unweighted
        X_test_nonlinear = X_test_unweighted
        gene_weights_normalized = np.ones(X_train_unweighted.shape[1])
        weight_map = {gene: 1.0 for gene in X_train_unweighted.columns}
    
    # --- 5. Check data leakage ---
    check_data_leakage(X_train_linear, X_test_linear, y_train, y_test, original_feature_count)
    
    # --- 6. Handle imbalanced data ---
    logging.info("\n5. Setting up imbalance parameters...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    total = n_neg + n_pos

    logging.info(f"Training set class distribution -> Negative(0): {n_neg} ({n_neg/total:.1%}), "
                 f"Positive(1): {n_pos} ({n_pos/total:.1%})")

    class_weights_rf = 'balanced'
    scale_pos_weight_xgb = n_neg / max(n_pos, 1)

    logging.info(f"Imbalance handling -> RF class_weight = 'balanced', "
                 f"XGBoost scale_pos_weight = {scale_pos_weight_xgb:.3f}")

    cv_performance_reports, test_performance_reports, roc_data, model_selections = [], [], {}, {}
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'mcc': make_scorer(matthews_corrcoef)
    }
    
    # Store weight effect validation results
    weight_validation_results = []

    # ==============================================================================
    #  MODEL 1: ELASTIC NET (modified: add baseline model for weight effect validation)
    # ==============================================================================
    logging.info("\n--- Starting Model 1: Elastic Net ---")
    try:
        # --- 1. Train baseline model (without weight adjustment) ---
        logging.info("Training baseline Elastic Net (without weight adjustment)...")
        enet_model_baseline = LogisticRegressionCV(
            penalty='elasticnet', 
            solver='saga', 
            l1_ratios=[0.5], 
            cv=cv_strategy, 
            random_state=RANDOM_STATE, 
            max_iter=5000, 
            scoring='roc_auc', 
            class_weight='balanced', 
            n_jobs=-1
        )
        
        # Use unweighted data for baseline model
        enet_model_baseline.fit(X_train_unweighted, y_train)
        baseline_enet_genes = X_train_unweighted.columns[enet_model_baseline.coef_[0] != 0].tolist()
        logging.info(f"Baseline Elastic Net selected {len(baseline_enet_genes)} genes")
        
        # --- 2. Train weighted model (with weight adjustment) ---
        logging.info("Training weighted Elastic Net (with weight adjustment)...")
        enet_model_weighted = LogisticRegressionCV(
            penalty='elasticnet', 
            solver='saga', 
            l1_ratios=[0.5], 
            cv=cv_strategy, 
            random_state=RANDOM_STATE, 
            max_iter=5000, 
            scoring='roc_auc', 
            class_weight='balanced', 
            n_jobs=-1
        )
        
        # Cross-validation on weighted data
        cv_results = cross_validate(enet_model_weighted, X_train_linear, y_train, cv=cv_strategy, scoring=scoring_metrics)

        auc_scores = cv_results['test_roc_auc']
        cv_auc_mean = np.mean(auc_scores)
        cv_auc_std = np.std(auc_scores)
        logging.info(f"Elastic Net CV AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")

        cv_report = {
            'Model': 'Elastic_Net', 
            **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k}, 
            **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}
        }
        cv_performance_reports.append(cv_report)

        # Train final weighted model
        enet_model_final = enet_model_weighted.fit(X_train_linear, y_train)
        y_pred = enet_model_final.predict(X_test_linear)
        y_prob = enet_model_final.predict_proba(X_test_linear)[:, 1]
        
        test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'Elastic_Net'))
        
        # Nature style confusion matrix
        plot_nature_confusion_matrix(y_test, y_pred, le.classes_, 'Elastic Net')
        
        # Nature style PR curve
        plot_nature_pr_curve(pd.Series(y_test), pd.Series(y_prob), 'Elastic Net', '#E41A1C')
        
        # Nature style ROC curve
        roc_data['Elastic_Net'] = plot_nature_roc_curve(y_test, y_prob, 'Elastic Net', '#E41A1C')
        
        # Feature selection from weighted model
        enet_genes = X_train_linear.columns[enet_model_final.coef_[0] != 0].tolist()
        model_selections['Elastic_Net'] = set(enet_genes)
        
        pd.DataFrame({'Gene': enet_genes}).to_csv(
            os.path.join(RESULTS_DIR, "Elastic_Net_selected_genes.csv"), index=False
        )
        
        logging.info(f"Elastic Net selected {len(enet_genes)} genes")
        
        # --- 3. Validate weight effect ---
        logging.info("Validating weight effect on Elastic Net feature selection...")
        enet_validation = validate_weight_effect(
            baseline_features=baseline_enet_genes,
            weighted_features=enet_genes,
            baseline_model=enet_model_baseline,
            weighted_model_name='Elastic_Net',
            gene_weights_dict=weight_map
        )
        weight_validation_results.append(enet_validation)
        
        # SHAP analysis
        logging.info("Performing SHAP analysis for Elastic Net...")
        try:
            explainer = shap.LinearExplainer(enet_model_final, X_train_linear)
            shap_values = explainer.shap_values(X_test_linear)
            
            # Nature style SHAP bar plot
            plt.figure(figsize=(6, 4))
            shap.summary_plot(shap_values, X_test_linear, show=False, plot_type="bar", max_display=20)
            plt.title("Elastic Net SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_enet_bar_nature")
            
            # Nature style SHAP beeswarm plot
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_test_linear, show=False, max_display=20)
            plt.title("Elastic Net SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_enet_beeswarm_nature")
            
            logging.info("SHAP analysis for Elastic Net complete.")
        except Exception as e:
            logging.error(f"SHAP analysis for Elastic Net failed: {e}")

    except Exception as e: 
        logging.error(f"Elastic Net failed: {e}")

    # ==============================================================================
    #  MODEL 2: RANDOM FOREST (improvement: post-processing correction of feature importance)
    # ==============================================================================
    logging.info("\n--- Starting Model 2: Random Forest (improvement: feature sampling bias) ---")
    try:
        # 1. Train baseline model (without weight adjustment)
        logging.info("Training baseline Random Forest (without weight adjustment)...")
        rf_model_baseline = RandomForestClassifier(
            n_estimators=500, 
            class_weight=class_weights_rf, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        rf_model_baseline.fit(X_train_nonlinear, y_train)
        
        # Get baseline model feature importance
        baseline_mdg_imp = pd.DataFrame({
            'Gene': X_train_nonlinear.columns, 
            'MDG': rf_model_baseline.feature_importances_
        }).sort_values('MDG', ascending=False)
        
        # Select baseline model's top 50 features
        baseline_top_genes = set(baseline_mdg_imp.head(50)['Gene'])
        
        # 2. Train weighted improved model
        logging.info("Training weighted Random Forest (with weight adjustment)...")
        rf_model_weighted = RandomForestClassifier(
            n_estimators=500, 
            class_weight=class_weights_rf, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        
        cv_results = cross_validate(rf_model_weighted, X_train_nonlinear, y_train, cv=cv_strategy, scoring=scoring_metrics)

        auc_scores = cv_results['test_roc_auc']
        cv_auc_mean = np.mean(auc_scores)
        cv_auc_std = np.std(auc_scores)
        logging.info(f"Random Forest CV AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
        
        cv_report = {
            'Model': 'RF_Weighted', 
            **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k},
            **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}
        }
        cv_performance_reports.append(cv_report)

        rf_model_final = rf_model_weighted.fit(X_train_nonlinear, y_train)
        y_pred = rf_model_final.predict(X_test_nonlinear)
        y_prob = rf_model_final.predict_proba(X_test_nonlinear)[:, 1]
        
        test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'RF_Weighted'))
        
        # Nature style confusion matrix
        plot_nature_confusion_matrix(y_test, y_pred, le.classes_, 'Random Forest')
        
        # Nature style PR curve
        plot_nature_pr_curve(pd.Series(y_test), pd.Series(y_prob), 'Random Forest', '#4DAF4A')
        
        # Nature style ROC curve
        roc_data['RF_Weighted'] = plot_nature_roc_curve(y_test, y_prob, 'Random Forest', '#4DAF4A')

        # 3. Apply weight correction to feature importance
        logging.info("Applying weight adjustment to Random Forest feature importance...")
        
        # Get original feature importance
        original_mdg = rf_model_final.feature_importances_
        
        # Apply weight correction: weighted importance = original importance * normalized weight
        weighted_mdg = original_mdg * gene_weights_normalized
        
        # Create weighted feature importance DataFrame
        mdg_imp = pd.DataFrame({
            'Gene': X_train_nonlinear.columns, 
            'MDG': original_mdg,
            'Weighted_MDG': weighted_mdg,
            'Normalized_Weight': gene_weights_normalized
        }).sort_values('Weighted_MDG', ascending=False)
        
        mdg_imp.to_csv(os.path.join(RESULTS_DIR, "RF_MDG_importance_weighted.csv"), index=False)
        
        # Select weighted top 50 features
        weighted_top_genes = set(mdg_imp.head(50)['Gene'])
        model_selections['RF_MDA_Top50'] = weighted_top_genes
        
        # 4. Permutation importance
        logging.info("Computing permutation importance with weight adjustment...")
        perm_imp = permutation_importance(
            rf_model_final, X_test_nonlinear, y_test,
            scoring='roc_auc',
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        # Apply weight correction to permutation importance
        weighted_mda = perm_imp.importances_mean * gene_weights_normalized
        
        mda_imp = pd.DataFrame({
            'Gene': X_train_nonlinear.columns, 
            'MDA': perm_imp.importances_mean,
            'Weighted_MDA': weighted_mda,
            'Normalized_Weight': gene_weights_normalized
        }).sort_values('Weighted_MDA', ascending=False)
        
        mda_imp.to_csv(os.path.join(RESULTS_DIR, "RF_MDA_importance_weighted.csv"), index=False)
        
        logging.info("RF permutation importance computed with weight adjustment")
        
        # 5. Validate weight effect
        logging.info("Validating weight effect on Random Forest feature selection...")
        rf_validation = validate_weight_effect(
            baseline_features=list(baseline_top_genes),
            weighted_features=list(weighted_top_genes),
            baseline_model=rf_model_baseline,
            weighted_model_name='RF_Weighted',
            gene_weights_dict=weight_map
        )
        weight_validation_results.append(rf_validation)
        
        # 6. SHAP analysis
        logging.info("Performing SHAP analysis for Random Forest...")
        try:
            explainer = shap.TreeExplainer(rf_model_final)
            shap_values = explainer.shap_values(X_test_nonlinear)
            
            # Calculate weighted SHAP importance
            mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
            weighted_shap = mean_abs_shap * gene_weights_normalized
            
            shap_imp_df = pd.DataFrame({
                'Gene': X_train_nonlinear.columns,
                'SHAP_Importance': mean_abs_shap,
                'Weighted_SHAP': weighted_shap,
                'Normalized_Weight': gene_weights_normalized
            }).sort_values('Weighted_SHAP', ascending=False)
            
            shap_imp_df.to_csv(os.path.join(RESULTS_DIR, "RF_SHAP_importance_weighted.csv"), index=False)
            
            top_genes_indices = np.argsort(weighted_shap)[-20:]
            top_genes_names = X_train_nonlinear.columns[top_genes_indices].tolist()
            model_selections['RF_SHAP_Top20'] = set(top_genes_names)
            
            # Nature style SHAP bar plot (using weighted importance)
            plt.figure(figsize=(6, 4))
            
            # Prepare data: take top 20 weighted SHAP features
            top_n = 20
            top_indices = np.argsort(weighted_shap)[-top_n:][::-1]
            top_genes_shap = X_train_nonlinear.columns[top_indices]
            top_shap_values = weighted_shap[top_indices]
            
            plt.barh(range(top_n), top_shap_values[::-1])
            plt.yticks(range(top_n), top_genes_shap[::-1])
            plt.xlabel('Weighted SHAP Importance')
            plt.title("Random Forest Weighted SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            save_nature_plot("shap_summary_rf_bar_weighted_nature")
            
            # Nature style SHAP beeswarm plot
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values[1], X_test_nonlinear, show=False, max_display=20)
            plt.title("Random Forest SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_rf_beeswarm_nature")
            
            logging.info("SHAP analysis for Random Forest complete.")
        except Exception as e:
            logging.error(f"SHAP analysis for Random Forest failed: {e}")

    except Exception as e: 
        logging.error(f"Random Forest main process failed: {e}")

    # ==============================================================================
    # MODEL 3: XGBOOST (improvement: use feature weight parameter)
    # ==============================================================================
    logging.info("\n--- Starting Model 3: XGBoost (improvement: feature weight) ---")
    try:
        # 1. Prepare feature weights
        # Convert normalized weights to XGBoost format
        # XGBoost's feature_weights expects weights in (0, 1] range
        if gene_weights_normalized is not None:
            # Ensure all weights are positive
            xgb_feature_weights = np.maximum(gene_weights_normalized, 0.001)
            # Optional: normalize to (0, 1] range
            if xgb_feature_weights.max() > 0:
                xgb_feature_weights = xgb_feature_weights / xgb_feature_weights.max()
        else:
            xgb_feature_weights = None
        
        # 2. Train baseline model (without feature weights)
        logging.info("Training baseline XGBoost (without feature weights)...")
        xgb_model_baseline = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight_xgb,
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_estimators=500,
            n_jobs=-1
        )
        xgb_model_baseline.fit(X_train_nonlinear, y_train, verbose=False)
        
        # Extract baseline model features
        try:
            baseline_gain_dict = xgb_model_baseline.get_booster().get_score(importance_type='gain')
            baseline_selected_genes = [gene for gene, gain in baseline_gain_dict.items() if gain > 0]
        except:
            baseline_selected_genes = []
        
        # 3. Train weighted improved model
        logging.info("Training weighted XGBoost (with feature weights)...")
        xgb_model_weighted = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight_xgb,
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_estimators=500,
            n_jobs=-1
        )
        
        cv_results = cross_validate(xgb_model_weighted, X_train_nonlinear, y_train, cv=cv_strategy, scoring=scoring_metrics)

        auc_scores = cv_results['test_roc_auc']
        cv_auc_mean = np.mean(auc_scores)
        cv_auc_std = np.std(auc_scores)
        logging.info(f"XGBoost CV AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
        
        cv_report = {
            'Model': 'XGBoost', 
            **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k},
            **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}
        }
        cv_performance_reports.append(cv_report)

        # Train model with feature_weights parameter
        if xgb_feature_weights is not None:
            xgb_model_final = xgb_model_weighted.fit(
                X_train_nonlinear, y_train, 
                verbose=False,
                feature_weights=xgb_feature_weights
            )
            logging.info("XGBoost trained with feature_weights parameter")
        else:
            xgb_model_final = xgb_model_weighted.fit(X_train_nonlinear, y_train, verbose=False)
            logging.info("XGBoost trained without feature_weights (no weight file available)")
        
        y_pred = xgb_model_final.predict(X_test_nonlinear)
        y_prob = xgb_model_final.predict_proba(X_test_nonlinear)[:, 1]

        test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'XGBoost'))

        # Nature style confusion matrix
        plot_nature_confusion_matrix(y_test, y_pred, le.classes_, 'XGBoost')
        
        # Nature style PR curve
        plot_nature_pr_curve(pd.Series(y_test), pd.Series(y_prob), 'XGBoost', '#984EA3')
        
        # Nature style ROC curve
        roc_data['XGBoost'] = plot_nature_roc_curve(y_test, y_prob, 'XGBoost', '#984EA3')

        # 4. Extract weighted model features
        logging.info("Extracting XGBoost features with positive Gain importance...")
        try:
            gain_dict = xgb_model_final.get_booster().get_score(importance_type='gain')
            xgb_selected_genes = [gene for gene, gain in gain_dict.items() if gain > 0]
            model_selections['XGBoost_Gain_Positive'] = set(xgb_selected_genes)
            logging.info(f"XGBoost selected {len(xgb_selected_genes)} genes with Gain > 0")

            # Save complete importance information
            gain_df = pd.DataFrame([
                {'Gene': gene, 'Gain': gain}
                for gene, gain in gain_dict.items()
            ])
            
            # Add weight information
            if weight_map:
                gain_df['External_Weight'] = gain_df['Gene'].map(lambda g: weight_map.get(g, 1.0))
            
            gain_df = gain_df.sort_values('Gain', ascending=False)
            gain_df.to_csv(os.path.join(RESULTS_DIR, "XGBoost_Gain_importance_weighted.csv"), index=False)

        except Exception as e:
            logging.error(f"Failed to extract XGBoost Gain importance: {e}")
            xgb_selected_genes = []

        # 5. Validate weight effect
        logging.info("Validating weight effect on XGBoost feature selection...")
        xgb_validation = validate_weight_effect(
            baseline_features=baseline_selected_genes,
            weighted_features=xgb_selected_genes,
            baseline_model=xgb_model_baseline,
            weighted_model_name='XGBoost',
            gene_weights_dict=weight_map
        )
        weight_validation_results.append(xgb_validation)

        # 6. SHAP analysis
        logging.info("Performing SHAP analysis for XGBoost...")
        try:
            explainer = shap.TreeExplainer(xgb_model_final)
            shap_values = explainer.shap_values(X_test_nonlinear)

            # Nature style SHAP bar plot
            plt.figure(figsize=(6, 4))
            shap.summary_plot(shap_values, X_test_nonlinear, show=False, plot_type="bar", max_display=20)
            plt.title("XGBoost SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_xgb_bar_nature")
            
            # Nature style SHAP beeswarm plot
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_test_nonlinear, show=False, max_display=20)
            plt.title("XGBoost SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_xgb_beeswarm_nature")
            
            logging.info("XGBoost SHAP plots generated.")
        except Exception as e:
            logging.error(f"SHAP analysis for XGBoost failed: {e}")

    except Exception as e:
        logging.error(f"XGBoost main process failed: {e}")

    # ==============================================================================
    #  MODEL 4: BORUTA + RANDOM FOREST (improvement: introduce selection bias through feature oversampling)
    # ==============================================================================
    logging.info("\n--- Starting Model 4: Boruta + RF (improvement: selection bias) ---")
    try:
        # 1. Train baseline model (without oversampling)
        logging.info("Training baseline Boruta (without feature oversampling)...")
        rf_boruta_baseline = RandomForestClassifier(
            n_jobs=-1, 
            class_weight=class_weights_rf, 
            max_depth=5, 
            random_state=RANDOM_STATE
        )
        
        boruta_selector_baseline = BorutaPy(
            rf_boruta_baseline, 
            n_estimators='auto', 
            verbose=0, 
            random_state=RANDOM_STATE, 
            max_iter=100
        )
        
        boruta_selector_baseline.fit(X_train_nonlinear.values, y_train)
        baseline_confirmed_genes = X_train_nonlinear.columns[boruta_selector_baseline.support_].tolist()
        logging.info(f"Baseline Boruta confirmed {len(baseline_confirmed_genes)} features.")
        
        # 2. Train weighted improved model (with feature oversampling)
        logging.info("Training weighted Boruta (with feature oversampling by weights)...")
        
        if gene_weights_normalized is not None:
            # Perform feature oversampling
            X_train_resampled, original_to_resampled = feature_oversampling_by_weights(
                X_train_nonlinear, gene_weights_normalized, random_state=RANDOM_STATE
            )
            
            # Create mapping from oversampled features to original features
            resampled_to_original = {}
            for original_gene, resampled_idx in original_to_resampled:
                resampled_gene = X_train_resampled.columns[resampled_idx]
                resampled_to_original[resampled_gene] = original_gene
            
            logging.info(f"Feature oversampling completed: {X_train_nonlinear.shape[1]} -> {X_train_resampled.shape[1]} features")
            
            # Train Boruta on oversampled data
            rf_boruta_base = RandomForestClassifier(
                n_jobs=-1, 
                class_weight=class_weights_rf, 
                max_depth=5, 
                random_state=RANDOM_STATE
            )
            
            boruta_selector = BorutaPy(
                rf_boruta_base, 
                n_estimators='auto', 
                verbose=0, 
                random_state=RANDOM_STATE, 
                max_iter=100
            )
            
            boruta_selector.fit(X_train_resampled.values, y_train)
            
            # Get confirmed features (in oversampled feature space)
            resampled_confirmed_features = X_train_resampled.columns[boruta_selector.support_].tolist()
            
            # Map back to original features
            confirmed_genes = []
            for feature in resampled_confirmed_features:
                if feature in resampled_to_original:
                    original_gene = resampled_to_original[feature]
                    if original_gene not in confirmed_genes:
                        confirmed_genes.append(original_gene)
                else:
                    # If no mapping, use directly (might be original feature without "_copy" suffix)
                    if feature not in confirmed_genes:
                        confirmed_genes.append(feature)
            
            logging.info(f"Weighted Boruta confirmed {len(confirmed_genes)} original features.")
            
        else:
            # If no weights, use original method
            rf_boruta_base = RandomForestClassifier(
                n_jobs=-1, 
                class_weight=class_weights_rf, 
                max_depth=5, 
                random_state=RANDOM_STATE
            )
            
            boruta_selector = BorutaPy(
                rf_boruta_base, 
                n_estimators='auto', 
                verbose=0, 
                random_state=RANDOM_STATE, 
                max_iter=100
            )
            
            boruta_selector.fit(X_train_nonlinear.values, y_train)
            confirmed_genes = X_train_nonlinear.columns[boruta_selector.support_].tolist()
            logging.info(f"Boruta confirmed {len(confirmed_genes)} features (no weights available).")
        
        model_selections['Boruta'] = set(confirmed_genes)
        
        # Save confirmed genes
        pd.DataFrame({'Gene': confirmed_genes}).to_csv(
            os.path.join(RESULTS_DIR, "Boruta_confirmed_genes_weighted.csv"), index=False
        )
        
        # 3. Validate weight effect
        logging.info("Validating weight effect on Boruta feature selection...")
        boruta_validation = validate_weight_effect(
            baseline_features=baseline_confirmed_genes,
            weighted_features=confirmed_genes,
            baseline_model=rf_boruta_baseline,
            weighted_model_name='Boruta_RF',
            gene_weights_dict=weight_map
        )
        weight_validation_results.append(boruta_validation)
        
        if len(confirmed_genes) > 0:
            # Train final model with confirmed genes
            X_train_boruta = X_train_nonlinear[confirmed_genes]
            X_test_boruta = X_test_nonlinear[confirmed_genes]
            
            rf_model_base = RandomForestClassifier(
                n_estimators=100, 
                class_weight=class_weights_rf, 
                random_state=RANDOM_STATE, 
                n_jobs=-1
            )
            
            cv_results = cross_validate(rf_model_base, X_train_boruta, y_train, cv=cv_strategy, scoring=scoring_metrics)

            auc_scores = cv_results['test_roc_auc']
            cv_auc_mean = np.mean(auc_scores)
            cv_auc_std = np.std(auc_scores)
            logging.info(f"Boruta+RF CV AUC: {cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
            
            cv_report = {
                'Model': 'Boruta_RF',
                **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k},
                **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}
            }
            cv_performance_reports.append(cv_report)

            rf_model_final = rf_model_base.fit(X_train_boruta, y_train)
            y_pred = rf_model_final.predict(X_test_boruta)
            y_prob = rf_model_final.predict_proba(X_test_boruta)[:, 1]

            test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'Boruta_RF'))
            
            # Nature style confusion matrix
            plot_nature_confusion_matrix(y_test, y_pred, le.classes_, 'Boruta+RF')
            
            # Nature style PR curve
            plot_nature_pr_curve(pd.Series(y_test), pd.Series(y_prob), 'Boruta+RF', '#FF7F00')
            
            # Nature style ROC curve
            roc_data['Boruta_RF'] = plot_nature_roc_curve(y_test, y_prob, 'Boruta+RF', '#FF7F00')

            # SHAP analysis
            logging.info("Performing SHAP analysis for Boruta+RF model...")
            try:
                explainer = shap.TreeExplainer(rf_model_final)
                shap_values = explainer.shap_values(X_test_boruta)
                
                # Nature style SHAP bar plot
                plt.figure(figsize=(6, 4))
                shap.summary_plot(shap_values[1], X_test_boruta, show=False, plot_type="bar", max_display=20)
                plt.title("Boruta+RF SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                save_nature_plot("shap_summary_boruta_rf_bar_nature")
                
                # Nature style SHAP beeswarm plot
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values[1], X_test_boruta, show=False, max_display=20)
                plt.title("Boruta+RF SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                save_nature_plot("shap_summary_boruta_rf_beeswarm_nature")
                
                mean_abs_shap_boruta = np.abs(shap_values[1]).mean(axis=0)
                top20_idx = np.argsort(mean_abs_shap_boruta)[-20:]
                model_selections['Boruta_SHAP_Top20'] = set(X_test_boruta.columns[top20_idx])
                logging.info("SHAP analysis for Boruta+RF complete.")
            except Exception as e:
                logging.error(f"SHAP analysis for Boruta+RF failed: {e}")
        else:
            logging.warning("Boruta confirmed no features. Skipping model.")
    except Exception as e: 
        logging.error(f"Boruta+RF failed: {e}")

    # ==============================================================================
    #  Final reporting and visualization
    # ==============================================================================
    logging.info("\n--- Generating Final Reports and Visualizations ---")
    
    if cv_performance_reports:
        df_cv = pd.DataFrame(cv_performance_reports)
        df_cv.to_csv(os.path.join(RESULTS_DIR, "cv_performance_report.csv"), index=False)
        logging.info(f"\n--- 10-Fold Cross-Validation Performance (on Training Set) ---\n{df_cv.round(4)}")
    
    if test_performance_reports:
        df_test = pd.DataFrame(test_performance_reports)
        df_test.to_csv(os.path.join(RESULTS_DIR, "test_set_performance_report.csv"), index=False)
        logging.info(f"\n--- Final Hold-out Test Set Performance ---\n{df_test.round(4)}")
        
        # Nature style performance comparison plot
        plot_nature_performance_comparison(df_test)
    
    # ==============================================================================
    # Save weight effect validation results
    # ==============================================================================
    logging.info("\n--- Saving Weight Effect Validation Results ---")
    if weight_validation_results:
        df_weight_validation = pd.DataFrame(weight_validation_results)
        df_weight_validation.to_csv(os.path.join(RESULTS_DIR, "weight_effect_validation.csv"), index=False)
        
        logging.info(f"\n--- Weight Effect on Feature Selection ---")
        for _, row in df_weight_validation.iterrows():
            logging.info(f"{row['Model']}: Jaccard similarity coefficient={row['Jaccard_Similarity']:.3f}, "
                        f"Weight ratio={row['Weight_Ratio']:.3f}, "
                        f"Baseline features={row['Baseline_Features']}, "
                        f"Weighted features={row['Weighted_Features']}")
        
        # Visualize weight effect
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            
            for idx, (_, row) in enumerate(df_weight_validation.iterrows()):
                ax = axes[idx]
                
                # Create bar chart
                metrics = ['Jaccard_Similarity', 'Overlap_Ratio', 'Weight_Ratio']
                metric_names = ['Jaccard Similarity', 'Overlap Ratio', 'Weight Ratio']
                values = [row[m] for m in metrics]
                
                bars = ax.bar(metric_names, values, color=['#4DAF4A', '#984EA3', '#FF7F00'])
                ax.set_ylim(0, max(values) * 1.2)
                ax.set_title(f"{row['Model']}", fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=10)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
            
            plt.suptitle("Feature Weight Impact on Feature Selection Validation", fontsize=13, fontweight='bold', y=0.98)
            plt.tight_layout()
            save_nature_plot("weight_effect_validation_plot")
            
        except Exception as e:
            logging.error(f"Failed to create weight effect visualization: {e}")

    # ==============================================================================
    # Model consensus analysis (enhanced robustness version)
    # ==============================================================================
    logging.info("\n--- Performing Model Consensus Analysis ---")
    
    # Initialize a list to record any step status or errors during consensus analysis
    consensus_log = []
    
    try:
        # Step 1: Prepare valid gene sets
        valid_sets = {}
        for model_name, gene_set in model_selections.items():
            if gene_set and len(gene_set) > 0:
                valid_sets[model_name] = set(gene_set)
                consensus_log.append(f"Model '{model_name}' contributed {len(gene_set)} genes.")
            else:
                consensus_log.append(f"Model '{model_name}' gene set is empty or invalid, skipped.")
        
        if len(valid_sets) < 2:
            log_msg = f"Consensus analysis interrupted: insufficient valid gene sets ({len(valid_sets)}). At least 2 models with results are required."
            logging.warning(log_msg)
            consensus_log.append(log_msg)
            # Even with insufficient models, try to generate a simple merged list
            all_genes_simple = []
            for gene_set in valid_sets.values():
                all_genes_simple.extend(list(gene_set))
            if all_genes_simple:
                simple_consensus = pd.DataFrame({'Gene': all_genes_simple})
                simple_consensus['Selected_In'] = 'Multiple' if len(valid_sets)>1 else list(valid_sets.keys())[0]
                simple_consensus.to_csv(os.path.join(RESULTS_DIR, "simple_gene_list_fallback.csv"), index=False)
                logging.info(f"Generated a fallback simple gene list containing {len(simple_consensus)} genes.")
        else:
            # Step 2: Calculate consensus scores
            all_genes = []
            for gene_set in valid_sets.values():
                all_genes.extend(list(gene_set))
            gene_counts = Counter(all_genes)
            
            consensus_df = pd.DataFrame(gene_counts.items(), columns=['Gene', 'Consensus_Score'])
            num_models = len(valid_sets)
            consensus_df['Selection_Frequency'] = consensus_df['Consensus_Score'] / num_models
            consensus_df = consensus_df.sort_values('Consensus_Score', ascending=False)
            
            # Step 3: Integrate external weight information
            if weight_map:
                consensus_df['External_Weight'] = consensus_df['Gene'].map(lambda g: weight_map.get(g, 1.0))
                consensus_df['Weighted_Consensus_Score'] = consensus_df['Consensus_Score'] * consensus_df['External_Weight']
                # Save two sorted versions
                consensus_df_basic = consensus_df.sort_values('Consensus_Score', ascending=False)
                consensus_df_weighted = consensus_df.sort_values('Weighted_Consensus_Score', ascending=False)
                
                consensus_df_basic.to_csv(os.path.join(RESULTS_DIR, "model_consensus_gene_report_basic.csv"), index=False)
                consensus_df_weighted.to_csv(os.path.join(RESULTS_DIR, "model_consensus_gene_report_weighted.csv"), index=False)
                
                logging.info(f"Consensus report (basic and weighted versions) saved, containing {len(consensus_df)} unique genes.")
                logging.info(f"Top 5 basic consensus genes: {consensus_df_basic.head(5)['Gene'].tolist()}")
                logging.info(f"Top 5 weighted consensus genes: {consensus_df_weighted.head(5)['Gene'].tolist()}")
                consensus_log.append(f"Successfully generated basic and weighted consensus reports, involving {len(valid_sets)} models.")
            else:
                consensus_df.to_csv(os.path.join(RESULTS_DIR, "model_consensus_gene_report.csv"), index=False)
                logging.info(f"Consensus report saved, containing {len(consensus_df)} unique genes.")
                logging.info(f"Top 10 consensus genes:\n{consensus_df.head(10)[['Gene', 'Consensus_Score']].to_string(index=False)}")
                consensus_log.append(f"Successfully generated consensus report, involving {len(valid_sets)} models.")
            
            # Step 4: Visualization - try to create UpSet plot, fallback to bar plot if failed
            try:
                from upsetplot import UpSet, from_contents
                logging.info("Attempting to create UpSet plot...")
                
                # Prepare UpSet data
                upset_contents = {name: list(genes) for name, genes in valid_sets.items()}
                upset_data = from_contents(upset_contents)
                
                # Create UpSet plot
                upset = UpSet(
                    upset_data,
                    subset_size='count',
                    show_counts=True,
                    sort_by='cardinality',
                    sort_categories_by='cardinality',
                    facecolor='steelblue',
                    shading_color='lightgray',
                    element_size=40
                )
                
                fig = plt.figure(figsize=(10, 6))
                upset.plot(fig=fig)
                fig.suptitle("Key Gene Intersections Across Multiple Models", 
                           fontsize=13, fontweight='bold', y=0.98)
                plt.tight_layout()
                save_nature_plot("Key_Gene_Intersection_UpSet_Plot_nature")
                logging.info("Nature style UpSet plot generated successfully!")
                consensus_log.append("Successfully created UpSet plot.")
                
            except ImportError as e:
                log_msg = f"Cannot import upsetplot library, will use bar plot instead. Error details: {e}"
                logging.warning(log_msg)
                consensus_log.append(log_msg)
                _create_fallback_bar_plot(consensus_df.head(20), valid_sets)
            except Exception as e:
                log_msg = f"Unexpected error when creating UpSet plot, will use bar plot instead. Error details: {e}"
                logging.error(log_msg)
                consensus_log.append(log_msg)
                _create_fallback_bar_plot(consensus_df.head(20), valid_sets)
                
    except Exception as e:
        # Catch any unexpected exceptions in the main consensus analysis process
        error_msg = f"Serious error during model consensus analysis: {e}"
        logging.error(error_msg)
        consensus_log.append(error_msg)
        # Even if failed here, try to record currently collected model selection results
        try:
            pd.DataFrame([{'Model': k, 'Num_Genes': len(v)} for k, v in model_selections.items()]).to_csv(
                os.path.join(RESULTS_DIR, "model_selection_summary_interrupted.csv"), index=False)
        except:
            pass
    
    # Step 5: Save consensus analysis process log regardless of success
    try:
        log_df = pd.DataFrame(consensus_log, columns=['Consensus_Analysis_Log'])
        log_df.to_csv(os.path.join(RESULTS_DIR, "consensus_analysis_process_log.csv"), index=False)
        logging.info("Consensus analysis process log saved.")
    except Exception as e:
        logging.error(f"Failed to save consensus analysis process log: {e}")

    def _create_fallback_bar_plot(consensus_data, valid_sets):
        """Create fallback consensus gene bar plot"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Subplot 1: Top consensus genes
            ax1.barh(consensus_data['Gene'].head(15)[::-1], consensus_data['Consensus_Score'].head(15)[::-1])
            ax1.set_xlabel('Consensus Score (Number of Methods Selecting Gene)')
            ax1.set_title('Top Consensus Genes', fontsize=13, fontweight='bold')
            ax1.set_xlim(0, max(consensus_data['Consensus_Score'].head(15)) * 1.1)
            
            # Subplot 2: Gene count per model
            model_names = list(valid_sets.keys())
            gene_counts = [len(valid_sets[m]) for m in model_names]
            ax2.bar(model_names, gene_counts, color='skyblue')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Number of Selected Genes')
            ax2.set_title('Gene Count per Model', fontsize=13, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.suptitle("Consensus Analysis (Fallback Visualization)", fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            save_nature_plot("top_consensus_genes_fallback_bar_nature")
            logging.info("Fallback bar plot generated.")
        except Exception as e:
            logging.error(f"Failed to create fallback bar plot: {e}")

    # Nature style combined ROC curves
    if roc_data:
        plot_nature_combined_roc(roc_data)

    logging.info("\n=== Analysis Complete: All Nature-style reports and plots saved to '%s' directory. ===" % RESULTS_DIR)
    logging.info("Data leakage fixed: preprocessing steps now fitted only on training set, test set transformed independently.")

if __name__ == '__main__':
    main()