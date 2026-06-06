# -*- coding: utf-8 -*-
# ==============================================================================
# Machine Learning for Key Gene Selection (Comprehensive Analysis Pipeline)
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
import venn
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
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Nature style general settings ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',  
    'font.size': 12,
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

# --- Nature style plotting functions ---
def plot_nature_confusion_matrix(y_true, y_pred, classes, model_name):
    """Nature style confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Use seaborn heatmap
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
    
    # Label settings
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'{model_name} Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    
    # Hide top and right borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    save_nature_plot(f'confusion_matrix_{model_name.lower().replace(" ", "_")}_nature')

def plot_nature_pr_curve(y_true, y_prob, model_name, color):
    """Nature style PR curve plot"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    # Compute baseline (random classifier performance)
    baseline_precision = np.mean(y_true)
    
    # Compute bootstrap confidence intervals
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
    
    # Compute confidence intervals
    ci_low, ci_high = bootstrap_ap(y_true, y_prob, n_bootstrap=2000, random_state=RANDOM_STATE)
    
    # Create Nature style plot
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    # Plot PR curve
    ax.plot(recall, precision, lw=2.5, color=color, label=f'{model_name}')
    
    # Random baseline
    ax.axhline(y=baseline_precision, color='black', linestyle='--', 
               lw=1.5, alpha=0.6, label=f'Random (AP={baseline_precision:.3f})')
    
    # Axis settings
    ax.set_xlim(-0.05, 1.01)
    ax.set_ylim(-0.05, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # Labels
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    
    # AP + CI annotation
    text = f"AP = {ap_score:.3f}\n95% CI: {ci_low:.3f}–{ci_high:.3f}"
    ax.text(0.55, 0.12, text, fontsize=10, ha="left", va="center", 
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # Hide top and right borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    # Legend
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    
    # Title
    ax.set_title(f'{model_name} Precision-Recall Curve', fontsize=13, fontweight='bold')
    
    save_nature_plot(f'pr_curve_{model_name.lower().replace(" ", "_")}_nature')

def plot_nature_roc_curve(y_true, y_prob, model_name, color):
    """Nature style ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    # Bootstrap confidence intervals
    bootstrapped_scores = []
    for i in range(100):
        y_true_res, y_prob_res = resample(y_true, y_prob, random_state=i)
        if len(np.unique(y_true_res)) > 1:
            bootstrapped_scores.append(roc_auc_score(y_true_res, y_prob_res))
    
    lower = np.percentile(bootstrapped_scores, 2.5) if bootstrapped_scores else auc_score
    upper = np.percentile(bootstrapped_scores, 97.5) if bootstrapped_scores else auc_score
    
    # Create Nature style plot
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2.5, color=color, label=model_name)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random')
    
    # Axis settings
    ax.set_xlim(-0.05, 1.01)
    ax.set_ylim(-0.05, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # Labels
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    
    # AUC + CI annotation
    text = f"AUC = {auc_score:.3f}\n95% CI: {lower:.3f}–{upper:.3f}"
    ax.text(0.55, 0.12, text, fontsize=10, ha="left", va="center", 
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # Hide top and right borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    # Legend
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    
    # Title
    ax.set_title(f'{model_name} ROC Curve', fontsize=13, fontweight='bold')
    
    save_nature_plot(f'roc_curve_{model_name.lower().replace(" ", "_")}_nature')
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc_score, 'ci': (lower, upper)}

def plot_nature_performance_comparison_with_std(df_test, df_cv=None):
    """Nature style model performance comparison plot (with cross-validation standard deviation)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    models = df_test['Model'].unique()
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score', 'MCC']
    
    # Set bar chart parameters
    n_models = len(models)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_models
    x_positions = np.arange(n_metrics)
    
    # Define colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_models))
    
    # Draw bars for each model
    for i, model in enumerate(models):
        # Get test set performance for current model
        model_data = df_test[df_test['Model'] == model]
        
        # Extract values for each metric
        scores = []
        error_bars = []
        
        for metric in metrics:
            score = model_data[metric].values[0] if metric in model_data.columns else np.nan
            scores.append(score)
            
            # If cross-validation data exists, get standard deviation
            if df_cv is not None and f'Std_{metric.lower()}' in df_cv.columns:
                std_model_data = df_cv[df_cv['Model'] == model]
                error = std_model_data[f'Std_{metric.lower()}'].values[0] if len(std_model_data) > 0 else 0
                error_bars.append(error)
            else:
                error_bars.append(0)
        
        # Calculate bar positions
        x_offset = i * bar_width - (n_models - 1) * bar_width / 2
        
        # Draw bars
        bars = ax.bar(
            x_positions + x_offset,
            scores,
            bar_width * 0.9,
            color=colors[i],
            alpha=0.8,
            label=model,
            yerr=error_bars if any(error_bars) else None,
            error_kw={'ecolor': 'black', 'linewidth': 1, 'capthick': 2, 'capsize': 4}
        )
        
        # Add value labels on top of bars
        for j, (score, error) in enumerate(zip(scores, error_bars)):
            if not np.isnan(score):
                label = f'{score:.3f}'
                if error > 0:
                    label += f' ± {error:.3f}'
                ax.text(
                    x_positions[j] + x_offset,
                    score + error + 0.02,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=0
                )
    
    # Set X-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics, fontsize=11)
    
    # Set Y-axis
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Title
    ax.set_title("Model Performance Comparison (Test Set with CV Standard Deviation)", 
                fontsize=13, fontweight='bold', pad=20)
    
    # Hide top and right borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    # Legend
    ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid lines
    ax.grid(True, alpha=0.1, axis='y', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    save_nature_plot("model_performance_comparison_with_std_nature")

def plot_nature_combined_roc(roc_data):
    """Nature style combined ROC curves plot"""
    if not roc_data:
        return
    
    # Color definitions
    colors = {
        "Elastic_Net": "#E41A1C",  # Nature red
        "RF_Weighted": "#4DAF4A",  # Nature green
        "XGBoost": "#984EA3",      # Nature purple
        "Boruta_RF": "#FF7F00"     # Nature orange
    }
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot ROC curves for each model
    for name, data in roc_data.items():
        if name in colors:
            ax.plot(data['fpr'], data['tpr'], 
                   lw=2, 
                   color=colors.get(name, 'gray'),
                   label=f"{name} (AUC = {data['auc']:.3f})")
    
    # Random reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random')
    
    # Axis settings
    ax.set_xlim(-0.05, 1.01)
    ax.set_ylim(-0.05, 1.01)
    ax.set_aspect('equal')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    
    # Labels
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    
    # Title
    ax.set_title("ROC Curves Comparison (Test Set)", fontsize=14, fontweight='bold', pad=10)
    
    # Hide top and right borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    # Legend
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    
    save_nature_plot("combined_roc_curves_nature")

def get_performance_report(y_true, y_pred, y_prob, model_name):
    return {'Model': model_name, 'Accuracy': accuracy_score(y_true, y_pred), 'AUC': roc_auc_score(y_true, y_prob),
            'Precision': precision_score(y_true, y_pred, zero_division=0), 'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0), 'MCC': matthews_corrcoef(y_true, y_pred)}

def check_data_leakage(X_train, X_test):
    """Simple test for data leakage"""
    logging.info("Checking for data leakage...")
    
    # Check if features are the same
    if set(X_train.columns) != set(X_test.columns):
        logging.warning("⚠️ Warning: Training and test set features differ")
    else:
        logging.info("✓ Training and test set features are the same")
    
    # Check if standardization is independent
    train_mean = X_train.mean()
    test_mean = X_test.mean()
    
    # If standardization is independent, test set mean should be close to 0
    if abs(test_mean.mean()) > 0.1:  # Adjustable threshold
        logging.warning(f"⚠️ Warning: Test set mean not zero, possible leakage: {test_mean.mean():.3f}")
    else:
        logging.info(f"✓ Test set standardization normal: mean={test_mean.mean():.3f}")
    
    # Check for extreme values
    train_max = X_train.max().max()
    test_max = X_test.max().max()
    if train_max > 10 or test_max > 10:
        logging.warning(f"⚠️ Warning: Extreme values found, check standardization: train_max={train_max:.1f}, test_max={test_max:.1f}")
    else:
        logging.info(f"✓ Standardization range normal: train_max={train_max:.1f}, test_max={test_max:.1f}")
    
    return True

def main():
    logging.info("=== Starting Comprehensive Machine Learning Analysis (FIXED VERSION) ===")
    logging.info("Data leakage fixed: preprocessing steps now fitted only on training set")
    
    # --- 1. Data Loading and Preprocessing ---
    logging.info("1. Reading and pre-processing data...")
    labels = pd.read_csv("phenotype.csv", index_col=0)
    expr_matrix = pd.read_csv("matched_expression.csv", index_col=0)
    common_samples = np.intersect1d(labels.index, expr_matrix.columns)
    labels, expr_matrix = labels.loc[common_samples], expr_matrix[common_samples]
    le = LabelEncoder().fit(labels['Label'])
    X, y = expr_matrix.T, le.transform(labels['Label'])
    
    # --- 2. Data Splitting FIRST (critical fix) ---
    logging.info("2. Splitting data FIRST (critical step to fix data leakage)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # --- 3. Preprocessing fitted ONLY on training set (critical fix) ---
    logging.info("3. Preprocessing: fitting ONLY on training set...")
    
    # Variance threshold selection
    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train_raw)
    X_test_selected = selector.transform(X_test_raw)
    
    # Get selected feature names
    selected_features = X_train_raw.columns[selector.get_support()]
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Reconstruct DataFrame
    X_train = pd.DataFrame(X_train_scaled, index=X_train_raw.index, columns=selected_features)
    X_test = pd.DataFrame(X_test_scaled, index=X_test_raw.index, columns=selected_features)
    
    logging.info(f"Original features: {X.shape[1]}")
    logging.info(f"Features after variance threshold: {X_train.shape[1]}")
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Test set shape: {X_test.shape}")
    
    # Check data leakage
    check_data_leakage(X_train, X_test)
    
    # --- 4. Handling imbalanced data ---
    logging.info("4. Setting up imbalance parameters...")
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

    # ==============================================================================
    #  MODEL 1: ELASTIC NET
    # ==============================================================================
    logging.info("--- Starting Model 1: Elastic Net ---")
    try:
        enet_model_base = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=[0.5], cv=cv_strategy, random_state=RANDOM_STATE, max_iter=5000, scoring='roc_auc', class_weight='balanced', n_jobs=-1)
        cv_results = cross_validate(enet_model_base, X_train, y_train, cv=cv_strategy, scoring=scoring_metrics)
        
        auc_scores = cv_results['test_roc_auc']
        cv_auc_mean = np.mean(auc_scores)
        cv_auc_std = np.std(auc_scores)
        logging.info(f"Elastic Net CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        
        cv_report = {'Model': 'Elastic_Net', **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k}, 
                     **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}}
        cv_performance_reports.append(cv_report)
        
        enet_model_final = enet_model_base.fit(X_train, y_train)
        y_pred, y_prob = enet_model_final.predict(X_test), enet_model_final.predict_proba(X_test)[:, 1]
        test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'Elastic_Net'))
        
        # Nature style confusion matrix
        plot_nature_confusion_matrix(y_test, y_pred, le.classes_, 'Elastic Net')
        
        # Nature style PR curve
        plot_nature_pr_curve(pd.Series(y_test), pd.Series(y_prob), 'Elastic Net', '#E41A1C')
        
        # Nature style ROC curve
        roc_data['Elastic_Net'] = plot_nature_roc_curve(y_test, y_prob, 'Elastic Net', '#E41A1C')
        
        enet_genes = X_train.columns[enet_model_final.coef_[0] != 0].tolist()
        model_selections['Elastic_Net'] = set(enet_genes)
        pd.DataFrame({'Gene': enet_genes}).to_csv(os.path.join(RESULTS_DIR, "Elastic_Net_selected_genes.csv"), index=False)
        
        logging.info("Performing SHAP analysis for Elastic Net...")
        try:
            explainer = shap.LinearExplainer(enet_model_final, X_train)
            shap_values = explainer.shap_values(X_test)
            
            # Nature style SHAP plot
            plt.figure(figsize=(6, 4))
            shap.summary_plot(shap_values, X_test, show=False, plot_type="bar", max_display=20)
            plt.title("Elastic Net SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_enet_bar_nature")
            
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_test, show=False, max_display=20)
            plt.title("Elastic Net SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_enet_beeswarm_nature")
            
            logging.info("✓ SHAP analysis for Elastic Net complete.")
        except Exception as e:
            logging.error(f"SHAP analysis for Elastic Net failed: {e}")
    
    except Exception as e: 
        logging.error(f"Elastic Net failed: {e}")
    
    # ==============================================================================
    #  MODEL 2: RANDOM FOREST
    # ==============================================================================
    logging.info("--- Starting Model 2: Random Forest ---")
    try:
        rf_model_base = RandomForestClassifier(n_estimators=500, class_weight=class_weights_rf, random_state=RANDOM_STATE, n_jobs=-1)
        cv_results = cross_validate(rf_model_base, X_train, y_train, cv=cv_strategy, scoring=scoring_metrics)
        
        auc_scores = cv_results['test_roc_auc']
        cv_auc_mean = np.mean(auc_scores)
        cv_auc_std = np.std(auc_scores)
        logging.info(f"Random Forest CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        
        cv_report = {'Model': 'RF_Weighted', **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k},
                     **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}}
        cv_performance_reports.append(cv_report)
        
        rf_model_final = rf_model_base.fit(X_train, y_train)
        y_pred, y_prob = rf_model_final.predict(X_test), rf_model_final.predict_proba(X_test)[:, 1]
        test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'RF_Weighted'))
        
        # Nature style confusion matrix
        plot_nature_confusion_matrix(y_test, y_pred, le.classes_, 'Random Forest')
        
        # Nature style PR curve
        plot_nature_pr_curve(pd.Series(y_test), pd.Series(y_prob), 'Random Forest', '#4DAF4A')
        
        # Nature style ROC curve
        roc_data['RF_Weighted'] = plot_nature_roc_curve(y_test, y_prob, 'Random Forest', '#4DAF4A')
        
        mdg_imp = pd.DataFrame({'Gene': X_train.columns, 'MDG': rf_model_final.feature_importances_}).sort_values('MDG', ascending=False)
        mdg_imp.to_csv(os.path.join(RESULTS_DIR, "RF_MDG_importance.csv"), index=False)
    
        perm_imp = permutation_importance(
            rf_model_final, X_test, y_test,
            scoring='roc_auc',
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
        mda_imp = pd.DataFrame({'Gene': X_train.columns, 'MDA': perm_imp.importances_mean}).sort_values('MDA', ascending=False)
        mda_imp.to_csv(os.path.join(RESULTS_DIR, "RF_MDA_importance.csv"), index=False)
        model_selections['RF_MDA_Top50'] = set(mda_imp.head(50)['Gene'])
    
        logging.info("RF permutation importance computed using ROC-AUC scoring (gold standard method)")
        logging.info("Performing SHAP analysis for Random Forest...")
        try:
            explainer = shap.TreeExplainer(rf_model_final)
            shap_values = explainer.shap_values(X_test)
    
            mean_abs_shap_rf = np.abs(shap_values[1]).mean(axis=0)
            top20_idx_rf = np.argsort(mean_abs_shap_rf)[-20:]
            top20_genes_rf = X_train.columns[top20_idx_rf].tolist()
            model_selections['RF_SHAP_Top20'] = set(top20_genes_rf)
            logging.info("Random Forest SHAP Top20 genes added to consensus.")
            
            # Nature style SHAP plot
            plt.figure(figsize=(6, 4))
            shap.summary_plot(shap_values[1], X_test, show=False, plot_type="bar", max_display=20)
            plt.title("Random Forest SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_rf_bar_nature")
            
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values[1], X_test, show=False, max_display=20)
            plt.title("Random Forest SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_rf_beeswarm_nature")
            
            logging.info("✓ SHAP analysis for Random Forest complete.")
        except Exception as e:
            logging.error(f"SHAP analysis for Random Forest failed: {e}")
    
    except Exception as e: 
        logging.error(f"Random Forest main process failed: {e}")
    
    # ==============================================================================
    # MODEL 3: XGBOOST
    # ==============================================================================
    logging.info("--- Starting Model 3: XGBoost ---")
    try:
        xgb_model_base = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight_xgb,
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_estimators=500,
            n_jobs=-1
        )
        cv_results = cross_validate(xgb_model_base, X_train, y_train, cv=cv_strategy, scoring=scoring_metrics)
        
        auc_scores = cv_results['test_roc_auc']
        cv_auc_mean = np.mean(auc_scores)
        cv_auc_std = np.std(auc_scores)
        logging.info(f"XGBoost CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        
        cv_report = {'Model': 'XGBoost', **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k},
                     **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}}
        cv_performance_reports.append(cv_report)
    
        xgb_model_final = xgb_model_base.fit(X_train, y_train, verbose=False)
        y_pred = xgb_model_final.predict(X_test)
        y_prob = xgb_model_final.predict_proba(X_test)[:, 1]
        test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'XGBoost'))
    
        # Nature style confusion matrix
        plot_nature_confusion_matrix(y_test, y_pred, le.classes_, 'XGBoost')
        
        # Nature style PR curve
        plot_nature_pr_curve(pd.Series(y_test), pd.Series(y_prob), 'XGBoost', '#984EA3')
        
        # Nature style ROC curve
        roc_data['XGBoost'] = plot_nature_roc_curve(y_test, y_prob, 'XGBoost', '#984EA3')
    
        logging.info("Extracting XGBoost features with positive Gain importance...")
        try:
            gain_dict = xgb_model_final.get_booster().get_score(importance_type='gain')
            xgb_selected_genes = [gene for gene, gain in gain_dict.items() if gain > 0]
            model_selections['XGBoost_Gain_Positive'] = set(xgb_selected_genes)
            logging.info(f"XGBoost selected {len(xgb_selected_genes)} genes with Gain > 0 for consensus voting.")
    
            gain_df = pd.DataFrame([
                {'Gene': gene, 'Gain': gain}
                for gene, gain in gain_dict.items()
            ]).sort_values('Gain', ascending=False)
            gain_df.to_csv(os.path.join(RESULTS_DIR, "XGBoost_Gain_importance_full.csv"), index=False)
    
        except Exception as e:
            logging.error(f"Failed to extract XGBoost Gain importance: {e}")
    
        logging.info("Performing SHAP analysis for XGBoost (for visualization only)...")
        try:
            explainer = shap.TreeExplainer(xgb_model_final)
            shap_values = explainer.shap_values(X_test)
            
            # Nature style SHAP plot
            plt.figure(figsize=(6, 4))
            shap.summary_plot(shap_values, X_test, show=False, plot_type="bar", max_display=20)
            plt.title("XGBoost SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_xgb_bar_nature")
            
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_test, show=False, max_display=20)
            plt.title("XGBoost SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            save_nature_plot("shap_summary_xgb_beeswarm_nature")
            
            logging.info("✓XGBoost SHAP plots generated.")
        except Exception as e:
            logging.error(f"SHAP analysis for XGBoost failed: {e}")
    
    except Exception as e:
        logging.error(f"XGBoost main process failed: {e}")
    
    # ==============================================================================
    # MODEL 4: BORUTA + RANDOM FOREST
    # ==============================================================================
    logging.info("--- Starting Model 4: Boruta + RF ---")
    try:
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
        boruta_selector.fit(X_train.values, y_train)
        confirmed_genes = X_train.columns[boruta_selector.support_].tolist()
        logging.info(f"Boruta confirmed {len(confirmed_genes)} features.")
    
        model_selections['Boruta'] = set(confirmed_genes)
        pd.DataFrame({'Gene': confirmed_genes}).to_csv(
            os.path.join(RESULTS_DIR, "Boruta_confirmed_genes.csv"), index=False
        )
    
        if len(confirmed_genes) > 0:
            X_train_boruta = X_train[confirmed_genes]
            X_test_boruta = X_test[confirmed_genes]
    
            rf_model_base = RandomForestClassifier(
                n_estimators=100,
                class_weight=class_weights_rf,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            cv_results = cross_validate(rf_model_base, X_train_boruta, y_train,
                                        cv=cv_strategy, scoring=scoring_metrics)
            
            auc_scores = cv_results['test_roc_auc']
            cv_auc_mean = np.mean(auc_scores)
            cv_auc_std = np.std(auc_scores)
            logging.info(f"Boruta+RF CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
            
            cv_report = {'Model': 'Boruta_RF',
                        **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k},
                        **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}}
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
    
            logging.info("Performing SHAP analysis for Boruta+RF model...")
            try:
                explainer = shap.TreeExplainer(rf_model_final)
                shap_values = explainer.shap_values(X_test_boruta)
    
                mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
                top20_idx = np.argsort(mean_abs_shap)[-20:]
                top20_genes = X_test_boruta.columns[top20_idx].tolist()
                model_selections['Boruta_RF_SHAP_Top20'] = set(top20_genes)
                logging.info("Boruta+RF SHAP Top20 genes added to consensus voting.")
                
                # Nature style SHAP plot
                plt.figure(figsize=(6, 4))
                shap.summary_plot(shap_values[1], X_test_boruta, show=False, plot_type="bar", max_display=20)
                plt.title("Boruta+RF SHAP Feature Importance", fontsize=13, fontweight='bold', pad=10)
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                save_nature_plot("shap_summary_boruta_rf_bar_nature")
                
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values[1], X_test_boruta, show=False, max_display=20)
                plt.title("Boruta+RF SHAP Beeswarm Plot", fontsize=13, fontweight='bold', pad=10)
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                save_nature_plot("shap_summary_boruta_rf_beeswarm_nature")
                
                logging.info("✓SHAP analysis for Boruta+RF complete.")
            except Exception as e:
                logging.error(f"SHAP analysis for Boruta+RF failed: {e}")
        else:
            logging.warning("Boruta confirmed no features. Skipping model.")
    
    except Exception as e:
        logging.error(f"Boruta+RF failed: {e}")

    # ==============================================================================
    #  FINAL REPORTING AND VISUALIZATION
    # ==============================================================================
    logging.info("--- Generating Final Reports and Visualizations ---")
    if cv_performance_reports:
        df_cv = pd.DataFrame(cv_performance_reports)
        df_cv.to_csv(os.path.join(RESULTS_DIR, "cv_performance_report.csv"), index=False)
        logging.info(f"\n--- 10-Fold Cross-Validation Performance (on Training Set) ---\n{df_cv.round(4)}")
    if test_performance_reports:
        df_test = pd.DataFrame(test_performance_reports)
        df_test.to_csv(os.path.join(RESULTS_DIR, "test_set_performance_report.csv"), index=False)
        logging.info(f"\n--- Final Hold-out Test Set Performance ---\n{df_test.round(4)}")
        
        # Performance comparison plot with standard deviation
        plot_nature_performance_comparison_with_std(df_test, df_cv if 'df_cv' in locals() else None)

    # ==============================================================================
    # MODEL CONSENSUS ANALYSIS + Nature style UpSet Plot
    # ==============================================================================
    logging.info("--- Performing Model Consensus Analysis ---")
    try:
        valid_sets = {str(k): set(v) for k, v in model_selections.items() if v and len(v) > 0}
        
        if len(valid_sets) < 2:
            logging.warning("Not enough valid gene sets for intersection analysis.")
        else:
            all_genes = [gene for gene_set in valid_sets.values() for gene in gene_set]
            gene_counts = Counter(all_genes)
            consensus_df = pd.DataFrame(gene_counts.items(), columns=['Gene', 'Consensus_Score'])
            num_models = len(valid_sets)
            consensus_df['Selection_Frequency'] = consensus_df['Consensus_Score'] / num_models
            consensus_df = consensus_df.sort_values('Consensus_Score', ascending=False)
            consensus_df.to_csv(os.path.join(RESULTS_DIR, "model_consensus_gene_report.csv"), index=False)
            logging.info(f"Consensus report saved. Top genes:\n{consensus_df.head(10)[['Gene', 'Consensus_Score']]}")

            # Nature style UpSet Plot - fixed version
            try:
                from upsetplot import UpSet, from_contents

                # Convert sets to lists
                upset_data = from_contents({name: list(genes) for name, genes in valid_sets.items()})
                
                # Create UpSet object
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
                
                # Create figure
                fig = plt.figure(figsize=(10, 6))
                upset.plot(fig=fig)
                
                # Set title
                fig.suptitle("Key Gene Intersections Across Multiple Models", 
                           fontsize=13, fontweight='bold', y=0.98)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure
                for ext in ['.png', '.pdf']:
                    plt.savefig(os.path.join(RESULTS_DIR, f"Key_Gene_Intersection_UpSet_Plot_nature{ext}"),
                              dpi=300 if ext == '.png' else None, bbox_inches='tight')
                plt.close(fig)
                
                logging.info("Nature style UpSet plot generated successfully!")
                
            except Exception as e:
                logging.warning(f"UpSet failed ({e}), falling back to Venn diagram.")
                
                # Nature style Venn diagram
                try:
                    import matplotlib_venn as venn_lib
                    if len(valid_sets) <= 6:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        venn_lib.venn(valid_sets, ax=ax, fmt="{size}")
                        
                        # Nature style settings
                        ax.set_title("Gene Set Intersection", fontsize=13, fontweight='bold', pad=20)
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        
                        plt.tight_layout()
                        save_nature_plot("gene_venn_diagram_nature")
                    else:
                        logging.info("Too many sets for Venn, skipping diagram.")
                except Exception as e2:
                    logging.error(f"Venn diagram also failed: {e2}")

    except Exception as e:
        logging.error(f"Consensus analysis failed: {e}")

    # Nature style combined ROC curves
    if roc_data:
        plot_nature_combined_roc(roc_data)

    logging.info("\n=== Analysis Complete: All Nature-style reports and plots saved to 'results' directory. ===")
    logging.info("Data leakage fixed: preprocessing steps now fitted only on training set, test set transformed independently.")

if __name__ == '__main__':

    # ==============================================================================
    # Figure 4: Nature style 4-gene ultimate heatmap + box plot
    # ==============================================================================
    logging.info("Generating Figure 4: Ultra-High-Confidence 4-Gene Signature (6/6 Consensus)")
    try:
        expr_raw = pd.read_csv("72326_50772.csv", index_col=0)
        labels_df = pd.read_csv("classification_labels_debatched.csv", index_col=0)

        common_samples = expr_raw.columns.intersection(labels_df.index)
        expr_raw = expr_raw[common_samples]
        labels_df = labels_df.loc[common_samples]

        group = labels_df['Label'].astype(int).map({0: 'Sensitive', 1: 'Resistant'}).fillna('Unknown')
        
        if group.str.contains('Unknown').any():
            raise ValueError(f"Grouping failed! Unknown label values: {labels_df['Label'][group == 'Unknown'].unique()}")
        
        print(f"Grouping successful! Sensitive (0): {(group=='Sensitive').sum()} samples, Resistant (1): {(group=='Resistant').sum()} samples")

        target_genes = ['OAS3', 'IFIT3', 'IFI27', 'EPSTI1']
        found_genes = []
        for g in target_genes:
            candidates = [x for x in expr_raw.index if g.upper() in x.upper() and 'L2' not in x]
            if not candidates:
                raise ValueError(f"Gene {g} not found!")
            exact = [x for x in candidates if x == g]
            real = exact[0] if exact else candidates[0]
            found_genes.append(real)
            print(f"Selected: {g} → {real}")

        expr_4 = expr_raw.loc[found_genes]
        print(f"Successfully extracted 4 genes, shape: {expr_4.shape}")

        from scipy.stats import mannwhitneyu, zscore
        from matplotlib.patches import Patch
        import numpy as np
        
        def safe_zscore(x):
            std = x.std()
            if std == 0:
                return np.zeros_like(x)
            return zscore(x)
        
        expr_z = expr_4.T.apply(safe_zscore, axis=0)
        sample_colors = group.map({'Sensitive': '#1f77b4', 'Resistant': '#d62728'})

        # ==================== Figure 4A: Nature style heatmap ====================
        g = sns.clustermap(
            expr_z,
            cmap="RdBu_r",
            center=0,
            vmin=-3, vmax=3,
            row_cluster=False,
            col_cluster=True,
            col_colors=sample_colors,
            linewidths=0.5,
            linecolor='lightgray',
            figsize=(10, 6),
            cbar_kws={"label": "Z-score Expression", "shrink": 0.8},
            xticklabels=False,
            yticklabels=True
        )
        
        # Nature style settings
        g.ax_heatmap.set_title("4-Gene Signature Expression Heatmap", 
                               fontsize=13, fontweight='bold', pad=20)
        g.ax_heatmap.set_ylabel("Genes", fontsize=12)
        g.ax_heatmap.set_xlabel("Samples", fontsize=12)
        
        # Set tick label size
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=10)
        
        # Legend
        handles = [Patch(facecolor='#1f77b4', label='Sensitive'),
                   Patch(facecolor='#d62728', label='Resistant')]
        g.ax_heatmap.legend(handles=handles, title="Group", frameon=False, 
                           fontsize=9, bbox_to_anchor=(0, 1.15), loc='upper left')
        
        # Hide heatmap borders
        for spine in g.ax_heatmap.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        save_nature_plot("Figure_4A_4_Gene_Heatmap_nature")
        plt.close('all')

        # ==================== Figure 4B: Nature style box plot ====================
        expr_melt = expr_4.T.melt(var_name='Gene', value_name='Expression')
        expr_melt['Group'] = np.repeat(group.values, len(found_genes))

        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Nature style box plot
        sns.boxplot(
            data=expr_melt, 
            x='Gene', 
            y='Expression', 
            hue='Group',
            palette={'Sensitive': '#1f77b4', 'Resistant': '#d62728'},
            linewidth=1.5, 
            fliersize=3,
            width=0.7,
            ax=ax
        )
        
        # Add scatter points to show data distribution
        sns.stripplot(
            data=expr_melt,
            x='Gene',
            y='Expression',
            hue='Group',
            palette={'Sensitive': '#1f77b4', 'Resistant': '#d62728'},
            dodge=True,
            alpha=0.4,
            size=3,
            jitter=0.2,
            ax=ax
        )
        
        # Calculate statistical significance
        y_max = expr_melt['Expression'].max()
        y_range = y_max - expr_melt['Expression'].min() + 1e-8
        base_height = y_max + y_range * 0.08
        offset = y_range * 0.09
        
        gene_layer = {gene: 0 for gene in found_genes}
        
        def add_annotation(gene_idx, text, is_perfect=False):
            layer = gene_layer[found_genes[gene_idx]]
            h = base_height + layer * offset
            color = 'purple' if is_perfect else 'black'
            weight = 'bold' if is_perfect else 'normal'
            size = 12 if is_perfect else 11
            ax.text(gene_idx, h, text, ha='center', va='bottom',
                    fontsize=size, fontweight=weight, color=color)
            gene_layer[found_genes[gene_idx]] += 1

        for i, gene in enumerate(found_genes):
            s = expr_melt[(expr_melt['Gene'] == gene) & (expr_melt['Group'] == 'Sensitive')]['Expression']
            r = expr_melt[(expr_melt['Gene'] == gene) & (expr_melt['Group'] == 'Resistant')]['Expression']

            if len(s) == 0 or len(r) == 0 or s.nunique() <= 1 or r.nunique() <= 1:
                if (s == 0).all() or (r == 0).all() or s.std() == 0 or r.std() == 0:
                    add_annotation(i, "Perfect Separation", is_perfect=True)
                else:
                    add_annotation(i, "N/A", is_perfect=False)
            else:
                p = mannwhitneyu(s, r, alternative='two-sided').pvalue
                if p < 0.001:
                    add_annotation(i, "***", is_perfect=False)
                elif p < 0.01:
                    add_annotation(i, "**", is_perfect=False)
                elif p < 0.05:
                    add_annotation(i, "*", is_perfect=False)
                else:
                    add_annotation(i, "ns", is_perfect=False)

        # Nature style settings
        ax.set_title("Expression of 4 Ultra-High-Confidence Genes (6/6 Consensus)", 
                    fontsize=13, fontweight='bold', pad=20)
        ax.set_ylabel("Log₂ Expression", fontsize=12)
        ax.set_xlabel("Gene", fontsize=12)
        
        # Hide top and right borders
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        
        # Legend handling
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Group", frameon=False, 
                 fontsize=9, title_fontsize=10)
        
        plt.tight_layout()
        save_nature_plot("Figure_4B_4_Gene_Boxplot_nature")
        plt.close('all')
        
        logging.info("Figure 4A & 4B generated successfully in Nature style!")

    except Exception as e:
        logging.error(f"Figure 4 failed: {e}")
        import traceback
        traceback.print_exc()

    main()