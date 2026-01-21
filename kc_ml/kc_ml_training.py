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
RESULTS_DIR = "results_KC-ML" # Use a new directory for weighted results
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# (Helper functions are unchanged)
def save_plot(filename, tight_layout=True):
    if tight_layout: plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close('all')

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f'{model_name} Confusion Matrix (Test Set)')
    save_plot(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')

def plot_pr_curve(y_true, y_prob, model_name, color):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 8)); ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.plot(recall, precision, color=color, lw=2.5, label=f'AP = {ap_score:.3f}')
    plt.xlabel('Recall (Sensitivity)'); plt.ylabel('Precision'); plt.title(f'{model_name} Precision-Recall Curve (Test Set)')
    plt.grid(linestyle=':'); plt.legend(); plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
    save_plot(f'pr_curve_{model_name.lower().replace(" ", "_")}.png')

def plot_roc_curve(y_true, y_prob, model_name, color):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    bootstrapped_scores = [roc_auc_score(*resample(y_true, y_prob, random_state=i)) for i in range(100) if len(np.unique(resample(y_true, y_prob, random_state=i)[0])) > 1]
    lower = np.percentile(bootstrapped_scores, 2.5) if bootstrapped_scores else auc_score
    upper = np.percentile(bootstrapped_scores, 97.5) if bootstrapped_scores else auc_score
    plt.figure(figsize=(8, 8)); ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    plt.plot(fpr, tpr, color=color, lw=2.5)
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.title(f'{model_name} ROC Curve (Test Set)')
    plt.grid(color='lightgray', linestyle=':'); legend_text = f"AUC (95% CI)\n{auc_score:.2f} ({lower:.2f}-{upper:.2f})"
    plt.legend([plt.Line2D([0], [0], color=color, lw=2.5)], [legend_text], loc='lower right', frameon=False)
    save_plot(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}

def get_performance_report(y_true, y_pred, y_prob, model_name):
    return {'Model': model_name, 'Accuracy': accuracy_score(y_true, y_pred), 'AUC': roc_auc_score(y_true, y_prob),
            'Precision': precision_score(y_true, y_pred, zero_division=0), 'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0), 'MCC': matthews_corrcoef(y_true, y_pred)}

def main():
    logging.info("=== Starting Knowledge-Guided Machine Learning Analysis ===")
    
    # --- 1. Data Loading and Preprocessing ---
    logging.info("1. Reading and pre-processing data...")
    labels = pd.read_csv("phenotype.csv", index_col=0)
    expr_matrix = pd.read_csv("genes_for_KC_ML_matched_expression.csv", index_col=0)
    common_samples = np.intersect1d(labels.index, expr_matrix.columns)
    labels, expr_matrix = labels.loc[common_samples], expr_matrix[common_samples]
    le = LabelEncoder().fit(labels['Label'])
    X, y = expr_matrix.T, le.transform(labels['Label'])
    selector = VarianceThreshold()
    X = pd.DataFrame(selector.fit_transform(X), index=X.index, columns=X.columns[selector.get_support()])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    
    # ==============================================================================
    # --- CORE MODIFICATION: APPLY GENE WEIGHTS ---
    # ==============================================================================
    logging.info("Applying knowledge-guided gene weights...")
    try:
        df_weights = pd.read_csv('final_external_data_weights.csv', encoding='utf-8-sig')
        # Create a dictionary for fast lookup: Gene -> Weight
        weight_map = df_weights.set_index('Gene')['Final_Weight'].to_dict()
        
        # Create a weight vector aligned with the columns of X_scaled
        gene_weights = X_scaled.columns.map(lambda gene: weight_map.get(gene, 1.0))
        
        # Apply the weights to the data
        X_weighted = X_scaled * gene_weights
        
        num_weighted = sum(gene_weights > 1.0)
        logging.info(f"✓ Successfully applied custom weights to {num_weighted} genes. Using weighted data for modeling.")
        
    except FileNotFoundError:
        logging.warning("✗ 'knowledge_weights.csv' not found. Proceeding with unweighted data.")
        X_weighted = X_scaled # Fallback to unweighted data
    except Exception as e:
        logging.error(f"✗ Failed to apply weights: {e}. Proceeding with unweighted data.")
        X_weighted = X_scaled # Fallback to unweighted data

    # --- 2. Data Splitting & Imbalance Setup ---
    logging.info("2. Splitting data and setting up imbalance parameters...")
    X_train, X_test, y_train, y_test = train_test_split(
    X_weighted, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # ==================== 类别不平衡参数自动计算（推荐写法）===================
    # 统计训练集中的正负类数量
    n_neg = (y_train == 0).sum()      # 负类（0）样本数
    n_pos = (y_train == 1).sum()      # 正类（1）样本数
    total = n_neg + n_pos

    logging.info(f"Training set class distribution -> Negative(0): {n_neg} ({n_neg/total:.1%}), "
             f"Positive(1): {n_pos} ({n_pos/total:.1%})")

    # 方法1：Random Forest 用 sklearn 官方推荐的 'balanced'（自动按比例加权）
    # 效果等价于 {0: n_pos/total, 1: n_neg/total}，少数类权重更高
    class_weights_rf = 'balanced'          # 直接写死这一行，永远不会反！

    # 方法2：XGBoost 用 scale_pos_weight（官方推荐）
    # 含义：正类（1）的权重视为1，负类（0）的权重 = n_neg / n_pos
    # 当负类少时 >1；当正类少时 <1，自动适应
    scale_pos_weight_xgb = n_neg / max(n_pos, 1)   # 防止除0

    logging.info(f"Imbalance handling -> RF class_weight = 'balanced', "
             f"XGBoost scale_pos_weight = {scale_pos_weight_xgb:.3f}")

    # 如果你仍想手动控制一个"强度因子"（比如原来想×5），可以这样写（可选）：
    # WEIGHT_FACTOR = 5.0
    # scale_pos_weight_xgb = WEIGHT_FACTOR * (n_neg / max(n_pos, 1))

    # ======================================================================

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

        # 添加CV AUC ± SD输出
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
        plot_confusion_matrix(y_test, y_pred, le.classes_, 'Elastic Net'); plot_pr_curve(y_test, y_prob, 'Elastic Net', '#E41A1C')
        roc_data['Elastic_Net'] = plot_roc_curve(y_test, y_prob, 'Elastic Net', '#E41A1C')
        
        enet_genes = X_train.columns[enet_model_final.coef_[0] != 0].tolist()
        model_selections['Elastic_Net'] = set(enet_genes)
        pd.DataFrame({'Gene': enet_genes}).to_csv(os.path.join(RESULTS_DIR, "Elastic_Net_selected_genes.csv"), index=False)
        
        logging.info("Performing SHAP analysis for Elastic Net...")
        try:
            explainer = shap.LinearExplainer(enet_model_final, X_train)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, show=False, plot_type="bar", max_display=20)
            save_plot("shap_summary_enet_bar.png")
            shap.summary_plot(shap_values, X_test, show=False, max_display=20)
            save_plot("shap_summary_enet_beeswarm.png")
            logging.info("✓ SHAP analysis for Elastic Net complete.")
        except Exception as e:
            logging.error(f"SHAP analysis for Elastic Net failed: {e}")

    except Exception as e: logging.error(f"Elastic Net failed: {e}")

    # ==============================================================================
    #  MODEL 2: RANDOM FOREST
    # ==============================================================================
    logging.info("--- Starting Model 2: Random Forest ---")
    try:
        rf_model_base = RandomForestClassifier(n_estimators=500, class_weight=class_weights_rf, random_state=RANDOM_STATE, n_jobs=-1)
        cv_results = cross_validate(rf_model_base, X_train, y_train, cv=cv_strategy, scoring=scoring_metrics)

        # 添加CV AUC ± SD输出
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
        plot_confusion_matrix(y_test, y_pred, le.classes_, 'Random Forest'); plot_pr_curve(y_test, y_prob, 'Random Forest', '#4DAF4A')
        roc_data['RF_Weighted'] = plot_roc_curve(y_test, y_prob, 'Random Forest', '#4DAF4A')

        mdg_imp = pd.DataFrame({'Gene': X_train.columns, 'MDG': rf_model_final.feature_importances_}).sort_values('MDG', ascending=False)
        mdg_imp.to_csv(os.path.join(RESULTS_DIR, "RF_MDG_importance.csv"), index=False)
        #perm_imp = permutation_importance(rf_model_final, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)

        # === 关键诊断代码（加在这段前面）===
       # print("RF 在测试集上的预测分布:", np.bincount(rf_model_final.predict(X_test)))
       # print("测试集真实标签分布:", np.bincount(y_test))
       # print("测试集 Accuracy:", accuracy_score(y_test, rf_model_final.predict(X_test)))
       # print("测试集 AUC:", roc_auc_score(y_test, rf_model_final.predict_proba(X_test)[:,1]))

        # 重点：改用 AUC 作为评分指标重新计算 permutation importance
        perm_imp = permutation_importance(
            rf_model_final, X_test, y_test,
            scoring='roc_auc',               # ← 改成这一行！（原来默认是 accuracy）
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        # ================================

        mda_imp = pd.DataFrame({'Gene': X_train.columns, 'MDA': perm_imp.importances_mean}).sort_values('MDA', ascending=False)
        mda_imp.to_csv(os.path.join(RESULTS_DIR, "RF_MDA_importance.csv"), index=False)
        model_selections['RF_MDA_Top50'] = set(mda_imp.head(50)['Gene'])
        
        logging.info("RF permutation importance computed using ROC-AUC scoring (gold standard method)")
        logging.info("Performing SHAP analysis for Random Forest...")
        try:
            explainer = shap.TreeExplainer(rf_model_final)
            shap_values = explainer.shap_values(X_test)
            # Get top 20 genes based on mean absolute SHAP value
            mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
            top_genes_indices = np.argsort(mean_abs_shap)[-20:]
            top_genes_names = X_train.columns[top_genes_indices].tolist()
            model_selections['RF_SHAP_Top20'] = set(top_genes_names)
            
            shap.summary_plot(shap_values[1], X_test, show=False, plot_type="bar", max_display=20)
            save_plot("shap_summary_rf_bar.png")
            shap.summary_plot(shap_values[1], X_test, show=False, max_display=20)
            save_plot("shap_summary_rf_beeswarm.png")
            logging.info("✓ SHAP analysis for Random Forest complete.")
        except Exception as e:
            logging.error(f"SHAP analysis for Random Forest failed: {e}")

    except Exception as e: logging.error(f"Random Forest main process failed: {e}")

    # ==============================================================================
    # MODEL 3: XGBOOST（改为 Gain > 0 的基因参与共识）
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

        # 添加CV AUC ± SD输出
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

        plot_confusion_matrix(y_test, y_pred, le.classes_, 'XGBoost')
        plot_pr_curve(y_test, y_prob, 'XGBoost', '#984EA3')
        roc_data['XGBoost'] = plot_roc_curve(y_test, y_prob, 'XGBoost', '#984EA3')

        # ==================== 关键修改：用 Gain > 0 的基因作为 XGBoost 的共识投票集合 ====================
        logging.info("Extracting XGBoost features with positive Gain importance...")
        try:
            # 获取每个特征的 Gain 值（XGBoost 官方最推荐的重要性指标）
            gain_dict = xgb_model_final.get_booster().get_score(importance_type='gain')
            # 将所有 Gain > 0 的基因加入共识
            xgb_selected_genes = [gene for gene, gain in gain_dict.items() if gain > 0]
            model_selections['XGBoost_Gain_Positive'] = set(xgb_selected_genes)
            logging.info(f"XGBoost selected {len(xgb_selected_genes)} genes with Gain > 0 for consensus voting.")

            # 保存完整 Gain 重要性表格（可选）
            gain_df = pd.DataFrame([
                {'Gene': gene, 'Gain': gain}
                for gene, gain in gain_dict.items()
            ]).sort_values('Gain', ascending=False)
            gain_df.to_csv(os.path.join(RESULTS_DIR, "XGBoost_Gain_importance_full.csv"), index=False)

        except Exception as e:
            logging.error(f"Failed to extract XGBoost Gain importance: {e}")

        # ==================== SHAP 分析（仅用于可视化，不参与共识） ====================
        logging.info("Performing SHAP analysis for XGBoost (for visualization only)...")
        try:
            explainer = shap.TreeExplainer(xgb_model_final)
            shap_values = explainer.shap_values(X_test)

            shap.summary_plot(shap_values, X_test, show=False, plot_type="bar", max_display=20)
            save_plot("shap_summary_xgb_bar.png")
            shap.summary_plot(shap_values, X_test, show=False, max_display=20)
            save_plot("shap_summary_xgb_beeswarm.png")
            logging.info("XGBoost SHAP plots generated.")
        except Exception as e:
            logging.error(f"SHAP analysis for XGBoost failed: {e}")

    except Exception as e:
        logging.error(f"XGBoost main process failed: {e}")

    # ==============================================================================
    #  MODEL 4: BORUTA + RANDOM FOREST
    # ==============================================================================
    logging.info("--- Starting Model 4: Boruta + RF ---")
    try:
        rf_boruta_base = RandomForestClassifier(n_jobs=-1, class_weight=class_weights_rf, max_depth=5, random_state=RANDOM_STATE)
        boruta_selector = BorutaPy(rf_boruta_base, n_estimators='auto', verbose=0, random_state=RANDOM_STATE, max_iter=100)
        # Boruta runs on unweighted data to find naturally important features
        boruta_selector.fit(X_train.values, y_train)
        confirmed_genes = X_train.columns[boruta_selector.support_].tolist()
        logging.info(f"Boruta confirmed {len(confirmed_genes)} features.")
        model_selections['Boruta'] = set(confirmed_genes)
        pd.DataFrame({'Gene': confirmed_genes}).to_csv(os.path.join(RESULTS_DIR, "Boruta_confirmed_genes.csv"), index=False)
        
        if len(confirmed_genes) > 0:
            X_train_boruta, X_test_boruta = X_train[confirmed_genes], X_test[confirmed_genes]
            rf_model_base = RandomForestClassifier(n_estimators=100, class_weight=class_weights_rf, random_state=RANDOM_STATE, n_jobs=-1)
            cv_results = cross_validate(rf_model_base, X_train_boruta, y_train, cv=cv_strategy, scoring=scoring_metrics)

            # 添加CV AUC ± SD输出
            auc_scores = cv_results['test_roc_auc']
            cv_auc_mean = np.mean(auc_scores)
            cv_auc_std = np.std(auc_scores)
            logging.info(f"Boruta+RF CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
            
            cv_report = {'Model': 'Boruta_RF',
                        **{f"Mean_{k.split('_')[-1]}": np.mean(v) for k, v in cv_results.items() if 'test_' in k},
                        **{f"Std_{k.split('_')[-1]}": np.std(v) for k, v in cv_results.items() if 'test_' in k}}
            cv_performance_reports.append(cv_report)

            rf_model_final = rf_model_base.fit(X_train_boruta, y_train)
            y_pred, y_prob = rf_model_final.predict(X_test_boruta), rf_model_final.predict_proba(X_test_boruta)[:, 1]

            test_performance_reports.append(get_performance_report(y_test, y_pred, y_prob, 'Boruta_RF'))
            plot_confusion_matrix(y_test, y_pred, le.classes_, 'Boruta+RF'); 
            plot_pr_curve(y_test, y_prob, 'Boruta+RF', '#FF7F00')
            roc_data['Boruta_RF'] = plot_roc_curve(y_test, y_prob, 'Boruta+RF', '#FF7F00')

            logging.info("Performing SHAP analysis for Boruta+RF model...")
            try:
                explainer = shap.TreeExplainer(rf_model_final)
                shap_values = explainer.shap_values(X_test_boruta)
                shap.summary_plot(shap_values[1], X_test_boruta, show=False, plot_type="bar", max_display=20)
                save_plot("shap_summary_boruta_rf_bar.png")
                shap.summary_plot(shap_values[1], X_test_boruta, show=False, max_display=20)
                save_plot("shap_summary_boruta_rf_beeswarm.png")
                mean_abs_shap_boruta = np.abs(shap_values[1]).mean(axis=0)
                top20_idx = np.argsort(mean_abs_shap_boruta)[-20:]
                model_selections['Boruta_SHAP_Top20'] = set(X_test_boruta.columns[top20_idx])
                logging.info("✓ SHAP analysis for Boruta+RF complete.")
            except Exception as e:
                logging.error(f"SHAP analysis for Boruta+RF failed: {e}")
        else:
            logging.warning("Boruta confirmed no features. Skipping model.")
    except Exception as e: logging.error(f"Boruta+RF failed: {e}")

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
        df_melted = df_test.melt(id_vars='Model', var_name='Metric', value_name='Score')
        plt.figure(figsize=(15, 8)); sns.barplot(x='Metric', y='Score', hue='Model', data=df_melted, palette='viridis')
        plt.title("Comprehensive Model Performance Comparison (Test Set)", fontsize=16)
        save_plot("final_performance_comparison_detailed.png")

    # ==============================================================================
    # MODEL CONSENSUS ANALYSIS + UpSet Plot（终极无敌版）
    # ==============================================================================
    logging.info("--- Performing Model Consensus Analysis ---")
    try:
        # 强制 key 为字符串 + 过滤空集合
        valid_sets = {str(k): set(v) for k, v in model_selections.items() if v and len(v) > 0}
        
        if len(valid_sets) < 2:
            logging.warning("Not enough valid gene sets for intersection analysis.")
        else:
            # 生成共识报告（不变）
            all_genes = [gene for gene_set in valid_sets.values() for gene in gene_set]
            gene_counts = Counter(all_genes)
            consensus_df = pd.DataFrame(gene_counts.items(), columns=['Gene', 'Consensus_Score'])
            num_models = len(valid_sets)
            consensus_df['Selection_Frequency'] = consensus_df['Consensus_Score'] / num_models
            consensus_df = consensus_df.sort_values('Consensus_Score', ascending=False)
            consensus_df.to_csv(os.path.join(RESULTS_DIR, "model_consensus_gene_report.csv"), index=False)
            logging.info(f"Consensus report saved. Top genes:\n{consensus_df.head(10)[['Gene', 'Consensus_Score']]}")

            # ==================== UpSet Plot（完美版）===================
            try:
                from upsetplot import UpSet, from_contents

                upset_data = from_contents({name: list(genes) for name, genes in valid_sets.items()})
                
                fig = plt.figure(figsize=(14, 9))
                upset = UpSet(
                    upset_data,
                    subset_size='count',
                    show_counts=True,
                    sort_by='cardinality',
                    sort_categories_by='cardinality',
                    facecolor='steelblue',
                    shading_color='lightgray'
                )
                upset.plot(fig=fig)
                plt.suptitle("Key Gene Intersections Across Multiple Models (UpSet Plot)", fontsize=18, y=0.98)
                
                # 修改部分：同时保存PNG和SVG格式
                plt.tight_layout()
                
                # 保存PNG格式
                plt.savefig(os.path.join(RESULTS_DIR, "Key_Gene_Intersection_UpSet_Plot.png"), dpi=300, bbox_inches='tight')
                
                # 保存SVG格式
                plt.savefig(os.path.join(RESULTS_DIR, "Key_Gene_Intersection_UpSet_Plot.svg"), format='svg', 
                           bbox_inches='tight', dpi=300)
                
                logging.info("UpSet plot generated successfully (publication-ready)!")
                logging.info("Saved UpSet plot: Key_Gene_Intersection_UpSet_Plot.png and Key_Gene_Intersection_UpSet_Plot.svg")
                
                plt.close('all')

            except Exception as e:
                logging.warning(f"UpSet failed ({e}), falling back to Venn diagram.")
                # 降级到传统 Venn（最多支持 6 集合，但会很丑）
                try:
                    import matplotlib_venn as venn_lib
                    if len(valid_sets) <= 6:
                        fig, ax = plt.subplots(figsize=(12, 12))
                        venn_lib.venn(valid_sets, ax=ax, fmt="{size}")
                        ax.set_title("Gene Intersection (Venn Fallback)", fontsize=16)
                        save_plot("gene_venn_diagram_fallback.png")
                    else:
                        logging.info("Too many sets for Venn, skipping diagram.")
                except:
                    pass

    except Exception as e:
        logging.error(f"Consensus analysis failed: {e}")

    # --- Continue with existing final visualizations ---
    if len(valid_sets) >= 2:
        fig, ax = plt.subplots(figsize=(10, 10)); venn.venn(valid_sets, ax=ax, fmt="{size}")
        ax.set_title("Intersection of Key Genes", fontsize=16); save_plot("gene_venn_diagram.png", tight_layout=False)
    if roc_data:
        plt.figure(figsize=(10, 10)); ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
        colors = {"Elastic_Net": "#E41A1C", "RF_Weighted": "#4DAF4A", "XGBoost": "#984EA3", "Boruta_RF": "#FF7F00"}
        for name, data in roc_data.items(): plt.plot(data['fpr'], data['tpr'], color=colors.get(name, 'black'), lw=2, label=f"{name} (AUC = {data['auc']:.3f})")
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--'); plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
        plt.title("Combined ROC Curves (Test Set Performance)"); plt.legend(loc="lower right"); plt.grid(linestyle=':')
        save_plot("roc_combined.png")

    logging.info("\n=== Analysis Complete: All reports and plots saved to 'results' directory. ===")

if __name__ == '__main__':
    main()