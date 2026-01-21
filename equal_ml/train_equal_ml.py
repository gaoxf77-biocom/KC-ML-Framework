# -*- coding: utf-8 -*-
# ==============================================================================
# Machine Learning for Key Gene Selection (Comprehensive Analysis Pipeline)
# Python Version 12 (Final Stable Version)
#
# This script is designed to run in a controlled environment to ensure library
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

# --- Helper Functions ---
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
    logging.info("=== Starting Comprehensive Machine Learning Analysis ===")
    
    # --- 1. Data Loading and Preprocessing ---
    logging.info("1. Reading and pre-processing data...")
    labels = pd.read_csv("phenotype.csv", index_col=0)
    expr_matrix = pd.read_csv("matched_expression.csv", index_col=0)
    common_samples = np.intersect1d(labels.index, expr_matrix.columns)
    labels, expr_matrix = labels.loc[common_samples], expr_matrix[common_samples]
    le = LabelEncoder().fit(labels['Label'])
    X, y = expr_matrix.T, le.transform(labels['Label'])
    selector = VarianceThreshold()
    X = pd.DataFrame(selector.fit_transform(X), index=X.index, columns=X.columns[selector.get_support()])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    
    # --- 2. Data Splitting & Imbalance Setup ---
    logging.info("2. Splitting data and setting up imbalance parameters...")
    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # Automatic calculation of class imbalance parameters
    n_neg = (y_train == 0).sum()      # Negative class (0) count
    n_pos = (y_train == 1).sum()      # Positive class (1) count
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
        plot_confusion_matrix(y_test, y_pred, le.classes_, 'Elastic Net'); plot_pr_curve(y_test, y_prob, 'Elastic Net', '#E41A1C')
        roc_data['Elastic_Net'] = plot_roc_curve(y_test, y_prob, 'Elastic Net', '#E41A1C')
        enet_genes = X_train.columns[enet_model_final.coef_[0] != 0].tolist()
        model_selections['Elastic_Net'] = set(enet_genes)
        pd.DataFrame({'Gene': enet_genes}).to_csv(os.path.join(RESULTS_DIR, "Elastic_Net_selected_genes.csv"), index=False)
        
    except Exception as e: logging.error(f"Elastic Net failed: {e}")
    
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
        plot_confusion_matrix(y_test, y_pred, le.classes_, 'Random Forest'); plot_pr_curve(y_test, y_prob, 'Random Forest', '#4DAF4A')
        roc_data['RF_Weighted'] = plot_roc_curve(y_test, y_prob, 'Random Forest', '#4DAF4A')
        
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
    
        logging.info("RF permutation importance computed using ROC-AUC scoring")
        
        # Keep RF SHAP analysis for gene selection but remove visualization
        logging.info("Performing SHAP analysis for Random Forest (gene selection only)...")
        try:
            explainer = shap.TreeExplainer(rf_model_final)
            shap_values = explainer.shap_values(X_test)

            mean_abs_shap_rf = np.abs(shap_values[1]).mean(axis=0)
            top20_idx_rf = np.argsort(mean_abs_shap_rf)[-20:]
            top20_genes_rf = X_train.columns[top20_idx_rf].tolist()
            model_selections['RF_SHAP_Top20'] = set(top20_genes_rf)
            logging.info("Random Forest SHAP Top20 genes added to consensus.")
            
        except Exception as e:
            logging.error(f"SHAP analysis for Random Forest failed: {e}")
    
    except Exception as e: logging.error(f"Random Forest main process failed: {e}")
    
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
    
        plot_confusion_matrix(y_test, y_pred, le.classes_, 'XGBoost')
        plot_pr_curve(y_test, y_prob, 'XGBoost', '#984EA3')
        roc_data['XGBoost'] = plot_roc_curve(y_test, y_prob, 'XGBoost', '#984EA3')
    
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
    
            plot_confusion_matrix(y_test, y_pred, le.classes_, 'Boruta+RF')
            plot_pr_curve(y_test, y_prob, 'Boruta+RF', '#FF7F00')
            roc_data['Boruta_RF'] = plot_roc_curve(y_test, y_prob, 'Boruta+RF', '#FF7F00')
    
            # Keep Boruta+RF SHAP analysis for gene selection but remove visualization
            logging.info("Performing SHAP analysis for Boruta+RF model (gene selection only)...")
            try:
                explainer = shap.TreeExplainer(rf_model_final)
                shap_values = explainer.shap_values(X_test_boruta)

                mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
                top20_idx = np.argsort(mean_abs_shap)[-20:]
                top20_genes = X_test_boruta.columns[top20_idx].tolist()
                model_selections['Boruta_RF_SHAP_Top20'] = set(top20_genes)
                logging.info("Boruta+RF SHAP Top20 genes added to consensus voting.")
                
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
        df_melted = df_test.melt(id_vars='Model', var_name='Metric', value_name='Score')
        plt.figure(figsize=(15, 8)); sns.barplot(x='Metric', y='Score', hue='Model', data=df_melted, palette='viridis')
        plt.title("Comprehensive Model Performance Comparison (Test Set)", fontsize=16)
        save_plot("final_performance_comparison_detailed.png")

    # ==============================================================================
    # MODEL CONSENSUS ANALYSIS + UpSet Plot
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

            # Generate comprehensive gene lists table
            logging.info("--- Generating Comprehensive Gene Lists Table ---")
            
            expected_models = [
                'Elastic_Net',
                'RF_MDA_Top50', 
                'RF_SHAP_Top20',
                'XGBoost_Gain_Positive',
                'Boruta',
                'Boruta_RF_SHAP_Top20'
            ]
            
            all_genes_dict = {}
            max_length = 0
            
            for model_name in expected_models:
                if model_name in model_selections:
                    gene_list = list(model_selections[model_name])
                else:
                    gene_list = []
                    logging.warning(f"Model {model_name} not found in model_selections, using empty list")
                
                all_genes_dict[model_name] = gene_list
                max_length = max(max_length, len(gene_list))
            
            union_genes = set()
            for gene_list in all_genes_dict.values():
                union_genes.update(gene_list)
            all_genes_dict['Union'] = list(union_genes)
            max_length = max(max_length, len(union_genes))
            
            gene_lists_df = pd.DataFrame()
            
            for col_name, gene_list in all_genes_dict.items():
                padded_list = gene_list + [''] * (max_length - len(gene_list))
                gene_lists_df[col_name] = padded_list
            
            output_path = os.path.join(RESULTS_DIR, "all_model_gene_lists.csv")
            gene_lists_df.to_csv(output_path, index=False)
            logging.info(f"Comprehensive gene lists table saved to: {output_path}")
            
            stats_df = pd.DataFrame({
                'Model': list(all_genes_dict.keys()),
                'Number_of_Genes': [len(gene_list) for gene_list in all_genes_dict.values()]
            })
            stats_path = os.path.join(RESULTS_DIR, "model_gene_counts.csv")
            stats_df.to_csv(stats_path, index=False)
            logging.info(f"Gene counts by model:\n{stats_df.to_string(index=False)}")

# UpSet Plot
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
                save_plot("Key_Gene_Intersection_UpSet_Plot.png")
                logging.info("UpSet plot generated successfully!")

            except Exception as e:
                logging.warning(f"UpSet failed ({e}), falling back to Venn diagram.")
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

    # Combined ROC curves
    if roc_data:
        plt.figure(figsize=(10, 10)); ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
        colors = {"Elastic_Net": "#E41A1C", "RF_Weighted": "#4DAF4A", "XGBoost": "#984EA3", "Boruta_RF": "#FF7F00"}
        for name, data in roc_data.items(): 
            plt.plot(data['fpr'], data['tpr'], color=colors.get(name, 'black'), lw=2, label=f"{name} (AUC = {data['auc']:.3f})")
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
        plt.title("Combined ROC Curves (Test Set Performance)")
        plt.legend(loc="lower right")
        plt.grid(linestyle=':')
        save_plot("roc_combined.png")

    logging.info("\n=== Analysis Complete: All reports and plots saved to 'results' directory. ===")

if __name__ == '__main__':
    main()