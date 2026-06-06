# ==============================================================================
# Ablation study – External validation (Nature style complete version)
# Using Nature journal official color scheme and style standards
# Added consensus score AUC analysis functionality
# Modification: Smooth consensus score curves and use shading to represent variance
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import matplotlib as mpl
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import itertools
from scipy import stats
from scipy.interpolate import make_interp_spline  # For curve smoothing

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ====================
# Nature journal style settings
# ====================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 3,
    'figure.dpi': 600,
    'axes.linewidth': 0.6,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 0.8,
    'lines.markersize': 4,
    'legend.frameon': False,
})

# ----------------------------- Nature color scheme -----------------------------
NATURE_COLORS = {
    "blue": "#1F77B4",       # Nature blue
    "orange": "#FF7F0E",     # Nature orange  
    "green": "#2CA02C",       # Nature green
    "red": "#D62728",         # Nature red
    "purple": "#9467BD",      # Nature purple
    "brown": "#8C564B",       # Nature brown
    "pink": "#E377C2",        # Nature pink
    "gray": "#7F7F7F",        # Nature gray
    "olive": "#BCBD22",       # Nature olive green
    "cyan": "#17BECF",        # Nature cyan
}

# Method-specific Nature colors
METHOD_COLORS = {
    "ML": NATURE_COLORS["blue"],           # Base method in blue
    "KC_ML": NATURE_COLORS["orange"],      # Knowledge-constrained method in orange
    "Bio_only": NATURE_COLORS["green"]      # Biological method in green
}

METHOD_MARKERS = {
    "ML": "o",      # Circle
    "KC_ML": "s",   # Square
    "Bio_only": "^"  # Triangle
}

# ----------------------------- Configuration -----------------------------
SIGNATURES = {
    "ML": "signatures/ml_equal.csv",
    "KC_ML": "signatures/KC_ML_AUC_weights.csv", 
    "Bio_only": "signatures/bio_only.csv"
}

# Consensus gene list file paths
CONSENSUS_GENE_FILES = {
    "KC_ML": "model_consensus_gene_kc-ml.csv",
    "ML": "model_consensus_gene_ml.csv"
}

COHORTS = {
    "GSE81622": {
        "expr": "external/81622_expression.csv",
        "pheno": "external/81622_phenotype.csv",
        "sample_col": "Sample",
        "label_col": "Group",
        "positive": "1"
    },
    "GSE61635": {
        "expr": "external/61635_expression.csv",
        "pheno": "external/61635_phenotype.csv",
        "sample_col": "Sample",
        "label_col": "Group",
        "positive": "1"
    }
}

OUTPUT_DIR = "ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
N_FEATURES = 50

# ----------------------------- Utility functions -----------------------------
def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each row (gene)"""
    return df.apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=1)

def bootstrap_auc_ci(y, score, n_resamples=2000, seed=42, alpha=0.05):
    """Paired bootstrap confidence interval"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    score = np.asarray(score)

    mask = ~np.isnan(score)
    y = y[mask]
    score = score[mask]
    
    if len(y) < 5 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    n = len(y)
    aucs = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        s_b = score[idx]
        
        if len(np.unique(y_b)) >= 2:
            try:
                aucs.append(roc_auc_score(y_b, s_b))
            except:
                continue

    if len(aucs) < 100:
        return np.nan, np.nan, np.nan

    auc = roc_auc_score(y, score)
    ci_low = np.percentile(aucs, 100 * alpha / 2)
    ci_high = np.percentile(aucs, 100 * (1 - alpha / 2))

    return auc, ci_low, ci_high

# ==================== PATCH START ====================
def bootstrap_delta_auc_pvalue(y, score1, score2,
                               n_resamples=2000, seed=42):
    """
    Paired bootstrap test for AUC difference.
    H0: AUC_KC_ML = AUC_ML
    Returns: mean(ΔAUC), one-sided P value
    """
    rng = np.random.default_rng(seed)

    y = np.asarray(y)
    score1 = np.asarray(score1)
    score2 = np.asarray(score2)

    mask = ~(np.isnan(score1) | np.isnan(score2))
    y, score1, score2 = y[mask], score1[mask], score2[mask]

    if len(y) < 10 or len(np.unique(y)) < 2:
        return np.nan, np.nan

    n = len(y)
    delta_aucs = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        s1_b = score1[idx]
        s2_b = score2[idx]

        try:
            auc1 = roc_auc_score(y_b, s1_b)
            auc2 = roc_auc_score(y_b, s2_b)
            delta_aucs.append(auc1 - auc2)
        except:
            continue

    if len(delta_aucs) < 100:
        return np.nan, np.nan

    mean_delta = float(np.mean(delta_aucs))
    p_value = float(np.mean(np.array(delta_aucs) <= 0))

    return round(mean_delta, 3), round(p_value, 4)
# ==================== PATCH END ====================

def save_nature_figure(filename, output_dir, dpi=600, formats=['png', 'svg']):
    """Save multiple formats compliant with Nature requirements"""
    for fmt in formats:
        plt.savefig(
            os.path.join(output_dir, f"{filename}.{fmt}"),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            format=fmt
        )

# ----------------------------- Core evaluation function -----------------------------
# ==================== PATCH START ====================
def evaluate_signature(method, sig_file, cohort_name, cfg):
    logging.info(f"Evaluating {method} on {cohort_name}")

    try:
        sig = pd.read_csv(sig_file)
        if "Gene" not in sig.columns:
            raise ValueError(f"{sig_file} must contain Gene column")

        sig["Gene"] = sig["Gene"].astype(str)
        genes = sig["Gene"].tolist()

        weighted = "Weight" in sig.columns
        if weighted:
            sig = sig.set_index("Gene")

        expr = pd.read_csv(cfg["expr"], index_col=0)
        expr.index = expr.index.astype(str)

        pheno = pd.read_csv(cfg["pheno"])
        pheno[cfg["sample_col"]] = pheno[cfg["sample_col"]].astype(str)

        common_samples = expr.columns.intersection(pheno[cfg["sample_col"]])
        expr = expr[common_samples]
        pheno = pheno.set_index(cfg["sample_col"]).loc[common_samples]

        if expr.shape[1] < 10:
            return None

        common_genes = expr.index.intersection(genes)
        if len(common_genes) < 3:
            return None

        expr_sub = expr.loc[common_genes]
        expr_sub = zscore_rows(expr_sub)

        if weighted:
            w = sig.loc[common_genes, "Weight"].values.astype(float)
        else:
            w = np.ones(len(common_genes))

        score = np.dot(w, expr_sub.values) / np.sum(np.abs(w))

        y = (pheno[cfg["label_col"]].astype(str) == str(cfg["positive"])).astype(int).values

        if y.sum() == 0 or y.sum() == len(y):
            return None

        auc, ci_low, ci_high = bootstrap_auc_ci(
            y, score,
            n_resamples=N_BOOTSTRAP,
            seed=RANDOM_SEED
        )

        if auc < 0.5:
            auc = 1 - auc
            ci_low, ci_high = 1 - ci_high, 1 - ci_low

        return {
            "Method": method,
            "Cohort": cohort_name,
            "Samples": len(y),
            "Genes_Used": len(common_genes),
            "AUC": round(auc, 3),
            "CI_low": round(ci_low, 3),
            "CI_high": round(ci_high, 3),
            "score": score,          # ✅ PATCH
            "y": y                   # ✅ PATCH
        }

    except Exception as e:
        logging.error(f"Error evaluating {method} on {cohort_name}: {str(e)}")
        return None
# ==================== PATCH END ====================

def evaluate_consensus_signature(method, gene_df, score_threshold, cohort_name, cfg):
    """Evaluate signature using consensus gene list"""
    try:
        # Filter genes
        if score_threshold == "all":
            selected_genes = gene_df["Gene"].tolist()
        else:
            selected_genes = gene_df[gene_df["Consensus_Score"] >= score_threshold]["Gene"].tolist()
        
        if not selected_genes:
            logging.warning(f"{method} on {cohort_name}: Number of genes with consensus score ≥{score_threshold} is 0")
            return None
        
        # Create signature DataFrame
        sig = pd.DataFrame({
            "Gene": selected_genes,
            "Weight": 1.0  # Equal weights
        })
        
        # Read expression data
        expr = pd.read_csv(cfg["expr"], index_col=0)
        expr.index = expr.index.astype(str)
        
        # Read phenotype data
        pheno = pd.read_csv(cfg["pheno"])
        pheno[cfg["sample_col"]] = pheno[cfg["sample_col"]].astype(str)
        
        # Align samples
        common_samples = expr.columns.intersection(pheno[cfg["sample_col"]])
        expr = expr[common_samples]
        pheno = pheno.set_index(cfg["sample_col"]).loc[common_samples]
        
        if expr.shape[1] < 10:
            logging.warning(f"{cohort_name}: Too few samples after alignment")
            return None
        
        # Gene intersection
        common_genes = expr.index.intersection(selected_genes)
        if len(common_genes) < 3:
            logging.warning(f"{cohort_name}: Too few common genes ({len(common_genes)})")
            return None
        
        # Prepare expression data
        expr_sub = expr.loc[common_genes]
        expr_sub = zscore_rows(expr_sub)
        
        # Calculate signature score (equal weights)
        w = np.ones(len(common_genes))
        score = np.dot(w, expr_sub.values) / np.sum(np.abs(w))
        
        # Prepare labels
        y = (pheno[cfg["label_col"]].astype(str) == str(cfg["positive"])).astype(int).values
        
        if y.sum() == 0 or y.sum() == len(y):
            logging.warning(f"{cohort_name}: Single class labels")
            return None
        
        # Calculate AUC and CI
        auc, ci_low, ci_high = bootstrap_auc_ci(
            y, score,
            n_resamples=N_BOOTSTRAP,
            seed=RANDOM_SEED
        )
        
        # Direction correction
        if auc < 0.5:
            auc = 1 - auc
            ci_low, ci_high = 1 - ci_high, 1 - ci_low
        
        return {
            "Method": method,
            "Cohort": cohort_name,
            "Score_Threshold": score_threshold,
            "Samples": len(y),
            "Genes_Used": len(common_genes),
            "AUC": round(auc, 3),
            "CI_low": round(ci_low, 3),
            "CI_high": round(ci_high, 3)
        }
        
    except Exception as e:
        logging.error(f"Error evaluating {method} on {cohort_name} with threshold {score_threshold}: {str(e)}")
        return None

# ----------------------------- Consensus score analysis function -----------------------------
def analyze_consensus_scores(output_dir):
    """Analyze AUC performance under different consensus score thresholds"""
    logging.info("Starting consensus score analysis")
    
    # Read consensus gene lists
    kc_ml_consensus = pd.read_csv(CONSENSUS_GENE_FILES["KC_ML"])
    ml_consensus = pd.read_csv(CONSENSUS_GENE_FILES["ML"])
    
    # Define consensus score thresholds
    score_thresholds = [6, 5, 4, 3, 2, 1]
    
    results = []
    
    # Evaluate performance under different thresholds
    for cohort_name, cfg in COHORTS.items():
        for threshold in score_thresholds:
            # Evaluate KC-ML
            kc_ml_result = evaluate_consensus_signature(
                "KC_ML", kc_ml_consensus, threshold, cohort_name, cfg
            )
            if kc_ml_result:
                results.append(kc_ml_result)
                logging.info(f"✓ KC_ML on {cohort_name} (threshold ≥{threshold}): AUC = {kc_ml_result['AUC']}, Genes = {kc_ml_result['Genes_Used']}")
            
            # Evaluate ML
            ml_result = evaluate_consensus_signature(
                "ML", ml_consensus, threshold, cohort_name, cfg
            )
            if ml_result:
                results.append(ml_result)
                logging.info(f"✓ ML on {cohort_name} (threshold ≥{threshold}): AUC = {ml_result['AUC']}, Genes = {ml_result['Genes_Used']}")
    
    if not results:
        logging.error("No valid results from consensus score analysis")
        return None
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(output_dir, "consensus_score_analysis.csv")
    results_df.to_csv(results_file, index=False)
    logging.info(f"Consensus score analysis saved to: {results_file}")
    
    # Generate visualizations
    generate_consensus_score_plots(results_df, output_dir)
    
    return results_df

def generate_consensus_score_plots(results_df, output_dir):
    """Generate plots for consensus score analysis - Modified: added curve smoothing and shading"""
    
    # 1. Plot separate curves for each validation set
    for cohort in results_df['Cohort'].unique():
        cohort_data = results_df[results_df['Cohort'] == cohort]
        
        plt.figure(figsize=(8.9/2.54, 6.7/2.54))  # Nature standard size
        
        # Prepare data
        kc_ml_data = cohort_data[cohort_data['Method'] == 'KC_ML']
        ml_data = cohort_data[cohort_data['Method'] == 'ML']
        
        # Define x-axis order
        threshold_order = [5, 4, 3, 2, 1]   # [6, 5, 4, 3, 2, 1]
        threshold_labels = ['≥5', '≥4', '≥3', '≥2', '≥1']   #['≥6', '≥5', '≥4', '≥3', '≥2', '≥1']
        threshold_positions = list(range(len(threshold_order)))
        
        # Extract AUC values and confidence intervals
        kc_ml_aucs = []
        kc_ml_ci_low = []
        kc_ml_ci_high = []
        
        ml_aucs = []
        ml_ci_low = []
        ml_ci_high = []
        
        for threshold in threshold_order:
            kc_ml_row = kc_ml_data[kc_ml_data['Score_Threshold'] == threshold]
            ml_row = ml_data[ml_data['Score_Threshold'] == threshold]
            
            if not kc_ml_row.empty:
                kc_ml_aucs.append(kc_ml_row['AUC'].values[0])
                kc_ml_ci_low.append(kc_ml_row['CI_low'].values[0])
                kc_ml_ci_high.append(kc_ml_row['CI_high'].values[0])
            else:
                kc_ml_aucs.append(np.nan)
                kc_ml_ci_low.append(np.nan)
                kc_ml_ci_high.append(np.nan)
            
            if not ml_row.empty:
                ml_aucs.append(ml_row['AUC'].values[0])
                ml_ci_low.append(ml_row['CI_low'].values[0])
                ml_ci_high.append(ml_row['CI_high'].values[0])
            else:
                ml_aucs.append(np.nan)
                ml_ci_low.append(np.nan)
                ml_ci_high.append(np.nan)
        
        # Data cleaning: remove NaN values
        kc_ml_valid_idx = ~np.isnan(kc_ml_aucs)
        ml_valid_idx = ~np.isnan(ml_aucs)
        
        # Create smoothed curve for KC-ML
        if np.sum(kc_ml_valid_idx) >= 3:  # Need at least 3 points for smoothing
            kc_ml_positions = np.array(threshold_positions)[kc_ml_valid_idx]
            kc_ml_aucs_valid = np.array(kc_ml_aucs)[kc_ml_valid_idx]
            kc_ml_ci_low_valid = np.array(kc_ml_ci_low)[kc_ml_valid_idx]
            kc_ml_ci_high_valid = np.array(kc_ml_ci_high)[kc_ml_valid_idx]
            
            # Create smoothed x-axis
            kc_ml_smooth_x = np.linspace(min(kc_ml_positions), max(kc_ml_positions), 100)
            
            # Create spline interpolation function
            try:
                # Smooth AUC values
                kc_ml_spline_auc = make_interp_spline(kc_ml_positions, kc_ml_aucs_valid, k=3)
                kc_ml_smooth_auc = kc_ml_spline_auc(kc_ml_smooth_x)
                
                # Smooth lower confidence bound
                kc_ml_spline_low = make_interp_spline(kc_ml_positions, kc_ml_ci_low_valid, k=3)
                kc_ml_smooth_low = kc_ml_spline_low(kc_ml_smooth_x)
                
                # Smooth upper confidence bound
                kc_ml_spline_high = make_interp_spline(kc_ml_positions, kc_ml_ci_high_valid, k=3)
                kc_ml_smooth_high = kc_ml_spline_high(kc_ml_smooth_x)
                
                # Plot KC-ML smoothed curve
                plt.plot(kc_ml_smooth_x, kc_ml_smooth_auc, 
                        linewidth=0.8,   # ← This parameter controls curve width
                        color=NATURE_COLORS["orange"], label='KC-ML', 
                        zorder=3)
                
                # Plot KC-ML confidence interval shading
                plt.fill_between(kc_ml_smooth_x, kc_ml_smooth_low, kc_ml_smooth_high,
                                alpha=0.1, # ← This parameter controls shading transparency
                                color=NATURE_COLORS["orange"], 
                                linewidth=0.1,  # Edge line width
                                linestyle='--',  # Edge line style
                                label='KC-ML 95% CI', zorder=2)
                
                # Mark original data points on smoothed curve
                plt.scatter(kc_ml_positions, kc_ml_aucs_valid, 
                           s=7, color=NATURE_COLORS["orange"], 
                           marker='s', edgecolors='white', linewidth=0.5, 
                           label='KC-ML data points', zorder=4)
                
            except Exception as e:
                logging.warning(f"Cannot smooth KC-ML curve: {e}")
                # If smoothing fails, use original line plot
                plt.plot(threshold_positions, kc_ml_aucs, 
                        marker='s', markersize=4, linewidth=1, 
                        color=NATURE_COLORS["orange"], label='KC-ML', 
                        markeredgewidth=0.3, zorder=3)
                
                # Draw original error bars
                kc_ml_yerr_low = [a - l for a, l in zip(kc_ml_aucs, kc_ml_ci_low)]
                kc_ml_yerr_high = [h - a for a, h in zip(kc_ml_aucs, kc_ml_ci_high)]
                plt.errorbar(threshold_positions, kc_ml_aucs,
                            yerr=[kc_ml_yerr_low, kc_ml_yerr_high],
                            fmt='none', ecolor=NATURE_COLORS["orange"], 
                            elinewidth=0.5, capsize=2, capthick=0.5,
                            alpha=0.7, zorder=2)
        else:
            # If insufficient data points, use original line plot
            plt.plot(threshold_positions, kc_ml_aucs, 
                    marker='s', markersize=4, linewidth=1, 
                    color=NATURE_COLORS["orange"], label='KC-ML', 
                    markeredgewidth=0.3, zorder=3)
            
            # Draw original error bars
            kc_ml_yerr_low = [a - l for a, l in zip(kc_ml_aucs, kc_ml_ci_low)]
            kc_ml_yerr_high = [h - a for a, h in zip(kc_ml_aucs, kc_ml_ci_high)]
            plt.errorbar(threshold_positions, kc_ml_aucs,
                        yerr=[kc_ml_yerr_low, kc_ml_yerr_high],
                        fmt='none', ecolor=NATURE_COLORS["orange"], 
                        elinewidth=0.5, capsize=2, capthick=0.5,
                        alpha=0.7, zorder=2)
        
        # Create smoothed curve for ML
        if np.sum(ml_valid_idx) >= 3:  # Need at least 3 points for smoothing
            ml_positions = np.array(threshold_positions)[ml_valid_idx]
            ml_aucs_valid = np.array(ml_aucs)[ml_valid_idx]
            ml_ci_low_valid = np.array(ml_ci_low)[ml_valid_idx]
            ml_ci_high_valid = np.array(ml_ci_high)[ml_valid_idx]
            
            # Create smoothed x-axis
            ml_smooth_x = np.linspace(min(ml_positions), max(ml_positions), 100)
            
            # Create spline interpolation function
            try:
                # Smooth AUC values
                ml_spline_auc = make_interp_spline(ml_positions, ml_aucs_valid, k=3)
                ml_smooth_auc = ml_spline_auc(ml_smooth_x)
                
                # Smooth lower confidence bound
                ml_spline_low = make_interp_spline(ml_positions, ml_ci_low_valid, k=3)
                ml_smooth_low = ml_spline_low(ml_smooth_x)
                
                # Smooth upper confidence bound
                ml_spline_high = make_interp_spline(ml_positions, ml_ci_high_valid, k=3)
                ml_smooth_high = ml_spline_high(ml_smooth_x)
                
                # Plot ML smoothed curve
                plt.plot(ml_smooth_x, ml_smooth_auc, 
                        linewidth=0.8,    # ← This parameter controls curve width
                        color=NATURE_COLORS["blue"], label='Equal-ML', 
                        zorder=3)
                
                # Plot ML confidence interval shading
                plt.fill_between(ml_smooth_x, ml_smooth_low, ml_smooth_high,
                                alpha=0.1,   # ← This parameter controls shading transparency
                                linewidth=0.1,  # Edge line width
                                linestyle='--',  # Edge line style
                                color=NATURE_COLORS["blue"], 
                                label='Equal-ML 95% CI', zorder=1)
                
                # Mark original data points on smoothed curve
                plt.scatter(ml_positions, ml_aucs_valid, 
                           s=7, color=NATURE_COLORS["blue"], 
                           marker='o', edgecolors='white', linewidth=0.5, 
                           label='ML data points', zorder=4)
                
            except Exception as e:
                logging.warning(f"Cannot smooth ML curve: {e}")
                # If smoothing fails, use original line plot
                plt.plot(threshold_positions, ml_aucs, 
                        marker='o', markersize=4, linewidth=1, 
                        color=NATURE_COLORS["blue"], label='Equal-ML', 
                        markeredgewidth=0.3, zorder=3)
                
                # Draw original error bars
                ml_yerr_low = [a - l for a, l in zip(ml_aucs, ml_ci_low)]
                ml_yerr_high = [h - a for a, h in zip(ml_aucs, ml_ci_high)]
                plt.errorbar(threshold_positions, ml_aucs,
                            yerr=[ml_yerr_low, ml_yerr_high],
                            fmt='none', ecolor=NATURE_COLORS["blue"], 
                            elinewidth=0.5, capsize=2, capthick=0.5,
                            alpha=0.7, zorder=2)
        else:
            # If insufficient data points, use original line plot
            plt.plot(threshold_positions, ml_aucs, 
                    marker='o', markersize=4, linewidth=1, 
                    color=NATURE_COLORS["blue"], label='Equal-ML', 
                    markeredgewidth=0.3, zorder=3)
            
            # Draw original error bars
            ml_yerr_low = [a - l for a, l in zip(ml_aucs, ml_ci_low)]
            ml_yerr_high = [h - a for a, h in zip(ml_aucs, ml_ci_high)]
            plt.errorbar(threshold_positions, ml_aucs,
                        yerr=[ml_yerr_low, ml_yerr_high],
                        fmt='none', ecolor=NATURE_COLORS["blue"], 
                        elinewidth=0.5, capsize=2, capthick=0.5,
                        alpha=0.7, zorder=2)
        
        # Add random line
        plt.axhline(0.5, linestyle='--', color=NATURE_COLORS["gray"], 
                   linewidth=1, alpha=0.5, label='Random (AUC=0.5)', zorder=0)
        
        # Set x-axis
        plt.xticks(threshold_positions, threshold_labels)
        plt.xlabel('Consensus Score Threshold')
        plt.ylabel('AUC (95% CI)')
        
        # Set y-axis range and ticks
        plt.ylim(0.4, 1.15)
        plt.yticks(np.arange(0.4, 1.16, 0.1))
        
        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        # Create custom legend
        legend_elements = [
            Line2D([0], [0], color=NATURE_COLORS["orange"], lw=1.5, label='KC-ML'),
            Patch(facecolor=NATURE_COLORS["orange"], alpha=0.3, label='KC-ML 95% CI'),
            Line2D([0], [0], color=NATURE_COLORS["blue"], lw=1.5, label='ML'),
            Patch(facecolor=NATURE_COLORS["blue"], alpha=0.3, label='ML 95% CI'),
            Line2D([0], [0], color=NATURE_COLORS["gray"], linestyle='--', lw=1, 
                  label='Random (AUC=0.5)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=NATURE_COLORS["orange"], 
                  markersize=5, label='KC-ML data points'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=NATURE_COLORS["blue"], 
                  markersize=5, label='ML data points')
        ]
        
        plt.legend(handles=legend_elements, loc='best', frameon=False, fontsize=6)
        plt.title(f'Consensus Score Analysis - {cohort}')
        
        # Add grid
        plt.grid(True, alpha=0.2, linestyle='--', axis='y')
        
        # Remove top and right borders
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        save_nature_figure(f'consensus_score_curve_smooth_{cohort}', output_dir)
        plt.close()
    
    # 2. Combined figure - both validation sets in one plot
    plt.figure(figsize=(12/2.54, 8/2.54))
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12/2.54, 5/2.54), sharey=True)
    
    for idx, cohort in enumerate(results_df['Cohort'].unique()):
        ax = axes[idx]
        cohort_data = results_df[results_df['Cohort'] == cohort]
        
        # Prepare data
        kc_ml_data = cohort_data[cohort_data['Method'] == 'KC_ML']
        ml_data = cohort_data[cohort_data['Method'] == 'ML']
        
        # Define x-axis order
        threshold_order = [5, 4, 3, 2, 1]    #[6, 5, 4, 3, 2, 1]
        threshold_labels = ['≥5', '≥4', '≥3', '≥2', '≥1']   # ['≥6', '≥5', '≥4', '≥3', '≥2', '≥1']
        threshold_positions = list(range(len(threshold_order)))
        
        # Extract AUC values
        kc_ml_aucs = []
        kc_ml_ci_low = []
        kc_ml_ci_high = []
        
        ml_aucs = []
        ml_ci_low = []
        ml_ci_high = []
        
        for threshold in threshold_order:
            kc_ml_row = kc_ml_data[kc_ml_data['Score_Threshold'] == threshold]
            ml_row = ml_data[ml_data['Score_Threshold'] == threshold]
            
            if not kc_ml_row.empty:
                kc_ml_aucs.append(kc_ml_row['AUC'].values[0])
                kc_ml_ci_low.append(kc_ml_row['CI_low'].values[0])
                kc_ml_ci_high.append(kc_ml_row['CI_high'].values[0])
            else:
                kc_ml_aucs.append(np.nan)
                kc_ml_ci_low.append(np.nan)
                kc_ml_ci_high.append(np.nan)
            
            if not ml_row.empty:
                ml_aucs.append(ml_row['AUC'].values[0])
                ml_ci_low.append(ml_row['CI_low'].values[0])
                ml_ci_high.append(ml_row['CI_high'].values[0])
            else:
                ml_aucs.append(np.nan)
                ml_ci_low.append(np.nan)
                ml_ci_high.append(np.nan)
        
        # Data cleaning: remove NaN values
        kc_ml_valid_idx = ~np.isnan(kc_ml_aucs)
        ml_valid_idx = ~np.isnan(ml_aucs)
        
        # Create smoothed curve for KC-ML
        if np.sum(kc_ml_valid_idx) >= 3:
            kc_ml_positions = np.array(threshold_positions)[kc_ml_valid_idx]
            kc_ml_aucs_valid = np.array(kc_ml_aucs)[kc_ml_valid_idx]
            kc_ml_ci_low_valid = np.array(kc_ml_ci_low)[kc_ml_valid_idx]
            kc_ml_ci_high_valid = np.array(kc_ml_ci_high)[kc_ml_valid_idx]
            
            kc_ml_smooth_x = np.linspace(min(kc_ml_positions), max(kc_ml_positions), 100)
            
            try:
                kc_ml_spline_auc = make_interp_spline(kc_ml_positions, kc_ml_aucs_valid, k=3)
                kc_ml_smooth_auc = kc_ml_spline_auc(kc_ml_smooth_x)
                
                kc_ml_spline_low = make_interp_spline(kc_ml_positions, kc_ml_ci_low_valid, k=3)
                kc_ml_smooth_low = kc_ml_spline_low(kc_ml_smooth_x)
                
                kc_ml_spline_high = make_interp_spline(kc_ml_positions, kc_ml_ci_high_valid, k=3)
                kc_ml_smooth_high = kc_ml_spline_high(kc_ml_smooth_x)
                
                ax.plot(kc_ml_smooth_x, kc_ml_smooth_auc, 
                       linewidth=0.8, # ← This parameter controls curve width
                       color=NATURE_COLORS["orange"], 
                       label='KC-ML', zorder=3)
                
                ax.fill_between(kc_ml_smooth_x, kc_ml_smooth_low, kc_ml_smooth_high,
                              alpha=0.1,   # ← This parameter controls shading transparency
                              linewidth=0.1,  # Edge line width
                              linestyle='--',  # Edge line style
                              color=NATURE_COLORS["orange"], 
                              label='KC-ML 95% CI', zorder=2)
                
                ax.scatter(kc_ml_positions, kc_ml_aucs_valid, 
                          s=7, color=NATURE_COLORS["orange"], 
                          marker='s', edgecolors='white', linewidth=0.3, zorder=4)
                
            except Exception as e:
                logging.warning(f"Cannot smooth KC-ML curve ({cohort}): {e}")
                ax.plot(threshold_positions, kc_ml_aucs, 
                       marker='s', markersize=3, linewidth=0.8, 
                       color=NATURE_COLORS["orange"], label='KC-ML', 
                       markeredgewidth=0.3, zorder=3)
        else:
            ax.plot(threshold_positions, kc_ml_aucs, 
                   marker='s', markersize=3, linewidth=0.8, 
                   color=NATURE_COLORS["orange"], label='KC-ML', 
                   markeredgewidth=0.3, zorder=3)
        
        # Create smoothed curve for ML
        if np.sum(ml_valid_idx) >= 3:
            ml_positions = np.array(threshold_positions)[ml_valid_idx]
            ml_aucs_valid = np.array(ml_aucs)[ml_valid_idx]
            ml_ci_low_valid = np.array(ml_ci_low)[ml_valid_idx]
            ml_ci_high_valid = np.array(ml_ci_high)[ml_valid_idx]
            
            ml_smooth_x = np.linspace(min(ml_positions), max(ml_positions), 100)
            
            try:
                ml_spline_auc = make_interp_spline(ml_positions, ml_aucs_valid, k=3)
                ml_smooth_auc = ml_spline_auc(ml_smooth_x)
                
                ml_spline_low = make_interp_spline(ml_positions, ml_ci_low_valid, k=3)
                ml_smooth_low = ml_spline_low(ml_smooth_x)
                
                ml_spline_high = make_interp_spline(ml_positions, ml_ci_high_valid, k=3)
                ml_smooth_high = ml_spline_high(ml_smooth_x)
                
                ax.plot(ml_smooth_x, ml_smooth_auc, 
                       linewidth=0.8, # ← This parameter controls curve width
                       color=NATURE_COLORS["blue"], 
                       label='ML', zorder=3)
                
                ax.fill_between(ml_smooth_x, ml_smooth_low, ml_smooth_high,
                              alpha=0.1,  # ← This parameter controls shading transparency
                              linewidth=0.1,  # Edge line width
                              linestyle='--',  # Edge line style
                              color=NATURE_COLORS["blue"], 
                              label='ML 95% CI', zorder=1)
                
                ax.scatter(ml_positions, ml_aucs_valid, 
                          s=7, color=NATURE_COLORS["blue"], 
                          marker='o', edgecolors='white', linewidth=0.3, zorder=4)
                
            except Exception as e:
                logging.warning(f"Cannot smooth ML curve ({cohort}): {e}")
                ax.plot(threshold_positions, ml_aucs, 
                       marker='o', markersize=3, linewidth=0.8, 
                       color=NATURE_COLORS["blue"], label='Equal-ML', 
                       markeredgewidth=0.3, zorder=3)
        else:
            ax.plot(threshold_positions, ml_aucs, 
                   marker='o', markersize=3, linewidth=0.8, 
                   color=NATURE_COLORS["blue"], label='Equal-ML', 
                   markeredgewidth=0.3, zorder=3)
        
        # Add random line
        ax.axhline(0.5, linestyle='--', color=NATURE_COLORS["gray"], 
                  linewidth=0.8, alpha=0.5, label='Random (AUC=0.5)', zorder=0)
        
        # Set x-axis
        ax.set_xticks(threshold_positions)
        ax.set_xticklabels(threshold_labels, rotation=45)
        ax.set_title(cohort)
        
        # Set y-axis range and ticks
        ax.set_ylim(0.4, 1.15)
        ax.set_yticks(np.arange(0.4, 1.16, 0.1))
        
        # Add grid
        #ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        
        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Only add y-label to first subplot
        if idx == 0:
            ax.set_ylabel('AUC (95% CI)')
        
        # Only add legend to second subplot
        if idx == 1:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            
            legend_elements = [
                Line2D([0], [0], color=NATURE_COLORS["orange"], lw=0.8, label='KC-ML'),
                Patch(facecolor=NATURE_COLORS["orange"], alpha=0.1, label='KC-ML 95% CI'),
                Line2D([0], [0], color=NATURE_COLORS["blue"], lw=0.8, label='Equal-ML'),
                Patch(facecolor=NATURE_COLORS["blue"], alpha=0.1, label='Equal-ML 95% CI'),
                Line2D([0], [0], color=NATURE_COLORS["gray"], linestyle='--', lw=0.8, 
                      label='Random (AUC=0.5)'),
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', frameon=False, 
                     fontsize=5, bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    save_nature_figure('consensus_score_curves_smooth_combined', output_dir)
    plt.close()
    
    # 3. Create performance summary table
    summary_data = []
    for cohort in results_df['Cohort'].unique():
        cohort_data = results_df[results_df['Cohort'] == cohort]
        
        for method in ['KC_ML', 'ML']:
            method_data = cohort_data[cohort_data['Method'] == method]
            
            for threshold in threshold_order:
                threshold_data = method_data[method_data['Score_Threshold'] == threshold]
                
                if not threshold_data.empty:
                    summary_data.append({
                        'Cohort': cohort,
                        'Method': method,
                        'Score_Threshold': threshold,
                        'AUC': threshold_data['AUC'].values[0],
                        'Genes_Used': threshold_data['Genes_Used'].values[0]
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, "consensus_score_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Consensus score summary saved to: {summary_file}")
    
    logging.info("Consensus score plots with smooth curves generated")

# ----------------------------- Analysis functions -----------------------------
def add_effect_size_analysis(df, output_dir):
    effect_sizes = []

    for cohort in df['Cohort'].unique():
        cohort_data = df[df['Cohort'] == cohort]

        kc_ml = cohort_data[cohort_data['Method'] == 'KC_ML']
        ml = cohort_data[cohort_data['Method'] == 'ML']

        if kc_ml.empty or ml.empty:
            continue

        kc_row = kc_ml.iloc[0]
        ml_row = ml.iloc[0]

        y = kc_row["y"]
        score_kc = kc_row["score"]
        score_ml = ml_row["score"]

        delta_auc, p_value = bootstrap_delta_auc_pvalue(
            y, score_kc, score_ml,
            n_resamples=N_BOOTSTRAP,
            seed=RANDOM_SEED
        )

        effect_sizes.append({
            "Cohort": cohort,
            "AUC_KC_ML": kc_row["AUC"],
            "AUC_ML": ml_row["AUC"],
            "Delta_AUC": delta_auc,
            "P_value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No"
        })

    if effect_sizes:
        effect_df = pd.DataFrame(effect_sizes)
        effect_df.to_csv(
            os.path.join(output_dir, "effect_sizes.csv"),
            index=False
        )
        logging.info("Effect size analysis with P-values saved.")
        return effect_df

    return None

def generate_interpretation_summary(df, output_dir):
    """Generate readable results summary"""
    summary = []
    
    for cohort in df['Cohort'].unique():
        cohort_data = df[df['Cohort'] == cohort]
        
        # Find best performing method
        best_idx = cohort_data['AUC'].idxmax()
        best_method = cohort_data.loc[best_idx]
        
        # Get ML baseline
        ml_data = cohort_data[cohort_data['Method'] == 'ML']
        ml_auc = ml_data['AUC'].values[0] if not ml_data.empty else 0
        
        improvement = best_method['AUC'] - ml_auc if not ml_data.empty else 0
        
        summary.append({
            'Cohort': cohort,
            'Best_Performing_Method': best_method['Method'],
            'Best_AUC': best_method['AUC'],
            'ML_Baseline_AUC': ml_auc,
            'Improvement_Over_ML': round(improvement, 3),
            'Sample_Size': best_method['Samples'],
            'Genes_Used': best_method['Genes_Used'],
            'Performance_Category': 'Excellent' if best_method['AUC'] > 0.9 else 
                                  'Good' if best_method['AUC'] > 0.8 else 
                                  'Moderate' if best_method['AUC'] > 0.7 else 'Poor'
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    
    # Generate text summary
    with open(os.path.join(output_dir, 'results_interpretation.txt'), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ABLATION STUDY RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERVIEW:\n")
        f.write(f"- Total comparisons: {len(df)}\n")
        f.write(f"- Cohorts analyzed: {len(df['Cohort'].unique())}\n")
        f.write(f"- Methods compared: {', '.join(df['Method'].unique())}\n\n")
        
        f.write("COHORT-SPECIFIC RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        for item in summary:
            f.write(f"\n{item['Cohort']}:\n")
            f.write(f"  * Best method: {item['Best_Performing_Method']}\n")
            f.write(f"  * AUC: {item['Best_AUC']} ({item['Performance_Category']})\n")
            f.write(f"  * Improvement over ML: +{item['Improvement_Over_ML']}\n")
            f.write(f"  * Sample size: {item['Sample_Size']}\n")
            f.write(f"  * Genes used: {item['Genes_Used']}\n")
        
        f.write(f"\nGLOBAL OBSERVATIONS:\n")
        f.write("-" * 40 + "\n")
        
        # Overall best performance
        overall_best = df.loc[df['AUC'].idxmax()]
        f.write(f"* Overall best performance: {overall_best['Method']} on {overall_best['Cohort']} (AUC = {overall_best['AUC']})\n")
        
        # Method consistency
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            avg_auc = method_data['AUC'].mean()
            f.write(f"* {method} average AUC: {avg_auc:.3f}\n")
    
    logging.info("Interpretation summary generated.")
    return summary_df

# ----------------------------- Nature style visualization functions -----------------------------
def generate_nature_style_visualizations(df, output_dir):
    """Generate visualizations compliant with Nature journal standards"""
    
    # 1. Bar chart - Nature style
    plt.figure(figsize=(8.9/2.54, 6.7/2.54))  # Convert to inches
    
    cohorts = df['Cohort'].unique()
    x = np.arange(len(cohorts))
    width = 0.25
    
    for i, method in enumerate(METHOD_COLORS.keys()):
        method_data = df[df['Method'] == method]
        
        auc_values = []
        ci_low_values = []
        ci_high_values = []
        
        for cohort in cohorts:
            cohort_data = method_data[method_data['Cohort'] == cohort]
            if not cohort_data.empty:
                auc_values.append(cohort_data['AUC'].values[0])
                ci_low_values.append(cohort_data['CI_low'].values[0])
                ci_high_values.append(cohort_data['CI_high'].values[0])
            else:
                auc_values.append(0)
                ci_low_values.append(0)
                ci_high_values.append(0)
        
        # Calculate error bars
        yerr_low = [auc - low for auc, low in zip(auc_values, ci_low_values)]
        yerr_high = [high - auc for auc, high in zip(auc_values, ci_high_values)]
        yerr = [yerr_low, yerr_high]
        
        plt.bar(
            x + (i - 1) * width,
            auc_values,
            width,
            yerr=yerr,
            capsize=2,
            color=METHOD_COLORS[method],
            edgecolor='black',
            linewidth=0.3,
            label=method,
            alpha=0.9,
            error_kw={
                'elinewidth': 0.5,
                'capsize': 2,
                'capthick': 0.5
            }
        )
    
    plt.axhline(0.5, linestyle='--', color=NATURE_COLORS["gray"], 
               linewidth=0.5, alpha=0.7, label='Random (AUC=0.5)')
    plt.xticks(x, cohorts)
    plt.ylabel('AUC (95% CI)')
    plt.ylim(0.4, 1.05)
    plt.legend(loc='lower right')
    plt.title('AUC Performance Comparison')
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_nature_figure('AUC_bar_nature_style', output_dir)
    plt.close()

    # 2. Dot plot - Nature style
    plt.figure(figsize=(8.9/2.54, 6.7/2.54))
    
    for i, cohort in enumerate(cohorts):
        cohort_data = df[df['Cohort'] == cohort]
        
        for j, method in enumerate(METHOD_COLORS.keys()):
            method_data = cohort_data[cohort_data['Method'] == method]
            if not method_data.empty:
                x_pos = i + (j - 1) * 0.2
                
                auc = method_data['AUC'].values[0]
                ci_low = method_data['CI_low'].values[0]
                ci_high = method_data['CI_high'].values[0]
                
                plt.errorbar(
                    x_pos, auc,
                    yerr=[[auc - ci_low], [ci_high - auc]],
                    fmt=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    markersize=4,
                    capsize=1.5,
                    capthick=0.5,
                    elinewidth=0.5,
                    label=method if i == 0 else "",
                    alpha=0.9,
                    markeredgewidth=0.3
                )
    
    plt.xticks(range(len(cohorts)), cohorts)
    plt.ylabel('AUC (95% CI)')
    plt.ylim(0.4, 1.15)
    plt.axhline(0.5, linestyle='--', color=NATURE_COLORS["gray"], 
               linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.title('AUC with Confidence Intervals')
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_nature_figure('AUC_dot_nature_style', output_dir)
    plt.close()

    # 3. Heatmap - Nature style
    plt.figure(figsize=(6/2.54, 4/2.54))
    
    pivot_df = df.pivot(index="Method", columns="Cohort", values="AUC")
    annot_df = pivot_df.round(3).astype(str)
    
    sns.heatmap(
        pivot_df,
        annot=annot_df,
        fmt='',
        cmap="YlOrRd",
        center=0.75,
        square=True,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={
            'label': 'AUC',
            'shrink': 0.8
        }
    )
    
    plt.title('AUC Performance Heatmap')
    plt.tight_layout()
    save_nature_figure('AUC_heatmap_nature_style', output_dir)
    plt.close()

    # 4. Method consistency plot
    plt.figure(figsize=(5/2.54, 4/2.54))
    
    # Calculate average performance
    method_stats = []
    for method in METHOD_COLORS.keys():
        method_data = df[df['Method'] == method]
        avg_auc = method_data['AUC'].mean()
        std_auc = method_data['AUC'].std()
        method_stats.append({
            'Method': method,
            'Mean_AUC': avg_auc,
            'Std_AUC': std_auc
        })
    
    stats_df = pd.DataFrame(method_stats)
    
    plt.bar(
        stats_df['Method'],
        stats_df['Mean_AUC'],
        yerr=stats_df['Std_AUC'],
        capsize=2,
        color=[METHOD_COLORS[m] for m in stats_df['Method']],
        edgecolor='black',
        linewidth=0.3,
        alpha=0.9,
        error_kw={
            'elinewidth': 0.5,
            'capsize': 2,
            'capthick': 0.5
        }
    )
    
    plt.ylabel('Average AUC ± SD')
    plt.ylim(0.5, 1.0)
    plt.axhline(0.5, linestyle='--', color=NATURE_COLORS["gray"], 
               linewidth=0.5, alpha=0.7)
    plt.title('Method Performance Consistency')
    plt.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (method, mean_auc) in enumerate(zip(stats_df['Method'], stats_df['Mean_AUC'])):
        plt.text(i, mean_auc + 0.01, f'{mean_auc:.3f}', 
                ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    save_nature_figure('method_consistency_nature_style', output_dir)
    plt.close()

    # 5. Performance trend plot
    plt.figure(figsize=(8.9/2.54, 6.7/2.54))
    
    for method in METHOD_COLORS.keys():
        method_data = df[df['Method'] == method]
        
        # Sort by cohort
        cohort_order = sorted(method_data['Cohort'].unique())
        auc_values = [method_data[method_data['Cohort'] == c]['AUC'].values[0] for c in cohort_order]
        
        plt.plot(
            range(len(cohort_order)), 
            auc_values,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markersize=4,
            linewidth=1,
            label=method,
            markeredgewidth=0.3
        )
    
    plt.xticks(range(len(cohort_order)), cohort_order)
    plt.ylabel('AUC')
    plt.ylim(0.5, 1.0)
    plt.axhline(0.5, linestyle='--', color=NATURE_COLORS["gray"], 
               linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.title('Method Performance Across Cohorts')
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_nature_figure('method_trend_nature_style', output_dir)
    plt.close()

    logging.info("Nature style visualizations generated")

# ----------------------------- Main function -----------------------------
def main():
    """Main execution function"""
    logging.info("Starting Nature-style ablation study analysis")
    
    results = []

    # Evaluate all method-cohort combinations
    for method, sig_file in SIGNATURES.items():
        for cohort, cfg in COHORTS.items():
            res = evaluate_signature(method, sig_file, cohort, cfg)
            if res is not None:
                results.append(res)
                logging.info(f"✓ {method} on {cohort}: AUC = {res['AUC']}")

    if not results:
        logging.error("No valid results generated. Aborting.")
        return

    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Save raw results
    results_file = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df.to_csv(results_file, index=False)
    logging.info(f"Results saved to: {results_file}")

    # Generate Nature style visualizations
    generate_nature_style_visualizations(df, OUTPUT_DIR)
    
    # Generate analysis results
    effect_df = add_effect_size_analysis(df, OUTPUT_DIR)
    summary_df = generate_interpretation_summary(df, OUTPUT_DIR)
    
    # Run consensus score analysis
    consensus_results = analyze_consensus_scores(OUTPUT_DIR)

    # Print console summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    for cohort in df['Cohort'].unique():
        cohort_data = df[df['Cohort'] == cohort]
        best_method = cohort_data.loc[cohort_data['AUC'].idxmax()]
        
        print(f"\n{cohort}:")
        print(f"  Best method: {best_method['Method']} (AUC = {best_method['AUC']})")
        
        for _, row in cohort_data.iterrows():
            ml_auc = cohort_data[cohort_data['Method'] == 'ML']['AUC'].values[0]
            improvement = row['AUC'] - ml_auc
            print(f"  {row['Method']}: {row['AUC']} (95% CI: {row['CI_low']}-{row['CI_high']})")
            if row['Method'] != 'ML':
                print(f"    Improvement over ML: +{improvement:.3f}")

    # Print global statistics
    print("\n" + "="*60)
    print("GLOBAL PERFORMANCE SUMMARY")
    print("="*60)
    
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        avg_auc = method_data['AUC'].mean()
        std_auc = method_data['AUC'].std()
        print(f"{method}: Average AUC = {avg_auc:.3f} ± {std_auc:.3f}")
    
    # Best performing method overall
    best_overall = df.loc[df['AUC'].idxmax()]
    print(f"\nOverall best: {best_overall['Method']} on {best_overall['Cohort']} (AUC = {best_overall['AUC']})")
    
    # Consensus score analysis summary
    if consensus_results is not None:
        print("\n" + "="*60)
        print("CONSENSUS SCORE ANALYSIS SUMMARY")
        print("="*60)
        
        for cohort in consensus_results['Cohort'].unique():
            print(f"\n{cohort}:")
            cohort_data = consensus_results[consensus_results['Cohort'] == cohort]
            
            for method in ['KC_ML', 'ML']:
                method_data = cohort_data[cohort_data['Method'] == method]
                if not method_data.empty:
                    best_threshold = method_data.loc[method_data['AUC'].idxmax()]
                    print(f"  {method}:")
                    print(f"    Best threshold: ≥{best_threshold['Score_Threshold']}")
                    print(f"    Best AUC: {best_threshold['AUC']} (using {best_threshold['Genes_Used']} genes)")
    
    logging.info("Nature-style ablation study completed successfully!")

# ----------------------------- run -----------------------------
if __name__ == "__main__":
    main()