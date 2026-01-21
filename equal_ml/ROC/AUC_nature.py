# ==============================================================================
# External validation with Nature-style ROC figures
# ==============================================================================

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import bootstrap

# ----------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------- Configuration -----------------------------
signature_file = "for_auc_gene.csv"      #Consensus>=2

cohorts = {
    "GSE81622": {
        "expr": "81622_expression.csv",
        "labels": "81622_phenotype.csv",
        "group_col": "Group",
        "positive": "1",
        "color": "#1F77B4"  # Nature blue
    },
    "GSE61635": {
        "expr": "61635_expression.csv",
        "labels": "61635_phenotype.csv",
        "group_col": "Group",
        "positive": "1",
        "color": "#D62728"  # Nature red
    }
}

output_dir = "external_validation_results_nature"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------Read tag-----------------------------
sig_df = pd.read_csv(signature_file)
sig_df["Gene"] = sig_df["Gene"].astype(str)
sig_genes = sig_df["Gene"].tolist()

# ----------------------------- Utility function -----------------------------
def zscore_rows(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=1)

def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)

# ----------------------------- ROC Plotting function -----------------------------
def plot_nature_roc(fpr, tpr, auc_val, ci_low, ci_high, label, color, outpath):
    plt.figure(figsize=(4.5, 4.5))

    # ROC 
    plt.plot(
        fpr,
        tpr,
        lw=2.8,
        color=color
    )

    # 
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        lw=1.5,
        alpha=0.6
    )

    # 
    plt.xlim(-0.05, 1.01)
    plt.ylim(-0.05, 1.01)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], fontsize=12)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], fontsize=12)

    plt.xlabel("False positive rate", fontsize=14)
    plt.ylabel("True positive rate", fontsize=14)

    # === Core：AUC + CI  ===
    text = (
        f"AUC = {auc_val:.3f}\n"
        f"95% CI: {ci_low:.3f}–{ci_high:.3f}"
    )

    plt.text(
        0.55, 0.11,              
        text,
        fontsize=10,
        ha="left",
        va="center",
        transform=plt.gca().transAxes
    )

    # 
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()

    # 
    plt.savefig(outpath.replace(".png", ".pdf"))
    plt.savefig(outpath.replace(".png", ".svg"))
    plt.savefig(outpath, dpi=300)
    plt.close()


# ----------------------------- Verification function -----------------------------
def validate_cohort(name, cfg):
    expr = pd.read_csv(cfg["expr"], index_col=0)
    labels = pd.read_csv(cfg["labels"])

    expr.index = expr.index.astype(str)
    labels = labels.set_index("Sample")

    common_samples = expr.columns.intersection(labels.index)
    expr = expr[common_samples]
    labels = labels.loc[common_samples]

    common_genes = expr.index.intersection(sig_genes)
    sig_sub = sig_df.set_index("Gene").loc[common_genes]

    weights = sig_sub["Weight"].values if "Weight" in sig_sub.columns else np.ones(len(common_genes))
    expr_sub = zscore_rows(expr.loc[common_genes])

    score = np.dot(weights, expr_sub.values) / np.sum(np.abs(weights))
    y_true = (labels[cfg["group_col"]].astype(str) == cfg["positive"]).astype(int)

    auc_val = roc_auc_score(y_true, score)
    if auc_val < 0.5:
        score = -score
        auc_val = roc_auc_score(y_true, score)

    ci = bootstrap(
        (y_true.values, score),
        safe_auc,
        n_resamples=2000,
        confidence_level=0.95,
        method="percentile",
        random_state=123
    ).confidence_interval

    fpr, tpr, _ = roc_curve(y_true, score)

    plot_nature_roc(
        fpr,
        tpr,
        auc_val,
        ci.low,
        ci.high,
        name,
        cfg["color"],
        os.path.join(output_dir, f"ROC_{name}.png")
    )

    return fpr, tpr, auc_val, ci

# ----------------------------- Run -----------------------------
roc_cache = {}

for name, cfg in cohorts.items():
    fpr, tpr, auc_val, ci = validate_cohort(name, cfg)
    roc_cache[name] = (fpr, tpr, auc_val, ci, cfg["color"])

# ----------------------------- Combined ROC（Nature ） -----------------------------
plt.figure(figsize=(4.8, 4.8))

for name, (fpr, tpr, auc_val, ci, color) in roc_cache.items():
    plt.plot(
        fpr,
        tpr,
        lw=2.8,
        color=color,
        label=f"{name} (AUC={auc_val:.3f})"
    )

plt.plot([0, 1], [0, 1], "--", color="black", lw=1.5, alpha=0.6)

plt.xlim(-0.05, 1.01)
plt.ylim(-0.05, 1.01)
plt.xlabel("False positive rate", fontsize=14)
plt.ylabel("True positive rate", fontsize=14)
plt.legend(frameon=False, fontsize=10, loc="lower right")

for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Combined_External_ROC.pdf"))
plt.savefig(os.path.join(output_dir, "Combined_External_ROC.svg"))
plt.savefig(os.path.join(output_dir, "Combined_External_ROC.png"), dpi=300)
plt.close()

logging.info("Nature-style ROC figures generated successfully")
