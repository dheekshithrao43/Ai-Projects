"""
Generate 10 publication-ready PNG plots for the project report.
Combines 5 preprocessing/EDA plots and 5 model comparison plots.

Usage:
    python generate_report_plots.py

Output: outputs/report_plots/  (PNG files, ~150-300 dpi)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, ".")
from src.data.loader import load_job_descriptions, load_matching_labels, load_resumes
from src.data.preprocess import clean_text
from src.evaluation.metrics import LABEL_MAP, evaluate_classifier, score_to_label
from src.matching.classical import ClassicalMatcher
from src.matching.transformer import TransformerMatcher
from sklearn.metrics.pairwise import cosine_similarity as _cos

OUT = Path("outputs/report_plots")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "No Fit": "#e74c3c",
    "Potential Fit": "#f39c12",
    "Good Fit": "#2ecc71",
}
LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"]
MODEL_COLORS = {
    "Classical (TF-IDF+BM25)": "#3498db",
    "Transformer (SBERT)": "#9b59b6",
    "Hybrid (0.3+0.7)": "#2ecc71",
}

LAYOUT = dict(
    font=dict(family="Arial, sans-serif", size=13),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=60, r=40, t=70, b=60),
)


def save_png(fig: go.Figure, name: str, width: int = 1100, height: int = 600) -> None:
    path = OUT / f"{name}.png"
    fig.write_image(str(path), width=width, height=height, scale=2)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\nLoading datasets...")
resumes = load_resumes()
jds = load_job_descriptions(software_only=True)
train_df, test_df = load_matching_labels()
scores_csv = Path("outputs/training/model_scores.csv")
if scores_csv.exists():
    scores_df = pd.read_csv(scores_csv)
    print(f"  Loaded pre-computed scores: {len(scores_df):,} rows")
else:
    print("  model_scores.csv not found — computing scores on a 200-sample subset...")
    print("  Fitting Classical Matcher...")
    classical = ClassicalMatcher()
    classical.fit(jds)
    print("  Fitting Transformer Matcher (SBERT)...")
    transformer = TransformerMatcher()
    transformer.fit(jds, show_progress=True)

    eval_sample = test_df.sample(200, random_state=42).reset_index(drop=True)
    rows = []
    from tqdm import tqdm
    for _, row in tqdm(eval_sample.iterrows(), total=len(eval_sample), desc="  Scoring"):
        q  = row["resume_text"]
        jd = row["job_description_text"]

        q_emb  = transformer.encode([q])
        jd_emb = transformer.encode([jd])
        sbert_score = float((q_emb @ jd_emb.T).flatten()[0])

        q_vec       = classical.tfidf.transform([q])
        tfidf_score = float(_cos(q_vec, classical.tfidf_matrix).flatten().max())

        bm25_raw   = np.array(classical.bm25.get_scores(q.lower().split()))
        bm25_score = float(bm25_raw.max() / (bm25_raw.max() or 1))

        hybrid_score = 0.3 * tfidf_score + 0.7 * sbert_score
        rows.append({
            "true_label":       row["label"],
            "sbert_score":      sbert_score,
            "tfidf_score":      tfidf_score,
            "bm25_score":       bm25_score,
            "hybrid_score":     hybrid_score,
            "pred_classical":   score_to_label(tfidf_score),
            "pred_transformer": score_to_label(sbert_score),
            "pred_hybrid":      score_to_label(hybrid_score),
        })
    scores_df = pd.DataFrame(rows)
    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(scores_csv, index=False)
    print(f"  Scores saved → {scores_csv}")

print(
    f"  Resumes: {len(resumes):,} | JDs: {len(jds):,} | "
    f"Train labels: {len(train_df):,} | Test labels: {len(test_df):,} | "
    f"Scored samples: {len(scores_df):,}"
)

# ─────────────────────────────────────────────────────────────
# ── PLOT 1: Resume Category Distribution ────────────────────
# ─────────────────────────────────────────────────────────────
print("\n[1/10] Resume category distribution...")
cat_counts = resumes["category"].value_counts().reset_index()
cat_counts.columns = ["Category", "Count"]

fig = px.bar(
    cat_counts,
    x="Count",
    y="Category",
    orientation="h",
    title="Figure 1 — Resume Category Distribution (2,484 resumes)",
    color="Count",
    color_continuous_scale="Blues",
    labels={"Count": "Number of Resumes"},
)
fig.update_layout(
    **LAYOUT,
    height=620,
    yaxis={"autorange": "reversed"},
    coloraxis_showscale=False,
)
fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
save_png(fig, "01_resume_category_distribution", height=620)

# ─────────────────────────────────────────────────────────────
# ── PLOT 2: Resume Length + JD Length side by side ──────────
# ─────────────────────────────────────────────────────────────
print("[2/10] Text length distributions (resumes & JDs)...")
resumes["text_len"] = resumes["resume_text"].str.split().str.len()
jds["jd_len"] = jds["job_description"].str.split().str.len()

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Resume Word-Count Distribution",
        "Job Description Word-Count Distribution",
    ],
)
fig.add_trace(
    go.Histogram(
        x=resumes["text_len"],
        nbinsx=40,
        marker_color="#3498db",
        opacity=0.8,
        name="Resumes",
    ),
    row=1,
    col=1,
)
fig.add_vline(
    x=resumes["text_len"].median(),
    line_dash="dash",
    line_color="red",
    annotation_text=f"Median {int(resumes['text_len'].median())}",
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=jds["jd_len"],
        nbinsx=30,
        marker_color="#2ecc71",
        opacity=0.8,
        name="JDs",
    ),
    row=1,
    col=2,
)
fig.add_vline(
    x=jds["jd_len"].median(),
    line_dash="dash",
    line_color="red",
    annotation_text=f"Median {int(jds['jd_len'].median())}",
    row=1,
    col=2,
)
fig.update_layout(
    **LAYOUT,
    title_text="Figure 2 — Text Length Distributions: Resumes vs Job Descriptions",
    showlegend=False,
    height=420,
)
fig.update_xaxes(title_text="Word Count")
fig.update_yaxes(title_text="Count")
save_png(fig, "02_text_length_distributions", height=420)

# ─────────────────────────────────────────────────────────────
# ── PLOT 3: Label Distribution (train / test split) ──────────
# ─────────────────────────────────────────────────────────────
print("[3/10] Label distribution (train/test)...")
label_train = train_df["label"].value_counts().reset_index()
label_train.columns = ["Label", "Count"]
label_train["Split"] = "Train"
label_test = test_df["label"].value_counts().reset_index()
label_test.columns = ["Label", "Count"]
label_test["Split"] = "Test"
label_all = pd.concat([label_train, label_test])

fig = px.bar(
    label_all,
    x="Label",
    y="Count",
    color="Split",
    barmode="group",
    title="Figure 3 — Matching Label Distribution: Train vs Test Split",
    color_discrete_map={"Train": "#3498db", "Test": "#e67e22"},
    text="Count",
    category_orders={"Label": LABEL_ORDER},
)
fig.update_traces(textposition="outside")
fig.update_layout(**LAYOUT, height=420, yaxis_range=[0, label_all["Count"].max() * 1.18])
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
save_png(fig, "03_label_distribution", height=420)

# ─────────────────────────────────────────────────────────────
# ── PLOT 4: Resume Length by Category (box) ──────────────────
# ─────────────────────────────────────────────────────────────
print("[4/10] Resume length by category (box plot)...")
fig = px.box(
    resumes,
    x="text_len",
    y="category",
    title="Figure 4 — Resume Length by Category",
    color="category",
    labels={"text_len": "Word Count", "category": ""},
)
fig.update_layout(
    **LAYOUT,
    height=620,
    showlegend=False,
    yaxis={"autorange": "reversed"},
)
fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
save_png(fig, "04_resume_length_by_category", height=620)

# ─────────────────────────────────────────────────────────────
# ── PLOT 5: CV Length vs JD Length scatter by label ──────────
# ─────────────────────────────────────────────────────────────
print("[5/10] CV length vs JD length scatter...")
train_df["cv_len"] = train_df["resume_text"].str.split().str.len()
train_df["jd_len2"] = train_df["job_description_text"].str.split().str.len()
sample = train_df.sample(500, random_state=42)

fig = px.scatter(
    sample,
    x="cv_len",
    y="jd_len2",
    color="label",
    color_discrete_map=COLORS,
    title="Figure 5 — CV Word Count vs JD Word Count by Fit Label (500-sample)",
    labels={"cv_len": "CV Word Count", "jd_len2": "JD Word Count", "label": "Label"},
    opacity=0.55,
    category_orders={"label": LABEL_ORDER},
)
fig.update_layout(**LAYOUT, height=450)
fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
save_png(fig, "05_cv_vs_jd_length_scatter", height=450)

# ─────────────────────────────────────────────────────────────
# ── PLOT 6: Model Comparison — key metrics ───────────────────
# ─────────────────────────────────────────────────────────────
print("[6/10] Model performance comparison...")

metrics_classical = evaluate_classifier(
    scores_df["true_label"].tolist(), scores_df["pred_classical"].tolist()
)
metrics_transformer = evaluate_classifier(
    scores_df["true_label"].tolist(), scores_df["pred_transformer"].tolist()
)
metrics_hybrid = evaluate_classifier(
    scores_df["true_label"].tolist(), scores_df["pred_hybrid"].tolist()
)

metric_keys = ["f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
metric_labels = {
    "f1_macro": "F1 (Macro)",
    "f1_weighted": "F1 (Weighted)",
    "precision_macro": "Precision (Macro)",
    "recall_macro": "Recall (Macro)",
}
metric_rows = []
for k in metric_keys:
    metric_rows += [
        {
            "Metric": metric_labels[k],
            "Model": "Classical (TF-IDF+BM25)",
            "Score": metrics_classical[k],
        },
        {
            "Metric": metric_labels[k],
            "Model": "Transformer (SBERT)",
            "Score": metrics_transformer[k],
        },
        {
            "Metric": metric_labels[k],
            "Model": "Hybrid (0.3+0.7)",
            "Score": metrics_hybrid[k],
        },
    ]
mdf = pd.DataFrame(metric_rows)

fig = px.bar(
    mdf,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    title="Figure 6 — Model Performance: Classical vs Transformer vs Hybrid",
    color_discrete_map=MODEL_COLORS,
    text=mdf["Score"].apply(lambda x: f"{x:.3f}"),
)
fig.update_traces(textposition="outside")
fig.update_layout(
    **LAYOUT,
    height=480,
    yaxis_range=[0, 1.08],
    uniformtext_minsize=9,
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", title_text="Score")
save_png(fig, "06_model_comparison", height=480)

# ─────────────────────────────────────────────────────────────
# ── PLOT 7: Confusion Matrices (3 models) ────────────────────
# ─────────────────────────────────────────────────────────────
print("[7/10] Confusion matrices...")
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Classical (TF-IDF+BM25)", "Transformer (SBERT)", "Hybrid (0.3+0.7)"],
    horizontal_spacing=0.08,
)
for col_i, pred_col in enumerate(
    ["pred_classical", "pred_transformer", "pred_hybrid"]
):
    y_true = [LABEL_MAP.get(l, 0) for l in scores_df["true_label"]]
    y_pred = [LABEL_MAP.get(l, 0) for l in scores_df[pred_col]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # row-normalise for readability
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text_vals = [[f"{v:.2f}" for v in row] for row in cm_norm]
    fig.add_trace(
        go.Heatmap(
            z=cm_norm,
            x=LABEL_ORDER,
            y=LABEL_ORDER,
            colorscale="Blues",
            text=text_vals,
            texttemplate="%{text}",
            showscale=col_i == 2,
            colorbar=dict(title="Proportion") if col_i == 2 else None,
            name=pred_col,
            zmin=0,
            zmax=1,
        ),
        row=1,
        col=col_i + 1,
    )
fig.update_layout(
    **LAYOUT,
    height=420,
    title_text="Figure 7 — Normalised Confusion Matrices: All Three Models",
)
fig.update_xaxes(title_text="Predicted")
fig.update_yaxes(title_text="True", autorange="reversed")
save_png(fig, "07_confusion_matrices", height=420)

# ─────────────────────────────────────────────────────────────
# ── PLOT 8: F1 / Precision / Recall per class ────────────────
# ─────────────────────────────────────────────────────────────
print("[8/10] Per-class F1 / Precision / Recall...")
pr_rows = []
for model_name, pred_col in [
    ("Classical", "pred_classical"),
    ("Transformer", "pred_transformer"),
    ("Hybrid", "pred_hybrid"),
]:
    y_true = [LABEL_MAP.get(l, 0) for l in scores_df["true_label"]]
    y_pred = [LABEL_MAP.get(l, 0) for l in scores_df[pred_col]]
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    for idx, cls in enumerate(LABEL_ORDER):
        pr_rows.append(
            {
                "Model": model_name,
                "Class": cls,
                "Precision": round(p[idx], 4),
                "Recall": round(r[idx], 4),
                "F1": round(f[idx], 4),
            }
        )

pr_df = pd.DataFrame(pr_rows)
pr_melt = pr_df.melt(
    id_vars=["Model", "Class"], value_vars=["Precision", "Recall", "F1"], var_name="Metric", value_name="Score"
)

fig = px.bar(
    pr_melt,
    x="Class",
    y="Score",
    color="Model",
    facet_col="Metric",
    barmode="group",
    title="Figure 8 — Precision / Recall / F1 per Class for All Models",
    color_discrete_map={
        "Classical": "#3498db",
        "Transformer": "#9b59b6",
        "Hybrid": "#2ecc71",
    },
    category_orders={"Class": LABEL_ORDER, "Metric": ["Precision", "Recall", "F1"]},
    text=pr_melt["Score"].apply(lambda x: f"{x:.2f}"),
)
fig.update_traces(textposition="outside", textfont_size=9)
fig.update_layout(**LAYOUT, height=450, yaxis_range=[0, 1.18])
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
save_png(fig, "08_per_class_metrics", height=450)

# ─────────────────────────────────────────────────────────────
# ── PLOT 9: Score Distributions by label (all 3 models) ──────
# ─────────────────────────────────────────────────────────────
print("[9/10] Score distributions by true label...")
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Classical (TF-IDF)",
        "Transformer (SBERT)",
        "Hybrid",
    ],
    shared_yaxes=False,
)
for col_i, (score_col, _) in enumerate(
    [
        ("tfidf_score", "Classical"),
        ("sbert_score", "Transformer"),
        ("hybrid_score", "Hybrid"),
    ]
):
    for label, color in COLORS.items():
        sub = scores_df[scores_df["true_label"] == label]
        fig.add_trace(
            go.Histogram(
                x=sub[score_col],
                name=label,
                marker_color=color,
                opacity=0.65,
                nbinsx=20,
                legendgroup=label,
                showlegend=(col_i == 0),
            ),
            row=1,
            col=col_i + 1,
        )
fig.update_layout(
    **LAYOUT,
    height=420,
    barmode="overlay",
    title_text="Figure 9 — Score Distributions by True Label (All Three Models)",
)
fig.update_xaxes(title_text="Score", showgrid=True, gridcolor="#eeeeee")
fig.update_yaxes(title_text="Count", showgrid=True, gridcolor="#eeeeee")
save_png(fig, "09_score_distributions", height=420)

# ─────────────────────────────────────────────────────────────
# ── PLOT 10: Precision-Recall Curves ─────────────────────────
# ─────────────────────────────────────────────────────────────
print("[10/10] Precision-Recall curves...")
y_true_bin = label_binarize(
    [LABEL_MAP.get(l, 0) for l in scores_df["true_label"]], classes=[0, 1, 2]
)
color_map = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
dash_map = {"Hybrid": "solid", "SBERT": "dash", "TF-IDF": "dot"}

fig = go.Figure()
for cls_idx, cls_name in enumerate(LABEL_ORDER):
    y_bin = y_true_bin[:, cls_idx]
    for score_col, model_name in [
        ("hybrid_score", "Hybrid"),
        ("sbert_score", "SBERT"),
        ("tfidf_score", "TF-IDF"),
    ]:
        prec, rec, _ = precision_recall_curve(y_bin, scores_df[score_col])
        ap = average_precision_score(y_bin, scores_df[score_col])
        fig.add_trace(
            go.Scatter(
                x=rec,
                y=prec,
                mode="lines",
                name=f"{cls_name} / {model_name} (AP={ap:.2f})",
                line=dict(color=color_map[cls_idx], dash=dash_map[model_name], width=2),
            )
        )
fig.update_layout(
    **LAYOUT,
    title="Figure 10 — Precision-Recall Curves per Class and Model",
    xaxis_title="Recall",
    yaxis_title="Precision",
    height=500,
    legend=dict(x=1.01, y=1, xanchor="left"),
)
fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", range=[0, 1])
fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", range=[0, 1.05])
save_png(fig, "10_precision_recall_curves", width=1200, height=500)

# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("Done — 10 PNG plots saved to:", OUT.resolve())
print("=" * 55)
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name:45}  {f.stat().st_size/1024:.0f} KB")
