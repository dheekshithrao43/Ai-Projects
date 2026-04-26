"""
Training pipeline for the AI Career Recommendation System.

Runs data exploration, fits the matching models, evaluates performance,
and saves all preprocessing + model performance plots to outputs/training/.

Usage:
    python train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm import tqdm

sys.path.insert(0, ".")

from src.data.loader import (
    load_job_descriptions,
    load_matching_labels,
    load_resumes,
)
from src.data.preprocess import clean_text, parse_skills_string
from src.evaluation.metrics import (
    LABEL_MAP,
    evaluate_classifier,
    evaluate_on_dataset,
    score_to_label,
)
from src.extraction.rule_based import extract_skills, extract_skills_from_section
from src.gap_analysis.analyzer import analyze_gap
from src.matching.classical import ClassicalMatcher
from src.matching.hybrid import HybridMatcher
from src.matching.transformer import TransformerMatcher
from src.normalisation.normaliser import normalise_skills

OUT = Path("outputs/training")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "No Fit":       "#e74c3c",
    "Potential Fit":"#f39c12",
    "Good Fit":     "#2ecc71",
}


def save(fig: go.Figure, name: str) -> None:
    path = OUT / f"{name}.html"
    fig.write_html(str(path))
    print(f"  [saved] {path}")


# ============================================================
# PHASE 1 — DATA EXPLORATION & PREPROCESSING PLOTS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1 — Data Exploration & Preprocessing")
print("=" * 60)

resumes = load_resumes()
jds     = load_job_descriptions(software_only=True)
train_df, test_df = load_matching_labels()

print(f"  Resumes : {len(resumes):,}  |  unique categories: {resumes['category'].nunique()}")
print(f"  JDs     : {len(jds):,}  |  unique roles: {jds['role'].nunique()}")
print(f"  Labels  : train={len(train_df):,}  test={len(test_df):,}")

# ── Plot 1: Resume category distribution ────────────────────────────────────
cat_counts = resumes["category"].value_counts().reset_index()
cat_counts.columns = ["Category", "Count"]
fig = px.bar(
    cat_counts, x="Count", y="Category", orientation="h",
    title="Resume Category Distribution (2,484 resumes)",
    color="Count", color_continuous_scale="Blues",
    labels={"Count": "Number of Resumes"},
)
fig.update_layout(height=600, yaxis={"autorange": "reversed"},
                  coloraxis_showscale=False)
save(fig, "01_resume_category_distribution")

# ── Plot 2: Resume text length distribution ──────────────────────────────────
resumes["text_len"] = resumes["resume_text"].str.split().str.len()
fig = px.histogram(
    resumes, x="text_len", nbins=40, color_discrete_sequence=["#3498db"],
    title="Resume Length Distribution (word count)",
    labels={"text_len": "Word Count"},
)
fig.add_vline(x=resumes["text_len"].median(), line_dash="dash", line_color="red",
              annotation_text=f"Median: {int(resumes['text_len'].median())} words")
fig.update_layout(height=380)
save(fig, "02_resume_length_distribution")

# ── Plot 3: Resume word count by category (box plot) ────────────────────────
fig = px.box(
    resumes, x="text_len", y="category",
    title="Resume Length by Category",
    color="category",
    labels={"text_len": "Word Count", "category": ""},
)
fig.update_layout(height=600, showlegend=False, yaxis={"autorange": "reversed"})
save(fig, "03_resume_length_by_category")

# ── Plot 4: Job role distribution ─────────────────────────────────────────────
role_counts = jds["role"].value_counts().head(30).reset_index()
role_counts.columns = ["Role", "Count"]
fig = px.bar(
    role_counts, x="Count", y="Role", orientation="h",
    title="Top 30 Software Job Roles",
    color="Count", color_continuous_scale="Greens",
)
fig.update_layout(height=700, yaxis={"autorange": "reversed"},
                  coloraxis_showscale=False)
save(fig, "04_job_role_distribution")

# ── Plot 5: JD text length distribution ──────────────────────────────────────
jds["jd_len"] = jds["job_description"].str.split().str.len()
fig = px.histogram(
    jds, x="jd_len", nbins=30, color_discrete_sequence=["#2ecc71"],
    title="Job Description Length Distribution (word count)",
    labels={"jd_len": "Word Count"},
)
fig.add_vline(x=jds["jd_len"].median(), line_dash="dash", line_color="red",
              annotation_text=f"Median: {int(jds['jd_len'].median())} words")
fig.update_layout(height=380)
save(fig, "05_jd_length_distribution")

# ── Plot 6: Matching labels distribution ─────────────────────────────────────
label_train = train_df["label"].value_counts().reset_index()
label_train.columns  = ["Label", "Count"]
label_train["Split"] = "Train"
label_test  = test_df["label"].value_counts().reset_index()
label_test.columns   = ["Label", "Count"]
label_test["Split"]  = "Test"
label_all = pd.concat([label_train, label_test])

fig = px.bar(
    label_all, x="Label", y="Count", color="Split", barmode="group",
    title="Matching Labels Distribution (Train vs Test)",
    color_discrete_map={"Train": "#3498db", "Test": "#e67e22"},
)
fig.update_layout(height=380)
save(fig, "06_label_distribution")

# ── Plot 7: Top skills in resumes ─────────────────────────────────────────────
print("\n  Extracting skills from 200 sample resumes...")
sample_resumes = resumes.sample(200, random_state=42)
skill_freq: dict[str, int] = {}
for _, row in tqdm(sample_resumes.iterrows(), total=200, desc="  Skill extraction"):
    skills = extract_skills_from_section(clean_text(row["resume_text"]))
    for s in normalise_skills(skills):
        skill_freq[s["canonical"]] = skill_freq.get(s["canonical"], 0) + 1

skill_df = (
    pd.DataFrame(list(skill_freq.items()), columns=["Skill", "Frequency"])
    .sort_values("Frequency", ascending=False)
    .head(30)
)
fig = px.bar(
    skill_df, x="Frequency", y="Skill", orientation="h",
    title="Top 30 Skills Mentioned in Resumes (200-resume sample)",
    color="Frequency", color_continuous_scale="Oranges",
)
fig.update_layout(height=650, yaxis={"autorange": "reversed"},
                  coloraxis_showscale=False)
save(fig, "07_top_skills_in_resumes")

# ── Plot 8: Top skills in job descriptions ────────────────────────────────────
print("  Extracting skills from all job descriptions...")
jd_skill_freq: dict[str, int] = {}
for _, row in tqdm(jds.iterrows(), total=len(jds), desc="  JD skill extraction"):
    raw_text = str(row["job_description"]) + " " + str(row["skills"])
    skills = extract_skills(clean_text(raw_text))
    for s in normalise_skills(skills):
        jd_skill_freq[s["canonical"]] = jd_skill_freq.get(s["canonical"], 0) + 1

jd_skill_df = (
    pd.DataFrame(list(jd_skill_freq.items()), columns=["Skill", "Frequency"])
    .sort_values("Frequency", ascending=False)
    .head(30)
)
fig = px.bar(
    jd_skill_df, x="Frequency", y="Skill", orientation="h",
    title="Top 30 Skills Required in Job Descriptions",
    color="Frequency", color_continuous_scale="Purples",
)
fig.update_layout(height=650, yaxis={"autorange": "reversed"},
                  coloraxis_showscale=False)
save(fig, "08_top_skills_in_jds")

# ── Plot 9: Skills overlap (resumes vs JDs) ───────────────────────────────────
top_resume_skills = set(skill_df["Skill"].tolist())
top_jd_skills     = set(jd_skill_df["Skill"].tolist())
overlap   = top_resume_skills & top_jd_skills
only_cv   = top_resume_skills - top_jd_skills
only_jd   = top_jd_skills     - top_resume_skills

venn_df = pd.DataFrame([
    {"Category": "In Both (overlap)",      "Count": len(overlap)},
    {"Category": "Only in Resumes",        "Count": len(only_cv)},
    {"Category": "Only in Job Descriptions","Count": len(only_jd)},
])
fig = px.pie(venn_df, values="Count", names="Category",
             title="Top-30 Skill Overlap: Resumes vs Job Descriptions",
             color_discrete_sequence=["#2ecc71", "#3498db", "#e74c3c"])
fig.update_layout(height=380)
save(fig, "09_skill_overlap_venn")

# ── Plot 10: Matching label pair text length ──────────────────────────────────
train_df["cv_len"]  = train_df["resume_text"].str.split().str.len()
train_df["jd_len2"] = train_df["job_description_text"].str.split().str.len()
fig = px.scatter(
    train_df.sample(500, random_state=42),
    x="cv_len", y="jd_len2", color="label",
    color_discrete_map=COLORS,
    title="CV Length vs JD Length by Fit Label (500-sample)",
    labels={"cv_len": "CV Word Count", "jd_len2": "JD Word Count", "label": "Label"},
    opacity=0.6,
)
fig.update_layout(height=420)
save(fig, "10_cv_vs_jd_length_scatter")

print(f"\n  [Phase 1 complete] 10 preprocessing plots saved.\n")


# ============================================================
# PHASE 2 — MODEL FITTING
# ============================================================
print("=" * 60)
print("PHASE 2 — Model Fitting")
print("=" * 60)

print("  Fitting Classical Matcher (TF-IDF + BM25)...")
classical = ClassicalMatcher()
classical.fit(jds)
print("  Classical matcher ready.")

print("  Fitting Transformer Matcher (SBERT)...")
transformer = TransformerMatcher()
transformer.fit(jds, show_progress=True)
print("  Transformer matcher ready.")

print("  Assembling Hybrid Matcher (0.3 classical + 0.7 transformer)...")
hybrid = HybridMatcher(classical_weight=0.3, transformer_weight=0.7)
hybrid.classical   = classical
hybrid.transformer = transformer
hybrid.corpus_df   = jds.reset_index(drop=True)
hybrid._fitted     = True
print("  Hybrid matcher ready.")

MODEL_DIR = Path("models/hybrid_matcher")
hybrid.save(MODEL_DIR)
print(f"  Models saved to {MODEL_DIR}/\n")


# ============================================================
# PHASE 3 — MODEL EVALUATION & PERFORMANCE PLOTS
# ============================================================
print("=" * 60)
print("PHASE 3 — Evaluation & Performance Plots")
print("=" * 60)

eval_sample = test_df.sample(300, random_state=42).reset_index(drop=True)
print(f"  Evaluating on {len(eval_sample)} test samples…")

# Score all samples with each matcher
rows = []
for _, row in tqdm(eval_sample.iterrows(), total=len(eval_sample), desc="  Scoring"):
    q  = row["resume_text"]
    jd = row["job_description_text"]
    true_label = row["label"]

    q_emb  = hybrid.transformer.encode([q])
    jd_emb = hybrid.transformer.encode([jd])
    sbert_score = float((q_emb @ jd_emb.T).flatten()[0])

    q_vec  = classical.tfidf.transform([q])
    import numpy as _np
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    tfidf_score = float(_cos(q_vec, classical.tfidf_matrix).flatten().max())

    bm25_raw   = _np.array(classical.bm25.get_scores(q.lower().split()))
    bm25_score = float(bm25_raw.max() / (bm25_raw.max() or 1))

    hybrid_score = 0.3 * tfidf_score + 0.7 * sbert_score

    rows.append({
        "true_label":    true_label,
        "sbert_score":   sbert_score,
        "tfidf_score":   tfidf_score,
        "bm25_score":    bm25_score,
        "hybrid_score":  hybrid_score,
        "pred_classical":  score_to_label(tfidf_score),
        "pred_transformer":score_to_label(sbert_score),
        "pred_hybrid":     score_to_label(hybrid_score),
    })

scores_df = pd.DataFrame(rows)
scores_df.to_csv(OUT / "model_scores.csv", index=False)
print(f"  Scores saved → {OUT}/model_scores.csv")

# Compute metrics for each matcher
metrics_classical   = evaluate_classifier(scores_df["true_label"].tolist(), scores_df["pred_classical"].tolist())
metrics_transformer = evaluate_classifier(scores_df["true_label"].tolist(), scores_df["pred_transformer"].tolist())
metrics_hybrid      = evaluate_classifier(scores_df["true_label"].tolist(), scores_df["pred_hybrid"].tolist())

print("\n  === Metric Comparison ===")
print(f"  {'Metric':25} {'Classical':>12} {'Transformer':>14} {'Hybrid':>10}")
print("  " + "-"*65)
for k in ("f1_macro","f1_weighted","precision_macro","recall_macro"):
    print(f"  {k:25} {metrics_classical[k]:>12.4f} {metrics_transformer[k]:>14.4f} {metrics_hybrid[k]:>10.4f}")

# ── Plot 11: Metric comparison bar chart ──────────────────────────────────────
metric_keys = ["f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
metric_rows = []
for k in metric_keys:
    metric_rows += [
        {"Metric": k, "Model": "Classical (TF-IDF+BM25)", "Score": metrics_classical[k]},
        {"Metric": k, "Model": "Transformer (SBERT)",     "Score": metrics_transformer[k]},
        {"Metric": k, "Model": "Hybrid (0.3+0.7)",        "Score": metrics_hybrid[k]},
    ]
metric_plot_df = pd.DataFrame(metric_rows)
fig = px.bar(
    metric_plot_df, x="Metric", y="Score", color="Model", barmode="group",
    title="Model Performance Comparison — Classical vs Transformer vs Hybrid",
    color_discrete_map={
        "Classical (TF-IDF+BM25)": "#3498db",
        "Transformer (SBERT)":     "#9b59b6",
        "Hybrid (0.3+0.7)":        "#2ecc71",
    },
    text=metric_plot_df["Score"].apply(lambda x: f"{x:.3f}"),
)
fig.update_traces(textposition="outside")
fig.update_layout(height=450, yaxis_range=[0, 1], uniformtext_minsize=9)
save(fig, "11_model_comparison")

# ── Plot 12: Score distributions for each model ───────────────────────────────
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=["Classical (TF-IDF)", "Transformer (SBERT)", "Hybrid"])
for col_i, (score_col, name) in enumerate(
    [("tfidf_score","Classical"), ("sbert_score","Transformer"), ("hybrid_score","Hybrid")]
):
    for label, color in COLORS.items():
        sub = scores_df[scores_df["true_label"] == label]
        fig.add_trace(
            go.Histogram(x=sub[score_col], name=label, marker_color=color,
                         opacity=0.6, nbinsx=20, legendgroup=label,
                         showlegend=(col_i == 0)),
            row=1, col=col_i + 1,
        )
fig.update_layout(height=400, barmode="overlay",
                  title_text="Score Distributions by True Label — All Three Models")
save(fig, "12_score_distributions")

# ── Plot 13: Confusion matrices (3 models) ────────────────────────────────────
label_order = ["No Fit", "Potential Fit", "Good Fit"]
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=["Classical", "Transformer", "Hybrid"])
for col_i, pred_col in enumerate(["pred_classical", "pred_transformer", "pred_hybrid"]):
    y_true = [LABEL_MAP.get(l, 0) for l in scores_df["true_label"]]
    y_pred = [LABEL_MAP.get(l, 0) for l in scores_df[pred_col]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig.add_trace(
        go.Heatmap(z=cm, x=label_order, y=label_order, colorscale="Blues",
                   text=cm, texttemplate="%{text}", showscale=False,
                   name=pred_col),
        row=1, col=col_i + 1,
    )
fig.update_layout(height=380, title_text="Confusion Matrices — Classical vs Transformer vs Hybrid")
save(fig, "13_confusion_matrices")

# ── Plot 14: Precision / Recall / F1 per class ───────────────────────────────
from sklearn.metrics import precision_recall_fscore_support

pr_rows = []
for model_name, pred_col in [("Classical","pred_classical"),
                              ("Transformer","pred_transformer"),
                              ("Hybrid","pred_hybrid")]:
    y_true = [LABEL_MAP.get(l,0) for l in scores_df["true_label"]]
    y_pred = [LABEL_MAP.get(l,0) for l in scores_df[pred_col]]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
    for idx, cls in enumerate(label_order):
        pr_rows.append({"Model":model_name,"Class":cls,"Precision":p[idx],"Recall":r[idx],"F1":f[idx]})

pr_df = pd.DataFrame(pr_rows)
fig = px.bar(pr_df, x="Class", y="F1", color="Model", barmode="group",
             facet_col=None,
             title="F1 Score per Class — All Models",
             color_discrete_map={"Classical":"#3498db","Transformer":"#9b59b6","Hybrid":"#2ecc71"},
             text=pr_df["F1"].apply(lambda x: f"{x:.2f}"))
fig.update_traces(textposition="outside")
fig.update_layout(height=420, yaxis_range=[0, 1])
save(fig, "14_f1_per_class")

# ── Plot 15: Score threshold sweep — F1 vs threshold ──────────────────────────
thresholds_good = np.arange(0.3, 0.9, 0.05)
thresholds_potential = np.arange(0.2, 0.7, 0.05)
sweep_rows = []
for tg in thresholds_good:
    for tp in thresholds_potential:
        if tp >= tg:
            continue
        preds = scores_df["hybrid_score"].apply(
            lambda s: "Good Fit" if s >= tg else ("Potential Fit" if s >= tp else "No Fit")
        ).tolist()
        m = evaluate_classifier(scores_df["true_label"].tolist(), preds)
        sweep_rows.append({"tg": round(tg,2), "tp": round(tp,2), "f1_macro": m["f1_macro"]})

sweep_df = pd.DataFrame(sweep_rows)
best_thresh = sweep_df.loc[sweep_df["f1_macro"].idxmax()]
print(f"\n  Best threshold — Good≥{best_thresh['tg']}  Potential≥{best_thresh['tp']}  F1={best_thresh['f1_macro']:.4f}")

fig = px.scatter(
    sweep_df, x="tg", y="tp", color="f1_macro",
    color_continuous_scale="Viridis",
    title="F1 Score vs Classification Thresholds (Good Fit / Potential Fit)",
    labels={"tg": "Good Fit threshold", "tp": "Potential Fit threshold", "f1_macro": "F1 macro"},
)
fig.add_scatter(x=[best_thresh["tg"]], y=[best_thresh["tp"]],
                mode="markers", marker=dict(size=14, color="red", symbol="star"),
                name="Best threshold")
fig.update_layout(height=450)
save(fig, "15_threshold_sweep")

# ── Plot 16: SBERT embedding similarity — labelled pairs ─────────────────────
fig = px.box(
    scores_df, x="true_label", y="sbert_score", color="true_label",
    color_discrete_map=COLORS,
    title="SBERT Score Distribution by True Label (Box Plot)",
    labels={"true_label": "Label", "sbert_score": "SBERT Cosine Score"},
    points="all",
)
fig.update_layout(height=420, showlegend=False)
save(fig, "16_sbert_score_boxplot")

# ── Plot 17: Hybrid score distribution violin ─────────────────────────────────
fig = px.violin(
    scores_df, x="true_label", y="hybrid_score", color="true_label",
    color_discrete_map=COLORS, box=True, points="outliers",
    title="Hybrid Score Distribution by True Label (Violin Plot)",
    labels={"true_label": "Label", "hybrid_score": "Hybrid Score"},
)
fig.update_layout(height=420, showlegend=False)
save(fig, "17_hybrid_score_violin")

# ── Plot 18: Per-class Precision-Recall curve ─────────────────────────────────
from sklearn.preprocessing import label_binarize

y_true_bin = label_binarize([LABEL_MAP.get(l,0) for l in scores_df["true_label"]], classes=[0,1,2])
score_matrix = scores_df[["tfidf_score","sbert_score","hybrid_score"]].values

fig = go.Figure()
color_map = {0:"#e74c3c",1:"#f39c12",2:"#2ecc71"}
for cls_idx, cls_name in enumerate(label_order):
    y_bin = y_true_bin[:, cls_idx]
    for score_col, model_name, dash in [
        ("hybrid_score","Hybrid","solid"),
        ("sbert_score","SBERT","dash"),
        ("tfidf_score","TF-IDF","dot"),
    ]:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        prec, rec, _ = precision_recall_curve(y_bin, scores_df[score_col])
        ap = average_precision_score(y_bin, scores_df[score_col])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{cls_name} / {model_name} (AP={ap:.2f})",
            line=dict(color=color_map[cls_idx], dash=dash),
        ))
fig.update_layout(
    title="Precision-Recall Curves per Class and Model",
    xaxis_title="Recall", yaxis_title="Precision",
    height=500,
)
save(fig, "18_precision_recall_curves")

# ── Summary report ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TRAINING COMPLETE — Summary")
print("=" * 60)
print(f"\n  Model performance on {len(eval_sample)} test samples:\n")
print(f"  {'':25} {'Classical':>12} {'Transformer':>14} {'Hybrid':>10}")
print("  " + "-"*65)
for k in ("f1_macro","f1_weighted","precision_macro","recall_macro","precision_weighted","recall_weighted"):
    print(f"  {k:25} {metrics_classical[k]:>12.4f} {metrics_transformer[k]:>14.4f} {metrics_hybrid[k]:>10.4f}")

print()
print(f"  All charts saved to: {OUT.resolve()}")
print()
files = sorted(OUT.iterdir())
for f in files:
    print(f"    {f.name:55} {f.stat().st_size/1024:.0f} KB")
