"""
AI Career Recommendation System — Streamlit Web App
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_job_descriptions, load_matching_labels
from src.data.preprocess import clean_text, parse_skills_string
from src.extraction.rule_based import extract_skills, extract_skills_from_section
from src.gap_analysis.analyzer import analyze_gap, format_gap_summary, prioritise_upskilling
from src.matching.hybrid import HybridMatcher
from src.normalisation.normaliser import normalise_skills

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="💼",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Demo CVs
# ---------------------------------------------------------------------------
DEMO_CVS = {
    "Alice – Python Backend Engineer": """
Senior Backend Engineer with 6 years of experience building scalable web services.
Skills: Python, Django, FastAPI, PostgreSQL, AWS, Docker, Redis, REST API, Git, CI/CD, microservices.
Experience deploying containerised applications on AWS ECS and managing PostgreSQL databases at scale.
""".strip(),
    "Bob – Machine Learning Engineer": """
Machine Learning Engineer with 4 years of experience in NLP and deep learning.
Skills: Python, PyTorch, TensorFlow, scikit-learn, MLflow, SQL, AWS, Docker, NLP, deep learning, Hugging Face.
Built and deployed transformer-based text classification models and recommendation systems.
""".strip(),
    "Carol – Frontend Developer": """
Frontend Developer with 3 years building modern single-page applications.
Skills: React, TypeScript, JavaScript, HTML, CSS, Webpack, Redux, Node.js, Figma, Jest, Git.
Strong experience with React hooks, state management with Redux, and responsive design.
""".strip(),
    "Dave – DevOps / SRE Engineer": """
DevOps and Site Reliability Engineer with 5 years of cloud infrastructure experience.
Skills: AWS, Kubernetes, Docker, Terraform, Jenkins, CI/CD, Linux, Bash, Prometheus, Grafana, Ansible, Python.
Managed Kubernetes clusters serving millions of requests per day and built full CI/CD pipelines.
""".strip(),
}

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading job database and AI models…")
def load_matcher() -> tuple[HybridMatcher, pd.DataFrame]:
    jd_df = load_job_descriptions(software_only=True)
    matcher = HybridMatcher(classical_weight=0.3, transformer_weight=0.7)
    matcher.fit(jd_df)
    return matcher, jd_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def process_cv(text: str) -> dict:
    cleaned = clean_text(text)
    raw = extract_skills_from_section(cleaned)
    normed = normalise_skills(raw)
    return {"cleaned": cleaned, "skills": [s["canonical"] for s in normed]}


def fit_label(score: float) -> str:
    if score >= 0.65:
        return "🟢 Good Fit"
    elif score >= 0.40:
        return "🟡 Potential Fit"
    return "🔴 Low Fit"


def fit_color(score: float) -> str:
    if score >= 0.65:
        return "#2ecc71"
    elif score >= 0.40:
        return "#f39c12"
    return "#e74c3c"


def make_score_gauge(score: float, title: str = "Match Score") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 32}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": fit_color(score)},
            "steps": [
                {"range": [0, 40],  "color": "#fde8e8"},
                {"range": [40, 65], "color": "#fef3cd"},
                {"range": [65, 100],"color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": score * 100,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def make_gap_bar(gap: dict) -> go.Figure:
    categories = (
        [m["job_skill"] for m in gap["matched"]] +
        [p["job_skill"] for p in gap["partial"]] +
        gap["missing"]
    )
    colors = (
        ["#2ecc71"] * len(gap["matched"]) +
        ["#f39c12"] * len(gap["partial"]) +
        ["#e74c3c"] * len(gap["missing"])
    )
    status = (
        ["Matched"] * len(gap["matched"]) +
        ["Partial"] * len(gap["partial"]) +
        ["Missing"] * len(gap["missing"])
    )
    scores_vals = (
        [s["score"] for s in gap["matched"]] +
        [s["score"] for s in gap["partial"]] +
        [0] * len(gap["missing"])
    )

    fig = px.bar(
        x=categories, y=scores_vals,
        color=status,
        color_discrete_map={"Matched": "#2ecc71", "Partial": "#f39c12", "Missing": "#e74c3c"},
        labels={"x": "Skill", "y": "Match score", "color": "Status"},
        title="Skill Gap Analysis",
    )
    fig.update_layout(height=300, margin=dict(t=50, b=60, l=20, r=20), xaxis_tickangle=-30)
    return fig


def make_match_bar(results: pd.DataFrame, score_col: str) -> go.Figure:
    fig = px.bar(
        results,
        x=score_col,
        y="role",
        orientation="h",
        color=score_col,
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        range_color=[0, 1],
        labels={score_col: "Hybrid Score", "role": "Job Role"},
        title="Top Matching Job Roles",
        text=results[score_col].apply(lambda x: f"{x:.0%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, margin=dict(t=50, b=20, l=160, r=60),
                      coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    return fig


def make_skills_radar(cv_skills: list[str], top_job_skills: list[str]) -> go.Figure:
    all_skills = list(set(cv_skills[:8] + top_job_skills[:8]))[:12]
    cv_vals  = [1 if s in cv_skills else 0 for s in all_skills]
    jd_vals  = [1 if s in top_job_skills else 0 for s in all_skills]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cv_vals + [cv_vals[0]], theta=all_skills + [all_skills[0]],
                                   fill="toself", name="Your Skills", line_color="#3498db"))
    fig.add_trace(go.Scatterpolar(r=jd_vals + [jd_vals[0]], theta=all_skills + [all_skills[0]],
                                   fill="toself", name="Job Requires", line_color="#e74c3c", opacity=0.5))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        title="Skills Overlap Radar",
        height=350,
        margin=dict(t=60, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def run_match(cv_text: str, matcher: HybridMatcher, top_k: int) -> tuple[dict, pd.DataFrame]:
    cv_data = process_cv(cv_text)
    results = matcher.rank(cv_text, top_k=top_k)
    return cv_data, results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
top_k = st.sidebar.slider("Top K results", 3, 15, 8)
show_breakdown = st.sidebar.checkbox("Show score breakdown", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**CSV formats**\n\n"
                    "CV CSV: `resume_text`, `name` (optional)\n\n"
                    "JD CSV: `job_title`, `job_description`, `skills` (optional)")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💼 AI Career Recommendation System")
st.caption("NLP-powered CV ↔ Job matching • Skill-gap analysis • Batch processing")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_demo, tab_match, tab_batch, tab_eval = st.tabs(
    ["🎯 Live Demo", "🔍 Match Your CV", "📂 Batch Matching", "📊 Evaluation"]
)

# ===========================================================================
# TAB 0 — LIVE DEMO (auto-runs on page load)
# ===========================================================================
with tab_demo:
    st.subheader("Live Demo — Pre-loaded Candidate Profiles")
    st.markdown("Select a sample candidate below to instantly see matching results, "
                "skill-gap charts, and recommendations.")

    matcher, jd_df = load_matcher()

    selected_name = st.selectbox("Choose a demo candidate", list(DEMO_CVS.keys()))
    demo_cv = DEMO_CVS[selected_name]

    with st.expander("📄 CV Text", expanded=False):
        st.text(demo_cv)

    with st.spinner("Matching…"):
        cv_data, results = run_match(demo_cv, matcher, top_k)

    cv_skills = cv_data["skills"]
    score_col = "hybrid_score"

    # ── Row 1: KPIs ────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    top_score = float(results[score_col].max())
    k1.metric("Skills detected", len(cv_skills))
    k2.metric("Top match score", f"{top_score:.0%}")
    k3.metric("Jobs analysed", len(jd_df))
    k4.metric("Top role", results.iloc[0]["role"])

    # ── Row 2: Gauge + Match bar ────────────────────────────────────────────
    col_gauge, col_bar = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(make_score_gauge(top_score, "Top Match Score"), use_container_width=True, key="demo_gauge")
    with col_bar:
        st.plotly_chart(make_match_bar(results.head(8), score_col), use_container_width=True, key="demo_match_bar")

    # ── Row 3: Skill chips ──────────────────────────────────────────────────
    st.markdown("**Detected skills from CV:**")
    st.markdown(" ".join(f"`{s}`" for s in cv_skills) or "_No skills detected_")

    # ── Row 4: Top match detail ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Top Match Deep-Dive")

    top_row = results.iloc[0]
    top_jd_raw = extract_skills(
        str(top_row.get("job_description", "")) + " " + str(top_row.get("skills", ""))
    )
    top_jd_skills = [s["canonical"] for s in normalise_skills(top_jd_raw)]
    gap = analyze_gap(cv_skills, top_jd_skills or parse_skills_string(str(top_row.get("skills", ""))))

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"**{top_row['job_title']}** — {top_row['role']}")
        st.caption(f"Fit label: {fit_label(top_score)}")
        st.markdown(f"Match score: **{top_score:.0%}**")
        st.markdown(f"Skill coverage: **{gap['coverage_pct']}%** of required skills")

        g1, g2, g3 = st.columns(3)
        g1.metric("✅ Matched", len(gap["matched"]))
        g2.metric("⚠️ Partial",  len(gap["partial"]))
        g3.metric("❌ Missing",  len(gap["missing"]))

    with col_r:
        if top_jd_skills:
            st.plotly_chart(make_skills_radar(cv_skills, top_jd_skills), use_container_width=True, key="demo_radar")

    # Gap bar
    if gap["total_job_skills"] > 0:
        st.plotly_chart(make_gap_bar(gap), use_container_width=True, key="demo_gap_bar")

    # Upskilling table
    recs = prioritise_upskilling(gap["missing"], gap["partial"])
    if recs:
        st.markdown("#### 📚 Upskilling Recommendations")
        rec_df = pd.DataFrame(recs)[["skill", "status", "priority"]]
        rec_df.columns = ["Skill to Learn", "Status", "Priority"]
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # ── All results table ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### All Matches")
    display_cols = ["job_title", "role", score_col]
    if show_breakdown and "transformer_score" in results.columns:
        display_cols += ["tfidf_score", "transformer_score"]
    display_df = results[display_cols].copy()
    display_df["fit_label"] = display_df[score_col].apply(fit_label)
    display_df[score_col] = display_df[score_col].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = results.to_csv(index=False).encode()
    st.download_button("⬇️ Download results CSV", csv, "demo_results.csv", "text/csv")


# ===========================================================================
# TAB 1 — MATCH YOUR CV
# ===========================================================================
with tab_match:
    st.subheader("Match Your Own CV to Jobs")

    cv_input_mode = st.radio("Input method", ["Paste text", "Upload CSV"], horizontal=True)

    cv_text = ""
    cv_name = "Candidate"

    if cv_input_mode == "Paste text":
        cv_text = st.text_area("Paste CV text", height=250,
                               placeholder="Python developer with 5 years experience in Django, AWS…")
    else:
        cv_file = st.file_uploader("Upload CV CSV", type="csv")
        if cv_file:
            cv_df = pd.read_csv(cv_file)
            st.dataframe(cv_df.head(3), use_container_width=True)
            str_cols = [c for c in cv_df.columns if cv_df[c].dtype == object]
            text_col = st.selectbox("CV text column", str_cols)
            row_idx  = st.number_input("Row index", 0, len(cv_df)-1, 0)
            cv_text  = str(cv_df[text_col].iloc[row_idx])
            if "name" in cv_df.columns:
                cv_name = str(cv_df["name"].iloc[row_idx])

    col_jd_src, _ = st.columns([2, 1])
    with col_jd_src:
        jd_source = st.radio("Job source", ["Built-in database", "Upload JD CSV"], horizontal=True)
        custom_jd_df = None
        if jd_source == "Upload JD CSV":
            jd_file = st.file_uploader("Upload JD CSV", type="csv", key="jd_up")
            if jd_file:
                custom_jd_df = pd.read_csv(jd_file)
                for col in ("job_title", "job_description", "skills", "role"):
                    if col not in custom_jd_df.columns:
                        custom_jd_df[col] = ""
                st.dataframe(custom_jd_df.head(3), use_container_width=True)

    if st.button("🚀 Find Matching Jobs", type="primary", use_container_width=True):
        if not cv_text.strip():
            st.warning("Please provide CV text.")
        else:
            with st.spinner("Analysing…"):
                if custom_jd_df is not None:
                    m = HybridMatcher(0.3, 0.7)
                    m.fit(custom_jd_df)
                    res = m.rank(cv_text, top_k=top_k)
                else:
                    m, _ = load_matcher()
                    res = m.rank(cv_text, top_k=top_k)

                cv_d = process_cv(cv_text)
                cv_s = cv_d["skills"]

            top_s = float(res["hybrid_score"].max())

            k1, k2, k3 = st.columns(3)
            k1.metric("Skills detected", len(cv_s))
            k2.metric("Top match score", f"{top_s:.0%}")
            k3.metric("Top role", res.iloc[0]["role"])

            col_g, col_b = st.columns([1, 2])
            with col_g:
                st.plotly_chart(make_score_gauge(top_s), use_container_width=True, key="match_gauge")
            with col_b:
                st.plotly_chart(make_match_bar(res.head(8), "hybrid_score"), use_container_width=True, key="match_match_bar")

            st.markdown("**Detected skills:**")
            st.markdown(" ".join(f"`{s}`" for s in cv_s) or "_None_")

            for i, row in res.iterrows():
                score = float(row["hybrid_score"])
                jd_raw = extract_skills(str(row.get("job_description","")) + " " + str(row.get("skills","")))
                jd_skills = [s["canonical"] for s in normalise_skills(jd_raw)]
                gap = analyze_gap(cv_s, jd_skills or parse_skills_string(str(row.get("skills",""))))

                with st.expander(f"#{i+1} {row['job_title']} ({row['role']}) — {fit_label(score)}", expanded=(i==0)):
                    cl, cr = st.columns(2)
                    with cl:
                        st.markdown(f"Hybrid score: **{score:.0%}**")
                        g1,g2,g3 = st.columns(3)
                        g1.metric("✅ Matched", len(gap["matched"]))
                        g2.metric("⚠️ Partial",  len(gap["partial"]))
                        g3.metric("❌ Missing",  len(gap["missing"]))
                        if gap["missing"]:
                            st.markdown("**Missing:** " + ", ".join(f"`{s}`" for s in gap["missing"][:8]))
                    with cr:
                        if jd_skills:
                            st.plotly_chart(make_skills_radar(cv_s, jd_skills), use_container_width=True, key=f"match_radar_{i}")

            csv = res.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv, "results.csv", "text/csv")


# ===========================================================================
# TAB 2 — BATCH MATCHING
# ===========================================================================
with tab_batch:
    st.subheader("Batch: Match Multiple CVs at Once")
    st.markdown("Upload a CSV with a `resume_text` column. Each row is matched independently.")

    batch_file = st.file_uploader("Upload CV CSV (multiple rows)", type="csv", key="batch_up")
    batch_k = st.slider("Top K per CV", 1, 10, 3)

    if batch_file:
        bdf = pd.read_csv(batch_file)
        st.dataframe(bdf.head(5), use_container_width=True)
        str_cols = [c for c in bdf.columns if bdf[c].dtype == object]
        tcol = st.selectbox("CV text column", str_cols, key="batch_tcol")

        if st.button("🚀 Run Batch", type="primary"):
            m, _ = load_matcher()
            all_res = []
            prog = st.progress(0)
            status_txt = st.empty()
            for idx, row in bdf.iterrows():
                name_i = str(row.get("name", f"Candidate {idx+1}"))
                status_txt.text(f"Processing {name_i}…")
                try:
                    r = m.rank(str(row[tcol]), top_k=batch_k)
                    r["candidate"] = name_i
                    all_res.append(r)
                except Exception as e:
                    st.warning(f"Skipped {name_i}: {e}")
                prog.progress((idx+1)/len(bdf))
            prog.empty(); status_txt.empty()

            if all_res:
                combined = pd.concat(all_res, ignore_index=True)
                combined["fit_label"] = combined["hybrid_score"].apply(fit_label)

                st.success(f"Matched {len(bdf)} CVs → {len(combined)} results")

                # Best score per candidate
                best = combined.groupby("candidate")["hybrid_score"].max().reset_index()
                best["fit_label"] = best["hybrid_score"].apply(fit_label)

                fig_batch = px.bar(
                    best, x="candidate", y="hybrid_score",
                    color="hybrid_score",
                    color_continuous_scale=["#e74c3c","#f39c12","#2ecc71"],
                    range_color=[0,1],
                    title="Best Match Score per Candidate",
                    text=best["hybrid_score"].apply(lambda x: f"{x:.0%}"),
                )
                fig_batch.update_traces(textposition="outside")
                fig_batch.update_layout(height=350, coloraxis_showscale=False)
                st.plotly_chart(fig_batch, use_container_width=True, key="batch_bar")

                # Fit label distribution
                label_counts = combined["fit_label"].value_counts().reset_index()
                label_counts.columns = ["Fit Label", "Count"]
                fig_pie = px.pie(label_counts, values="Count", names="Fit Label",
                                 color="Fit Label",
                                 color_discrete_map={"🟢 Good Fit":"#2ecc71","🟡 Potential Fit":"#f39c12","🔴 Low Fit":"#e74c3c"},
                                 title="Overall Fit Distribution")
                st.plotly_chart(fig_pie, use_container_width=True, key="batch_pie")

                st.dataframe(combined[["candidate","job_title","role","hybrid_score","fit_label"]],
                             use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download", combined.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv")


# ===========================================================================
# TAB 3 — EVALUATION
# ===========================================================================
with tab_eval:
    st.subheader("Model Evaluation on Labelled Dataset")
    st.markdown(
        "Evaluates the hybrid matcher on the **8,000 labelled CV–job pairs** "
        "(No Fit / Potential Fit / Good Fit) from HuggingFace."
    )

    eval_n = st.slider("Number of test samples", 50, 500, 150, step=50)

    if st.button("▶ Run Evaluation", type="primary"):
        with st.spinner("Running…"):
            try:
                m, _ = load_matcher()
                _, test_df = load_matching_labels()
                sample = test_df.sample(eval_n, random_state=42)

                from src.evaluation.metrics import evaluate_on_dataset, score_to_label
                metrics = evaluate_on_dataset(m, sample)

                st.success("Evaluation complete!")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("F1 (macro)",     f"{metrics['f1_macro']:.3f}")
                m2.metric("F1 (weighted)",  f"{metrics['f1_weighted']:.3f}")
                m3.metric("Precision",      f"{metrics['precision_macro']:.3f}")
                m4.metric("Recall",         f"{metrics['recall_macro']:.3f}")

                st.markdown("#### Classification Report")
                st.code(metrics.get("report",""), language=None)

                # Score distribution on the test sample
                scores_list = []
                for _, row in sample.iterrows():
                    q  = row["resume_text"]
                    jd = row["job_description_text"]
                    q_emb  = m.transformer.encode([q])
                    jd_emb = m.transformer.encode([jd])
                    sc = float((q_emb @ jd_emb.T).flatten()[0])
                    scores_list.append({"score": sc, "true_label": row["label"],
                                        "pred_label": score_to_label(sc)})

                score_df = pd.DataFrame(scores_list)
                fig_dist = px.histogram(
                    score_df, x="score", color="true_label",
                    barmode="overlay", nbins=30,
                    color_discrete_map={"No Fit":"#e74c3c","Potential Fit":"#f39c12","Good Fit":"#2ecc71"},
                    title="SBERT Score Distribution by True Label",
                    labels={"score":"Cosine Similarity Score","true_label":"Label"},
                )
                fig_dist.update_layout(height=350)
                st.plotly_chart(fig_dist, use_container_width=True, key="eval_dist")

                # Confusion matrix heatmap
                from sklearn.metrics import confusion_matrix
                from src.evaluation.metrics import LABEL_MAP, LABEL_NAMES
                y_true = [LABEL_MAP.get(l, 0) for l in score_df["true_label"]]
                y_pred = [LABEL_MAP.get(p, 0) for p in score_df["pred_label"]]
                cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
                labels_present = ["No Fit","Potential Fit","Good Fit"]
                fig_cm = px.imshow(
                    cm, x=labels_present, y=labels_present,
                    text_auto=True, color_continuous_scale="Blues",
                    title="Confusion Matrix",
                    labels={"x":"Predicted","y":"Actual"},
                )
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, use_container_width=True, key="eval_cm")

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)
