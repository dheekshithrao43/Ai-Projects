"""Dataset loaders for resumes, job descriptions, labels, and taxonomy."""

import json
import pandas as pd
from pathlib import Path
from datasets import load_from_disk

DATA_DIR = Path("data/raw")

SW_KEYWORDS = (
    r"software|developer|engineer|data|machine.learning|ml\b|ai\b|devops|cloud|"
    r"python|java\b|javascript|backend|frontend|full.?stack|security|database|"
    r"architect|analyst|qa\b|testing|sre\b|platform|infrastructure|mobile|ios|android"
)


def load_resumes(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the resume corpus.
    Expected columns: ID, Resume_str, Resume_html, Category
    Returns: ID, resume_text, category
    """
    path = Path(path) if path else DATA_DIR / "resumes" / "Resume" / "Resume.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Resume_str": "resume_text", "Category": "category"})
    df["resume_text"] = df["resume_text"].fillna("").astype(str)
    return df[["ID", "resume_text", "category"]].reset_index(drop=True)


def load_job_descriptions(
    path: str | Path | None = None,
    software_only: bool = True,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Load job descriptions dataset.
    If software_only=True, filter to software/data/engineering roles.
    Returns: job_id, job_title, role, job_description, skills, responsibilities
    """
    path = Path(path) if path else DATA_DIR / "job_descriptions" / "job_descriptions.csv"
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {
        "job_id": "job_id",
        "job_title": "job_title",
        "role": "role",
        "job_description": "job_description",
        "skills": "skills",
        "responsibilities": "responsibilities",
    }
    # Handle 'job id' vs 'job_id'
    if "job_id" not in df.columns and "job id" in [c.lower() for c in df.columns]:
        df = df.rename(columns={"job id": "job_id"})

    for col in col_map:
        if col not in df.columns:
            df[col] = ""

    df = df[list(col_map.keys())].copy()
    df = df.fillna("")

    if software_only:
        mask = (
            df["role"].str.contains(SW_KEYWORDS, case=False, na=False)
            | df["job_title"].str.contains(SW_KEYWORDS, case=False, na=False)
        )
        df = df[mask].reset_index(drop=True)

    # Always deduplicate — the dataset has many synthetic duplicate rows
    df = df.drop_duplicates(subset=["job_description"]).reset_index(drop=True)

    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    return df.reset_index(drop=True)


def load_matching_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the HuggingFace resume-job matching labels dataset.
    Returns (train_df, test_df) with columns: resume_text, job_description_text, label
    Labels: 'No Fit', 'Potential Fit', 'Good Fit'
    """
    ds = load_from_disk(str(DATA_DIR / "matching_labels"))
    return ds["train"].to_pandas(), ds["test"].to_pandas()


def load_onet_tech_skills() -> list[str]:
    """
    Load O*NET Tools & Technology examples as a flat skill vocabulary list.
    """
    path = DATA_DIR / "onet_taxonomy" / "db_23_1_text" / "Tools and Technology.txt"
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t")
    skills = df["T2 Example"].dropna().str.lower().str.strip().unique().tolist()
    return skills


def load_onet_occupations() -> pd.DataFrame:
    """Load O*NET occupation titles and descriptions."""
    path = DATA_DIR / "onet_taxonomy" / "db_23_1_text" / "Occupation Data.txt"
    return pd.read_csv(path, sep="\t")


def load_onet_alternate_titles() -> pd.DataFrame:
    """Load O*NET alternate job title mappings."""
    path = DATA_DIR / "onet_taxonomy" / "db_23_1_text" / "Alternate Titles.txt"
    return pd.read_csv(path, sep="\t")


def load_ner_annotations() -> list[dict]:
    """Load the NER-annotated resume JSON file (220 annotated resumes)."""
    path = DATA_DIR / "ner_annotations" / "Entity Recognition in Resumes.json"
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def load_user_csv(file_obj, text_col: str | None = None) -> pd.DataFrame:
    """
    Load a user-uploaded CSV file (from Streamlit file_uploader).
    Auto-detects the text column if not specified.
    """
    df = pd.read_csv(file_obj)
    df.columns = [c.strip() for c in df.columns]

    if text_col and text_col in df.columns:
        df["_text"] = df[text_col].fillna("").astype(str)
    else:
        # Auto-detect: pick the longest average text column
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if not str_cols:
            raise ValueError("No text columns found in uploaded CSV.")
        avg_lens = {col: df[col].fillna("").astype(str).str.len().mean() for col in str_cols}
        best_col = max(avg_lens, key=avg_lens.get)
        df["_text"] = df[best_col].fillna("").astype(str)
        df["_detected_col"] = best_col

    return df
