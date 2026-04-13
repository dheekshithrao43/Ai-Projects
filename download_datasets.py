"""
Dataset download script for AI Career Recommendation System.

Requirements:
    pip install kaggle datasets requests tqdm

Kaggle setup:
    1. Go to kaggle.com -> Account -> Create New API Token
    2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"


def run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        sys.exit(1)
    print(f"  Done.")


def download_file(url: str, dest: Path, desc: str) -> None:
    print(f"\n>>> {desc}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved to {dest}")


def download_kaggle(slug: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"],
        f"Kaggle: {slug}",
    )


def download_huggingface(repo_id: str, out_dir: Path) -> None:
    print(f"\n>>> HuggingFace: {repo_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    ds = load_dataset(repo_id)
    ds.save_to_disk(str(out_dir))
    print(f"  Saved to {out_dir}")


def download_onet(out_dir: Path) -> None:
    url = "https://www.onetcenter.org/dl_files/database/db_23_1_text.zip"
    zip_path = out_dir / "onet_23_1.zip"
    download_file(url, zip_path, "O*NET 23.1 taxonomy (CC BY 4.0)")
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    zip_path.unlink()
    print(f"  Extracted to {out_dir}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Resume corpus (2,484 resumes, 24 role labels) ──────────────────────
    download_kaggle(
        "snehaanbhawal/resume-dataset",
        RAW_DIR / "resumes",
    )

    # ── 2. Job descriptions (skills + responsibilities columns) ───────────────
    download_kaggle(
        "ravindrasinghrana/job-description-dataset",
        RAW_DIR / "job_descriptions",
    )

    # ── 3. NER training data (220 annotated resumes) ──────────────────────────
    download_kaggle(
        "dataturks/resume-entities-for-ner",
        RAW_DIR / "ner_annotations",
    )

    # ── 4. Ground truth matching labels (8,000 CV-job pairs) ──────────────────
    #    Labels: No Fit / Partial Fit / Good Fit
    download_huggingface(
        "cnamuangtoun/resume-job-description-fit",
        RAW_DIR / "matching_labels",
    )

    # ── 5. Fine-grained scoring with justifications (1,031 pairs) ─────────────
    #    GPT-4o scored with numeric macro/micro scores + text justifications
    download_huggingface(
        "netsol/resume-score-details",
        RAW_DIR / "score_details",
    )

    # ── 6. O*NET skills taxonomy (CC BY 4.0, no registration needed) ──────────
    download_onet(RAW_DIR / "onet_taxonomy")

    print("\n" + "=" * 60)
    print("All datasets downloaded to ./data/raw/")
    print()
    print("Directory layout:")
    print("  data/raw/resumes/           <- Resume corpus (Kaggle)")
    print("  data/raw/job_descriptions/  <- Job descriptions (Kaggle)")
    print("  data/raw/ner_annotations/   <- NER training data (Kaggle)")
    print("  data/raw/matching_labels/   <- CV-job pairs + labels (HuggingFace)")
    print("  data/raw/score_details/     <- Scored pairs + justifications (HuggingFace)")
    print("  data/raw/onet_taxonomy/     <- O*NET 23.1 skills taxonomy")
    print()
    print("Next: run  python -m src.data.preprocess  to clean and prepare the data.")


if __name__ == "__main__":
    main()

