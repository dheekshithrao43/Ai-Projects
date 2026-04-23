"""
CLI entry point for the AI Career Recommendation System.

Commands:
    python main.py app           -- launch Streamlit web app
    python main.py evaluate      -- run evaluation on test set
    python main.py match <text>  -- quick CLI match for a CV snippet
"""

from __future__ import annotations

import argparse
import sys


def cmd_app(_: argparse.Namespace) -> None:
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
        check=True,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from datasets import load_from_disk

    from src.data.loader import load_job_descriptions
    from src.evaluation.metrics import evaluate_on_dataset
    from src.matching.hybrid import HybridMatcher

    print(f"Loading job descriptions (sample={args.sample})…")
    jd_df = load_job_descriptions(software_only=True, sample_size=args.sample)

    print("Fitting hybrid matcher…")
    matcher = HybridMatcher()
    matcher.fit(jd_df)

    print("Loading test labels…")
    ds = load_from_disk("data/raw/matching_labels")
    test_df = ds["test"].to_pandas()
    if args.test_sample:
        test_df = test_df.sample(args.test_sample, random_state=42)

    print(f"Evaluating on {len(test_df)} samples…")
    results = evaluate_on_dataset(matcher, test_df)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if k != "report":
            print(f"  {k}: {v}")
    print("\n" + results.get("report", ""))


def cmd_match(args: argparse.Namespace) -> None:
    from src.data.loader import load_job_descriptions
    from src.data.preprocess import clean_text
    from src.extraction.rule_based import extract_skills_from_section
    from src.matching.hybrid import HybridMatcher
    from src.normalisation.normaliser import normalise_skills

    cv_text = " ".join(args.text)
    print(f"\nCV text: {cv_text[:200]}…\n")

    cleaned = clean_text(cv_text)
    raw_skills = extract_skills_from_section(cleaned)
    skills = [s["canonical"] for s in normalise_skills(raw_skills)]
    print(f"Detected skills: {', '.join(skills) or 'none'}\n")

    print(f"Loading job database (sample={args.sample})…")
    jd_df = load_job_descriptions(software_only=True, sample_size=args.sample)

    print("Fitting matcher…")
    matcher = HybridMatcher()
    matcher.fit(jd_df)

    results = matcher.rank(cv_text, top_k=args.top_k)
    print(f"\nTop {args.top_k} matches:")
    for i, row in results.iterrows():
        score = row.get("hybrid_score", row.get("transformer_score", 0))
        print(f"  {i+1}. {row['job_title']} ({row['role']}) — score: {score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Career Recommendation System")
    sub = parser.add_subparsers(dest="command")

    # app
    sub.add_parser("app", help="Launch Streamlit web app")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation on test set")
    p_eval.add_argument("--sample", type=int, default=5000, help="JD corpus sample size")
    p_eval.add_argument("--test-sample", type=int, default=200, help="Test set sample size")

    # match
    p_match = sub.add_parser("match", help="Quick CLI match for a CV snippet")
    p_match.add_argument("text", nargs="+", help="CV text (quoted string)")
    p_match.add_argument("--top-k", type=int, default=5)
    p_match.add_argument("--sample", type=int, default=3000, help="JD corpus sample size")

    args = parser.parse_args()

    if args.command == "app":
        cmd_app(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "match":
        cmd_match(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
