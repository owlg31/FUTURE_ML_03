"""Explainable resume screening and ranking CLI.

Example:
    python resume_screening.py --job data/job_description.txt --resumes data/resumes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ModuleNotFoundError:  # Lets the sample run in lightweight Python environments.
    HAS_SKLEARN = False


# Add or remove items to suit the role being screened. Aliases map to one label.
SKILL_LEXICON = {
    "python": ["python"], "sql": ["sql"], "machine learning": ["machine learning", "ml"],
    "statistics": ["statistics", "statistical"], "pandas": ["pandas"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "data visualization": ["data visualization", "visualization", "tableau", "power bi"],
    "git": ["git", "github"], "nlp": ["nlp", "natural language processing"],
    "aws": ["aws", "amazon web services"], "docker": ["docker", "containerization"],
    "tableau": ["tableau"], "tensorflow": ["tensorflow"], "excel": ["excel"],
    "java": ["java"], "javascript": ["javascript"], "react": ["react"],
}

# Important capabilities for the sample data-scientist job. Missing values default to 1.
CRITICAL_SKILLS = {"python": 2.0, "sql": 1.8, "machine learning": 2.0,
                   "statistics": 1.5, "pandas": 1.5, "scikit-learn": 1.5}


def normalize(text: str) -> str:
    """Lowercase text and retain spaces between alphanumeric terms."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.]", " ", text.lower())).strip()


def contains_term(normalized_text: str, term: str) -> bool:
    """Match terms as words so `java` does not match `javascript`."""
    return bool(re.search(r"(?<!\w)" + re.escape(normalize(term)) + r"(?!\w)", normalized_text))


def extract_skills(text: str) -> set[str]:
    normalized = normalize(text)
    return {skill for skill, aliases in SKILL_LEXICON.items()
            if any(contains_term(normalized, alias) for alias in aliases)}


@dataclass
class CandidateResult:
    rank: int
    candidate_name: str
    final_score: float
    skill_coverage_score: float
    text_similarity_score: float
    matched_skills: list[str]
    missing_required_skills: list[str]
    additional_skills: list[str]


def weighted_coverage(found: set[str], required: set[str]) -> float:
    """Return weighted required-skill coverage from 0 to 100."""
    if not required:
        return 0.0
    possible = sum(CRITICAL_SKILLS.get(skill, 1.0) for skill in required)
    earned = sum(CRITICAL_SKILLS.get(skill, 1.0) for skill in found & required)
    return 100 * earned / possible


def tfidf_similarities(job_text: str, resume_texts: list[str]) -> list[float]:
    """Return cosine similarities, using scikit-learn when it is available."""
    documents = [job_text] + resume_texts
    if HAS_SKLEARN:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(documents)
        return list(cosine_similarity(matrix[0:1], matrix[1:]).ravel())

    # Small standard-library fallback: unigram TF-IDF cosine similarity.
    tokens = [re.findall(r"[a-z0-9+#.]+", normalize(document)) for document in documents]
    document_frequency: dict[str, int] = {}
    for terms in tokens:
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    total_documents = len(tokens)

    def vector(terms: list[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        length = max(len(terms), 1)
        return {term: (count / length) * (math.log((1 + total_documents) /
                (1 + document_frequency[term])) + 1) for term, count in counts.items()}

    job_vector = vector(tokens[0])
    job_norm = math.sqrt(sum(value * value for value in job_vector.values()))
    scores = []
    for terms in tokens[1:]:
        resume_vector = vector(terms)
        numerator = sum(value * resume_vector.get(term, 0.0) for term, value in job_vector.items())
        denominator = job_norm * math.sqrt(sum(value * value for value in resume_vector.values()))
        scores.append(numerator / denominator if denominator else 0.0)
    return scores


def rank_candidates(job_text: str, resumes: Iterable[dict[str, str]]) -> tuple[set[str], list[CandidateResult]]:
    """Score resumes with a shared TF-IDF model and return sorted results."""
    resumes = list(resumes)
    if not resumes:
        raise ValueError("The resumes file contains no candidates.")
    if any(not row.get("candidate_name") or not row.get("resume_text") for row in resumes):
        raise ValueError("Every row needs non-empty candidate_name and resume_text columns.")

    required = extract_skills(job_text)
    similarities = tfidf_similarities(job_text, [row["resume_text"] for row in resumes])

    results = []
    for row, similarity in zip(resumes, similarities):
        candidate_skills = extract_skills(row["resume_text"])
        coverage = weighted_coverage(candidate_skills, required)
        similarity_score = float(similarity * 100)
        final = 0.55 * coverage + 0.45 * similarity_score
        results.append(CandidateResult(
            rank=0, candidate_name=row["candidate_name"], final_score=round(final, 1),
            skill_coverage_score=round(coverage, 1), text_similarity_score=round(similarity_score, 1),
            matched_skills=sorted(candidate_skills & required),
            missing_required_skills=sorted(required - candidate_skills),
            additional_skills=sorted(candidate_skills - required),
        ))
    results.sort(key=lambda item: (item.final_score, item.skill_coverage_score), reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return required, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank resumes against a job description.")
    parser.add_argument("--job", required=True, type=Path, help="UTF-8 job-description text file")
    parser.add_argument("--resumes", required=True, type=Path, help="CSV with candidate_name,resume_text")
    parser.add_argument("--output", type=Path, default=Path("reports/screening_report.json"))
    args = parser.parse_args()

    job_text = args.job.read_text(encoding="utf-8")
    with args.resumes.open(encoding="utf-8", newline="") as handle:
        resumes = list(csv.DictReader(handle))
    required_skills, results = rank_candidates(job_text, resumes)

    report = {"required_skills_detected": sorted(required_skills),
              "scoring": {"skill_coverage_weight": 0.55, "text_similarity_weight": 0.45},
              "candidates": [asdict(item) for item in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCandidate ranking")
    print("-" * 76)
    for item in results:
        missing = ", ".join(item.missing_required_skills) or "None"
        print(f"{item.rank}. {item.candidate_name}: {item.final_score:.1f}/100 "
              f"(skills {item.skill_coverage_score:.1f}, similarity {item.text_similarity_score:.1f})")
        print(f"   Missing required skills: {missing}")
    print(f"\nDetailed report written to: {args.output}")


if __name__ == "__main__":
    main()
