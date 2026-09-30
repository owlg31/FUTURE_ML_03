# Resume / Candidate Screening System

An explainable NLP-based screening tool that ranks candidate resumes against a job description. It combines text similarity with a transparent, weighted skill-match score and produces an HR-friendly report showing matched and missing skills.

## What it does

- Cleans resume and job-description text
- Extracts skills using a configurable skills lexicon
- Calculates TF-IDF cosine similarity for overall role relevance
- Gives extra weight to critical job skills
- Ranks candidates and flags missing required skills
- Exports a JSON report that can be shared with recruiters

## Scoring

The final score is deliberately interpretable:

```
final score = 55% weighted skill coverage + 45% TF-IDF similarity
```

Skill coverage is calculated from the skills explicitly required by the job description. Critical skills receive the configured importance weight, so a candidate cannot rank highly merely by repeating generic terms. Similarity is normalized to a 0–100 scale.

This is a decision-support demonstration, not an automated hiring decision-maker. A recruiter should review the ranked results and the underlying evidence.

## Run it

```powershell
cd outputs/resume-screening-system
python -m pip install -r requirements.txt
python resume_screening.py --job data/job_description.txt --resumes data/resumes.csv --output reports/screening_report.json
```

The command prints a ranking and writes a detailed report. The sample job is for a Data Scientist and has five simulated candidates.

## Use your own data

Provide a UTF-8 CSV with the columns `candidate_name` and `resume_text`:

```powershell
python resume_screening.py --job path/to/job.txt --resumes path/to/resumes.csv --output report.json
```

Update `SKILL_LEXICON` and `CRITICAL_SKILLS` in `resume_screening.py` for another role or domain. Multi-word skills, punctuation variants, and common aliases (for example, `scikit-learn` / `sklearn`) are supported.

## Limitations and responsible use

- Text similarity cannot verify depth of experience, project quality, or soft skills.
- The system should never use protected characteristics or proxies for them.
- Results can be affected by writing style and incomplete resumes; use them as one input to a human review process.
- The skill lexicon needs periodic review for the target role.

## Project structure

```
resume-screening-system/
├── data/                 # simulated example inputs
├── reports/              # generated output is written here
├── resume_screening.py   # CLI application and scoring logic
├── requirements.txt
└── README.md
```
