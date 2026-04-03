# Progress Log

## Goal
Track planning decisions, implementation progress, and next steps so all team members can follow project status.

## Team Rules
- Do not modify context/project_description.md.
- Do not modify context/project_report.md.
- Use this file as the shared planning/progress log.

## Status
- Date: 2026-04-04
- Current phase: Step-by-step restart implementation
- Owner: Copilot + project team

## Completed
- Added a reproducible quick start section to README.
- Pinned Python dependency versions in requirements.txt.
- Updated .gitignore to keep key sample assets/notebooks trackable.
- Fixed path portability and trailing invalid code content in analytics_project_jupyter_draft.ipynb.

## In Progress
- Single-step execution only.

## Next Actions
1. Step 2 only: Data Understanding (Section 3.2) with identical Orange3 and Jupyter outputs, then log exact parity metrics.

## Notes For Team
- exploratory_data:analysis.ipynb remains the strongest EDA reference.
- Keep modeling changes reproducible with relative paths only.

## Step 1 - Data Description
Date: 2026-04-03

Established one shared baseline dataset pipeline for Section 3.1 Data Description. In both Orange3 and Jupyter, with identical logic:
1. Load train.csv and items.csv (separator |).
2. Left-join on pid.
3. Run the same four checks: row count, column count, missing values per column, and click/basket/order mutual exclusivity.
4. Export one canonical output file: data/processed/step1_merged_baseline.csv.
5. Record the exact numbers in progress_log.md:

- Row count: 2756003
- Column count: 21
- Missing values per column:
lineID                   0
day                      0
pid                      0
adFlag                   0
availability             0
competitorPrice     100687
click                    0
basket                   0
order                    0
price                    0
revenue                  0
manufacturer             0
group                    0
content                  0
unit                     0
pharmForm           194124
genericProduct           0
salesIndex               0
category             87394
campaignIndex      2287968
rrp                      0
dtype: int64
- Click/basket/order mutual exclusivity: True

Orange3 and Jupyter produce the same row count, same column count, and the same validation numbers.

## Step 1 - Audit Artifacts (Implemented in Jupyter)
Date: 2026-04-04

Added strict and auditable Step 1 outputs in main.ipynb:
- Exact mutual exclusivity check: click+basket+order must equal 1.
- Exported baseline dataset: data/processed/step1_merged_baseline.csv
- Exported missing-value report: data/processed/step1_missing_values.csv
- Exported summary report: data/processed/step1_validation_summary.json

Validation results remain aligned with Step 1 log values:
- Row count: 2756003
- Column count: 21
- Mutual exclusivity (exactly one action): True