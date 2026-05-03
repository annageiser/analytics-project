# Progress Log

## Goal
Track planning decisions, implementation progress, and next steps so all team members can follow project status.

## Team Rules
- Do not modify context/project_description.md.
- Do not modify context/project_report.md.
- Use this file as the shared planning/progress log.

## Status
- Date: 2026-05-01
- Current phase: Project Documentation
- Owner: Copilot + project team

## Completed
- **Project Setup and Initialization**: Established the project structure, including data and context directories, and set up the initial `requirements.txt`.
- **Data Ingestion and Merging**: Loaded the `items.csv` and `train.csv` datasets, and merged them into a single baseline dataset (`step1_merged_baseline.csv`) for analysis.
- **Initial Data Validation**: Performed initial data quality checks, including row and column counts, identifying missing values, and verifying the mutual exclusivity of the `click`, `basket`, and `order` columns.
- **Workflow Hardening**: Improved the data processing workflow by adding PID integrity checks, standardizing table names, and creating a machine-readable validation summary (`step1_validation_summary.json`).
- Added a reproducible quick start section to README.
- Pinned Python dependency versions in requirements.txt.
- Updated .gitignore to keep key sample assets/notebooks trackable.
- Fixed path portability and trailing invalid code content in analytics_project_jupyter_draft.ipynb.

## In Progress
- Exploratory Data Analysis (EDA) to understand data distributions and relationships.

## Next Actions
1.  **Feature Engineering**: Create new attributes from existing data to improve model performance.
2.  **Model Selection and Training**: Implement and compare at least three different classification algorithms.
3.  **Model Evaluation**: Evaluate the models based on appropriate metrics and select the best-performing one.
4.  **Reporting**: Summarize the findings in a paper and presentation.

## Notes For Team
- The `exploratory_data:analysis.ipynb` notebook is the primary reference for EDA.
- All modeling changes should use relative paths to ensure reproducibility.
- The `step1_validation_summary.json` file is the main audit record for the initial data processing step.

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

## Step 1 - Validation Hardening and Naming Cleanup
Date: 2026-04-10

Refined the Step 1 notebook workflow for clearer validation, reproducibility, and future train/validation split readiness.

### Completed
- Renamed core tables for clarity:
  - train_raw_df
  - items_raw_df
- Added PID integrity checks before merge:
  - Null PID check in train and items
  - Duplicate PID check in items
- Kept merge safety net with validate="m:1".
- Improved artifact persistence:
  - Missing-value CSV now stores only non-zero missing columns
  - Stable empty-schema fallback for missing report
  - Added FORCE_REFRESH_BASELINE flag for controlled baseline rewrites
- Added generated_at_utc timestamp to summary payload.
- Fixed exploratory reporting cell to use renamed dataframe variables consistently.

### Current Step 1 Outputs
- data/processed/step1_merged_baseline.csv
- data/processed/step1_missing_values.csv
- data/processed/step1_validation_summary.json

### Notes
- Summary payload is now the main machine-readable audit record for Step 1.
- If progress_log is not updated frequently, the JSON summary should remain mandatory for traceability.

### Next Actions
1. Re-run the Step 1 validation cell and confirm parity outputs.
2. Start Section 4 Data Preparation with split-safe naming:
   - X_train, X_valid, y_train, y_valid