
# Analytics Project

A model that predicts a) whether a product will be bought in a session and (b) if bought, in which quantity (sales forecast) for a dynamically priced mail-order pharmacy

## Quick Start

1. Create and activate a virtual environment:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   - `pip install --upgrade pip && pip install -r requirements.txt`
3. Place input data in the `data/` folder:
   - `data/train.csv`
   - `data/items.csv`
4. Run notebooks in this order:
   - `exploratory_data:analysis.ipynb` (EDA and data quality checks)
   - `modeling_example.ipynb` (initial preprocessing sample)
   - `analytics_project_jupyter_draft.ipynb` (feature engineering draft)