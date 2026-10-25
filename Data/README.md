# Data

The raw files are not tracked in git (~3.5 GB). To reproduce:

1. Download **All Lending Club loan data** from Kaggle: https://www.kaggle.com/datasets/wordsmyth/lending-club
2. Unzip and place these two files in this folder:
   - `accepted_2007_to_2018Q4.csv`
   - `rejected_2007_to_2018Q4.csv`
3. Run any notebook. `src/lending_db.py` converts the CSVs to `accepted.parquet` / `rejected.parquet` on first use.
