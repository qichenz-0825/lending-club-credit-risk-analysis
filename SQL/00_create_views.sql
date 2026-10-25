-- =============================================================================
-- 00_create_views.sql
-- Analysis layer for the Lending Club portfolio study.
-- `{data_dir}` is substituted by src/lending_db.py at runtime.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- loans: raw accepted-loan file (2.26M rows, 151 columns), stored as Parquet
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW loans AS
SELECT *
FROM '{data_dir}/accepted.parquet';

-- -----------------------------------------------------------------------------
-- loans_clean: all accepted loans, cleaned
--   * drops 33 file-footer rows (loan_status IS NULL; summary text in id, all else empty)
--   * issue_d is stored as text ('Dec-2015'), so text sorting is wrong;
--     parse it to a real date (day defaults to the 1st of the month)
--   * term is stored as ' 36 months' with a leading space; normalize to an integer
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW loans_clean AS
SELECT *,
       strptime(issue_d, '%b-%Y')                        AS issue_date,
       CAST(TRIM(REPLACE(term, 'months', '')) AS INTEGER) AS term_months
FROM loans
WHERE loan_status IS NOT NULL;

-- -----------------------------------------------------------------------------
-- cohort: modeling / performance population
--   * 36-month loans issued Jan-2014 to Dec-2015
--   * the data ends Dec-2018, so every loan in this window has reached
--     maturity and has an observed final outcome (avoids right-censoring)
--   * bad_flag is the single, shared target definition:
--       1    = Charged Off, Late (31-120 days)
--       0    = Fully Paid
--       NULL = indeterminate (Current, In Grace Period, Late 16-30 days),
--              excluded from rates because AVG() skips NULLs
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW cohort AS
SELECT *,
       CASE
           WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1
           WHEN loan_status = 'Fully Paid'                           THEN 0
       END AS bad_flag
FROM loans_clean
WHERE issue_date BETWEEN '2014-01-01' AND '2015-12-31'
  AND term_months = 36;

-- -----------------------------------------------------------------------------
-- rejected: declined applications (27.6M rows, 9 columns)
-- Only application-time fields exist; no term, grade, rate or performance.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW rejected AS
SELECT *
FROM '{data_dir}/rejected.parquet';
