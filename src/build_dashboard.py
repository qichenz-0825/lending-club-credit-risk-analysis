"""Build the interactive results dashboard: docs/index.html (served by GitHub Pages).

Run from the project root:
    python src/build_dashboard.py
"""

import json
from pathlib import Path

from lending_db import PROJECT_ROOT, get_connection

OUT = PROJECT_ROOT / "docs" / "index.html"
TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def records(con, sql):
    """Run a query and return a list of dicts (JSON-friendly)."""
    df = con.sql(sql).df()
    for col in df.columns:
        if str(df[col].dtype).startswith("datetime"):
            df[col] = df[col].dt.strftime("%Y-%m")
    return json.loads(df.to_json(orient="records"))


def main():
    con = get_connection()
    data = {}

    data["kpis"] = records(con, """
        SELECT (SELECT COUNT(*) FROM loans_clean)                     AS n_loans,
               (SELECT SUM(funded_amnt) / 1e9 FROM loans_clean)       AS funded_bn,
               (SELECT COUNT(*) FROM rejected)                        AS n_rejected,
               (SELECT COUNT(*) FROM cohort)                          AS n_cohort,
               (SELECT AVG(bad_flag) FROM cohort)                     AS bad_rate
    """)[0]

    data["yearly"] = records(con, """
        WITH a AS (
            SELECT YEAR(issue_date) AS year, COUNT(*) AS n_accepted, SUM(funded_amnt) / 1e6 AS funded_mm
            FROM loans_clean GROUP BY year
        ),
        r AS (
            SELECT YEAR("Application Date") AS year, COUNT(*) AS n_rejected
            FROM rejected GROUP BY year
        )
        SELECT a.year, a.n_accepted, r.n_rejected, a.funded_mm,
               a.n_accepted * 1.0 / (a.n_accepted + r.n_rejected) AS approval_rate
        FROM a LEFT JOIN r ON a.year = r.year
        ORDER BY a.year
    """)

    data["monthly"] = records(con, """
        WITH m AS (
            SELECT DATE_TRUNC('month', issue_date) AS month, SUM(funded_amnt) / 1e6 AS funded_mm
            FROM loans_clean
            WHERE issue_date >= '2015-01-01' AND issue_date < '2018-01-01'
            GROUP BY month
        )
        SELECT month, funded_mm,
               funded_mm / LAG(funded_mm)     OVER (ORDER BY month) - 1 AS mom,
               funded_mm / LAG(funded_mm, 12) OVER (ORDER BY month) - 1 AS yoy
        FROM m ORDER BY month
    """)

    data["decomp"] = records(con, """
        WITH a AS (
            SELECT DATE_TRUNC('month', issue_date) AS month, COUNT(*) AS n_accepted, SUM(funded_amnt) AS funded
            FROM loans_clean GROUP BY month
        ),
        r AS (
            SELECT DATE_TRUNC('month', "Application Date") AS month, COUNT(*) AS n_rejected
            FROM rejected GROUP BY month
        ),
        m AS (
            SELECT a.*, r.n_rejected,
                   CASE WHEN a.month BETWEEN '2015-07-01' AND '2015-12-01' THEN 'H2-2015'
                        WHEN a.month BETWEEN '2016-07-01' AND '2016-12-01' THEN 'H2-2016' END AS period
            FROM a JOIN r ON a.month = r.month
        )
        SELECT period,
               SUM(n_accepted + n_rejected)                         AS applications,
               SUM(n_accepted) * 1.0 / SUM(n_accepted + n_rejected) AS approval_rate,
               SUM(funded) / SUM(n_accepted)                        AS avg_loan_size,
               SUM(funded) / 1e6                                    AS funded_mm
        FROM m WHERE period IS NOT NULL GROUP BY period ORDER BY period
    """)

    data["grade"] = records(con, """
        SELECT grade, COUNT(*) AS n_loans, AVG(int_rate) / 100 AS int_rate, AVG(bad_flag) AS bad_rate,
               SUM(total_pymnt) / SUM(funded_amnt) - 1 AS realized_return
        FROM cohort GROUP BY grade ORDER BY grade
    """)

    drivers = {
        "Annual income": """CASE WHEN annual_inc IS NULL THEN '0|Missing' WHEN annual_inc < 40000 THEN '1|<$40k'
                                 WHEN annual_inc < 60000 THEN '2|$40–60k' WHEN annual_inc < 80000 THEN '3|$60–80k'
                                 WHEN annual_inc < 100000 THEN '4|$80–100k' ELSE '5|$100k+' END""",
        "DTI": """CASE WHEN dti IS NULL THEN '9|Missing' WHEN dti < 10 THEN '1|0–10' WHEN dti < 20 THEN '2|10–20'
                       WHEN dti < 30 THEN '3|20–30' WHEN dti < 40 THEN '4|30–40' ELSE '5|40+' END""",
        "Employment length": """CASE WHEN emp_length IS NULL THEN '00|Missing' WHEN emp_length = '< 1 year' THEN '01|< 1 yr'
                                     WHEN emp_length = '10+ years' THEN '11|10+ yrs'
                                     WHEN emp_length = '1 year' THEN '02|1 yr'
                                     ELSE lpad(CAST(CAST(split_part(emp_length, ' ', 1) AS INTEGER) + 1 AS VARCHAR), 2, '0')
                                          || '|' || split_part(emp_length, ' ', 1) || ' yrs' END""",
        "Home ownership": "home_ownership || '|' || home_ownership",
        "Purpose": "purpose || '|' || replace(purpose, '_', ' ')",
    }
    data["drivers"] = {}
    for name, expr in drivers.items():
        rows = records(con, f"""
            SELECT {expr} AS bucket, COUNT(*) AS n_loans, AVG(bad_flag) AS bad_rate
            FROM cohort GROUP BY bucket ORDER BY bucket
        """)
        for row in rows:
            row["label"] = row.pop("bucket").split("|", 1)[1]
        if name == "Purpose":
            rows.sort(key=lambda r: -r["n_loans"])
        data["drivers"][name] = rows

    data["separation"] = [
        {"variable": name, "ratio": max(r["bad_rate"] for r in rows if r["n_loans"] >= 300)
                                   / min(r["bad_rate"] for r in rows if r["n_loans"] >= 300)}
        for name, rows in {**data["drivers"], "Grade (benchmark)": [
            {"bad_rate": g["bad_rate"], "n_loans": g["n_loans"]} for g in data["grade"]]}.items()
    ]
    data["separation"].sort(key=lambda r: r["ratio"])

    OUT.parent.mkdir(exist_ok=True)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
