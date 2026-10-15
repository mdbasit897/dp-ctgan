#!/usr/bin/env python3
"""Extract the MIMIC-III cohort from BigQuery into data/mimic3.csv.

Prerequisites
  * PhysioNet credentialed access to MIMIC-III, with your Google account linked so that it can
    read the `physionet-data` BigQuery project (https://physionet.org/settings/cloud/).
  * Application-default credentials for THAT Google account on the machine running this:
        gcloud auth application-default login            # laptop
        gcloud auth application-default login --no-launch-browser   # on a VM over SSH
    (The VM's service account does NOT have PhysioNet access.)
  * pip install google-cloud-bigquery   (db-dtypes/pyarrow optional: without them, or if they
    clash with NumPy 1.x, rows are read directly; do NOT upgrade NumPy in the experiment env)

Usage
  python data/prepare_mimic3.py --project YOUR_GCP_PROJECT --dry-run   # cost estimate only
  python data/prepare_mimic3.py --project YOUR_GCP_PROJECT

DATA USE: MIMIC-III is governed by the PhysioNet Credentialed Health Data Use Agreement.
Keep mimic3.csv (and any synthetic data derived from it) on access-controlled storage;
never commit it to git or upload it to a public bucket.
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GCP project that is billed for the query")
    ap.add_argument("--sql", default=os.path.join(HERE, "mimic3_cohort.sql"))
    ap.add_argument("--out", default=os.path.join(HERE, "mimic3.csv"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from google.cloud import bigquery
    client = bigquery.Client(project=a.project)
    sql = open(a.sql).read()

    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    gb = dry.total_bytes_processed / 1e9
    print(f"Query will process {gb:.1f} GB (on-demand price ~ ${gb / 1000 * 6.25:.2f}; "
          f"the first 1 TB per month is free).")
    if a.dry_run:
        return

    result = client.query(sql).result()
    try:
        df = result.to_dataframe(create_bqstorage_client=False)
    except (ImportError, ValueError) as ex:
        # to_dataframe needs db-dtypes/pyarrow, whose newest releases require NumPy >= 2. The experiment
        # environment keeps NumPy 1.x for PyTorch/synthcity, so build the frame from plain rows instead.
        print(f"  to_dataframe unavailable ({type(ex).__name__}); reading rows directly")
        import pandas as pd
        result = client.query(sql).result()
        cols = [f.name for f in result.schema]
        df = pd.DataFrame([dict(row.items()) for row in result], columns=cols)
    df.to_csv(a.out, index=False)
    os.chmod(a.out, 0o600)
    print(f"wrote {a.out}: {len(df):,} admissions, mortality {df['MORTALITY'].mean():.4f}")
    na = df.isna().sum()
    if na.any():
        print("missing values per column (rows with any missing value are dropped at load):")
        print(na[na > 0].to_string())
        print(f"rows remaining after dropna: {len(df.dropna()):,}")


if __name__ == "__main__":
    main()
