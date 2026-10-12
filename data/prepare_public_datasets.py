#!/usr/bin/env python3
"""Download and convert the three public datasets into ./data/:

  adult.csv              UCI Adult (adult.data + adult.test, 48,842 rows before NA removal)
  UCI_Credit_Card.csv    UCI Default of Credit Card Clients (Taiwan), 30,000 rows
  creditcard.csv         ULB Credit Card Fraud (Kaggle mlg-ulb/creditcardfraud), 284,807 rows

Credit Card Fraud: preferred source is Kaggle (needs ~/.kaggle/kaggle.json). If the Kaggle
CLI is not configured the script falls back to OpenML dataset 1597 (same data).

  python data/prepare_public_datasets.py            # all three
  python data/prepare_public_datasets.py --only adult taiwan
"""
import argparse
import io
import os
import shutil
import subprocess
import urllib.request
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
TAIWAN_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
              "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
              "hours-per-week", "native-country", "income"]


def _download(url):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def adult(out):
    z = zipfile.ZipFile(io.BytesIO(_download(ADULT_URL)))
    read = lambda name, skip: pd.read_csv(z.open(name), header=None, names=ADULT_COLS, skiprows=skip,
                                          skipinitialspace=True, na_values="?")
    df = pd.concat([read("adult.data", 0), read("adult.test", 1)], ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df = df.dropna(how="all")
    df.to_csv(out, index=False)
    print(f"  adult: {len(df):,} rows ({df.isna().any(axis=1).sum():,} with missing values, dropped at load) "
          f"-> {out}")


def taiwan(out):
    z = zipfile.ZipFile(io.BytesIO(_download(TAIWAN_URL)))
    name = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))][0]
    df = pd.read_excel(io.BytesIO(z.read(name)), header=1)
    df = df.rename(columns={"default payment next month": "default.payment.next.month"})
    df.to_csv(out, index=False)
    print(f"  taiwan: {len(df):,} rows, default rate {df['default.payment.next.month'].mean():.4f} -> {out}")


def credit_fraud(out):
    if shutil.which("kaggle"):
        tmp = os.path.join(HERE, "_kaggle_tmp")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud",
                            "-p", tmp, "--unzip"], check=True)
            shutil.move(os.path.join(tmp, "creditcard.csv"), out)
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"  credit_fraud (Kaggle) -> {out}")
            return
        except Exception as e:
            print(f"  Kaggle download failed ({e}); falling back to OpenML")
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = d.frame
    df["Class"] = df["Class"].astype(str).str.strip("'").astype(float).astype(int)
    df.to_csv(out, index=False)
    print(f"  credit_fraud (OpenML 1597): {len(df):,} rows, fraud rate {df['Class'].mean():.5f} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["adult", "taiwan", "credit_fraud"])
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    jobs = {"adult": (adult, "adult.csv"), "taiwan": (taiwan, "UCI_Credit_Card.csv"),
            "credit_fraud": (credit_fraud, "creditcard.csv")}
    for k in (a.only or jobs):
        fn, name = jobs[k]
        fn(os.path.join(a.out_dir, name))


if __name__ == "__main__":
    main()
