#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts tables from the ZUS PDFs and (optionally) builds a processed CSV approximating
the format app.py expects (incl. "71%69%73%" compact RR column if we can detect it).
- Saves raw tables to data/interim/table_XX.csv
- Writes a minimal combined CSV to data/processed/zus_forecast_all.csv if requested.

Requires: tabula-py (Java), else tries camelot, else pdfplumber fallback.
"""

import os, re, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "raw")
INTERIM = os.path.join(ROOT, "data", "interim")
PROCESSED = os.path.join(ROOT, "data", "processed")
os.makedirs(INTERIM, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)

PUB_PDF = os.path.join(RAW, "Publikacja_Fundusz_Emerytalny_2023-2080.pdf")
DETAILS_PDF = os.path.join(RAW, "DETAILS_ZUS_SymulatorEmerytalny.pdf")

def extract_with_tabula(pdf_path):
    import tabula
    try:
        dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, lattice=True)
    except Exception:
        dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, stream=True)
    return dfs or []

def extract_with_camelot(pdf_path):
    import camelot
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    except Exception:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    return [t.df for t in tables] if tables else []

def extract_with_pdfplumber(pdf_path):
    import pdfplumber
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                tbls = page.extract_tables()
                for t in tbls or []:
                    out.append(pd.DataFrame(t))
            except Exception:
                pass
    return out

def extract_all(pdf_path):
    for extractor in (extract_with_tabula, extract_with_camelot, extract_with_pdfplumber):
        try:
            dfs = extractor(pdf_path)
            if dfs:
                return dfs
        except Exception:
            continue
    return []

def save_interim_tables(dfs, prefix="tables"):
    saved = []
    for i, df in enumerate(dfs, 1):
        # normalize headers: use first row as header when it looks like a header
        if df.shape[0] > 1 and any(isinstance(x, str) for x in df.iloc[0].tolist()):
            df = df.rename(columns=df.iloc[0]).iloc[1:].reset_index(drop=True)
        path = os.path.join(INTERIM, f"{prefix}_{i:02d}.csv")
        df.to_csv(path, index=False)
        saved.append(path)
    return saved

def build_processed_csv_from_interim() -> pd.DataFrame:
    """
    Heuristic builder: scans interim CSVs for:
      - a 'rok' (indicator) column and numeric year columns: 2022..2080
      - a compact RR column (e.g., '71%69%73%') or a row containing 'wariant' with those values
    Produces a single wide CSV combining the most useful table.
    """
    csvs = [os.path.join(INTERIM, f) for f in os.listdir(INTERIM) if f.endswith(".csv")]
    if not csvs:
        raise RuntimeError("No interim CSVs found. Run extraction first.")

    candidates = []
    year_pat = re.compile(r"^(19|20)\d{2}$")
    for p in csvs:
        df = pd.read_csv(p, dtype=str)
        cols = [str(c) for c in df.columns]
        has_rok = any(c.strip().lower() == "rok" for c in cols)
        year_cols = [c for c in cols if year_pat.match(c)]
        if has_rok and len(year_cols) >= 5:
            candidates.append((p, df))

    if not candidates:
        # fallback: just stitch all tables vertically and write
        merged = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
        out = os.path.join(PROCESSED, "zus_forecast_all.csv")
        merged.to_csv(out, index=False)
        print("Wrote fallback processed CSV:", out)
        return merged

    # pick the largest candidate by area
    pth, df = max(candidates, key=lambda x: x[1].shape[0] * x[1].shape[1])
    print("Using candidate table:", os.path.basename(pth))

    # keep only 'rok' + year columns
    cols = [c for c in df.columns if str(c).strip().lower() == "rok" or year_pat.match(str(c))]
    core = df[cols].copy()

    # try to locate a compact replacement-rate cell/col like "71%69%73%"
    packed_rr_col = None
    for c in df.columns:
        if re.fullmatch(r"(\d+%){3}", str(c).replace(" ", "")):
            packed_rr_col = c; break
    if packed_rr_col is None:
        # search cell contents
        for c in df.columns:
            ser = df[c].astype(str).str.replace(" ", "")
            if ser.str.contains(r"^\d+%\d+%\d+%$", regex=True, na=False).any():
                packed_rr_col = c; break

    if packed_rr_col is not None and packed_rr_col not in core.columns:
        core[packed_rr_col] = df[packed_rr_col]
        print("Found compact RR column:", packed_rr_col)
    else:
        print("Compact RR column not found; app will still work if processed CSV already exists.")

    out = os.path.join(PROCESSED, "zus_forecast_all.csv")
    core.to_csv(out, index=False)
    print("Wrote processed CSV:", out)
    return core

def main():
    print("Extracting from:", PUB_PDF)
    dfs = extract_all(PUB_PDF)
    print(f"Extracted {len(dfs)} tables")
    save_interim_tables(dfs, prefix="tables")

    # Optional: extract details PDF too (kept for record)
    if os.path.exists(DETAILS_PDF):
        dfs2 = extract_all(DETAILS_PDF)
        print(f"Extracted {len(dfs2)} tables from DETAILS")
        save_interim_tables(dfs2, prefix="details")

    # Try to build processed CSV (heuristic). If you already have a curated CSV, keep it.
    try:
        build_processed_csv_from_interim()
    except Exception as e:
        print("Failed to build processed CSV automatically:", e)
        print("You can keep your curated data/processed/zus_forecast_all.csv as the source.")

if __name__ == "__main__":
    main()
