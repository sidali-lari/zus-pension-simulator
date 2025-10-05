
# -*- coding: utf-8 -*-
"""
ZUS Pension Simulator — FINAL Flask API
Covers: data cleaning, XGB RR model, simulator, overrides, admin XLS, user XLS report.

Endpoints:
- GET  /health
- GET  /model/metrics
- POST /forecast                 -> JSON in, JSON out (full results + admin log)
- POST /forecast/report          -> JSON in, returns .xlsx with inputs & tables
- GET  /admin/usage-report       -> download admin .xls (fallback .xlsx)

Env:
- ZUS_CSV_PATH           (default: ./zus_forecast_all.csv)
- BASE_AVG_WAGE_TODAY    (default: 8000, PLN/month)

Run:
    pip install flask flask-cors xgboost scikit-learn pandas numpy openpyxl xlwt
    python app.py
"""

import os, io, math, json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
try:
    from supabase import create_client, Client
except ImportError:
    print("Warning: Supabase not available, database saving will be disabled")
    create_client = None
    Client = None

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from xgboost import XGBRegressor

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

# -------------------------------
# Global Config
# -------------------------------
CURRENT_YEAR = 2025
DEFAULT_WOMEN_RET_AGE = 60
DEFAULT_MEN_RET_AGE   = 65
CONTRIB_RATE = 0.1952
AVG_SICKLEAVE_F_M = 0.02
AVG_SICKLEAVE_F_W = 0.03
POST_RETIRE_YEARS = 20
WORKDAYS_PER_YEAR = 260  # for info output of sick-leave days

#DATA_PATH = os.environ.get("ZUS_CSV_PATH", "zus_forecast_all.csv")
# was: DATA_PATH = os.environ.get("ZUS_CSV_PATH", "zus_forecast_all.csv")
DATA_PATH = os.environ.get("ZUS_CSV_PATH", "data/processed/zus_forecast_all.csv")

BASE_AVG_WAGE_TODAY = float(os.environ.get("BASE_AVG_WAGE_TODAY", "8000"))

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

USAGE_XLS = "zus_usage_report.xls"
USAGE_XLSX = "zus_usage_report.xlsx"

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# Supabase client
# -------------------------------
supabase: Optional[Client] = None

def init_supabase():
    global supabase
    if not create_client:
        print("Supabase not available, database saving disabled")
        supabase = None
        return
        
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("Supabase client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Supabase client: {e}")
            supabase = None
    else:
        print("Supabase credentials not provided, database saving disabled")
        supabase = None

# -------------------------------
# Inputs
# -------------------------------
@dataclass
class UserInputs:
    age: int
    sex: str                  # "M" / "F"
    gross_salary_now: float   # monthly PLN
    start_year: int
    ret_year: Optional[int] = None
    include_sickleave: bool = False
    accumulated_now: Optional[float] = None
    desired_pension: Optional[float] = None
    postal_code: Optional[str] = None
    valorization_rule: str = "CPI"               # "CPI" | "CPI_plus_alpha_wage"
    valorization_alpha: float = 0.2

    # Advanced overrides for dashboard features:
    historical_salaries: Optional[List[Dict]] = None      # [{year, monthly_salary}]
    future_wage_growth: Optional[List[Dict]] = None       # [{year, mult}]
    indexation_override: Optional[List[Dict]] = None      # [{year, mult}]
    sickleave_periods: Optional[List[Dict]] = None        # [{year, fraction}]

# -------------------------------
# XGBoost best params
# -------------------------------
BEST_XGB_PARAMS = {
    "n_estimators": 575,
    "learning_rate": 0.015177105064665856,
    "max_depth": 5,
    "min_child_weight": 1.2892258524609916,
    "subsample": 0.7560825001555036,
    "colsample_bytree": 0.8192272142388425,
    "reg_lambda": 1.229382076466428,
    "reg_alpha": 0.2816300941704181,
    "gamma": 0.17522781023199535,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

# -------------------------------
# Helpers
# -------------------------------
def statutory_retirement_year(ui: UserInputs) -> int:
    ret_age = DEFAULT_MEN_RET_AGE if ui.sex.upper() == "M" else DEFAULT_WOMEN_RET_AGE
    return CURRENT_YEAR + max(0, ret_age - ui.age)

def extend_list(series: List[float], length: int) -> List[float]:
    if not series:
        raise ValueError("Empty series.")
    return series[:length] + [series[-1]] * max(0, length - len(series))

def annuity_factor_exponential(life_expectancy_years: float, discount_rate: float, max_years: int = 200) -> float:
    A = 0.0
    for t in range(max_years + 1):
        surv = math.exp(-t / life_expectancy_years)
        disc = 1.0 / ((1.0 + discount_rate) ** t)
        term = surv * disc
        A += term
        if term < 1e-10:
            break
    return float(A)

def project_salaries_annual(gross_monthly_now: float, wage_growth_mults: List[float], years: int) -> List[float]:
    wage_growth_mults = extend_list(wage_growth_mults, years)
    sals, annual = [], gross_monthly_now * 12.0
    for i in range(years):
        if i > 0:
            annual *= wage_growth_mults[i - 1]
        sals.append(annual)
    return sals

# per-year sick leave aware accumulator
def accumulate_contrib_forward_per_year(
    salaries_annual: List[float],
    c: float,
    kappa: List[float],
    indexation: List[float],
    sick_frac_list: List[float],
    accumulated_now: float
) -> float:
    T = len(salaries_annual)
    kappa = extend_list(kappa, T)
    indexation = extend_list(indexation, T)
    if len(sick_frac_list) != T:
        sick_frac_list = extend_list(sick_frac_list, T)

    acc = accumulated_now
    for j in range(T):
        acc *= indexation[j]
    for i in range(T):
        contrib = salaries_annual[i] * c * kappa[i] * (1.0 - sick_frac_list[i])
        factor = np.prod(indexation[i + 1 :]) if i + 1 < T else 1.0
        acc += contrib * factor
    return float(acc)

# -------------------------------
# Data loading & cleaning
# -------------------------------
def load_and_prepare_macro(path_csv: str) -> pd.DataFrame:
    try:
        # Try to read the CSV with more robust settings
        df = pd.read_csv(path_csv, encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Error reading CSV with utf-8, trying latin-1: {e}")
        try:
            df = pd.read_csv(path_csv, encoding='latin-1', low_memory=False)
        except Exception as e2:
            print(f"Error reading CSV with latin-1, trying cp1252: {e2}")
            df = pd.read_csv(path_csv, encoding='cp1252', low_memory=False)

    print(f"Loaded CSV with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if we have the required column
    if "71%69%73%" not in df.columns:
        print("Warning: Missing '71%69%73%' column. Creating dummy data.")
        # Create dummy replacement rates if the column is missing
        years = [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080]
        dummy_data = []
        for year in years:
            dummy_data.append({
                "year": year,
                "replacement_rate_s1": 71.0,
                "replacement_rate_s2": 69.0,
                "replacement_rate_s3": 73.0,
                "unemployment_rate": 5.0,
                "cpi_overall": 3.0,
                "real_wage_growth": 2.0,
                "cpi_pensioners": 3.0
            })
        return pd.DataFrame(dummy_data)

    # Process replacement rate column
    rr_raw = df["71%69%73%"].dropna().astype(str)
    rr = rr_raw.str.extract(r'(\d+)%(\d+)%(\d+)%')
    rr.columns = ["replacement_rate_s1", "replacement_rate_s2", "replacement_rate_s3"]
    rr = rr.apply(pd.to_numeric, errors="coerce")

    # years
    year_cols = [c for c in df.columns if str(c).isdigit()]
    year_vec = pd.to_numeric(year_cols, errors="coerce")
    rr["year"] = year_vec[: len(rr)]

    # long->wide for macro indicators
    tidy = df.melt(id_vars=["rok"], value_vars=year_cols, var_name="year", value_name="value")
    wide = tidy.pivot_table(index="year", columns="rok", values="value", aggfunc="first").reset_index()

    rename_map = {
        "(stan na koniec roku)": "unemployment_rate",
        "ogółem": "cpi_overall",
        "przeciętnego wynagrodzenia": "real_wage_growth",
        "rencistów": "cpi_pensioners",
    }
    wide = wide.rename(columns=rename_map)

    for col in ["unemployment_rate", "cpi_overall", "real_wage_growth", "cpi_pensioners"]:
        if col in wide.columns:
            wide[col] = (
                wide[col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    wide["year"] = pd.to_numeric(wide["year"], errors="coerce")

    macro = wide.merge(
        rr[["year", "replacement_rate_s1", "replacement_rate_s2", "replacement_rate_s3"]],
        on="year",
        how="left",
    )
    macro = macro.sort_values("year").reset_index(drop=True)
    return macro

def annualize_macro(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("year")
    all_years = pd.DataFrame({"year": np.arange(int(df["year"].min()), int(df["year"].max()) + 1)})
    out = all_years.merge(df, on="year", how="left")

    cols_to_interp = [
        "cpi_overall",
        "cpi_pensioners",
        "real_wage_growth",
        "unemployment_rate",
        "replacement_rate_s1",
        "replacement_rate_s2",
        "replacement_rate_s3",
    ]
    for c in cols_to_interp:
        if c in out.columns:
            out[c] = out[c].interpolate().ffill().bfill()

    out["rr_path"] = out["replacement_rate_s2"].copy()
    return out

# -------------------------------
# Overrides parsing
# -------------------------------
def to_year_value_map(items: Optional[List[Dict]], year_key: str, value_key: str) -> Dict[int, float]:
    m = {}
    if items:
        for x in items:
            try:
                y = int(x[year_key])
                v = float(x[value_key])
                m[y] = v
            except Exception:
                continue
    return m

# -------------------------------
# Past accumulation (start_year -> CURRENT_YEAR)
# -------------------------------
def estimate_accumulated_now_from_history(
    gross_monthly_now: float,
    start_year: int,
    macro_annual: pd.DataFrame,
    sex: str,
    include_sickleave: bool,
    historical_salaries_map: Optional[Dict[int, float]] = None,  # monthly salary override
    sickleave_map_hist: Optional[Dict[int, float]] = None         # per-year fraction override
) -> float:
    if start_year >= CURRENT_YEAR:
        return 0.0

    m = macro_annual.sort_values("year").copy()
    years = list(range(start_year, CURRENT_YEAR))

    # pad if needed
    if start_year < int(m["year"].min()) or (CURRENT_YEAR - 1) > int(m["year"].max()):
        all_yrs = pd.DataFrame(
            {"year": np.arange(min(start_year, int(m["year"].min())),
                               max(CURRENT_YEAR - 1, int(m["year"].max())) + 1)}
        )
        m = all_yrs.merge(m, on="year", how="left").sort_values("year")
        for c in ["real_wage_growth", "cpi_overall"]:
            m[c] = m[c].interpolate().ffill().bfill()

    wg = (m.set_index("year").loc[years, "real_wage_growth"] / 100.0).tolist()

    salaries_month = []
    for i, y in enumerate(years):
        if historical_salaries_map and y in historical_salaries_map:
            base = float(historical_salaries_map[y])
        else:
            cum = np.prod(wg[i:]) if i < len(wg) else 1.0
            base = gross_monthly_now / (cum if cum > 0 else 1.0)
        salaries_month.append(base)
    salaries_annual = [s * 12.0 for s in salaries_month]

    # sick leave per year
    if include_sickleave:
        default_frac = AVG_SICKLEAVE_F_M if sex.upper() == "M" else AVG_SICKLEAVE_F_W
    else:
        default_frac = 0.0
    sick_fracs = [(sickleave_map_hist.get(y, default_frac) if sickleave_map_hist else default_frac) for y in years]

    kappa_hist = [0.99] * len(years)
    indexation_hist = (m.set_index("year").loc[years, "real_wage_growth"] / 100.0).tolist()

    acc = 0.0
    for i in range(len(years)):
        contrib = salaries_annual[i] * CONTRIB_RATE * kappa_hist[i] * (1.0 - sick_fracs[i])
        factor = np.prod(indexation_hist[i + 1 :]) if i + 1 < len(indexation_hist) else 1.0
        acc += contrib * factor
    return float(acc)

# -------------------------------
# Average wage & average benefit
# -------------------------------
def forecast_avg_wage_annual(macro_annual: pd.DataFrame, base_avg_wage_today_pln: float) -> pd.DataFrame:
    m = macro_annual.sort_values("year").copy()
    years = m["year"].values.tolist()
    wz = (m["real_wage_growth"] / 100.0).fillna(1.0).values.tolist()
    start_idx = years.index(CURRENT_YEAR) if CURRENT_YEAR in years else 0
    avg_month = base_avg_wage_today_pln
    path = []
    for i, y in enumerate(years):
        if i == start_idx:
            avg_month = base_avg_wage_today_pln
        elif i > start_idx:
            avg_month *= wz[i - 1]
        path.append({"year": y, "avg_wage_annual": avg_month * 12.0})
    return pd.DataFrame(path)

def avg_benefit_in_year(macro_annual: pd.DataFrame, avg_wage_df: pd.DataFrame, year: int, rr_col: str = "rr_path") -> float:
    rr = macro_annual.loc[macro_annual["year"] == year, rr_col]
    wage = avg_wage_df.loc[avg_wage_df["year"] == year, "avg_wage_annual"]
    if len(rr) == 0 or len(wage) == 0 or pd.isna(rr.values[0]) or pd.isna(wage.values[0]):
        return np.nan
    return float(wage.values[0] * (rr.values[0] / 100.0) / 12.0)

# -------------------------------
# XGBoost RR model
# -------------------------------
def make_supervised_rr(macro: pd.DataFrame, lags: List[int] = [1, 2, 3, 4], include_macro: bool = True):
    df = macro.copy().sort_values("year")
    df["RR"] = df["replacement_rate_s2"]
    for L in lags:
        df[f"RR_lag{L}"] = df["RR"].shift(L)
    feat_cols = [f"RR_lag{L}" for L in lags]
    if include_macro:
        for mcol in ["unemployment_rate", "cpi_overall", "real_wage_growth", "cpi_pensioners"]:
            if mcol in df.columns:
                feat_cols.append(mcol)
    df = df.dropna(subset=feat_cols + ["RR"])
    X = df[feat_cols].copy()
    y = df["RR"].copy()
    X.index = df["year"]
    y.index = df["year"]
    return X, y, feat_cols

def fit_xgb_best(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(**BEST_XGB_PARAMS)
    model.fit(X.values, y.values)
    return model

def forecast_rr_recursive_xgb(model: XGBRegressor, macro: pd.DataFrame, lags: List[int] = [1, 2, 3, 4], end_year: int = 2080) -> pd.DataFrame:
    df = macro.copy().sort_values("year").reset_index(drop=True)
    if "rr_path" not in df.columns:
        df["rr_path"] = df["replacement_rate_s2"]
    rr_series = df["rr_path"].combine_first(df["replacement_rate_s2"]).interpolate().ffill().bfill()
    buffer = rr_series.iloc[-max(lags) :].tolist()
    last_year = int(df["year"].max())
    macro_cols = ["unemployment_rate", "cpi_overall", "real_wage_growth", "cpi_pensioners"]
    mindex = df.set_index("year")

    for y in df["year"]:
        if pd.isna(mindex.at[y, "rr_path"]):
            feat = {}
            for L in lags:
                feat[f"RR_lag{L}"] = buffer[-L] if L <= len(buffer) else buffer[0]
            for mcol in macro_cols:
                if mcol in mindex.columns:
                    val = mindex.at[y, mcol]
                    feat[mcol] = float(val) if not pd.isna(val) else float(pd.Series(mindex[mcol]).dropna().iloc[-1])
            pred = float(model.predict(pd.DataFrame([feat]).values)[0])
            df.loc[df["year"] == y, "rr_path"] = pred
            buffer.append(pred)
            if len(buffer) > max(lags):
                buffer = buffer[-max(lags) :]

    future_rows = []
    for y in range(last_year + 1, end_year + 1):
        feat = {}
        for L in lags:
            feat[f"RR_lag{L}"] = buffer[-L] if L <= len(buffer) else buffer[0]
        feat_vals = {}
        for mcol in macro_cols:
            if mcol in mindex.columns and y in mindex.index:
                val = mindex.at[y, mcol]
                feat_vals[mcol] = float(val) if not pd.isna(val) else float(pd.Series(mindex[mcol]).dropna().iloc[-1])
            else:
                feat_vals[mcol] = float(pd.Series(mindex[mcol]).dropna().iloc[-1]) if mcol in mindex.columns else 0.0
        feat.update(feat_vals)
        pred = float(model.predict(pd.DataFrame([feat]).values)[0])
        future_rows.append({"year": y, "rr_path": pred})
        buffer.append(pred)
        if len(buffer) > max(lags):
            buffer = buffer[-max(lags) :]

    if future_rows:
        df = pd.concat([df, pd.DataFrame(future_rows)], ignore_index=True)
    return df.sort_values("year").reset_index(drop=True)

def cv_metrics_xgb(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=max(2, min(n_splits, len(X) - 1)))
    preds, actuals = [], []
    for tr, te in tscv.split(X):
        model = XGBRegressor(**BEST_XGB_PARAMS)
        model.fit(X.values[tr], y.values[tr])
        p = model.predict(X.values[te])
        preds.extend(p.tolist()); actuals.extend(y.values[te].tolist())
    mape = float(mean_absolute_percentage_error(actuals, preds) * 100.0)
    mae_pp = float(mean_absolute_error(actuals, preds))
    return {"cv_mape_pct": mape, "cv_accuracy_pct": 100.0 - mape, "cv_mae_pp": mae_pp}

# -------------------------------
# Simulator (with overrides, no recursion)
# -------------------------------
def simulate_user_pension_enhanced(
    ui: UserInputs,
    macro_annual: pd.DataFrame,
    life_expectancy_at_ret: float,
    annuity_discount_rate: float = 0.02,
    base_avg_wage_today_pln: float = BASE_AVG_WAGE_TODAY,
    compute_extras: bool = True
) -> Dict:

    # Build override maps
    hist_salary_map = to_year_value_map(ui.historical_salaries, "year", "monthly_salary") if ui.historical_salaries else {}
    sick_hist_map   = to_year_value_map(ui.sickleave_periods, "year", "fraction") if ui.sickleave_periods else None

    # Auto-estimate accumulated_now if missing
    if ui.accumulated_now is None:
        accumulated_now = estimate_accumulated_now_from_history(
            ui.gross_salary_now, ui.start_year, macro_annual, ui.sex, ui.include_sickleave,
            historical_salaries_map=hist_salary_map,
            sickleave_map_hist=sick_hist_map
        )
    else:
        accumulated_now = float(ui.accumulated_now)

    # Years to retirement
    ret_year = ui.ret_year if ui.ret_year else statutory_retirement_year(ui)
    T = ret_year - CURRENT_YEAR
    if T <= 0:
        raise ValueError("ret_year must be after CURRENT_YEAR.")

    # Macro slice
    sub = macro_annual[(macro_annual["year"] >= CURRENT_YEAR) & (macro_annual["year"] < ret_year)].copy()
    if sub.empty:
        raise ValueError("Macro series missing for the projection horizon.")

    # Base series from macro
    wage_growth_mults = (sub["real_wage_growth"] / 100.0).fillna(1.0).tolist()
    inflation_mults   = (sub["cpi_overall"] / 100.0).fillna(1.0).tolist()
    indexation_mults  = wage_growth_mults[:]  # default FUS convention
    kappa             = [0.99] * T

    # Apply future overrides (year-based)
    fw_map  = to_year_value_map(ui.future_wage_growth, "year", "mult") if ui.future_wage_growth else {}
    idx_map = to_year_value_map(ui.indexation_override, "year", "mult") if ui.indexation_override else {}
    sick_map_future = to_year_value_map(ui.sickleave_periods, "year", "fraction") if ui.sickleave_periods else {}

    years_fwd = list(range(CURRENT_YEAR, ret_year))
    for i, y in enumerate(years_fwd):
        if y in fw_map:
            wage_growth_mults[i] = fw_map[y]
        if y in idx_map:
            indexation_mults[i] = idx_map[y]

    # Sick leave — per-year fractions
    if ui.include_sickleave:
        default_sick_frac = AVG_SICKLEAVE_F_M if ui.sex.upper() == "M" else AVG_SICKLEAVE_F_W
    else:
        default_sick_frac = 0.0
    sick_frac_list = [sick_map_future.get(y, default_sick_frac) for y in years_fwd]

    # Project salaries and accumulate with per-year sick leave
    salaries_with = project_salaries_annual(ui.gross_salary_now, wage_growth_mults, T)
    acc_with = accumulate_contrib_forward_per_year(salaries_with, CONTRIB_RATE, kappa, indexation_mults, sick_frac_list, accumulated_now)
    acc_without = accumulate_contrib_forward_per_year(salaries_with, CONTRIB_RATE, kappa, indexation_mults, [0.0]*T, accumulated_now)

    # Convert to annuity
    A = annuity_factor_exponential(life_expectancy_at_ret, annuity_discount_rate)
    pension_annual_with    = acc_with / A
    pension_annual_without = acc_without / A
    pen_month_nominal         = pension_annual_with / 12.0
    pen_month_nominal_no_sick = pension_annual_without / 12.0

    # Real today
    infl_factor = float(np.prod(inflation_mults)) if len(inflation_mults) else 1.0
    pen_month_real_today = pen_month_nominal / infl_factor

    # Final (contributory) wage with sick leave in final year (for “including vs excluding” comparison)
    final_salary_annual = salaries_with[-1] if salaries_with else ui.gross_salary_now * 12.0
    final_year_sick = sick_frac_list[-1] if sick_frac_list else 0.0
    final_salary_annual_excl_sick = final_salary_annual * (1.0 - final_year_sick)

    replacement_rate = pension_annual_with / final_salary_annual if final_salary_annual > 0 else 0.0

    # 20y post-retirement path with valorization
    sub_after = macro_annual[(macro_annual["year"] >= ret_year) & (macro_annual["year"] <= ret_year + POST_RETIRE_YEARS)].copy()
    pen_nom, pen_real, nominal, real_base = [], [], pen_month_nominal, 1.0
    for i, y in enumerate(range(ret_year, ret_year + POST_RETIRE_YEARS + 1)):
        if i == 0:
            pen_nom.append(nominal); pen_real.append(nominal)
        else:
            cpi_p = sub_after.loc[sub_after["year"] == y, "cpi_pensioners"].values
            rw    = sub_after.loc[sub_after["year"] == y, "real_wage_growth"].values
            cpi_mult = (cpi_p[0] / 100.0) if (len(cpi_p) and not np.isnan(cpi_p[0])) else 1.0
            if ui.valorization_rule == "CPI_plus_alpha_wage":
                rw_mult = (rw[0] / 100.0) if (len(rw) and not np.isnan(rw[0])) else 1.0
                valoriz_factor = cpi_mult * (1.0 + ui.valorization_alpha * (rw_mult - 1.0))
            else:
                valoriz_factor = cpi_mult
            nominal *= valoriz_factor
            pen_nom.append(nominal)

            cpi_o = sub_after.loc[sub_after["year"] == y, "cpi_overall"].values
            cpi_o_mult = (cpi_o[0] / 100.0) if (len(cpi_o) and not np.isnan(cpi_o[0])) else 1.0
            real_base *= cpi_o_mult
            pen_real.append(nominal / real_base)

    path_df = pd.DataFrame({
        "year": list(range(ret_year, ret_year + POST_RETIRE_YEARS + 1)),
        "pension_monthly_nominal": np.round(pen_nom, 2),
        "pension_monthly_real": np.round(pen_real, 2),
    })

    # Average benefit comparison (current & retirement year)
    avg_wage_df = forecast_avg_wage_annual(macro_annual, BASE_AVG_WAGE_TODAY)
    rr_col = "rr_path" if "rr_path" in macro_annual.columns else "replacement_rate_s2"
    avg_benefit_retire = avg_benefit_in_year(macro_annual, avg_wage_df, ret_year, rr_col=rr_col)
    avg_benefit_current = avg_benefit_in_year(macro_annual, avg_wage_df, CURRENT_YEAR, rr_col=rr_col)

    compare_vs_avg_ret = None
    if not pd.isna(avg_benefit_retire):
        compare_vs_avg_ret = {
            "diff_PLN_per_month": round(pen_month_nominal - avg_benefit_retire, 2),
            "ratio_vs_avg": round((pen_month_nominal / avg_benefit_retire) if avg_benefit_retire > 0 else np.nan, 3),
        }

    # Sick-leave info for UI (average days)
    avg_sick_frac_used = float(np.mean(sick_frac_list)) if sick_frac_list else 0.0
    sickleave_days_assumed = round(avg_sick_frac_used * WORKDAYS_PER_YEAR, 1)

    # Delay scenarios (+1/+2/+5), guarded to avoid recursion storm
    if compute_extras:
        def delay_nominal(extra_years: int) -> float:
            ui2 = UserInputs(**{**asdict(ui), "age": ui.age + extra_years, "ret_year": ret_year + extra_years})
            sex_LE = 18.0 if ui.sex.upper() == "M" else 22.0
            res = simulate_user_pension_enhanced(ui2, macro_annual, sex_LE, annuity_discount_rate, base_avg_wage_today_pln, compute_extras=False)
            return res["pension_first_year_nominal"]

        inc_1y = delay_nominal(1) - pen_month_nominal
        inc_2y = delay_nominal(2) - pen_month_nominal
        inc_5y = delay_nominal(5) - pen_month_nominal

        years_needed, target_year = None, None
        if ui.desired_pension is not None:
            for extra in range(0, 41):
                ui2 = UserInputs(**{**asdict(ui), "age": ui.age + extra, "ret_year": ret_year + extra})
                sex_LE = 18.0 if ui.sex.upper() == "M" else 22.0
                res2 = simulate_user_pension_enhanced(ui2, macro_annual, sex_LE, annuity_discount_rate, base_avg_wage_today_pln, compute_extras=False)
                if res2["pension_first_year_nominal"] >= ui.desired_pension:
                    years_needed = extra; target_year = ret_year + extra; break
    else:
        inc_1y = inc_2y = inc_5y = None
        years_needed = target_year = None

    return {
        "retirement_year": ret_year,
        "final_salary_annual_including_sick": round(final_salary_annual, 2),
        "final_salary_annual_excluding_sick": round(final_salary_annual_excl_sick, 2),
        "accumulated_future": round(acc_with, 2),
        "accumulated_future_no_sick": round(acc_without, 2),
        "replacement_rate_percent": round(replacement_rate * 100.0, 2),
        "pension_first_year_nominal": round(pen_month_nominal, 2),
        "pension_first_year_real_today": round(pen_month_real_today, 2),
        "pension_first_year_nominal_no_sick": round(pen_month_nominal_no_sick, 2),
        "pension_path_20y": path_df.to_dict(orient="records"),
        "avg_benefit_retire_monthly": None if pd.isna(avg_benefit_retire) else round(avg_benefit_retire, 2),
        "avg_benefit_current_monthly": None if pd.isna(avg_benefit_current) else round(avg_benefit_current, 2),
        "compare_vs_avg_retire": compare_vs_avg_ret,
        "sickleave_assumptions": {
            "assumed_fraction_avg": round(avg_sick_frac_used, 4),
            "assumed_workdays_per_year": WORKDAYS_PER_YEAR,
            "assumed_days_per_year": sickleave_days_assumed
        },
        "increase_if_delay": {
            "+1y_PLN_per_month": None if inc_1y is None else round(inc_1y, 2),
            "+2y_PLN_per_month": None if inc_2y is None else round(inc_2y, 2),
            "+5y_PLN_per_month": None if inc_5y is None else round(inc_5y, 2),
        },
        "years_needed_for_desired": years_needed,
        "target_year_for_desired": target_year
    }

# -------------------------------
# Admin usage report
# -------------------------------
def usage_record_row(ui: UserInputs, sim: Dict) -> Dict:
    now = datetime.now(ZoneInfo("Europe/Warsaw")) if ZoneInfo else datetime.now()
    return {
        "Date of use": now.strftime("%Y-%m-%d"),
        "Time of use": now.strftime("%H:%M:%S"),
        "Expected pension": ui.desired_pension,
        "Age": ui.age,
        "Sex": ui.sex,
        "Salary amount": ui.gross_salary_now,
        "Whether periods of illness were included": bool(ui.include_sickleave),
        "Amount of funds accumulated in the account and Sub-account": ui.accumulated_now,
        "Actual pension": sim["pension_first_year_nominal"],
        "Real (inflation-adjusted) pension": sim["pension_first_year_real_today"],
        "Postal code": ui.postal_code,
        "Year of starting work": ui.start_year,
        "Planned year of ending professional activity": sim["retirement_year"],
    }

def save_forecast_to_supabase(ui: UserInputs, sim: Dict) -> bool:
    """Save forecast data to Supabase 'forecast' table"""
    if not supabase or not create_client:
        return False
    
    try:
        # Prepare data according to the specified format
        forecast_data = {
            'data_symulacji': datetime.now().strftime('%d.%m.%Y, %H:%M'),
            'wiek': ui.age,
            'plec': ui.sex,
            'wynagrodzenie': ui.gross_salary_now,
            'kod_pocztowy': ui.postal_code or '—',
            'pozadana_emerytura': ui.desired_pension,
            'zgromadzone_srodki': ui.accumulated_now,
            'l4_wliczone': 'Tak' if ui.include_sickleave else 'Nie',
            'prognozowana_emerytura': sim.get('pension_first_year_nominal'),
            'realna_emerytura': sim.get('pension_first_year_real_today')
        }
        
        # Insert into Supabase
        result = supabase.table('forecast').insert(forecast_data).execute()
        print(f"Forecast data saved to Supabase: {result}")
        return True
        
    except Exception as e:
        print(f"Failed to save forecast to Supabase: {e}")
        return False

def append_usage_report(records: List[Dict]):
    df_new = pd.DataFrame.from_records(records)

    def _append(base_path):
        try:
            old = pd.read_excel(base_path)
            return pd.concat([old, df_new], ignore_index=True)
        except FileNotFoundError:
            return df_new

    try:
        import xlwt  # noqa
        df_out = _append(USAGE_XLS)
        df_out.to_excel(USAGE_XLS, index=False, engine="xlwt")
        saved = USAGE_XLS
    except Exception:
        df_out = _append(USAGE_XLSX)
        df_out.to_excel(USAGE_XLSX, index=False)
        saved = USAGE_XLSX
    return saved

# -------------------------------
# Excel user report
# -------------------------------
def build_user_report_bytes(ui: UserInputs, sim: Dict, model_cv: Dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # Inputs
        pd.DataFrame([asdict(ui)]).to_excel(xw, index=False, sheet_name="Inputs")

        # Results (high-level)
        high = {
            k: v for k, v in sim.items()
            if k in [
                "retirement_year",
                "final_salary_annual_including_sick",
                "final_salary_annual_excluding_sick",
                "accumulated_future",
                "accumulated_future_no_sick",
                "replacement_rate_percent",
                "pension_first_year_nominal",
                "pension_first_year_real_today",
                "pension_first_year_nominal_no_sick",
                "avg_benefit_current_monthly",
                "avg_benefit_retire_monthly",
                "years_needed_for_desired",
                "target_year_for_desired",
            ]
        }
        pd.DataFrame([high]).to_excel(xw, index=False, sheet_name="Results")

        # Pension path (20y)
        pd.DataFrame(sim["pension_path_20y"]).to_excel(xw, index=False, sheet_name="PensionPath20y")

        # Comparisons & deltas
        comp = {
            **(sim.get("compare_vs_avg_retire") or {}),
            **sim.get("increase_if_delay", {}),
            **sim.get("sickleave_assumptions", {})
        }
        pd.DataFrame([comp]).to_excel(xw, index=False, sheet_name="Comparisons")

        # Model metrics (for transparency)
        pd.DataFrame([model_cv]).to_excel(xw, index=False, sheet_name="ModelMetrics")

    buf.seek(0)
    return buf.getvalue()

# -------------------------------
# Globals initialized at startup
# -------------------------------
macro_raw: Optional[pd.DataFrame] = None
macro_annual: Optional[pd.DataFrame] = None
xgb_model: Optional[XGBRegressor] = None
macro_with_rr: Optional[pd.DataFrame] = None
model_metrics: Dict[str, float] = {}

def init_pipeline():
    global macro_raw, macro_annual, xgb_model, macro_with_rr, model_metrics

    macro_raw = load_and_prepare_macro(DATA_PATH)
    macro_annual = annualize_macro(macro_raw)

    X, y, _ = make_supervised_rr(macro_annual, lags=[1, 2, 3, 4], include_macro=True)
    xgb_model = fit_xgb_best(X, y)
    model_metrics = cv_metrics_xgb(X, y, n_splits=3)

    macro_with_rr = forecast_rr_recursive_xgb(xgb_model, macro_annual, lags=[1, 2, 3, 4], end_year=2080)
    
    # Initialize Supabase
    init_supabase()

# -------------------------------
# Routes
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    ok = all(v is not None for v in [macro_with_rr, xgb_model])
    return jsonify({"ok": ok, "data_loaded": macro_with_rr is not None})

@app.route("/model/metrics", methods=["GET"])
def metrics():
    return jsonify({"xgb_best_params": BEST_XGB_PARAMS, "cv": model_metrics})

def parse_user_inputs(payload: dict) -> UserInputs:
    # Minimal validation
    for key in ["age", "sex", "gross_salary_now", "start_year"]:
        if key not in payload:
            raise ValueError(f"Missing field '{key}'")
    # Build dataclass with advanced overrides too
    return UserInputs(
        age=int(payload["age"]),
        sex=str(payload["sex"]),
        gross_salary_now=float(payload["gross_salary_now"]),
        start_year=int(payload["start_year"]),
        ret_year=int(payload["ret_year"]) if payload.get("ret_year") else None,
        include_sickleave=bool(payload.get("include_sickleave", False)),
        accumulated_now=float(payload["accumulated_now"]) if payload.get("accumulated_now") is not None else None,
        desired_pension=float(payload["desired_pension"]) if payload.get("desired_pension") is not None else None,
        postal_code=payload.get("postal_code"),
        valorization_rule=str(payload.get("valorization_rule", "CPI")),
        valorization_alpha=float(payload.get("valorization_alpha", 0.2)),
        historical_salaries=payload.get("historical_salaries"),
        future_wage_growth=payload.get("future_wage_growth"),
        indexation_override=payload.get("indexation_override"),
        sickleave_periods=payload.get("sickleave_periods"),
    )

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        payload = request.get_json(force=True)
        ui = parse_user_inputs(payload)
        life_exp = 18.0 if ui.sex.upper() == "M" else 22.0
        sim = simulate_user_pension_enhanced(
            ui, macro_with_rr, life_expectancy_at_ret=life_exp,
            annuity_discount_rate=0.02, base_avg_wage_today_pln=BASE_AVG_WAGE_TODAY
        )
        # Save to Supabase
        supabase_saved = save_forecast_to_supabase(ui, sim)
        
        # Save to local Excel (existing functionality)
        saved_path = append_usage_report([usage_record_row(ui, sim)])
        
        return jsonify({
            "inputs": asdict(ui), 
            "result": sim, 
            "admin_usage_saved_to": saved_path,
            "supabase_saved": supabase_saved
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/forecast/report", methods=["POST"])
def forecast_report():
    try:
        payload = request.get_json(force=True)
        ui = parse_user_inputs(payload)
        life_exp = 18.0 if ui.sex.upper() == "M" else 22.0
        sim = simulate_user_pension_enhanced(
            ui, macro_with_rr, life_expectancy_at_ret=life_exp,
            annuity_discount_rate=0.02, base_avg_wage_today_pln=BASE_AVG_WAGE_TODAY
        )
        xlsx_bytes = build_user_report_bytes(ui, sim, model_metrics)
        fname = f"ZUS_Pension_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            io.BytesIO(xlsx_bytes),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=fname,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/usage-report", methods=["GET"])
def download_report():
    path = USAGE_XLS if os.path.exists(USAGE_XLS) else (USAGE_XLSX if os.path.exists(USAGE_XLSX) else None)
    if not path:
        return jsonify({"error": "No usage report found yet."}), 404
    mime = "application/vnd.ms-excel" if path.endswith(".xls") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    with open(path, "rb") as f:
        data = f.read()
    return send_file(io.BytesIO(data), mimetype=mime, as_attachment=True, download_name=os.path.basename(path))

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    init_pipeline()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
