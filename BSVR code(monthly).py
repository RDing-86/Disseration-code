# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 11:24:54 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
Generate 4-panel plot: (a) Testing Scatter  (b) Training Scatter  (c) Testing Time Series  (d) Training Time Series
Model: FF3 or FF6 (includes WML)
Data:
  Six Factors: D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv
  Industry   : D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ---------------- Paths & Adjustable Parameters ----------------
PATH_FACTORS = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
MODEL = "FF3"          # Choose "FF3" or "FF6"
INDUSTRY = "Consumption"  # Choose Consumption/Manufacturing/Hi-Tech/Health/Other (case-insensitive, auto-matches Cnsmer/Manuf/HiTec/Hlth/Other)
TEST_SIZE = 0.30
FAST = True             # Fast mode (fewer iterations, only linear/rbf)
FAST_LAST_N = 240       # In FAST mode, use only last N months; set to None to use full sample
OUTFILE = f"plots_{MODEL}_{INDUSTRY}.png"
# ---------------- Utility Functions ----------------
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    if not os.path.exists(path): raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError("Header not detected, please check column names")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df.replace(-99.99, np.nan)
    return df
def align_on_date(dfs):
    common = dfs[0][["Date"]]
    for d in dfs[1:]:
        common = common.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(common["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols
def canonical_ind_name(s):
    s0 = s.lower()
    if s0.startswith(("cns","con","cnsmr")): return "Consumption"
    if s0.startswith(("man","mfg")):         return "Manufacturing"
    if s0.startswith(("hit","tec","high")):  return "Hi-Tech"
    if s0.startswith(("hlth","hea")):        return "Health"
    return "Other"
def match_industry(industry, ind_cols):
    target = industry.lower()
    for c in ind_cols:
        if canonical_ind_name(c).lower() == target:
            return c
    raise ValueError(f"Industry column not found: {industry}, available: {[canonical_ind_name(c) for c in ind_cols]}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_splits = 3 if fast else 5
    n_iter   = 12 if fast else 60
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel": Categorical(kernels),
        "svr__C": Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma": Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree": Integer(2,3),
        "svr__coef0": Real(0.0,1.0)
    }
    cv = TimeSeriesSplit(n_splits=n_splits)
    opt = BayesSearchCV(pipe, space, n_iter=n_iter, cv=cv,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def compute_corr(y, yhat, method="spearman"):
    if method == "spearman":
        rho, pval = spearmanr(y, yhat, nan_policy="omit")
    else:
        # Pearson
        from scipy.stats import pearsonr
        rho, pval = pearsonr(pd.Series(y).dropna().align(pd.Series(yhat).dropna(), join="inner")[0],
                             pd.Series(yhat).dropna().align(pd.Series(y).dropna(), join="inner")[0])
    return float(rho), float(pval)
# ---------------- Main Process ----------------
# Read data
fac = read_csv_auto_header(PATH_FACTORS, ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cns","Cnsmr","Cnsmer","Manuf","HiTec","Hlth","Other"])
fac, ind = align_on_date([fac, ind])
# Use only last N months (fast mode)
if FAST and FAST_LAST_N is not None and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
# Industry excess returns
ind_cols = infer_industry_cols(ind)
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values
# Select model and features
if MODEL.upper() == "FF3":
    X = fac[["Mkt-RF","SMB","HML"]].to_numpy()
    mdl_label = "3-Factor"
elif MODEL.upper() == "FF6":
    need = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
    miss = [c for c in need if c not in fac.columns]
    if miss: raise ValueError(f"Missing factor columns: {miss}")
    X = fac[need].to_numpy()
    mdl_label = "6-Factor"
else:
    raise ValueError("MODEL only supports FF3 or FF6")
# Match industry column
col_raw = match_industry(INDUSTRY, ind_cols)
y = ind[col_raw].to_numpy()
# Split train/test
n = len(y)
split = int(round(n*(1-TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Train
est = tune_and_fit(X_tr, y_tr, fast=FAST)
# Predict
yhat_tr = est.predict(X_tr)
yhat_te = est.predict(X_te)
# Correlation coefficient (Spearman, consistent with RHO in your figure)
rho_te, p_te = compute_corr(y_te, yhat_te, method="spearman")
rho_tr, p_tr = compute_corr(y_tr, yhat_tr, method="spearman")
print(f"[INFO] {mdl_label} - {INDUSTRY} | Samples={n} Train={split} Test={n-split}")
print(f"Testing  Spearman rho={rho_te:.5f}, pval={p_te:.3e}")
print(f"Training Spearman rho={rho_tr:.5f}, pval={p_tr:.3e}")
# ---------------- Plot 4-panel Figure ----------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
ax_sc_te, ax_sc_tr, ax_ts_te, ax_ts_tr = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
# (a) Testing Scatter
ax_sc_te.scatter(y_te, yhat_te, s=25)
ax_sc_te.set_xlabel("experimental crf")
ax_sc_te.set_ylabel("crf predicted")
ax_sc_te.set_title("(a) Testing")
ax_sc_te.text(0.02, 0.96,
              f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_te.transAxes, va="top")
# (b) Training Scatter
ax_sc_tr.scatter(y_tr, yhat_tr, s=15)
ax_sc_tr.set_xlabel("experimental crf")
ax_sc_tr.set_ylabel("crf predicted")
ax_sc_tr.set_title("(b) Training")
ax_sc_tr.text(0.02, 0.96,
              f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_tr.transAxes, va="top")
# (c) Testing Time Series
ax_ts_te.plot(range(len(y_te)), y_te, marker="o", linestyle="-", linewidth=1, markersize=3, label="experimental crf")
ax_ts_te.plot(range(len(y_te)), yhat_te, marker="o", linestyle="-", linewidth=1, markersize=3, label="predicted crf")
ax_ts_te.set_xlabel("crf points")
ax_ts_te.set_ylabel("crf")
ax_ts_te.set_title("(c) Testing")
ax_ts_te.legend(loc="upper right")
# (d) Training Time Series
ax_ts_tr.plot(range(len(y_tr)), y_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label="experimental crf")
ax_ts_tr.plot(range(len(y_tr)), yhat_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label="predicted crf")
ax_ts_tr.set_xlabel("crf points")
ax_ts_tr.set_ylabel("crf")
ax_ts_tr.set_title("(d) Training")
ax_ts_tr.legend(loc="upper right")
fig.suptitle(f"{mdl_label} — {INDUSTRY}", y=0.98, fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ================= User-modifiable Parameters =================
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
MODEL    = "FF3"          # Choose "FF3" or "FF6"
INDUSTRY = "manurf"       # e.g., "manurf"/"Manufacturing"/"Manuf"/"HiTec"/"hitecrf" etc.
TEST_SIZE = 0.30          # Test set ratio (split from end by time)
FAST = True               # Fast mode (fewer iterations, search only linear/rbf)
FAST_LAST_N = 240         # In FAST mode, use only last N months; set to None for full sample
RHO_METHOD = "spearman"   # "spearman" or "pearson"
# Output filename
OUTFILE = f"plots_{MODEL}_{INDUSTRY}.png"
# ================= Utility Functions =================
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    if not os.path.exists(path): raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError(f"Header not detected: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    first = df.columns[0]
    # Keep only YYYYMM rows
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df.replace(-99.99, np.nan)
    return df
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    out = []
    for d in dfs:
        out.append(d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True))
    return out
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols
def _norm_industry_name(s: str) -> str:
    """Normalize various spellings to standard labels: Cnsmr/Manuf/HiTec/Hlth/Other"""
    s = s.strip().lower().replace("-", "")
    # Remove trailing rf/crf
    s = re.sub(r"(crf|rf)$", "", s)
    if s.startswith(("cns","con","cnsmr")): return "Cnsmr"
    if s.startswith(("man","mfg","manu")):  return "Manuf"
    if s.startswith(("hit","tec","high")):  return "HiTec"
    if s.startswith(("hea","hlth")):        return "Hlth"
    return "Other"
def match_industry(target, ind_cols):
    want = _norm_industry_name(target)
    for c in ind_cols:
        if _norm_industry_name(c) == want:
            return c
    raise ValueError(f"Industry column not found: {target}; available: {ind_cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_splits = 3 if fast else 5
    n_iter   = 12 if fast else 60
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma":   Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree":  Integer(2, 3),
        "svr__coef0":   Real(0.0, 1.0)
    }
    cv = TimeSeriesSplit(n_splits=n_splits)
    opt = BayesSearchCV(pipe, space, n_iter=n_iter, cv=cv,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def corr_with_p(y, yhat, method="spearman"):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if method == "pearson":
        from scipy.stats import pearsonr
        r, p = pearsonr(y[mask], yhat[mask])
    else:
        r, p = spearmanr(y[mask], yhat[mask])
    return float(r), float(p)
# ================= Main Process =================
# Load data
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
# Standardize aliases
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cns","Cnsmr","Cnsmer","Manuf","HiTec","Hlth","Other"])
# Align by date
fac, ind = align_on_date([fac, ind])
# Fast mode: use only last N periods
if FAST and FAST_LAST_N is not None and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
# Industry columns & excess returns
ind_cols = infer_industry_cols(ind)
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values
# Feature matrix
if MODEL.upper() == "FF3":
    X = fac[["Mkt-RF","SMB","HML"]].to_numpy()
    model_label = "3-Factor"
elif MODEL.upper() == "FF6":
    need = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
    miss = [c for c in need if c not in fac.columns]
    if miss: raise ValueError(f"Missing factor columns: {miss}")
    X = fac[need].to_numpy()
    model_label = "6-Factor"
else:
    raise ValueError("MODEL only supports FF3 or FF6")
# Match industry column
col_name = match_industry(INDUSTRY, ind_cols)
y = ind[col_name].to_numpy()
# Split
n = len(y)
split = int(round(n * (1 - TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Train and predict
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr)
yhat_te = est.predict(X_te)
# Correlation coefficients
rho_te, p_te = corr_with_p(y_te, yhat_te, method=RHO_METHOD)
rho_tr, p_tr = corr_with_p(y_tr, yhat_tr, method=RHO_METHOD)
print(f"[INFO] {model_label} — {INDUSTRY} (Actual column name: {col_name})")
print(f"Samples={n} Train={split} Test={n-split}")
print(f"Testing  RHO={rho_te:.5f}, p={p_te:.3e}")
print(f"Training RHO={rho_tr:.5f}, p={p_tr:.3e}")
# ================= Plot 4-panel Figure =================
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
(ax_sc_te, ax_sc_tr), (ax_ts_te, ax_ts_tr) = axes
# (a) Testing Scatter
ax_sc_te.scatter(y_te, yhat_te, s=18)
ax_sc_te.set_xlabel(f"experimental {INDUSTRY}")
ax_sc_te.set_ylabel(f"{INDUSTRY} predicted")
ax_sc_te.set_title("(a) Testing")
ax_sc_te.text(0.02, 0.96,
              f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_te.transAxes, va="top")
# (b) Training Scatter
ax_sc_tr.scatter(y_tr, yhat_tr, s=12)
ax_sc_tr.set_xlabel(f"experimental {INDUSTRY}")
ax_sc_tr.set_ylabel(f"{INDUSTRY} predicted")
ax_sc_tr.set_title("(b) Training")
ax_sc_tr.text(0.02, 0.96,
              f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_tr.transAxes, va="top")
# (c) Testing Time Series
ax_ts_te.plot(range(len(y_te)), y_te,  marker="o", linestyle="-", linewidth=1, markersize=3, label=f"experimental {INDUSTRY}")
ax_ts_te.plot(range(len(y_te)), yhat_te, marker="o", linestyle="-", linewidth=1, markersize=3, label=f"predicted {INDUSTRY}")
ax_ts_te.set_xlabel(f"{INDUSTRY} points")
ax_ts_te.set_ylabel(INDUSTRY)
ax_ts_te.set_title("(c) Testing")
ax_ts_te.legend(loc="upper right")
# (d) Training Time Series
ax_ts_tr.plot(range(len(y_tr)), y_tr,  marker="o", linestyle="-", linewidth=1, markersize=2, label=f"experimental {INDUSTRY}")
ax_ts_tr.plot(range(len(y_tr)), yhat_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label=f"predicted {INDUSTRY}")
ax_ts_tr.set_xlabel(f"{INDUSTRY} points")
ax_ts_tr.set_ylabel(INDUSTRY)
ax_ts_tr.set_title("(d) Training")
ax_ts_tr.legend(loc="upper right")
fig.suptitle(f"{model_label} — { {'Cnsmr':'Consumption','Manuf':'Manufacturing','HiTec':'Hi-Tech','Hlth':'Health'}.get(_norm_industry_name(col_name), 'Other') }", y=0.98, fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))
# -*- coding: utf-8 -*-
# 4-panel plot for one industry (Testing/Training scatter + series)
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ====== paths & options ======
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
MODEL    = "FF3"          # "FF3" or "FF6"
INDUSTRY = "hitecrf"      # e.g., hitecrf / HiTec / Hi-Tech / Manuf / manurf / Cnsmr / ...
TEST_SIZE = 0.30
FAST = True               # faster search
FAST_LAST_N = 240         # only use last N months when FAST is True
RHO_METHOD = "spearman"   # "spearman" or "pearson"
OUTFILE = f"plots_{MODEL}_{INDUSTRY}.png"
# ====== helpers ======
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError(f"Header not found in: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).replace(-99.99, np.nan)
    return df.sort_values("Date").reset_index(drop=True)
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
def _norm_ind(s: str) -> str:
    s = s.lower().replace("-", "")
    s = re.sub(r"(crf|rf)$", "", s)   # strip trailing rf/crf
    if s.startswith(("cns","con","cnsmr")): return "Cnsmr"
    if s.startswith(("man","mfg","manu")):  return "Manuf"
    if s.startswith(("hit","tec","high")):  return "HiTec"
    if s.startswith(("hea","hlth")):        return "Hlth"
    return "Other"
def match_industry(target, cols):
    want = _norm_ind(target)
    for c in cols:
        if _norm_ind(c) == want: return c
    raise ValueError(f"Industry not found: {target}; choices: {cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_iter  = 12 if fast else 60
    n_splits= 3  if fast else 5
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3,1e3,prior="log-uniform"),
        "svr__epsilon": Real(1e-4,1.0,prior="log-uniform"),
        "svr__gamma":   Real(1e-4,1e2,prior="log-uniform"),
        "svr__degree":  Integer(2,3),
        "svr__coef0":   Real(0.0,1.0),
    }
    opt = BayesSearchCV(pipe, space, n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def corr_with_p(y, yhat, method="spearman"):
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if method == "pearson":
        from scipy.stats import pearsonr
        r, p = pearsonr(y[mask], yhat[mask])
    else:
        r, p = spearmanr(y[mask], yhat[mask])
    return float(r), float(p)
# ====== load & prep ======
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cns","Cnsmr","Cnsmer","Manuf","HiTec","Hlth","Other"])
fac, ind = align_on_date([fac, ind])
if FAST and FAST_LAST_N and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
ind_cols = infer_industry_cols(ind)
# convert to excess returns
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values
# features
if MODEL.upper()=="FF3":
    X = fac[["Mkt-RF","SMB","HML"]].to_numpy(); model_label="3-Factor"
elif MODEL.upper()=="FF6":
    X = fac[["Mkt-RF","SMB","HML","RMW","CMA","WML"]].to_numpy(); model_label="6-Factor"
else:
    raise ValueError("MODEL must be 'FF3' or 'FF6'.")
col = match_industry(INDUSTRY, ind_cols)
y = ind[col].to_numpy()
n = len(y); split = int(round(n*(1-TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]; y_tr, y_te = y[:split], y[split:]
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr); yhat_te = est.predict(X_te)
rho_te, p_te = corr_with_p(y_te, yhat_te, method=RHO_METHOD)
rho_tr, p_tr = corr_with_p(y_tr, yhat_tr, method=RHO_METHOD)
print(f"[INFO] {model_label} — industry input='{INDUSTRY}' -> column='{col}'")
print(f"Sample={n}, Train={split}, Test={n-split}")
print(f"Testing  rho={rho_te:.5f}, p={p_te:.3e}")
print(f"Training rho={rho_tr:.5f}, p={p_tr:.3e}")
# ====== plots ======
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(2,2, figsize=(11,9))
(ax_sc_te, ax_sc_tr), (ax_ts_te, ax_ts_tr) = axes
# (a) testing scatter
ax_sc_te.scatter(y_te, yhat_te, s=18)
ax_sc_te.set_xlabel(f"experimental {INDUSTRY}")
ax_sc_te.set_ylabel(f"{INDUSTRY} predicted")
ax_sc_te.set_title("(a) Testing")
ax_sc_te.text(0.02, 0.96, f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_te.transAxes, va="top")
# (b) training scatter
ax_sc_tr.scatter(y_tr, yhat_tr, s=12)
ax_sc_tr.set_xlabel(f"experimental {INDUSTRY}")
ax_sc_tr.set_ylabel(f"{INDUSTRY} predicted")
ax_sc_tr.set_title("(b) Training")
ax_sc_tr.text(0.02, 0.96, f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_tr.transAxes, va="top")
# (c) testing series
ax_ts_te.plot(range(len(y_te)), y_te, marker="o", linestyle="-", linewidth=1, markersize=3, label=f"experimental {INDUSTRY}")
ax_ts_te.plot(range(len(y_te)), yhat_te, marker="o", linestyle="-", linewidth=1, markersize=3, label=f"predicted {INDUSTRY}")
ax_ts_te.set_xlabel(f"{INDUSTRY} points"); ax_ts_te.set_ylabel(INDUSTRY); ax_ts_te.set_title("(c) Testing")
ax_ts_te.legend(loc="upper right")
# (d) training series
ax_ts_tr.plot(range(len(y_tr)), y_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label=f"experimental {INDUSTRY}")
ax_ts_tr.plot(range(len(y_tr)), yhat_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label=f"predicted {INDUSTRY}")
ax_ts_tr.set_xlabel(f"{INDUSTRY} points"); ax_ts_tr.set_ylabel(INDUSTRY); ax_ts_tr.set_title("(d) Training")
ax_ts_tr.legend(loc="upper right")
fig.suptitle(f"{model_label} — {_norm_ind(col)}", y=0.98, fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Saved:", os.path.abspath(OUTFILE))
# -*- coding: utf-8 -*-
"""
4-panel plot: healthrf (Health industry excess return) — Testing/Training Scatter + Time Series
Switchable between FF3 / FF6 (includes WML); supports aliases like healthrf/Health/Hlth.
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ======== Paths & Options (modify as needed) ========
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
MODEL    = "FF3"          # "FF3" or "FF6"
INDUSTRY = "healthrf"     # healthrf / Hlth / Health / hlt etc. are all acceptable
TEST_SIZE   = 0.30        # Test set ratio (split from end by time)
FAST        = True        # Fast search (linear/rbf, fewer iterations)
FAST_LAST_N = 240         # In FAST mode, use only last N months; set to None for full sample
RHO_METHOD  = "spearman"  # "spearman" or "pearson"
OUTFILE     = f"plots_{MODEL}_{INDUSTRY}.png"
# ======== Utility Functions ========
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    if not os.path.exists(path): raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError(f"Header not detected: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).replace(-99.99, np.nan)
    return df.sort_values("Date").reset_index(drop=True)
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
def _norm_industry(s: str) -> str:
    s = s.strip().lower().replace("-", "")
    s = re.sub(r"(crf|rf)$", "", s)           # Remove rf/crf suffix
    if s.startswith(("cns","con","cnsmr")):   return "Cnsmr"
    if s.startswith(("man","mfg","manu")):    return "Manuf"
    if s.startswith(("hit","tec","high")):    return "HiTec"
    if s.startswith(("hea","hlth","hlt")):    return "Hlth"
    return "Other"
def match_industry(target, cols):
    want = _norm_industry(target)
    for c in cols:
        if _norm_industry(c) == want: return c
    raise ValueError(f"Industry column not found: {target}; available: {cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_iter  = 12 if fast else 60
    n_splits= 3  if fast else 5
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma":   Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree":  Integer(2, 3),
        "svr__coef0":   Real(0.0, 1.0),
    }
    opt = BayesSearchCV(pipe, space, n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def corr_with_p(y, yhat, method="spearman"):
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if method == "pearson":
        from scipy.stats import pearsonr
        r, p = pearsonr(y[mask], yhat[mask])
    else:
        r, p = spearmanr(y[mask], yhat[mask])
    return float(r), float(p)
# ======== Data Preparation ========
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cns","Cnsmr","Cnsmer","Manuf","HiTec","Hlth","Other"])
fac, ind = align_on_date([fac, ind])
if FAST and FAST_LAST_N and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
ind_cols = infer_industry_cols(ind)
# Industry excess returns: Industry - RF
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values
# Features
if MODEL.upper() == "FF3":
    X = fac[["Mkt-RF","SMB","HML"]].to_numpy(); model_label = "3-Factor"
elif MODEL.upper() == "FF6":
    X = fac[["Mkt-RF","SMB","HML","RMW","CMA","WML"]].to_numpy(); model_label = "6-Factor"
else:
    raise ValueError("MODEL only supports 'FF3' or 'FF6'")
# Select Health column
col = match_industry(INDUSTRY, ind_cols)   # Maps healthrf/Hlth/Health etc. to actual column name
y = ind[col].to_numpy()
# Train/test split
n = len(y); split = int(round(n*(1-TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Fit and predict
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr); yhat_te = est.predict(X_te)
# Correlation
rho_te, p_te = corr_with_p(y_te, yhat_te, method=RHO_METHOD)
rho_tr, p_tr = corr_with_p(y_tr, yhat_tr, method=RHO_METHOD)
print(f"[INFO] {model_label} — HEALTH | Column: {col}  Samples={n} Train={split} Test={n-split}")
print(f"Testing  RHO={rho_te:.5f}, p={p_te:.3e}")
print(f"Training RHO={rho_tr:.5f}, p={p_tr:.3e}")
# ======== Plot 4-panel Figure ========
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
(ax_sc_te, ax_sc_tr), (ax_ts_te, ax_ts_tr) = axes
# (a) Testing Scatter
ax_sc_te.scatter(y_te, yhat_te, s=18)
ax_sc_te.set_xlabel("experimental healthrf")
ax_sc_te.set_ylabel("healthrf predicted")
ax_sc_te.set_title("(a) Testing")
ax_sc_te.text(0.02, 0.96,
              f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_te.transAxes, va="top")
# (b) Training Scatter
ax_sc_tr.scatter(y_tr, yhat_tr, s=12)
ax_sc_tr.set_xlabel("experimental healthrf")
ax_sc_tr.set_ylabel("healthrf predicted")
ax_sc_tr.set_title("(b) Training")
ax_sc_tr.text(0.02, 0.96,
              f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_tr.transAxes, va="top")
# (c) Testing Time Series
ax_ts_te.plot(range(len(y_te)), y_te, marker="o", linestyle="-", linewidth=1, markersize=3, label="experimental healthrf")
ax_ts_te.plot(range(len(y_te)), yhat_te, marker="o", linestyle="-", linewidth=1, markersize=3, label="predicted healthrf")
ax_ts_te.set_xlabel("healthrf points"); ax_ts_te.set_ylabel("healthrf"); ax_ts_te.set_title("(c) Testing")
ax_ts_te.legend(loc="upper right")
# (d) Training Time Series
ax_ts_tr.plot(range(len(y_tr)), y_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label="experimental healthrf")
ax_ts_tr.plot(range(len(y_tr)), yhat_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label="predicted healthrf")
ax_ts_tr.set_xlabel("healthrf points"); ax_ts_tr.set_ylabel("healthrf"); ax_ts_tr.set_title("(d) Training")
ax_ts_tr.legend(loc="upper right")
fig.suptitle(f"{model_label} — Health", y=0.98, fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 17:28:28 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
4-panel plot: Other (otherrf) — Testing/Training Scatter + Time Series (includes Spearman ρ and p-value)
Modify MODEL/INDUSTRY/paths to plot any industry: manurf/hitecrf/healthrf/otherrf/Cnsmr etc.
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ===== Configuration (modify as needed) =====
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
MODEL    = "FF3"        # "FF3" or "FF6"
INDUSTRY = "otherrf"    # Industry alias: otherrf/Other/other variations are all acceptable
TEST_SIZE   = 0.30
FAST        = True       # Fast search (only linear/rbf, fewer iterations)
FAST_LAST_N = 240        # In FAST mode, use only last N months; set to None for full sample
RHO_METHOD  = "spearman" # "spearman" or "pearson"
OUTFILE     = f"plots_{MODEL}_{INDUSTRY}.png"
# ===== Utility Functions =====
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    if not os.path.exists(path): raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError(f"Header not detected: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).replace(-99.99, np.isnan)
    return df.sort_values("Date").reset_index(drop=True)
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
def _norm_industry(s: str) -> str:
    """Normalize various spellings to: Cnsmr/Manuf/HiTec/Hlth/Other"""
    s = s.strip().lower().replace("-", "")
    s = re.sub(r"(crf|rf)$", "", s)  # Remove trailing rf/crf
    if s.startswith(("cns","con","cnsmr")): return "Cnsmr"
    if s.startswith(("man","mfg","manu")):  return "Manuf"
    if s.startswith(("hit","tec","high")):  return "HiTec"
    if s.startswith(("hea","hlth")):        return "Hlth"
    return "Other"
def match_industry(target, cols):
    want = _norm_industry(target)
    for c in cols:
        if _norm_industry(c) == want:
            return c
    raise ValueError(f"Industry column not found: {target}; available: {cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_iter  = 12 if fast else 60
    n_splits= 3  if fast else 5
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma":   Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree":  Integer(2, 3),
        "svr__coef0":   Real(0.0, 1.0),
    }
    opt = BayesSearchCV(pipe, space, n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def corr_with_p(y, yhat, method="spearman"):
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if method == "pearson":
        from scipy.stats import pearsonr
        r, p = pearsonr(y[mask], yhat[mask])
    else:
        r, p = spearmanr(y[mask], yhat[mask])
    return float(r), float(p)
# ===== Load and Prepare =====
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cns","Cnsmr","Cnsmer","Manuf","HiTec","Hlth","Other"])
fac, ind = align_on_date([fac, ind])
if FAST and FAST_LAST_N and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
ind_cols = infer_industry_cols(ind)
# Industry excess return = Industry - RF
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values
# Feature matrix
if MODEL.upper() == "FF3":
    X = fac[["Mkt-RF","SMB","HML"]].to_numpy(); model_label = "3-Factor"
elif MODEL.upper() == "FF6":
    need = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
    miss = [c for c in need if c not in fac.columns]
    if miss: raise ValueError(f"Missing factor columns: {miss}")
    X = fac[need].to_numpy(); model_label = "6-Factor"
else:
    raise ValueError("MODEL only supports FF3/FF6")
# Select industry column
col = match_industry(INDUSTRY, ind_cols)
y = ind[col].to_numpy()
# Split
n = len(y); split = int(round(n*(1-TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Train and predict
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr); yhat_te = est.predict(X_te)
# Correlation coefficients
rho_te, p_te = corr_with_p(y_te, yhat_te, method=RHO_METHOD)
rho_tr, p_tr = corr_with_p(y_tr, yhat_tr, method=RHO_METHOD)
print(f"[INFO] {model_label} — industry='{INDUSTRY}' -> column='{col}' | Samples={n} Train={split} Test={n-split}")
print(f"Testing  rho={rho_te:.5f}, p={p_te:.3e}")
print(f"Training rho={rho_tr:.5f}, p={p_tr:.3e}")
# ===== Plot (4-panel) =====
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
(ax_sc_te, ax_sc_tr), (ax_ts_te, ax_ts_tr) = axes
# (a) Testing Scatter
ax_sc_te.scatter(y_te, yhat_te, s=18)
ax_sc_te.set_xlabel("experimental otherrf")
ax_sc_te.set_ylabel("otherrf predicted")
ax_sc_te.set_title("(a) Testing")
ax_sc_te.text(0.02, 0.96, f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_te.transAxes, va="top")
# (b) Training Scatter
ax_sc_tr.scatter(y_tr, yhat_tr, s=12)
ax_sc_tr.set_xlabel("experimental otherrf")
ax_sc_tr.set_ylabel("otherrf predicted")
ax_sc_tr.set_title("(b) Training")
ax_sc_tr.text(0.02, 0.96, f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.3e}\nat ALPH-significant level: 0.05",
              transform=ax_sc_tr.transAxes, va="top")
# (c) Testing Time Series
ax_ts_te.plot(range(len(y_te)), y_te,  marker="o", linestyle="-", linewidth=1, markersize=3, label="experimental otherrf")
ax_ts_te.plot(range(len(y_te)), yhat_te, marker="o", linestyle="-", linewidth=1, markersize=3, label="predicted otherrf")
ax_ts_te.set_xlabel("otherrf points"); ax_ts_te.set_ylabel("otherrf"); ax_ts_te.set_title("(c) Testing")
ax_ts_te.legend(loc="upper right")
# (d) Training Time Series
ax_ts_tr.plot(range(len(y_tr)), y_tr,  marker="o", linestyle="-", linewidth=1, markersize=2, label="experimental otherrf")
ax_ts_tr.plot(range(len(y_tr)), yhat_tr, marker="o", linestyle="-", linewidth=1, markersize=2, label="predicted otherrf")
ax_ts_tr.set_xlabel("otherrf points"); ax_ts_tr.set_ylabel("otherrf"); ax_ts_tr.set_title("(d) Training")
ax_ts_tr.legend(loc="upper right")
fig.suptitle(f"{model_label} — Other", y=0.98, fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))
# -*- coding: utf-8 -*-
"""
FF6 (Six Factors) — Consumption Industry: Testing/Training Correlation Scatter Plots
Left: testing experimental consumption vs predicted
Right: training experimental consumption vs predicted
Annotations: Pearson correlation & p-value
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ========= Paths & Options (modify as needed) =========
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
TEST_SIZE   = 0.30        # Test set ratio (split from end by time)
FAST        = True        # Fast search (linear/rbf, fewer iterations)
FAST_LAST_N = 240         # In FAST mode, use only last N months; set to None for full sample
OUTFILE     = "ff6_consumption_corr.png"
# ========= Utility Functions =========
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None:
        raise ValueError(f"Header not detected: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).replace(-99.99, np.nan)
    return df.sort_values("Date").reset_index(drop=True)
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
def get_consumption_col(cols):
    # Your file usually has 'Cnsmr' (or Cnsmer/Cons), here we do fault-tolerant matching
    for c in cols:
        if c.lower().startswith(("cns","con","cnsmr","cnsmer","cons")):
            return c
    raise ValueError(f"Consumption industry column not found (may be called Cnsmr/Cnsmer/Cons etc.): {cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels  = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_iter   = 12 if fast else 60
    n_splits = 3  if fast else 5
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma":   Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree":  Integer(2,3),
        "svr__coef0":   Real(0.0,1.0),
    }
    opt = BayesSearchCV(pipe, space, n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def pearson_corr(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat)
    r, p = pearsonr(y[mask], yhat[mask])
    return float(r), float(p)
# ========= Load Data =========
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cns","Cnsmr","Cnsmer","Cons","Manuf","HiTec","Hlth","Other"])
fac, ind = align_on_date([fac, ind])
# Fast mode: use only last N months
if FAST and FAST_LAST_N and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
# Industry columns & excess returns
ind_cols = infer_industry_cols(ind)
cons_col = get_consumption_col(ind_cols)   # Consumption industry column name
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values  # Industry - RF
# Six-factor features
need = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
miss = [c for c in need if c not in fac.columns]
if miss:
    raise ValueError(f"Missing factor columns: {miss}")
X = fac[need].to_numpy()
y = ind[cons_col].to_numpy()
# Train/test split
n = len(y)
split = int(round(n * (1 - TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Train and predict (BSVR≈SVR with Bayesian hyperparameter search)
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr)
yhat_te = est.predict(X_te)
# Correlation coefficient (Pearson)
r_te, p_te = pearson_corr(y_te, yhat_te)
r_tr, p_tr = pearson_corr(y_tr, yhat_tr)
print(f"[INFO] FF6 — Consumption | Column: {cons_col}  Samples={n} Train={split} Test={n-split}")
print(f"Testing  correlation={r_te:.5f}, p-value={p_te:.3e}")
print(f"Training correlation={r_tr:.5f}, p-value={p_tr:.3e}")
# ========= Plot Two-panel Figure (consistent with screenshot style) =========
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.5))
def annotate(ax, r, p):
    ax.text(0.03, 0.97,
            f"correlation: {r:.5f}\np-value: {p:.3e}\nat alpha-significant level: 0.05",
            transform=ax.transAxes, va="top")
# (a) Testing
axL.scatter(y_te, yhat_te, s=28)
axL.set_xlabel("experimental cons.")
axL.set_ylabel("predicted cons.")
axL.set_title("testing experimental consumption vs predicted")
annotate(axL, r_te, p_te)
# (b) Training
axR.scatter(y_tr, yhat_tr, s=20)
axR.set_xlabel("experimental cons")
axR.set_ylabel("predicted cons")
axR.set_title("training experimental consumption vs predicted")
annotate(axR, r_tr, p_tr)
fig.suptitle("(a) Testing                                        (b) Training", y=0.02, fontsize=14)
fig.tight_layout(rect=[0,0.05,1,0.98])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))
"""
Created on Sat Aug 23 18:02:32 2025
@author: support huawei
"""
# -*- coding: utf-8  -*-
"""
FF6 (Six Factors) → Manufacturing Industry
Output: Two-panel plot (Left: Testing / Right: Training) — experimental vs predicted scatter (includes correlation coefficient and p-value)
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ============== Paths & Adjustable Parameters (modify as needed) ==============
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
TEST_SIZE   = 0.30         # Test set ratio (split from end by time)
FAST        = True         # True: Fast search (linear/rbf, fewer iterations); False: More thorough (slower)
FAST_LAST_N = 240          # In FAST mode, use only last N months; set to None for full sample
OUTFILE     = "ff6_manufacturing_corr.png"
# ============== Utility Functions ==============
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    """Automatically locate header and read, first column is YYYYMM, generate Date column"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError(f"Header not detected: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).replace(-99.99, np.nan)
    return df.sort_values("Date").reset_index(drop=True)
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
def get_manuf_col(cols):
    # Common names: Manuf / Manufacturing / manurf etc.
    for c in cols:
        if c.lower().startswith(("man","mfg","manuf","manufact")):
            return c
    raise ValueError(f"Manufacturing industry column not found (Manuf/Manufacturing etc.): {cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels  = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_iter   = 12 if fast else 60
    n_splits = 3  if fast else 5
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma":   Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree":  Integer(2, 3),
        "svr__coef0":   Real(0.0, 1.0),
    }
    opt = BayesSearchCV(pipe, space, n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def pearson_corr(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.isfinite(y) & np.isfinite(yhat)
    r, p = pearsonr(y[mask], yhat[mask])
    return float(r), float(p)
# ============== Load Data and Prepare ==============
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cnsmr","Cns","Cnsmr","Manuf","Manufacturing","HiTec","Hlth","Other"])
fac, ind = align_on_date([fac, ind])
# Fast mode: use only last N months
if FAST and FAST_LAST_N and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
# Excess returns (Industry - RF)
ind_cols = infer_industry_cols(ind)
man_col  = get_manuf_col(ind_cols)
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values
# FF6 Features
need = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
miss = [c for c in need if c not in fac.columns]
if miss: raise ValueError(f"Missing factor columns: {miss}")
X = fac[need].to_numpy()
y = ind[man_col].to_numpy()
# Time split
n = len(y); split = int(round(n*(1-TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Train and predict
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr); yhat_te = est.predict(X_te)
# Pearson Correlation
r_te, p_te = pearson_corr(y_te, yhat_te)
r_tr, p_tr = pearson_corr(y_tr, yhat_tr)
print(f"[INFO] FF6 — Manufacturing | Column: {man_col} | Samples={n} Train={split} Test={n-split}")
print(f"Testing  correlation={r_te:.5f}, p-value={p_te:.3e}")
print(f"Training correlation={r_tr:.5f}, p-value={p_tr:.3e}")
# ============== Plot Two-panel Figure (consistent with screenshot style) ==============
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.5))
def annotate(ax, r, p):
    ax.text(0.03, 0.97,
            f"correlation: {r:.5f}\np-value: {p:.3e}\nat alpha-significant level: 0.05",
            transform=ax.transAxes, va="top")
# (a) Testing
axL.scatter(y_te, yhat_te, s=28)
axL.set_xlabel("experimental manufacturing")
axL.set_ylabel("predicted manufacturing")
axL.set_title("testing experimental manufacturing vs predicted")
annotate(axL, r_te, p_te)
# (b) Training
axR.scatter(y_tr, yhat_tr, s=20)
axR.set_xlabel("experimental manufacturing")
axR.set_ylabel("predicted manufacturing")
axR.set_title("training experimental manufacturing vs predicted")
annotate(axR, r_tr, p_tr)
fig.suptitle("(a) Testing                                        (b) Training", y=0.02, fontsize=14)
fig.tight_layout(rect=[0,0.05,1,0.98])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 18:54:19 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF6 → Hi-Tech Industry: Two-panel plot (Testing/Training)
Left: testing experimental hi-tech vs predicted
Right: training experimental hi-tech vs predicted
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ======== Paths & Parameters (modify as needed) ========
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
TEST_SIZE   = 0.30     # Test set ratio (split from end)
FAST        = True     # True=Fast search (linear/rbf, fewer iterations); False=More thorough search
FAST_LAST_N = 240      # In FAST mode, use only last N months; set to None for full sample
OUTFILE     = "ff6_hitech_two_panel.png"
# ======== Utility Functions ========
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError(f"Header not detected: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    # Keep only YYYYMM rows
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).replace(-99.99, np.nan)
    return df.sort_values("Date").reset_index(drop=True)
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
def pick_hitech_col(cols):
    # Common: HiTec / Hi-Tech / Hitec / HighTech …
    for c in cols:
        if c.lower().startswith(("hit","hitec","hi-tec","tec","high")):
            return c
    raise ValueError(f"Hi-Tech column not found (e.g., HiTec/Hi-Tech): {cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels  = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_iter   = 12 if fast else 60
    n_splits = 3  if fast else 5
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma":   Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree":  Integer(2, 3),
        "svr__coef0":   Real(0.0, 1.0),
    }
    opt = BayesSearchCV(pipe, space, n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def pearson_corr(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    m = np.isfinite(y) & np.isfinite(yhat)
    r, p = pearsonr(y[m], yhat[m])
    return float(r), float(p)
# ======== Load and Prepare ========
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cnsmr","Manuf","HiTec","Hlth","Other"])
fac, ind = align_on_date([fac, ind])
# Fast: use only last N months
if FAST and FAST_LAST_N and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
# Industry columns and excess returns
ind_cols = infer_industry_cols(ind)
hi_col = pick_hitech_col(ind_cols)      # Hi-Tech column name
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values  # Industry - RF
# Six-factor features
need = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
miss = [c for c in need if c not in fac.columns]
if miss: raise ValueError(f"Missing factor columns: {miss}")
X = fac[need].to_numpy()
y = ind[hi_col].to_numpy()
# Time split
n = len(y); split = int(round(n*(1-TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Train+predict
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr); yhat_te = est.predict(X_te)
# Correlation coefficient (Pearson)
r_te, p_te = pearson_corr(y_te, yhat_te)
r_tr, p_tr = pearson_corr(y_tr, yhat_tr)
print(f"[INFO] FF6 — Hi-Tech | Column: {hi_col}  Samples={n} Train={split} Test={n-split}")
print(f"Testing  correlation={r_te:.5f}, p-value={p_te:.3e}")
print(f"Training correlation={r_tr:.5f}, p-value={p_tr:.3e}")
# ======== Plot Two-panel Figure (consistent with screenshot) ========
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.5))
def annotate(ax, r, p):
    ax.text(0.03, 0.97,
            f"correlation: {r:.5f}\np-value: {p:.3e}\nat alpha-significant level: 0.05",
            transform=ax.transAxes, va="top")
# To match screenshot, set axis limits more "loosely"
def pad_limits(a, b, pad=0.05):
    lo = np.nanmin([a.min(), b.min()])
    hi = np.nanmax([a.max(), b.max()])
    span = hi - lo
    return lo - pad*span, hi + pad*span
# (a) Testing
axL.scatter(y_te, yhat_te, s=28)
axL.set_xlabel("experimental hi-tech")
axL.set_ylabel("predicted hi-tech")
axL.set_title("testing experimental hi-tech vs predicted")
annotate(axL, r_te, p_te)
xl, xr = pad_limits(y_te, yhat_te); yl, yr = xl, xr  # Make xy ranges consistent
axL.set_xlim(xl, xr); axL.set_ylim(yl, yr)
# (b) Training
axR.scatter(y_tr, yhat_tr, s=20)
axR.set_xlabel("experimental hi-tech")
axR.set_ylabel("predicted hi-tech")
axR.set_title("training experimental hi-tech vs predicted")
annotate(axR, r_tr, p_tr)
xl, xr = pad_limits(y_tr, yhat_tr); yl, yr = xl, xr
axR.set_xlim(xl, xr); axR.set_ylim(yl, yr)
fig.suptitle("(a) Testing                                        (b) Training", y=0.02, fontsize=14)
fig.tight_layout(rect=[0,0.05,1,0.98])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 13:15:59 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
BSVR-lite: A computationally efficient Bayesian-optimized SVR
- First, use a small budget to pick the best kernel, then perform a small-scale BO on the winning kernel
- Narrow hyperparameter ranges + early stopping + optional downsampling
- Outputs C/ε/γ/kernel/bias/CV_RMSE for FF3/FF5 × five industries
"""
import re, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import DeltaYStopper
# ================== Basic Settings (can be made smaller or larger as needed) ==================
PATH_FAC = r"D:\Europe_5_Factors_CSV\Europe_5_Factors(monthly).csv"
PATH_IND = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
TEST_RATIO      = 0.10   # Train/Test = 90/10
N_SPLITS_CV     = 3      # Number of CV folds for training segment
STRIDE          = 1      # ← Downsampling stride (1=none; 2=take every 2nd month)
INCLUDE_POLY    = False  # ← To save computation, poly is disabled by default; enable if needed
KERNEL_PICK_IT  = 9      # ← First, evaluate each kernel with a very small budget
FINAL_BO_IT     = 18     # ← Then, perform a small-scale BO only on the winning kernel
EARLY_DELTA     = 1e-4   # ← Early stopping threshold (if improvement is less than this value)
EARLY_NBEST     = 8      # ← Stop if no significant improvement for N consecutive iterations
RANDOM_STATE    = 42
# Narrowed parameter ranges (empirically sufficient and stable)
C_BOUNDS        = (1e-2, 1e2)
EPS_BOUNDS      = (1e-3, 2e-1)
GAMMA_BOUNDS    = (1e-3, 1.0)
DEG_BOUNDS      = (2, 3)     # Only used if INCLUDE_POLY=True
# ================== Utility Function: Read French-style CSV ==================
def read_french_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = Path(path).read_text(encoding=enc); break
        except Exception: pass
    lines = text.splitlines()
    header = None
    for i, ln in enumerate(lines[:120]):
        low = ln.lower()
        if ("date" in low and "," in low) or any(k in low for k in
           ["mkt-rf","rm-rf","smb","hml","rmw","cma","rf",
            "cnsmr","manuf","hitec","hlth","other"]):
            header = i; break
    end = None
    for j in range((header or 0)+1, len(lines)):
        if lines[j].strip().lower().startswith("annual"):
            end = j; break
    nrows = None if end is None else (end - header - 1)
    df = pd.read_csv(path, skiprows=header, nrows=nrows)
    first = df.columns[0]
    idx = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    if idx.notna().sum() > len(df)*0.6:
        df = df.set_index(idx).drop(columns=[first])
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace(-99.99, np.nan)
    return df
def norm(s): return re.sub(r"[^A-Za-z0-9]","", str(s)).upper()
def find_col(df, aliases):
    for c in df.columns:
        if norm(c) in aliases: return c
    return None
# ================== Load and Align FF Factors and Industries ==================
fac = read_french_csv(PATH_FAC)
ind = read_french_csv(PATH_IND)
mktrf = find_col(fac, {"MKTRF","RMRF","MKTMINUSRF","RMMINUSRF","MARKETRF"})
smb   = find_col(fac, {"SMB"});  hml = find_col(fac, {"HML"})
rmw   = find_col(fac, {"RMW"});  cma = find_col(fac, {"CMA"})
rfcol = find_col(fac, {"RF","RISKFREE","RISKFREERATE"})
assert all([mktrf, smb, hml, rfcol]), "Factor file is missing Mkt-RF/SMB/HML/RF."
ind_cols = [c for c in ind.select_dtypes(include=[np.number]).columns if norm(c)!="RF"]
need = [mktrf, smb, hml, rmw, cma, rfcol]
data = fac[need].join(ind[ind_cols], how="inner").dropna(subset=need + ind_cols)
# Industry excess returns (Ri - Rf)
y_df = data[ind_cols].subtract(data[rfcol], axis=0).rename(columns={
    "Cnsmr":"Consumption", "Cnsumr":"Consumption",
    "Manuf":"Manufacturing",
    "HiTec":"Hi-Tech",
    "Hlth":"Health",
    "Other":"Other"
})
INDUSTRIES = ["Consumption","Manufacturing","Hi-Tech","Health","Other"]
y_df = y_df[INDUSTRIES]
X3_df = data[[mktrf, smb, hml]]
X5_df = data[[mktrf, smb, hml, rmw, cma]]
# ================== BayesSearch Wrapper (Small Budget) ==================
def make_spaces(kernel_name):
    if kernel_name == "linear":
        return {'svr__kernel': Categorical(['linear']),
                'svr__C': Real(*C_BOUNDS, prior='log-uniform'),
                'svr__epsilon': Real(*EPS_BOUNDS, prior='log-uniform')}
    if kernel_name == "rbf":
        return {'svr__kernel': Categorical(['rbf']),
                'svr__C': Real(*C_BOUNDS, prior='log-uniform'),
                'svr__epsilon': Real(*EPS_BOUNDS, prior='log-uniform'),
                'svr__gamma': Real(*GAMMA_BOUNDS, prior='log-uniform')}
    if kernel_name == "poly":
        return {'svr__kernel': Categorical(['poly']),
                'svr__degree': Integer(*DEG_BOUNDS),
                'svr__C': Real(*C_BOUNDS, prior='log-uniform'),
                'svr__epsilon': Real(*EPS_BOUNDS, prior='log-uniform'),
                'svr__gamma': Real(*GAMMA_BOUNDS, prior='log-uniform')}
    raise ValueError("unknown kernel")
def bayes_search(X, y, kernel_name, n_iter):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svr", SVR(cache_size=100))])
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    stopper = DeltaYStopper(delta=EARLY_DELTA, n_best=EARLY_NBEST)
    bo = BayesSearchCV(
        estimator=pipe,
        search_spaces=make_spaces(kernel_name),
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,                   # To save resources, single core is sufficient; set to -1 for speed
        random_state=RANDOM_STATE,
        verbose=0
    )
    bo.fit(X, y, callback=[stopper])
    return bo
# First pick kernel (linear/rbf/optional poly), then refine the winning kernel
def lite_bsvr(X_tr, y_tr):
    kernels = ["linear", "rbf"] + (["poly"] if INCLUDE_POLY else [])
    best = None
    for k in kernels:
        bo_k = bayes_search(X_tr, y_tr, k, n_iter=KERNEL_PICK_IT)
        score = -float(bo_k.best_score_)   # CV_RMSE
        if (best is None) or (score < best[0]):
            best = (score, k, bo_k)
    # Perform a small-scale BO only on the winning kernel
    final_bo = bayes_search(X_tr, y_tr, best[1], n_iter=FINAL_BO_IT)
    return final_bo
def summarize(industry, bo3, bo5):
    svr3 = bo3.best_estimator_.named_steps['svr']
    svr5 = bo5.best_estimator_.named_steps['svr']
    p3, p5 = bo3.best_params_, bo5.best_params_
    gamma3 = p3.get("svr__gamma", "—")
    gamma5 = p5.get("svr__gamma", "—")
    return {
        "Industry": industry,
        "FF3_C": p3["svr__C"], "FF3_ε": p3["svr__epsilon"], "FF3_γ": gamma3,
        "FF3_kernel": p3["svr__kernel"], "FF3_bias": float(svr3.intercept_[0]),
        "FF3_CV_RMSE": -float(bo3.best_score_),
        "FF5_C": p5["svr__C"], "FF5_ε": p5["svr__epsilon"], "FF5_γ": gamma5,
        "FF5_kernel": p5["svr__kernel"], "FF5_bias": float(svr5.intercept_[0]),
        "FF5_CV_RMSE": -float(bo5.best_score_)
    }
# ================== Main Loop (Lightweight) ==================
rows = []
for ind_name in INDUSTRIES:
    y = y_df[ind_name].values
    X3 = X3_df.values
    X5 = X5_df.values
    # Time-ordered split + optional downsampling
    n = len(y); split = int(n * (1 - TEST_RATIO))
    X3_tr, y_tr = X3[:split:STRIDE], y[:split:STRIDE]
    X5_tr       = X5[:split:STRIDE]
    print(f">>> {ind_name}: kernel picking & lite BO...")
    bo3 = lite_bsvr(X3_tr, y_tr)
    bo5 = lite_bsvr(X5_tr, y_tr)
    rows.append(summarize(ind_name, bo3, bo5))
# ================== Output and Save ==================
df = pd.DataFrame(rows).astype({
    "Industry":"object","FF3_kernel":"object","FF5_kernel":"object",
    "FF3_γ":"object","FF5_γ":"object"
})
num_cols = [c for c in df.columns if c not in ["Industry","FF3_kernel","FF5_kernel","FF3_γ","FF5_γ"]]
df[num_cols] = df[num_cols].apply(lambda s: np.round(s.astype(float), 4))
order = ["Industry",
         "FF3_C","FF3_ε","FF3_γ","FF3_kernel","FF3_bias","FF3_CV_RMSE",
         "FF5_C","FF5_ε","FF5_γ","FF5_kernel","FF5_bias","FF5_CV_RMSE"]
df = df[order]
print("\nOptimized (BSVR-lite) SVR parameters by industry:")
print(df.to_string(index=False))
out_path = "BSVR_params_lite.xlsx"
with pd.ExcelWriter(out_path) as xw:
    df.to_excel(xw, sheet_name="BSVR_lite", index=False)
print(f"\nSaved: {out_path}")
# -*- coding: utf-8 -*-
"""
FF6 (Six Factors) → Other Industry
Output: Two-panel plot (Left: Testing / Right: Training) — experimental vs predicted (includes Pearson correlation and p-value)
"""
import os, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # For Spearman: from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# ========= Paths & Parameters (modify as needed) =========
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
TEST_SIZE   = 0.30       # Test set ratio (split from end by time)
FAST        = True       # True=Fast tuning (only linear/rbf, fewer iterations); False=More thorough (slower)
FAST_LAST_N = 240        # In FAST mode, use only last N months; set to None for full sample
OUTFILE     = "ff6_other_corr.png"
# Fix axis limits to match paper-style plots (can be modified or disabled)
FIXED_AXES = True
TEST_XY_LIM  = (-15, 15), (-20, 20)   # (xlim, ylim)
TRAIN_XY_LIM = (-30, 20), (-20, 20)
# ========= Utility Functions =========
def _find_header_line(lines, tokens):
    pat = re.compile("|".join([re.escape(t) for t in tokens]), re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return None
def read_csv_auto_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h = _find_header_line(lines, tokens)
    if h is None: raise ValueError(f"Header not detected: {path}")
    df = pd.read_csv(path, header=h, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m", errors="coerce")
    df = df.drop(columns=[first]).dropna(subset=["Date"]).replace(-99.99, np.nan)
    return df.sort_values("Date").reset_index(drop=True)
def align_on_date(dfs):
    base = dfs[0][["Date"]]
    for d in dfs[1:]:
        base = base.merge(d[["Date"]], on="Date", how="inner")
    return [d[d["Date"].isin(base["Date"])].sort_values("Date").reset_index(drop=True) for d in dfs]
def infer_industry_cols(df):
    exclude = {"Date","RF","R_f","R_F","Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","MOM"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
def pick_other_col(cols):
    # Common names: Other / others / otherrf / other
    for c in cols:
        if c.lower().startswith("other"):
            return c
    raise ValueError(f"Other industry column not found (e.g., Other/otherrf): {cols}")
def tune_and_fit(X_train, y_train, fast=True):
    kernels  = ["linear","rbf"] if fast else ["linear","rbf","poly"]
    n_iter   = 12 if fast else 60
    n_splits = 3  if fast else 5
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    space = {
        "svr__kernel":  Categorical(kernels),
        "svr__C":       Real(1e-3, 1e3, prior="log-uniform"),
        "svr__epsilon": Real(1e-4, 1.0, prior="log-uniform"),
        "svr__gamma":   Real(1e-4, 1e2, prior="log-uniform"),
        "svr__degree":  Integer(2, 3),
        "svr__coef0":   Real(0.0, 1.0),
    }
    opt = BayesSearchCV(pipe, space, n_iter=n_iter,
                        cv=TimeSeriesSplit(n_splits=n_splits),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, random_state=42, verbose=0)
    m = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
    opt.fit(X_train[m], y_train[m])
    return opt.best_estimator_
def corr_with_p(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    m = np.isfinite(y) & np.isfinite(yhat)
    r, p = pearsonr(y[m], yhat[m])  # For Spearman: r,p = spearmanr(y[m], yhat[m])
    return float(r), float(p)
# ========= Load and Prepare Data =========
fac = read_csv_auto_header(PATH_FACTORS,  ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML"])
fac.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                    "Mom":"WML","MOM":"WML","R_F":"RF","R_f":"RF"}, inplace=True)
ind = read_csv_auto_header(PATH_INDUSTRY, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other","others"])
fac, ind = align_on_date([fac, ind])
# Fast mode: use only last N months
if FAST and FAST_LAST_N and len(fac) > FAST_LAST_N:
    fac = fac.iloc[-FAST_LAST_N:].reset_index(drop=True)
    ind = ind.iloc[-FAST_LAST_N:].reset_index(drop=True)
# Industry columns & excess returns (Industry - RF)
ind_cols  = infer_industry_cols(ind)
other_col = pick_other_col(ind_cols)
for c in ind_cols:
    ind[c] = ind[c].astype(float) - fac["RF"].astype(float).values
# Six-factor features
need = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
miss = [c for c in need if c not in fac.columns]
if miss: raise ValueError(f"Missing factor columns: {miss}")
X = fac[need].to_numpy()
y = ind[other_col].to_numpy()
# Train/test split
n = len(y); split = int(round(n*(1-TEST_SIZE)))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
# Train + Predict
est = tune_and_fit(X_tr, y_tr, fast=FAST)
yhat_tr = est.predict(X_tr); yhat_te = est.predict(X_te)
# Correlation (Pearson)
r_te, p_te = corr_with_p(y_te, yhat_te)
r_tr, p_tr = corr_with_p(y_tr, yhat_tr)
print(f"[INFO] FF6 — Other | Column: {other_col} | Samples={n} Train={split} Test={n-split}")
print(f"Testing  correlation={r_te:.5f}, p-value={p_te:.3e}")
print(f"Training correlation={r_tr:.5f}, p-value={p_tr:.3e}")
# ========= Plot Two-panel Figure =========
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.5))
def annotate(ax, r, p):
    ax.text(0.03, 0.97,
            f"correlation: {r:.5f}\np-value: {p:.3e}\nat alpha-significant level: 0.05",
            transform=ax.transAxes, va="top")
# (a) Testing
axL.scatter(y_te, yhat_te, s=28)
axL.set_xlabel("experimental others")
axL.set_ylabel("predicted others")
axL.set_title("testing experimental others vs predicted")
annotate(axL, r_te, p_te)
if FIXED_AXES:
    axL.set_xlim(*TEST_XY_LIM[0]); axL.set_ylim(*TEST_XY_LIM[1])
# (b) Training
axR.scatter(y_tr, yhat_tr, s=20)
axR.set_xlabel("experimental others")
axR.set_ylabel("predicted others")
axR.set_title("training experimental others vs predicted")
annotate(axR, r_tr, p_tr)
if FIXED_AXES:
    axR.set_xlim(*TRAIN_XY_LIM[0]); axR.set_ylim(*TRAIN_XY_LIM[1])
fig.suptitle("(a) Testing                                        (b) Training", y=0.02, fontsize=14)
fig.tight_layout(rect=[0,0.05,1,0.98])
plt.savefig(OUTFILE, dpi=300)
plt.show()
print("Plot saved: ", os.path.abspath(OUTFILE))

