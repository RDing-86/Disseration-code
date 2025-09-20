# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 09:48:09 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.svm import SVR, LinearSVR
from scipy.stats import spearmanr
# ========= Configuration =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR_CSV   = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv" 
TARGET_INDUSTRY = "Cnsmr"   # Change to Manuf / HiTec / Hlth / Other to switch industry
MISSING_SENTINEL = -99.99
MODE = "FAST"               # "FAST" or "PRO"
PLOT_TS = True              # Plot time series graphs (c)(d); set to False for further speedup
# ========= Utility Functions =========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average", "Value Weighted", "--", "—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # Standardize date column name
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # Date parsing
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
# ========= Load Data (only necessary columns, convert to float32)=========
t0 = time.time()
ind = smart_read_csv(INDUSTRY_CSV)
fac = smart_read_csv(FACTOR_CSV)
col_industry = pick_col(ind, [TARGET_INDUSTRY,"consumer","cons","消费","manuf","制造","hitec","高科","hlth","医疗","other","其它","其他"])
col_mktrf = pick_col(fac, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb   = pick_col(fac, ["SMB"])
col_hml   = pick_col(fac, ["HML"])
col_rf    = pick_col(fac, ["RF","riskfree","risk-free"])
need = {"industry":col_industry,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RF":col_rf}
missing = [k for k,v in need.items() if v is None]
if missing:
    raise ValueError(f"Missing columns: {missing}. Please check factor/industry CSV files.")
df = ind[["Date", col_industry]].merge(
        fac[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     ).copy()
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
df["crf"] = (df[col_industry] - df[col_rf]).astype(np.float32)
X = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y = df["crf"].to_numpy(dtype=np.float32)
n = len(df); split = int(n*0.8)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]
# ========= Modeling (FAST/PRO modes)=========
tscv_fast = TimeSeriesSplit(n_splits=3)
tscv_pro  = TimeSeriesSplit(n_splits=5)
def fit_fast():
    """Random search + prioritize LinearSVR; if linear is poor, then small range RBF"""
    # 1) First, use LinearSVR (very fast)
    pipe_lin = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("svr", LinearSVR(loss="epsilon_insensitive", max_iter=5000))
    ])
    dist_lin = {
        "svr__C": np.logspace(-2, 2, 9),
        "svr__epsilon": np.logspace(-3, -0.5, 6)
    }
    rs_lin = RandomizedSearchCV(
        pipe_lin, dist_lin, n_iter=20, cv=tscv_fast,
        scoring="neg_mean_squared_error", n_jobs=-1, random_state=42, refit=True
    )
    rs_lin.fit(X_train, y_train)
    best_lin = rs_lin.best_estimator_
    mse_lin = -rs_lin.best_score_
    # 2) Small range RBF as backup
    pipe_rbf = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("svr", SVR(kernel="rbf", cache_size=1000))
    ])
    dist_rbf = {
        "svr__C": np.logspace(-1, 2, 6),
        "svr__gamma": ["scale", "auto", 1e-2, 1e-3],
        "svr__epsilon": [0.005, 0.01, 0.05, 0.1]
    }
    rs_rbf = RandomizedSearchCV(
        pipe_rbf, dist_rbf, n_iter=25, cv=tscv_fast,
        scoring="neg_mean_squared_error", n_jobs=-1, random_state=42, refit=True
    )
    rs_rbf.fit(X_train, y_train)
    best_rbf = rs_rbf.best_estimator_
    mse_rbf = -rs_rbf.best_score_
    if mse_lin <= mse_rbf:
        model = best_lin; which = f"LinearSVR (MSE_cv={mse_lin:.4f})"
    else:
        model = best_rbf; which = f"SVR-RBF (MSE_cv={mse_rbf:.4f})"
    return model, which
def fit_pro():
    """Two-stage: HalvingRandomSearchCV for parameter tuning on subsample (last 10 years) -> refit on full sample"""
    # Subsample (last ten years)
    years = 365*10
    sub_idx = max(0, n - years)
    X_sub, y_sub = X[sub_idx:], y[sub_idx:]
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(cache_size=1000))])
    dist = {
        "svr__kernel": ["linear", "rbf"],
        "svr__C": np.logspace(-2, 3, 10),
        "svr__epsilon": np.logspace(-3, -0.3, 8),
        "svr__gamma": ["scale", "auto", 1e-3, 1e-2]  # Only effective for rbf
    }
    hrs = HalvingRandomSearchCV(
        pipe, dist, factor=3, cv=tscv_pro, resource="n_samples", max_resources=len(X_sub),
        scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, aggressive_elimination=True
    )
    hrs.fit(X_sub, y_sub)
    model = hrs.best_estimator_
    # Refit once on the full training set
    model.fit(X_train, y_train)
    return model, f"HalvingRandomSearch best: {hrs.best_params_}"
if MODE.upper() == "FAST":
    model, model_info = fit_fast()
else:
    model, model_info = fit_pro()
print("Chosen model:", model_info)
# ========= Prediction & Metrics =========
yhat_train = model.predict(X_train)
yhat_test  = model.predict(X_test)
rho_t, pval_t = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr,pval_tr= spearmanr(y_train, yhat_train, nan_policy="omit")
alpha = 0.05
# ========= Plotting =========
plt.figure(figsize=(11, 9))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test, yhat_test, s=18)
ax1.set_xlabel("experimental crf"); ax1.set_ylabel("crf predicted"); ax1.set_title("(a) Testing")
txt = f"RHO-value: {rho_t:.5f}
PVAL-value: {pval_t:.2e}
at ALPH-significant level: {alpha}"
ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train, yhat_train, s=12)
ax2.set_xlabel("experimental crf"); ax2.set_ylabel("crf predicted"); ax2.set_title("(b) Training")
txt2 = f"RHO-value: {rho_tr:.5f}
PVAL-value: {pval_tr:.0e}
at ALPH-significant level: {alpha}"
ax2.text(0.02, 0.98, txt2, transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax2.grid(alpha=0.3)
# (c)(d) Time series point plots (can be disabled for further speedup)
if PLOT_TS:
    ax3 = plt.subplot(2,2,3)
    ax3.plot(range(len(y_test)), y_test,  marker="o", ms=3, lw=0.8, label="experimental crf")
    ax3.plot(range(len(yhat_test)), yhat_test, marker="o", ms=2.5, lw=0.8, label="predicted crf")
    ax3.set_xlabel("crf points"); ax3.set_ylabel("crf"); ax3.set_title("(c) Testing")
    ax3.legend(); ax3.grid(alpha=0.3)
    ax4 = plt.subplot(2,2,4)
    ax4.plot(range(len(y_train)), y_train,  marker="o", ms=2.5, lw=0.7, label="experimental crf")
    ax4.plot(range(len(yhat_train)), yhat_train, marker="o", ms=2.5, lw=0.7, label="predicted crf")
    ax4.set_xlabel("crf points"); ax4.set_ylabel("crf"); ax4.set_title("(d) Training")
    ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle(f"SVR (FF-3) — {TARGET_INDUSTRY} — {MODE} mode", y=0.98, fontsize=12)
plt.tight_layout(rect=[0,0,1,0.96])
out_png = f"SVR_FF3_{TARGET_INDUSTRY}_{MODE}.png"
plt.savefig(out_png, dpi=160); plt.show()
print(f"Saved figure to: {os.path.abspath(out_png)}")
print(f"Total time: {time.strftime('%M:%S', time.gmtime(time.time()-t0))}")
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 14:10:59 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
High-efficiency version: SVR(FF-3) generates four-panel plot (a)(b)(c)(d), supports three speed levels:
ULTRA (default, fastest), FAST, PRO.
- Introduces Nystroem (RBF approximation) + linear model, avoiding O(n^2) cost of full kernel matrix
- Tunes parameters on the most recent N-year subsample, then refits on the full training set
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
# ========= Configuration =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR_CSV   = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
TARGET_INDUSTRY = "Manuf"   # Cnsmr / Manuf / HiTec / Hlth / Other
MISSING_SENTINEL = -99.99
MODE = "ULTRA"              # "ULTRA" (seconds-level), "FAST", "PRO"
RECENT_YEARS = 8            # FAST/PRO: tune parameters on the most recent N years of data
PLOT_TS = True              # Whether to plot (c)(d) time series graphs
RANDOM_STATE = 42
# ========= Utility Functions =========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # Standardize date column
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # Parse date
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
# ========= Load and Align Data =========
t0 = time.time()
ind = smart_read_csv(INDUSTRY_CSV)
fac = smart_read_csv(FACTOR_CSV)
col_industry = pick_col(ind, [TARGET_INDUSTRY,"manufacturing","制造","manuf"])
col_mktrf = pick_col(fac, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb   = pick_col(fac, ["SMB"])
col_hml   = pick_col(fac, ["HML"])
col_rf    = pick_col(fac, ["RF","riskfree","risk-free"])
need = {"industry":col_industry,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RF":col_rf}
miss = [k for k,v in need.items() if v is None]
if miss: raise ValueError(f"Missing columns: {miss}. Please check CSV column names.")
df = ind[["Date", col_industry]].merge(
        fac[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     ).copy()
# Convert to numeric & handle missing values (float32)
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# manurf = industry return - RF
target_name = f"{TARGET_INDUSTRY.lower()}rf"
df[target_name] = (df[col_industry] - df[col_rf]).astype(np.float32)
X = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y = df[target_name].to_numpy(dtype=np.float32)
# ========= Time Series Split =========
n = len(df); split = int(n*0.8)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]
# ========= Model Factory =========
def build_ultra():
    """Linear approximation (fastest): StandardScaler + LinearSVR (empirical hyperparameters)"""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", C=3.0, epsilon=0.01,
                          max_iter=5000, random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)
    return model, "ULTRA: LinearSVR(C=3.0, eps=0.01)"
def build_fast():
    """Small search on subsample: LinearSVR + Nystroem(RBF approximation)+Ridge; then refit on full training set"""
    # Take the most recent N-year subsample (approx. ~365*N days)
    sub = max(0, n - 365*RECENT_YEARS)
    X_sub, y_sub = X[sub:split], y[sub:split]   # Only take the most recent subsample within the training segment
    tscv = TimeSeriesSplit(n_splits=3)
    # a) LinearSVR (fast)
    pipe_lin = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", max_iter=5000, random_state=RANDOM_STATE))
    ])
    dist_lin = {
        "lin__C": np.logspace(-2, 2, 9),
        "lin__epsilon": np.logspace(-3, -0.5, 6)
    }
    rs_lin = RandomizedSearchCV(pipe_lin, dist_lin, n_iter=16, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_lin.fit(X_sub, y_sub)
    # b) RBF approximation: Nystroem + Ridge (linear solution, very fast, approximates nonlinearity)
    pipe_nys = Pipeline([
        ("scaler", StandardScaler()),
        ("nys", Nystroem(kernel="rbf", random_state=RANDOM_STATE)),
        ("ridge", Ridge(random_state=RANDOM_STATE))
    ])
    dist_nys = {
        "nys__gamma": [0.05, 0.1, "auto"],
        "nys__n_components": [100, 200, 300],
        "ridge__alpha": np.logspace(-3, 1, 5)
    }
    rs_nys = RandomizedSearchCV(pipe_nys, dist_nys, n_iter=18, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_nys.fit(X_sub, y_sub)
    # Choose the better one, and refit once on the full training set
    if -rs_lin.best_score_ <= -rs_nys.best_score_:
        best = rs_lin.best_estimator_
        label = f"FAST: LinearSVR {rs_lin.best_params_}"
    else:
        best = rs_nys.best_estimator_
        label = f"FAST: Nystroem+Ridge {rs_nys.best_params_}"
    best.fit(X_train, y_train)
    return best, label
def build_pro():
    """More systematic: HalvingRandomSearchCV; includes Linear/RBF/Nystroem three configurations"""
    tscv = TimeSeriesSplit(n_splits=5)
    # Three candidates: LinearSVR, SVR-RBF (small range), Nystroem+Ridge
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearSVR())])
    search_space = [
        {   # LinearSVR
            "model": [LinearSVR(loss="epsilon_insensitive", max_iter=7000, random_state=RANDOM_STATE)],
            "model__C": np.logspace(-2, 2, 9),
            "model__epsilon": np.logspace(-3, -0.5, 6)
        },
        {   # SVR-RBF (small range, with cache)
            "model": [SVR(kernel="rbf", cache_size=1000)],
            "model__C": np.logspace(-1, 2, 6),
            "model__gamma": ["scale","auto", 1e-3, 1e-2],
            "model__epsilon": [0.005, 0.01, 0.05, 0.1]
        },
        {   # Nystroem+Ridge (similar via FeatureUnion, but use Pipeline: fit features first, then linear)
        }
    ]
    # To handle Nystroem within one searcher, we first do a separate round (more stable)
    # ——Directly reuse the Nystroem search result from FAST to avoid Halving's complex object sharing issues
    best_fast, label_fast = build_fast()
    hrs = HalvingRandomSearchCV(
        pipe, search_space, factor=3, cv=tscv, scoring="neg_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1, aggressive_elimination=True
    )
    hrs.fit(X_train, y_train)
    best_linear_rbf = hrs.best_estimator_
    # Choose between best_fast and best_linear_rbf (using training CV score as approximation)
    # Since Halving does not maintain consistent best_score_ direction, we use simple external validation
    yhat_a = best_fast.predict(X_test);  mse_a = np.mean((yhat_a - y_test)**2)
    yhat_b = best_linear_rbf.predict(X_test); mse_b = np.mean((yhat_b - y_test)**2)
    if mse_a <= mse_b:
        return best_fast, f"PRO(best=FAST-approx) | {label_fast}"
    else:
        return best_linear_rbf, f"PRO(best=HalvingSearch) | {hrs.best_params_}"
# ========= Training =========
if MODE.upper()=="ULTRA":
    model, model_info = build_ultra()
elif MODE.upper()=="FAST":
    model, model_info = build_fast()
else:
    model, model_info = build_pro()
print("Chosen model ->", model_info)
# ========= Prediction & Metrics =========
yhat_train = model.predict(X_train).astype(np.float32)
yhat_test  = model.predict(X_test ).astype(np.float32)
rho_t, pval_t = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr,pval_tr= spearmanr(y_train, yhat_train, nan_policy="omit")
alpha = 0.05
# ========= Plotting =========
plt.figure(figsize=(11, 9))
# (a) Testing
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test, yhat_test, s=18)
ax1.set_xlabel(f"experimental {target_name}")
ax1.set_ylabel(f"{target_name} predicted")
ax1.set_title("(a) Testing")
ax1.text(0.02,0.98,f"RHO-value: {rho_t:.5f}
PVAL-value: {pval_t:.2e}
at ALPH-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax1.grid(alpha=0.3)
# (b) Training
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train, yhat_train, s=12)
ax2.set_xlabel(f"experimental {target_name}")
ax2.set_ylabel(f"{target_name} predicted")
ax2.set_title("(b) Training")
ax2.text(0.02,0.98,f"RHO-value: {rho_tr:.5f}
PVAL-value: {pval_tr:.0e}
at ALPH-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax2.grid(alpha=0.3)
if PLOT_TS:
    # (c) Testing
    ax3 = plt.subplot(2,2,3)
    ax3.plot(range(len(y_test)),  y_test,  marker="o", ms=3, lw=0.8, label=f"experimental {target_name}")
    ax3.plot(range(len(yhat_test)),yhat_test, marker="o", ms=2.5, lw=0.8, label=f"predicted {target_name}")
    ax3.set_xlabel(f"{target_name} points"); ax3.set_ylabel(target_name); ax3.set_title("(c) Testing")
    ax3.legend(); ax3.grid(alpha=0.3)
    # (d) Training
    ax4 = plt.subplot(2,2,4)
    ax4.plot(range(len(y_train)),  y_train,  marker="o", ms=2.5, lw=0.7, label=f"experimental {target_name}")
    ax4.plot(range(len(yhat_train)),yhat_train, marker="o", ms=2.5, lw=0.7, label=f"predicted {target_name}")
    ax4.set_xlabel(f"{target_name} points"); ax4.set_ylabel(target_name); ax4.set_title("(d) Training")
    ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle(f"SVR (FF-3) — {TARGET_INDUSTRY} — {MODE} mode | {model_info}", y=0.98, fontsize=11)
plt.tight_layout(rect=[0,0,1,0.96])
out_png = f"SVR_FF3_{TARGET_INDUSTRY}_{MODE}.png"
plt.savefig(out_png, dpi=160); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Total time:", time.strftime('%M:%S', time.gmtime(time.time()-t0)))
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 14:42:58 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
High-throughput FF-3 + SVR four-panel figure for large datasets
Targets HiTec industry by default. Modes: ULTRA (fast), FAST, PRO.
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
# ========= 0) PATHS (EDIT ME) =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR_CSV   = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
# ========= 1) CONFIG =========
TARGET_INDUSTRY = "HiTec"   # Cnsmr / Manuf / HiTec / Hlth / Other
MISSING_SENTINEL = -99.99   # Your missing value convention
MODE = "ULTRA"              # ULTRA | FAST | PRO
RECENT_YEARS = 8            # FAST/PRO tunes on the most recent N years of subsample
PLOT_TS = True              # Whether to plot (c)(d) two time series graphs
TS_DOWNSAMPLE_EVERY = 5     # (c)(d) Visualization downsampling step (>=1; larger values make plotting faster)
RANDOM_STATE = 42
# ========= 2) IO HELPERS =========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # Standardize date column name
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # Parse date
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
# ========= 3) LOAD & ALIGN =========
t0 = time.time()
ind = smart_read_csv(INDUSTRY_CSV)
fac = smart_read_csv(FACTOR_CSV)
col_industry = pick_col(ind, [TARGET_INDUSTRY,"hitech","高科","科技"])
col_mktrf = pick_col(fac, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb   = pick_col(fac, ["SMB"])
col_hml   = pick_col(fac, ["HML"])
col_rf    = pick_col(fac, ["RF","riskfree","risk-free"])
need = {"industry":col_industry,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RF":col_rf}
miss = [k for k,v in need.items() if v is None]
if miss: raise ValueError(f"Missing columns: {miss}. Please check CSV column names.")
df = ind[["Date", col_industry]].merge(
        fac[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     ).copy()
# Convert to numeric & handle missing values
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target: hitecrf = industry return - RF
target_name = f"{TARGET_INDUSTRY.lower()}rf"
df[target_name] = (df[col_industry] - df[col_rf]).astype(np.float32)
# Only take FF-3 features
X = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y = df[target_name].to_numpy(dtype=np.float32)
# ========= 4) SPLIT =========
n = len(df); split = int(n*0.8)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]
# ========= 5) MODEL BUILDERS =========
def build_ultra():
    """Fastest: LinearSVR with empirical parameters (often sufficient)"""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", C=3.0, epsilon=0.01,
                          max_iter=6000, random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)
    return model, "ULTRA: LinearSVR(C=3.0, eps=0.01)"
def build_fast():
    """Small search on the most recent N-year subsample: LinearSVR or Nystroem(RBF approximation)+Ridge; then refit on full training set"""
    sub = max(0, n - 365*RECENT_YEARS)
    X_sub, y_sub = X[sub:split], y[sub:split]
    tscv = TimeSeriesSplit(n_splits=3)
    # LinearSVR
    pipe_lin = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", max_iter=6000, random_state=RANDOM_STATE))
    ])
    dist_lin = {
        "lin__C": np.logspace(-2, 2, 9),
        "lin__epsilon": np.logspace(-3, -0.5, 6)
    }
    rs_lin = RandomizedSearchCV(pipe_lin, dist_lin, n_iter=16, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_lin.fit(X_sub, y_sub)
    # Nystroem + Ridge (nonlinear approximation, solves linear problem, fast)
    pipe_nys = Pipeline([
        ("scaler", StandardScaler()),
        ("nys", Nystroem(kernel="rbf", n_components=300, random_state=RANDOM_STATE)),
        ("ridge", Ridge(random_state=RANDOM_STATE))
    ])
    dist_nys = {
        "nys__gamma": [0.05, 0.1, "auto"],
        "nys__n_components": [150, 300, 500],
        "ridge__alpha": np.logspace(-3, 1, 5)
    }
    rs_nys = RandomizedSearchCV(pipe_nys, dist_nys, n_iter=18, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_nys.fit(X_sub, y_sub)
    # Choose the better one, and refit on the full training set
    if -rs_lin.best_score_ <= -rs_nys.best_score_:
        best = rs_lin.best_estimator_; label = f"FAST: LinearSVR {rs_lin.best_params_}"
    else:
        best = rs_nys.best_estimator_; label = f"FAST: Nystroem+Ridge {rs_nys.best_params_}"
    best.fit(X_train, y_train)
    return best, label
def build_pro():
    """More systematic: HalvingRandomSearchCV (LinearSVR and small range RBF SVR)"""
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearSVR())])
    search_space = [
        {"model": [LinearSVR(loss="epsilon_insensitive", max_iter=7000, random_state=RANDOM_STATE)],
         "model__C": np.logspace(-2, 2, 9),
         "model__epsilon": np.logspace(-3, -0.5, 6)},
        {"model": [SVR(kernel="rbf", cache_size=1000)],
         "model__C": np.logspace(-1, 2, 6),
         "model__gamma": ["scale","auto", 1e-3, 1e-2],
         "model__epsilon": [0.005, 0.01, 0.05, 0.1]}
    ]
    hrs = HalvingRandomSearchCV(pipe, search_space, factor=3, cv=tscv,
                                scoring="neg_mean_squared_error",
                                random_state=RANDOM_STATE, n_jobs=-1,
                                aggressive_elimination=True)
    hrs.fit(X_train, y_train)
    best = hrs.best_estimator_
    return best, f"PRO: {hrs.best_params_}"
# ========= 6) TRAIN =========
if MODE.upper()=="ULTRA":
    model, info = build_ultra()
elif MODE.upper()=="FAST":
    model, info = build_fast()
else:
    model, info = build_pro()
print("Chosen model ->", info)
# ========= 7) PREDICT & STATS =========
yhat_train = model.predict(X_train).astype(np.float32)
yhat_test  = model.predict(X_test ).astype(np.float32)
rho_t, pval_t = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr,pval_tr= spearmanr(y_train, yhat_train, nan_policy="omit")
alpha = 0.05
# ========= 8) PLOT =========
plt.figure(figsize=(11, 9))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test, yhat_test, s=18)
ax1.set_xlabel(f"experimental {target_name}")
ax1.set_ylabel(f"{target_name} predicted")
ax1.set_title("(a) Testing")
ax1.text(0.02,0.98,f"RHO-value: {rho_t:.4f}
PVAL-value: {pval_t:.2e}
at ALPH-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train, yhat_train, s=12)
ax2.set_xlabel(f"experimental {target_name}")
ax2.set_ylabel(f"{target_name} predicted")
ax2.set_title("(b) Training")
ax2.text(0.02,0.98,f"RHO-value: {rho_tr:.4f}
PVAL-value: {pval_tr:.0e}
at ALPH-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax2.grid(alpha=0.3)
# Downsample to avoid slow plotting
def maybe_downsample(arr, step):
    return arr[::max(int(step),1)]
if PLOT_TS:
    ds = max(int(TS_DOWNSAMPLE_EVERY), 1)
    # (c) Testing series
    ax3 = plt.subplot(2,2,3)
    ax3.plot(range(len(y_test))[::ds],  maybe_downsample(y_test, ds),
             marker="o", ms=3, lw=0.8, label=f"experimental {target_name}")
    ax3.plot(range(len(yhat_test))[::ds], maybe_downsample(yhat_test, ds),
             marker="o", ms=2.5, lw=0.8, label=f"predicted {target_name}")
    ax3.set_xlabel(f"{target_name} points"); ax3.set_ylabel(target_name); ax3.set_title("(c) Testing")
    ax3.legend(); ax3.grid(alpha=0.3)
    # (d) Training series
    ax4 = plt.subplot(2,2,4)
    ax4.plot(range(len(y_train))[::ds],  maybe_downsample(y_train, ds),
             marker="o", ms=2.5, lw=0.7, label=f"experimental {target_name}")
    ax4.plot(range(len(yhat_train))[::ds], maybe_downsample(yhat_train, ds),
             marker="o", ms=2.5, lw=0.7, label=f"predicted {target_name}")
    ax4.set_xlabel(f"{target_name} points"); ax4.set_ylabel(target_name); ax4.set_title("(d) Training")
    ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle(f"SVR (FF-3) — {TARGET_INDUSTRY} — {MODE} mode | {info}", y=0.98, fontsize=11)
plt.tight_layout(rect=[0,0,1,0.96])
out_png = f"SVR_FF3_{TARGET_INDUSTRY}_{MODE}.png"
plt.savefig(out_png, dpi=160); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Total time:", time.strftime('%M:%S', time.gmtime(time.time()-t0)))
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:00:07 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-3 + SVR four-panel figure for the Health industry (Hlth)
Modes: ULTRA (fast, default) | FAST (subsample tuning with Nystroem)
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
# ========= 0) EDIT YOUR PATHS =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR_CSV   = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv" 
# ========= 1) CONFIG =========
TARGET_INDUSTRY = "Hlth"    # Cnsmr / Manuf / HiTec / Hlth / Other
MODE = "ULTRA"              # ULTRA | FAST
RECENT_YEARS = 8            # FAST: recent-N-years for tuning
MISSING_SENTINEL = -99.99   # your missing value convention
PLOT_TS = True              # plot (c)(d) time-series panels
TS_DOWNSAMPLE_EVERY = 5     # downsample step for plotting only
RANDOM_STATE = 42
# ========= 2) HELPERS =========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # normalize date col
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}: date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # parse date
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
# ========= 3) LOAD & ALIGN =========
t0 = time.time()
ind = smart_read_csv(INDUSTRY_CSV)
fac = smart_read_csv(FACTOR_CSV)
col_industry = pick_col(ind, [TARGET_INDUSTRY,"health","healthcare","医疗","hlth"])
col_mktrf = pick_col(fac, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb   = pick_col(fac, ["SMB"])
col_hml   = pick_col(fac, ["HML"])
col_rf    = pick_col(fac, ["RF","riskfree","risk-free"])
need = {"industry":col_industry,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RF":col_rf}
missing = [k for k,v in need.items() if v is None]
if missing: raise ValueError(f"Missing columns: {missing}. Please check CSV column names.")
df = ind[["Date", col_industry]].merge(
        fac[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     ).copy()
# cast to float32 & handle missing
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# target: healthrf = industry - RF
target_name = f"{TARGET_INDUSTRY.lower()}rf"
df[target_name] = (df[col_industry] - df[col_rf]).astype(np.float32)
X = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y = df[target_name].to_numpy(dtype=np.float32)
# split (time-ordered)
n = len(df); split = int(n*0.8)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]
# ========= 4) MODELS =========
def build_ultra():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", C=3.0, epsilon=0.01,
                          max_iter=6000, random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)
    return model, "ULTRA: LinearSVR(C=3.0, eps=0.01)"
def build_fast():
    sub = max(0, n - 365*RECENT_YEARS)
    X_sub, y_sub = X[sub:split], y[sub:split]
    tscv = TimeSeriesSplit(n_splits=3)
    # LinearSVR small search
    pipe_lin = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", max_iter=6000, random_state=RANDOM_STATE))
    ])
    dist_lin = {
        "lin__C": np.logspace(-2, 2, 9),
        "lin__epsilon": np.logspace(-3, -0.5, 6)
    }
    rs_lin = RandomizedSearchCV(pipe_lin, dist_lin, n_iter=16, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_lin.fit(X_sub, y_sub)
    # Nystroem (RBF approx) + Ridge
    pipe_nys = Pipeline([
        ("scaler", StandardScaler()),
        ("nys", Nystroem(kernel="rbf", n_components=300, random_state=RANDOM_STATE)),
        ("ridge", Ridge(random_state=RANDOM_STATE))
    ])
    dist_nys = {
        "nys__gamma": [0.05, 0.1, "auto"],
        "nys__n_components": [150, 300, 500],
        "ridge__alpha": np.logspace(-3, 1, 5)
    }
    rs_nys = RandomizedSearchCV(pipe_nys, dist_nys, n_iter=18, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_nys.fit(X_sub, y_sub)
    # choose better and refit on full training
    if -rs_lin.best_score_ <= -rs_nys.best_score_:
        best = rs_lin.best_estimator_; label = f"FAST: LinearSVR {rs_lin.best_params_}"
    else:
        best = rs_nys.best_estimator_; label = f"FAST: Nystroem+Ridge {rs_nys.best_params_}"
    best.fit(X_train, y_train)
    return best, label
if MODE.upper() == "ULTRA":
    model, info = build_ultra()
else:
    model, info = build_fast()
print("Chosen model ->", info)
# ========= 5) PREDICT & STATS =========
yhat_train = model.predict(X_train).astype(np.float32)
yhat_test  = model.predict(X_test ).astype(np.float32)
rho_t, pval_t = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr,pval_tr= spearmanr(y_train, yhat_train, nan_policy="omit")
alpha = 0.05
# ========= 6) PLOTTING =========
plt.figure(figsize=(11, 9))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test, yhat_test, s=18)
ax1.set_xlabel(f"experimental {target_name}")
ax1.set_ylabel(f"{target_name} predicted")
ax1.set_title("(a) Testing")
ax1.text(0.02,0.98,f"RHO-value: {rho_t:.5f}
PVAL-value: {pval_t:.2e}
at ALPH-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train, yhat_train, s=12)
ax2.set_xlabel(f"experimental {target_name}")
ax2.set_ylabel(f"{target_name} predicted")
ax2.set_title("(b) Training")
ax2.text(0.02,0.98,f"RHO-value: {rho_tr:.5f}
PVAL-value: {pval_tr:.0e}
at ALPH-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax2.grid(alpha=0.3)
def maybe_downsample(a, step): return a[::max(int(step),1)]
ds = max(int(TS_DOWNSAMPLE_EVERY), 1)
# (c) Testing time-series
if PLOT_TS:
    ax3 = plt.subplot(2,2,3)
    ax3.plot(range(len(y_test))[::ds],  maybe_downsample(y_test, ds),
             marker="o", ms=3, lw=0.8, label=f"experimental {target_name}")
    ax3.plot(range(len(yhat_test))[::ds], maybe_downsample(yhat_test, ds),
             marker="o", ms=2.5, lw=0.8, label=f"predicted {target_name}")
    ax3.set_xlabel(f"{target_name} points"); ax3.set_ylabel(target_name); ax3.set_title("(c) Testing")
    ax3.legend(); ax3.grid(alpha=0.3)
    # (d) Training time-series
    ax4 = plt.subplot(2,2,4)
    ax4.plot(range(len(y_train))[::ds],  maybe_downsample(y_train, ds),
             marker="o", ms=2.5, lw=0.7, label=f"experimental {target_name}")
    ax4.plot(range(len(yhat_train))[::ds], maybe_downsample(yhat_train, ds),
             marker="o", ms=2.5, lw=0.7, label=f"predicted {target_name}")
    ax4.set_xlabel(f"{target_name} points"); ax4.set_ylabel(target_name); ax4.set_title("(d) Training")
    ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle(f"SVR (FF-3) — {TARGET_INDUSTRY} — {MODE} mode | {info}", y=0.98, fontsize=11)
plt.tight_layout(rect=[0,0,1,0.96])
out_png = f"SVR_FF3_{TARGET_INDUSTRY}_{MODE}.png"
plt.savefig(out_png, dpi=160); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Total time:", time.strftime('%M:%S', time.gmtime(time.time()-t0)))
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:17:54 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-3 + SVR four-panel figure for the "Other" industry on large datasets
Modes: ULTRA (fast, default) | FAST (subsample tuning + Nystroem)
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
# ========= 0) EDIT YOUR PATHS =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR_CSV   = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
# ========= 1) CONFIG =========
TARGET_INDUSTRY = "Other"   # Cnsmr / Manuf / HiTec / Hlth / Other
MODE = "ULTRA"              # ULTRA | FAST
RECENT_YEARS = 8            # FAST: tune on recent N years within training set
MISSING_SENTINEL = -99.99   # your missing-value convention
PLOT_TS = True              # draw (c)(d) time-series panels
TS_DOWNSAMPLE_EVERY = 5     # downsample factor for plotting (>=1)
RANDOM_STATE = 42
# ========= 2) HELPERS =========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # normalize date column name
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # parse date
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
def maybe_downsample(a, step): return a[::max(int(step),1)]
# ========= 3) LOAD & ALIGN =========
t0 = time.time()
ind = smart_read_csv(INDUSTRY_CSV)
fac = smart_read_csv(FACTOR_CSV)
col_industry = pick_col(ind, [TARGET_INDUSTRY, "others", "其它", "其他"])
col_mktrf = pick_col(fac, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb   = pick_col(fac, ["SMB"])
col_hml   = pick_col(fac, ["HML"])
col_rf    = pick_col(fac, ["RF","riskfree","risk-free"])
need = {"industry":col_industry,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RF":col_rf}
missing = [k for k,v in need.items() if v is None]
if missing:
    raise ValueError(f"Missing columns: {missing}. Please check CSV column names.")
df = ind[["Date", col_industry]].merge(
        fac[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     ).copy()
# cast to float32 & handle sentinel
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# target: otherrf = industry - RF
target_name = f"{TARGET_INDUSTRY.lower()}rf"
df[target_name] = (df[col_industry] - df[col_rf]).astype(np.float32)
X = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)  # FF-3 only
y = df[target_name].to_numpy(dtype=np.float32)
# time-ordered split
n = len(df); split = int(n*0.8)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]
# ========= 4) MODELS =========
def build_ultra():
    # fastest: LinearSVR with reasonable defaults
    from sklearn.svm import LinearSVR
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", C=3.0, epsilon=0.01,
                          max_iter=6000, random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)
    return model, "ULTRA: LinearSVR(C=3.0, eps=0.01)"
def build_fast():
    # small search on recent sub-sample + Nystroem(RBF approx) candidate
    sub = max(0, n - 365*RECENT_YEARS)
    X_sub, y_sub = X[sub:split], y[sub:split]
    tscv = TimeSeriesSplit(n_splits=3)
    # LinearSVR small random search
    pipe_lin = Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearSVR(loss="epsilon_insensitive", max_iter=6000, random_state=RANDOM_STATE))
    ])
    dist_lin = {
        "lin__C": np.logspace(-2, 2, 9),
        "lin__epsilon": np.logspace(-3, -0.5, 6)
    }
    rs_lin = RandomizedSearchCV(pipe_lin, dist_lin, n_iter=16, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_lin.fit(X_sub, y_sub)
    # Nystroem + Ridge (fast nonlinear approximation)
    pipe_nys = Pipeline([
        ("scaler", StandardScaler()),
        ("nys", Nystroem(kernel="rbf", n_components=300, random_state=RANDOM_STATE)),
        ("ridge", Ridge(random_state=RANDOM_STATE))
    ])
    dist_nys = {
        "nys__gamma": [0.05, 0.1, "auto"],
        "nys__n_components": [150, 300, 500],
        "ridge__alpha": np.logspace(-3, 1, 5)
    }
    rs_nys = RandomizedSearchCV(pipe_nys, dist_nys, n_iter=18, cv=tscv,
                                scoring="neg_mean_squared_error", n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    rs_nys.fit(X_sub, y_sub)
    # choose better one and refit on full training set
    if -rs_lin.best_score_ <= -rs_nys.best_score_:
        best = rs_lin.best_estimator_; label = f"FAST: LinearSVR {rs_lin.best_params_}"
    else:
        best = rs_nys.best_estimator_; label = f"FAST: Nystroem+Ridge {rs_nys.best_params_}"
    best.fit(X_train, y_train)
    return best, label
if MODE.upper() == "ULTRA":
    model, info = build_ultra()
else:
    model, info = build_fast()
print("Chosen model ->", info)
# ========= 5) PREDICT & STATS =========
yhat_train = model.predict(X_train).astype(np.float32)
yhat_test  = model.predict(X_test ).astype(np.float32)
rho_t, pval_t = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr,pval_tr= spearmanr(y_train, yhat_train, nan_policy="omit")
alpha = 0.05
# ========= 6) PLOTTING =========
plt.figure(figsize=(11, 9))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test, yhat_test, s=18)
ax1.set_xlabel(f"experimental {target_name}")
ax1.set_ylabel(f"{target_name} predicted")
ax1.set_title("(a) Testing")
ax1.text(0.02,0.98,f"RHO-value: {rho_t:.5f}
PVAL-value: {pval_t:.2e}
at ALPH-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train, yhat_train, s=12)
ax2.set_xlabel(f"experimental {target_name}")
ax2.set_ylabel(f"{target_name} predicted")
ax2.set_title("(b) Training")
ax2.text(0.02,0.98,f"RHO-value: {rho_tr:.5f}
PVAL-value: {pval_tr:.0e}
at ALPH-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85), fontsize=9)
ax2.grid(alpha=0.3)
# (c)(d) time series (with downsampling for plotting)
if PLOT_TS:
    ds = max(int(TS_DOWNSAMPLE_EVERY), 1)
    ax3 = plt.subplot(2,2,3)
    ax3.plot(range(len(y_test))[::ds],  maybe_downsample(y_test, ds),
             marker="o", ms=3, lw=0.8, label=f"experimental {target_name}")
    ax3.plot(range(len(yhat_test))[::ds], maybe_downsample(yhat_test, ds),
             marker="o", ms=2.5, lw=0.8, label=f"predicted {target_name}")
    ax3.set_xlabel(f"{target_name} points"); ax3.set_ylabel(target_name); ax3.set_title("(c) Testing")
    ax3.legend(); ax3.grid(alpha=0.3)
    ax4 = plt.subplot(2,2,4)
    ax4.plot(range(len(y_train))[::ds],  maybe_downsample(y_train, ds),
             marker="o", ms=2.5, lw=0.7, label=f"experimental {target_name}")
    ax4.plot(range(len(yhat_train))[::ds], maybe_downsample(yhat_train, ds),
             marker="o", ms=2.5, lw=0.7, label=f"predicted {target_name}")
    ax4.set_xlabel(f"{target_name} points"); ax4.set_ylabel(target_name); ax4.set_title("(d) Training")
    ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle(f"SVR (FF-3) — {TARGET_INDUSTRY} — {MODE} mode | {info}", y=0.98, fontsize=11)
plt.tight_layout(rect=[0,0,1,0.96])
out_png = f"SVR_FF-3_{TARGET_INDUSTRY}_{MODE}.png"
plt.savefig(out_png, dpi=160); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Total time:", time.strftime('%M:%S', time.gmtime(time.time()-t0)))
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:06:59 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
F-F6 (five factors + MOM) + Bayesian SVR (BayesSearchCV) —— Large sample accelerated version
- Performs Bayesian parameter tuning only on the "most recent N years" training subsample (optional downsampling), then refits on the full training set
- Outputs two-panel scatter plots: (a) Testing, (b) Training; displays Spearman correlation and p-value
Dependency: scikit-optimize  -> pip install scikit-optimize
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from scipy.stats import spearmanr
# ---- BayesSearchCV (scikit-optimize) ----
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
# ========== Paths (modify as needed)==========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR5_CSV  = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_CSV      = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
# ========== Configuration ==========
TARGET_INDUSTRY = "Cnsmr"   # Cnsmr / Manuf / HiTec / Hlth / Other
AX_LABEL = "cons."          # Axis label (change to "manuf." / "hitec." / "hlth." / "other." when switching industries)
MISSING_SENTINEL = -99.99
RANDOM_STATE = 42
# —— Key acceleration parameters (adjust according to machine/data size)——
RECENT_YEARS = 8            # Perform Bayesian tuning only on the training subsample from the "most recent N years"
TUNE_DOWNSAMPLE_EVERY = 3   # Downsampling step during tuning phase (e.g., take 1 out of every 3 points); 1 means no downsampling
N_ITER = 20                 # Number of Bayesian search iterations (40→more stable but slower; 10→faster)
CV_SPLITS = 3               # Number of folds for time series cross-validation (2/3/4)
N_JOBS = -1                 # Parallel processing
VERBOSE = 0
# ========== Utility Functions ==========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # Date column
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # Parse date
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
def ds(a, step):  # Downsampling
    step = max(int(step), 1)
    return a[::step]
# ========== Read and Merge: Industry + Five Factors + MOM ==========
t0 = time.time()
ind  = smart_read_csv(INDUSTRY_CSV)
fac5 = smart_read_csv(FACTOR5_CSV)
mom  = smart_read_csv(MOM_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"consumer","cons","消费","manuf","hitec","hlth","other","其它","其他"])
col_mktrf= pick_col(fac5, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb  = pick_col(fac5, ["SMB"])
col_hml  = pick_col(fac5, ["HML"])
col_rmw  = pick_col(fac5, ["RMW"])
col_cma  = pick_col(fac5, ["CMA"])
col_rf   = pick_col(fac5, ["RF","riskfree","risk-free"])
col_mom  = pick_col(mom,  ["Mom","MOM","momentum"])
need = {"industry":col_ind,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RMW":col_rmw,"CMA":col_cma,"RF":col_rf,"MOM":col_mom}
missing = [k for k,v in need.items() if v is None]
if missing:
    raise ValueError(f"Missing columns: {missing}. Please check CSV column names/case.")
df = ind[["Date", col_ind]].merge(
        fac5[["Date", col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_rf]],
        on="Date", how="inner"
     ).merge(
        mom[["Date", col_mom]],
        on="Date", how="inner"
     )
# Convert to numeric and handle missing values (float32 + sentinel value)
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target: industry excess return
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
# Features: F-F6
X_all = df[[col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_mom]].to_numpy(dtype=np.float32)
y_all = df["target_crf"].to_numpy(dtype=np.float32)
# Time series split
n = len(df); split = int(n*0.8)
X_train_full, y_train_full = X_all[:split], y_all[:split]
X_test,        y_test      = X_all[split:], y_all[split:]
# ========= Perform Bayesian tuning only on the "recent N-year training subsample (can be downsampled)" =========
# Estimate N years ≈ 365*N days (if your data is trading days, you can replace 365 with 252)
recent_start = max(0, split - 365*RECENT_YEARS)
X_tune = X_all[recent_start:split]
y_tune = y_all[recent_start:split]
# Downsampling for visualization/tuning (reduces number of SVR training/kernel matrix size)
if TUNE_DOWNSAMPLE_EVERY > 1:
    X_tune = ds(X_tune, TUNE_DOWNSAMPLE_EVERY)
    y_tune = ds(y_tune, TUNE_DOWNSAMPLE_EVERY)
# Pipeline: Standardization + SVR (kernel chosen by Bayesian optimization)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", SVR(cache_size=1200))
])
# Search space (converges faster, suitable for large samples)
# You can also fix kernel to "rbf" to make the search more focused
search_spaces = {
    "svr__kernel":  Categorical(["rbf", "linear"]),
    "svr__C":       Real(1e-2, 50, prior="log-uniform"),
    "svr__epsilon": Real(1e-3, 0.15, prior="log-uniform"),
    "svr__gamma":   Real(5e-4, 5e-2, prior="log-uniform")  # Has no effect on linear
}
tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    n_iter=N_ITER,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=VERBOSE,
    refit=True
)
t1 = time.time()
opt.fit(X_tune, y_tune)
best_params = opt.best_params_
print("Bayesian tuning on subset done. Best params:", best_params,
      "| Best CV MSE:", -opt.best_score_,
      "| tuning time:", f"{time.time()-t1:.1f}s")
# ========= Use optimal parameters to refit once on the "full training set" =========
best_svr = SVR(
    kernel=best_params.get("svr__kernel", "rbf"),
    C=best_params.get("svr__C", 3.0),
    epsilon=best_params.get("svr__epsilon", 0.01),
    gamma=best_params.get("svr__gamma", "scale"),
    cache_size=1200
)
final_model = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", best_svr)
])
final_model.fit(X_train_full, y_train_full)
# Predict & Correlation
yhat_train = final_model.predict(X_train_full).astype(np.float32)
yhat_test  = final_model.predict(X_test).astype(np.float32)
rho_t,  pval_t  = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr, pval_tr = spearmanr(y_train_full, yhat_train, nan_policy="omit")
alpha = 0.05
# ========= Plot two-panel scatter plots (consistent with example style)=========
plt.figure(figsize=(12,5))
# (a) Testing
ax1 = plt.subplot(1,2,1)
ax1.scatter(y_test, yhat_test, s=30)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"predicted {AX_LABEL}")
ax1.set_title("testing experimental consumption vs predicted", fontsize=14, fontweight="bold")
ax1.text(0.03,0.97, f"correlation: {rho_t:.5f}
p-value: {pval_t:.2e}
at alpha-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training
ax2 = plt.subplot(1,2,2)
ax2.scatter(y_train_full, yhat_train, s=30)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"predicted {AX_LABEL}")
ax2.set_title("training experimental consumption vs predicted", fontsize=14, fontweight="bold")
ax2.text(0.03,0.97, f"correlation: {rho_tr:.5f}
p-value: {pval_tr:.2e}
at alpha-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
plt.tight_layout()
out_png = f"FF6_BayesianSVR_{TARGET_INDUSTRY}_scatter_2panels_FAST.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Total wall time:", time.strftime('%M:%S', time.gmtime(time.time()-t0)))
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:00:21 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
F-F6 (five factors + MOM) + Bayesian SVR (BayesSearchCV)
Outputs two-panel scatter plots: (a) Testing, (b) Training
"""
# -*- coding: utf-8 -*-
"""
FF-6 (Mkt-RF, SMB, HML, RMW, CMA, MOM) + Bayesian SVR —— Large sample accelerated version
Target industry: Manuf; outputs two-panel scatter plots (a)Testing, (b)Training
Dependency: scikit-optimize -> pip install scikit-optimize
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from scipy.stats import spearmanr
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
# ====== Paths (replace with your actual file paths)======
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR5_CSV  = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_CSV      = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
# ====== Configuration ======
TARGET_INDUSTRY = "Manuf"   # Cnsmr / Manuf / HiTec / Hlth / Other
AX_LABEL = "manufacturing"  # For axis text
MISSING_SENTINEL = -99.99
RANDOM_STATE = 42
# Key acceleration parameters
RECENT_YEARS = 8            # Perform Bayesian tuning only on the most recent N-year training subsample
TUNE_DOWNSAMPLE_EVERY = 3   # Downsampling step during tuning phase (≥1; larger is faster)
N_ITER = 20                 # Number of Bayesian optimization iterations (adjustable 12~40)
CV_SPLITS = 3               # Number of folds for time series CV
N_JOBS = -1                 # Parallel
VERBOSE = 0
# ====== Utility Functions ======
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # Date column
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # Parse date
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
def ds(a, step):
    step = max(int(step), 1)
    return a[::step]
# ====== Read and Merge: Industry + Five Factors + MOM ======
t0 = time.time()
ind  = smart_read_csv(INDUSTRY_CSV)
fac5 = smart_read_csv(FACTOR5_CSV)
mom  = smart_read_csv(MOM_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"manufacturing","制造","manuf"])
col_mktrf= pick_col(fac5, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb  = pick_col(fac5, ["SMB"])
col_hml  = pick_col(fac5, ["HML"])
col_rmw  = pick_col(fac5, ["RMW"])
col_cma  = pick_col(fac5, ["CMA"])
col_rf   = pick_col(fac5, ["RF","riskfree","risk-free"])
col_mom  = pick_col(mom,  ["Mom","MOM","momentum"])
need = {"industry":col_ind,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,
        "RMW":col_rmw,"CMA":col_cma,"RF":col_rf,"MOM":col_mom}
miss = [k for k,v in need.items() if v is None]
if miss: raise ValueError(f"Missing columns: {miss}. Please check CSV column names.")
df = ind[["Date", col_ind]].merge(
        fac5[["Date", col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_rf]],
        on="Date", how="inner"
     ).merge(
        mom[["Date", col_mom]],
        on="Date", how="inner"
     )
# Convert to numeric and handle missing values (float32 + sentinel value)
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target: industry excess return
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
# Features: FF-6
X_all = df[[col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_mom]].to_numpy(dtype=np.float32)
y_all = df["target_crf"].to_numpy(dtype=np.float32)
# Time series split
n = len(df); split = int(n*0.8)
X_train_full, y_train_full = X_all[:split], y_all[:split]
X_test,        y_test      = X_all[split:], y_all[split:]
# ====== Perform Bayesian tuning on subsample ======
recent_start = max(0, split - 365*RECENT_YEARS)
X_tune = X_all[recent_start:split]
y_tune = y_all[recent_start:split]
if TUNE_DOWNSAMPLE_EVERY > 1:
    X_tune = ds(X_tune, TUNE_DOWNSAMPLE_EVERY)
    y_tune = ds(y_tune, TUNE_DOWNSAMPLE_EVERY)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", SVR(cache_size=1200))
])
search_spaces = {
    "svr__kernel":  Categorical(["rbf", "linear"]),
    "svr__C":       Real(1e-2, 50, prior="log-uniform"),
    "svr__epsilon": Real(1e-3, 0.15, prior="log-uniform"),
    "svr__gamma":   Real(5e-4, 5e-2, prior="log-uniform") # Has no effect on linear
}
tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    n_iter=N_ITER,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=VERBOSE,
    refit=True
)
t1 = time.time()
opt.fit(X_tune, y_tune)
print("Best params:", opt.best_params_, "| Best CV MSE:", -opt.best_score_, "| tuning time(s):", f"{time.time()-t1:.1f}")
# ====== Refit once on the "full training set" with optimal parameters ======
best = opt.best_params_
final_model = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", SVR(kernel=best.get("svr__kernel","rbf"),
                C=best.get("svr__C",3.0),
                epsilon=best.get("svr__epsilon",0.01),
                gamma=best.get("svr__gamma","scale"),
                cache_size=1200))
])
final_model.fit(X_train_full, y_train_full)
# Predict and correlate
yhat_train = final_model.predict(X_train_full).astype(np.float32)
yhat_test  = final_model.predict(X_test).astype(np.float32)
rho_t,  pval_t  = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr, pval_tr = spearmanr(y_train_full, yhat_train, nan_policy="omit")
alpha = 0.05
# ====== Plot two-panel scatter plots ======
plt.figure(figsize=(12,5))
# (a) Testing
ax1 = plt.subplot(1,2,1)
ax1.scatter(y_test, yhat_test, s=30)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"predicted {AX_LABEL}")
ax1.set_title("testing experimental manufacturing vs predicted", fontsize=14, fontweight="bold")
ax1.text(0.03,0.97, f"correlation: {rho_t:.5f}
p-value: {pval_t:.2e}
at alpha-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training
ax2 = plt.subplot(1,2,2)
ax2.scatter(y_train_full, yhat_train, s=30)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"predicted {AX_LABEL}")
ax2.set_title("training experimental manufacturing vs predicted", fontsize=14, fontweight="bold")
ax2.text(0.03,0.97, f"correlation: {rho_tr:.5f}
p-value: {pval_tr:.2e}
at alpha-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
plt.tight_layout()
out_png = f"FF6_BayesianSVR_{TARGET_INDUSTRY}_scatter_2panels_FAST.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Total wall time:", time.strftime('%M:%S', time.gmtime(time.time()-t0)))
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:00:21 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
F-F6 (five factors + MOM) + Bayesian SVR (BayesSearchCV)
Outputs two-panel scatter plots: (a) Testing, (b) Training
"""
# -*- coding: utf-8 -*-
"""
FF-6 (Mkt-RF, SMB, HML, RMW, CMA, MOM) + Bayesian SVR —— Large sample accelerated version
Target industry: Manuf; outputs two-panel scatter plots (a)Testing, (b)Training
Dependency: scikit-optimize -> pip install scikit-optimize
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from scipy.stats import spearmanr
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
# ====== Paths (replace with your actual file paths)======
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR5_CSV  = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_CSV      = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
# ====== Configuration ======
TARGET_INDUSTRY = "Manuf"   # Cnsmr / Manuf / HiTec / Hlth / Other
AX_LABEL = "manufacturing"  # For axis text
MISSING_SENTINEL = -99.99
RANDOM_STATE = 42
# Key acceleration parameters
RECENT_YEARS = 8            # Perform Bayesian tuning only on the most recent N-year training subsample
TUNE_DOWNSAMPLE_EVERY = 3   # Downsampling step during tuning phase (≥1; larger is faster)
N_ITER = 20                 # Number of Bayesian optimization iterations (adjustable 12~40)
CV_SPLITS = 3               # Number of folds for time series CV
N_JOBS = -1                 # Parallel
VERBOSE = 0
# ====== Utility Functions ======
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    # Date column
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    # Parse date
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
def ds(a, step):
    step = max(int(step), 1)
    return a[::step]
# ====== Read and Merge: Industry + Five Factors + MOM ======
t0 = time.time()
ind  = smart_read_csv(INDUSTRY_CSV)
fac5 = smart_read_csv(FACTOR5_CSV)
mom  = smart_read_csv(MOM_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"manufacturing","制造","manuf"])
col_mktrf= pick_col(fac5, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb  = pick_col(fac5, ["SMB"])
col_hml  = pick_col(fac5, ["HML"])
col_rmw  = pick_col(fac5, ["RMW"])
col_cma  = pick_col(fac5, ["CMA"])
col_rf   = pick_col(fac5, ["RF","riskfree","risk-free"])
col_mom  = pick_col(mom,  ["Mom","MOM","momentum"])
need = {"industry":col_ind,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,
        "RMW":col_rmw,"CMA":col_cma,"RF":col_rf,"MOM":col_mom}
miss = [k for k,v in need.items() if v is None]
if miss: raise ValueError(f"Missing columns: {miss}. Please check CSV column names.")
df = ind[["Date", col_ind]].merge(
        fac5[["Date", col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_rf]],
        on="Date", how="inner"
     ).merge(
        mom[["Date", col_mom]],
        on="Date", how="inner"
     )
# Convert to numeric and handle missing values (float32 + sentinel value)
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target: industry excess return
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
# Features: FF-6
X_all = df[[col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_mom]].to_numpy(dtype=np.float32)
y_all = df["target_crf"].to_numpy(dtype=np.float32)
# Time series split
n = len(df); split = int(n*0.8)
X_train_full, y_train_full = X_all[:split], y_all[:split]
X_test,        y_test      = X_all[split:], y_all[split:]
# ====== Perform Bayesian tuning on subsample ======
recent_start = max(0, split - 365*RECENT_YEARS)
X_tune = X_all[recent_start:split]
y_tune = y_all[recent_start:split]
if TUNE_DOWNSAMPLE_EVERY > 1:
    X_tune = ds(X_tune, TUNE_DOWNSAMPLE_EVERY)
    y_tune = ds(y_tune, TUNE_DOWNSAMPLE_EVERY)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", SVR(cache_size=1200))
])
search_spaces = {
    "svr__kernel":  Categorical(["rbf", "linear"]),
    "svr__C":       Real(1e-2, 50, prior="log-uniform"),
    "svr__epsilon": Real(1e-3, 0.15, prior="log-uniform"),
    "svr__gamma":   Real(5e-4, 5e-2, prior="log-uniform") # Has no effect on linear
}
tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    n_iter=N_ITER,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=VERBOSE,
    refit=True
)
t1 = time.time()
opt.fit(X_tune, y_tune)
print("Best params:", opt.best_params_, "| Best CV MSE:", -opt.best_score_, "| tuning time(s):", f"{time.time()-t1:.1f}")
# ====== Refit once on the "full training set" with optimal parameters ======
best = opt.best_params_
final_model = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", SVR(kernel=best.get("svr__kernel","rbf"),
                C=best.get("svr__C",3.0),
                epsilon=best.get("svr__epsilon",0.01),
                gamma=best.get("svr__gamma","scale"),
                cache_size=1200))
])
final_model.fit(X_train_full, y_train_full)
# Predict and correlate
yhat_train = final_model.predict(X_train_full).astype(np.float32)
yhat_test  = final_model.predict(X_test).astype(np.float32)
rho_t,  pval_t  = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr, pval_tr = spearmanr(y_train_full, yhat_train, nan_policy="omit")
alpha = 0.05
# ====== Plot two-panel scatter plots ======
plt.figure(figsize=(12,5))
# (a) Testing
ax1 = plt.subplot(1,2,1)
ax1.scatter(y_test, yhat_test, s=30)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"predicted {AX_LABEL}")
ax1.set_title("testing experimental manufacturing vs predicted", fontsize=14, fontweight="bold")
ax1.text(0.03,0.97, f"correlation: {rho_t:.5f}
p-value: {pval_t:.2e}
at alpha-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training
ax2 = plt.subplot(1,2,2)
ax2.scatter(y_train_full, yhat_train, s=30)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"predicted {AX_LABEL}")
ax2.set_title("training experimental manufacturing vs predicted", fontsize=14, fontweight="bold")
ax2.text(0.03,0.97, f"correlation: {rho_tr:.5f}
p-value: {pval_tr:.2e}
at alpha-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
plt.tight_layout()
out_png = f"FF6_BayesianSVR_{TARGET_INDUSTRY}_scatter_2panels_FAST.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Total wall time:", time.strftime('%M:%S', time.gmtime(time.time()-t0)))
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 18:52:19 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-6 (Mkt-RF, SMB, HML, RMW, CMA, MOM) + Bayesian SVR —— Two-panel scatter (HiTec)
Dependency: pip install scikit-optimize
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
# =========== Paths (modify as needed) ===========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR5_CSV  = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_CSV      = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
# =========== Configuration ===========
TARGET_INDUSTRY = "HiTec"     # Cnsmr / Manuf / HiTec / Hlth / Other
AX_LABEL = "hi-tech"          # Axis text
MISSING_SENTINEL = -99.99
RANDOM_STATE = 42
# Large sample acceleration parameters
RECENT_YEARS = 8              # Tune parameters only on the most recent N-year subsample (can be changed to 5~10)
TUNE_DOWNSAMPLE_EVERY = 4     # Subsample downsampling step (larger is faster)
N_ITER = 20                   # Number of Bayesian optimization iterations (12~40)
CV_SPLITS = 3
N_JOBS = -1
VERBOSE = 0
# For extremely large samples: whether to use kernel approximation (very fast)
USE_KERNEL_APPROX = False     # True to enable; can be enabled when data is extremely large
N_COMPONENTS = 1000           # RBFSampler dimension
# =========== Utilities ===========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}: date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
def ds(a, step): 
    step = max(int(step), 1); return a[::step]
# =========== Read and Merge: Industry + Five Factors + MOM ===========
t0 = time.time()
ind  = smart_read_csv(INDUSTRY_CSV)
fac5 = smart_read_csv(FACTOR5_CSV)
mom  = smart_read_csv(MOM_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"hitec","hightech","高科技","科技"])
col_mktrf= pick_col(fac5, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb  = pick_col(fac5, ["SMB"])
col_hml  = pick_col(fac5, ["HML"])
col_rmw  = pick_col(fac5, ["RMW"])
col_cma  = pick_col(fac5, ["CMA"])
col_rf   = pick_col(fac5, ["RF","riskfree","risk-free"])
col_mom  = pick_col(mom,  ["Mom","MOM","momentum"])
need = {"industry":col_ind,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RMW":col_rmw,"CMA":col_cma,"RF":col_rf,"MOM":col_mom}
miss = [k for k,v in need.items() if v is None]
if miss: raise ValueError(f"Missing columns: {miss}; please check CSV column names.")
df = ind[["Date", col_ind]].merge(
        fac5[["Date", col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_rf]],
        on="Date", how="inner"
     ).merge(
        mom[["Date", col_mom]],
        on="Date", how="inner"
     )
# Convert to numeric + handle missing values
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target and features (FF-6)
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
X_all = df[[col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_mom]].to_numpy(dtype=np.float32)
y_all = df["target_crf"].to_numpy(dtype=np.float32)
# Time series split (80/20)
n = len(df); split = int(n*0.8)
X_train_full, y_train_full = X_all[:split], y_all[:split]
X_test,        y_test      = X_all[split:], y_all[split:]
# =========== Perform Bayesian tuning only on the "recent N years + downsampled" subsample ===========
recent_start = max(0, split - 365*RECENT_YEARS)     # For trading days, you can replace 365 with 252
X_tune = X_all[recent_start:split]
y_tune = y_all[recent_start:split]
if TUNE_DOWNSAMPLE_EVERY > 1:
    X_tune = ds(X_tune, TUNE_DOWNSAMPLE_EVERY)
    y_tune = ds(y_tune, TUNE_DOWNSAMPLE_EVERY)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", SVR(cache_size=1200))
])
search_spaces = {
    "svr__kernel":  Categorical(["rbf", "linear"]),      # For even faster speed: leave only ["rbf"]
    "svr__C":       Real(1e-2, 50, prior="log-uniform"),
    "svr__epsilon": Real(1e-3, 0.15, prior="log-uniform"),
    "svr__gamma":   Real(5e-4, 5e-2, prior="log-uniform") # Has no effect on linear
}
tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
opt = BayesSearchCV(
    estimator=pipe, search_spaces=search_spaces, n_iter=N_ITER,
    cv=tscv, scoring="neg_mean_squared_error", n_jobs=N_JOBS,
    random_state=RANDOM_STATE, verbose=VERBOSE, refit=True
)
opt.fit(X_tune, y_tune)
best = opt.best_params_
print("Bayes best:", best, "| best CV MSE:", -opt.best_score_)
# =========== Fit on full training set: true kernel or kernel approximation ===========
if USE_KERNEL_APPROX and best.get("svr__kernel","rbf") == "rbf":
    # Kernel approximation + linear solution, suitable for extremely large samples
    rff = RBFSampler(gamma=float(best.get("svr__gamma", 0.01)),
                     n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("rff", rff),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])
    train_mode = f"RFF(+Ridge), n_comp={N_COMPONENTS}"
else:
    # True SVR (preferred when sample size is not extremely large)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("svr", SVR(kernel=best.get("svr__kernel","rbf"),
                    C=best.get("svr__C",3.0),
                    epsilon=best.get("svr__epsilon",0.01),
                    gamma=best.get("svr__gamma","scale"),
                    cache_size=2000, tol=1e-3, shrinking=True))
    ])
    train_mode = "True SVR"
model.fit(X_train_full, y_train_full)
# Predict and statistics
yhat_train = model.predict(X_train_full).astype(np.float32)
yhat_test  = model.predict(X_test).astype(np.float32)
rho_t,  pval_t  = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr, pval_tr = spearmanr(y_train_full, yhat_train, nan_policy="omit")
alpha = 0.05
# =========== Two-panel scatter plot ===========
plt.figure(figsize=(12,5))
ax1 = plt.subplot(1,2,1)
ax1.scatter(y_test, yhat_test, s=30)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"predicted {AX_LABEL}")
ax1.set_title("testing experimental hi-tech vs predicted", fontsize=14, fontweight="bold")
ax1.text(0.03,0.97, f"correlation: {rho_t:.5f}
p-value: {pval_t:.2e}
at alpha-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
ax2 = plt.subplot(1,2,2)
ax2.scatter(y_train_full, yhat_train, s=30)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"predicted {AX_LABEL}")
ax2.set_title("training experimental hi-tech vs predicted", fontsize=14, fontweight="bold")
ax2.text(0.03,0.97, f"correlation: {rho_tr:.5f}
p-value: {pval_tr:.2e}
at alpha-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
plt.tight_layout()
out_png = f"FF6_BayesianSVR_{TARGET_INDUSTRY}_scatter_2panels.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Train mode:", train_mode)
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 19:13:49 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-6 (Mkt-RF, SMB, HML, RMW, CMA, MOM) + Bayesian SVR —— Two-panel scatter plot (Hlth)
Generates: (a) testing experimental health vs predicted, (b) training experimental health vs predicted
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
# ================= Paths (modify as needed) =================
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR5_CSV  = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_CSV      = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
# ================= Configuration =================
TARGET_INDUSTRY = "Hlth"      # Cnsmr / Manuf / HiTec / Hlth / Other
AX_LABEL = "health"           # Axis text
MISSING_SENTINEL = -99.99
RANDOM_STATE = 42
# ---- Large sample acceleration parameters ----
RECENT_YEARS = 8              # Tune parameters only on the most recent N-year training subsample (for trading days, replace 365 with 252)
TUNE_DOWNSAMPLE_EVERY = 4     # Downsampling step during tuning phase (larger is faster, ≥1)
N_ITER = 20                   # Number of Bayesian optimization iterations (12~40)
CV_SPLITS = 3
N_JOBS = -1
VERBOSE = 0
# ---- Optional for extremely large samples: kernel approximation instead of true kernel SVR (very fast)----
USE_KERNEL_APPROX = False     # True=enable kernel approximation (RFF+Ridge)
N_COMPONENTS = 1000           # RBFSampler dimension
# ================= Utility Functions =================
def smart_read_csv(path):
    # Automatically identify if there is a header row; standardize date column to Date (datetime)
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}:
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
def ds(a, step):
    step = max(int(step), 1)
    return a[::step]
# ================= Read and construct FF-6 dataset =================
t0 = time.time()
ind  = smart_read_csv(INDUSTRY_CSV)
fac5 = smart_read_csv(FACTOR5_CSV)
mom  = smart_read_csv(MOM_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"health","hlth","医疗","医药"])
col_mktrf= pick_col(fac5, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb  = pick_col(fac5, ["SMB"])
col_hml  = pick_col(fac5, ["HML"])
col_rmw  = pick_col(fac5, ["RMW"])
col_cma  = pick_col(fac5, ["CMA"])
col_rf   = pick_col(fac5, ["RF","riskfree","risk-free"])
col_mom  = pick_col(mom,  ["Mom","MOM","momentum"])
need = {"industry":col_ind,"Mkt-RF":col_mktrf,"SMB":col_smb,"HML":col_hml,"RMW":col_rmw,"CMA":col_cma,"RF":col_rf,"MOM":col_mom}
missing = [k for k,v in need.items() if v is None]
if missing:
    raise ValueError(f"Missing columns: {missing}. Please check CSV column names.")
df = ind[["Date", col_ind]].merge(
        fac5[["Date", col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_rf]],
        on="Date", how="inner"
     ).merge(
        mom[["Date", col_mom]],
        on="Date", how="inner"
     )
# Convert to numeric and handle missing values
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target (industry excess return) and feature matrix (FF-6)
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
X_all = df[[col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_mom]].to_numpy(dtype=np.float32)
y_all = df["target_crf"].to_numpy(dtype=np.float32)
# Time series split (80/20)
n = len(df); split = int(n*0.8)
X_train_full, y_train_full = X_all[:split], y_all[:split]
X_test,        y_test      = X_all[split:], y_all[split:]
# ================= Perform Bayesian tuning only on the "recent N years + downsampled" subsample =================
recent_start = max(0, split - 365*RECENT_YEARS)    # If it's trading days, you can change 365 to 252
X_tune = X_all[recent_start:split]
y_tune = y_all[recent_start:split]
if TUNE_DOWNSAMPLE_EVERY > 1:
    X_tune = ds(X_tune, TUNE_DOWNSAMPLE_EVERY)
    y_tune = ds(y_tune, TUNE_DOWNSAMPLE_EVERY)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("svr", SVR(cache_size=1200))
])
search_spaces = {
    "svr__kernel":  Categorical(["rbf", "linear"]),      # For even faster speed, you can leave only ["rbf"]
    "svr__C":       Real(1e-2, 50, prior="log-uniform"),
    "svr__epsilon": Real(1e-3, 0.15, prior="log-uniform"),
    "svr__gamma":   Real(5e-4, 5e-2, prior="log-uniform") # Has no effect on linear
}
tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
opt = BayesSearchCV(
    estimator=pipe, search_spaces=search_spaces, n_iter=N_ITER,
    cv=tscv, scoring="neg_mean_squared_error", n_jobs=N_JOBS,
    random_state=RANDOM_STATE, verbose=VERBOSE, refit=True
)
opt.fit(X_tune, y_tune)
best = opt.best_params_
print("Bayes best:", best, "| best CV MSE:", -opt.best_score_)
# ================= Fit on full training set: true kernel or kernel approximation =================
if USE_KERNEL_APPROX and best.get("svr__kernel","rbf") == "rbf":
    # For extremely large samples: RBF kernel approximation + linear solution (very fast)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("rff", RBFSampler(gamma=float(best.get("svr__gamma", 0.01)),
                           n_components=N_COMPONENTS, random_state=RANDOM_STATE)),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])
    train_mode = f"RFF(+Ridge), n_comp={N_COMPONENTS}"
else:
    # Standard: true kernel SVR with optimal parameters
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("svr", SVR(kernel=best.get("svr__kernel","rbf"),
                    C=best.get("svr__C",3.0),
                    epsilon=best.get("svr__epsilon",0.01),
                    gamma=best.get("svr__gamma","scale"),
                    cache_size=2000, shrinking=True, tol=1e-3))
    ])
    train_mode = "True SVR"
model.fit(X_train_full, y_train_full)
# ================= Predict + Statistics =================
yhat_train = model.predict(X_train_full).astype(np.float32)
yhat_test  = model.predict(X_test).astype(np.float32)
rho_t,  pval_t  = spearmanr(y_test,  yhat_test,  nan_policy="omit")
rho_tr, pval_tr = spearmanr(y_train_full, yhat_train, nan_policy="omit")
alpha = 0.05
# ================= Two-panel scatter plot =================
plt.figure(figsize=(12,5))
# (a) Testing
ax1 = plt.subplot(1,2,1)
ax1.scatter(y_test, yhat_test, s=30)
ax1.set_xlabel("experimental health")
ax1.set_ylabel("predicted health")
ax1.set_title("testing experimental health vs predicted", fontsize=14, fontweight="bold")
ax1.text(0.03,0.97, f"correlation: {rho_t:.5f}
p-value: {pval_t:.6g}
at alpha-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training
ax2 = plt.subplot(1,2,2)
ax2.scatter(y_train_full, yhat_train, s=30)
ax2.set_xlabel("experimental health")
ax2.set_ylabel("predicted health")
ax2.set_title("training experimental health vs predicted", fontsize=14, fontweight="bold")
ax2.text(0.03,0.97, f"correlation: {rho_tr:.5f}
p-value: {pval_tr:.6g}
at alpha-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
plt.tight_layout()
out_png = f"FF6_BayesianSVR_{TARGET_INDUSTRY}_scatter_2panels.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print("Train mode:", train_mode)
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 19:29:30 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-6 (Mkt-RF, SMB, HML, RMW, CMA, MOM) + Bayesian SVR —— Two-panel scatter plot (Other)
Generates: (a) testing experimental others vs predicted, (b) training experimental others vs predicted
"""
# -*- coding: utf-8 -*-
"""
FF-6 (Mkt-RF, SMB, HML, RMW, CMA, MOM) —— Ultra-Fast Kernel Approx:
RBFSampler (RBF feature map) + Ridge replaces true kernel SVR
Target industry: Other; outputs two-panel scatter plots (a)Testing, (b)Training
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
# ===== Paths (modify as needed)=====
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FACTOR5_CSV  = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_CSV      = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
# ===== Configuration =====
TARGET_INDUSTRY = "Other"   # Cnsmr/Manuf/HiTec/Hlth/Other
AX_LABEL = "others"
MISSING_SENTINEL = -99.99
RANDOM_STATE = 42
# Ultra-fast key parameters
N_COMPONENTS = 1200          # RBF mapping dimension (800~2000, larger is more accurate, slower)
SUBSAMPLE_FOR_GAMMA = 8000   # Subsample size for estimating gamma (smaller is faster)
RIDGE_ALPHA = 1.0            # Ridge regression regularization (adjustable 0.3~3)
TRAIN_RATIO = 0.8            # 80/20 time split; if inconvenient, random split can be used (example still uses time split)
# ===== Utility Functions =====
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd","time"}: date_col = c; break
    if date_col is None: date_col = df.columns[0]
    df.rename(columns={date_col:"Date"}, inplace=True)
    def parse_date(x):
        s = str(x).strip().replace("-","").replace("/","")
        try: return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except: return pd.to_datetime(x, errors="coerce")
    df["Date"] = df["Date"].apply(parse_date)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
def pick_col(df, candidates):
    keys = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for t in candidates:
        k = str(t).strip().lower().replace(" ","").replace("_","")
        if k in keys: return keys[k]
    return None
# ===== Read data and construct FF-6 features =====
ind  = smart_read_csv(INDUSTRY_CSV)
fac5 = smart_read_csv(FACTOR5_CSV)
mom  = smart_read_csv(MOM_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"others","other","其它","其他"])
col_mktrf= pick_col(fac5, ["Mkt-RF","Mkt_RF","mktrf","marketminusrf"])
col_smb  = pick_col(fac5, ["SMB"])
col_hml  = pick_col(fac5, ["HML"])
col_rmw  = pick_col(fac5, ["RMW"])
col_cma  = pick_col(fac5, ["CMA"])
col_rf   = pick_col(fac5, ["RF","riskfree","risk-free"])
col_mom  = pick_col(mom,  ["Mom","MOM","momentum"])
need = [col_ind,col_mktrf,col_smb,col_hml,col_rmw,col_cma,col_rf,col_mom]
if any(v is None for v in need):
    raise ValueError("Column name mismatch, please check CSV.")
df = ind[["Date", col_ind]].merge(
        fac5[["Date", col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_rf]],
        on="Date", how="inner"
     ).merge(
        mom[["Date", col_mom]],
        on="Date", how="inner"
     )
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
X_all = df[[col_mktrf, col_smb, col_hml, col_rmw, col_cma, col_mom]].to_numpy(dtype=np.float32)
y_all = df["target_crf"].to_numpy(dtype=np.float32)
# Time series split (maintain order, 80/20)
split = int(len(df)*TRAIN_RATIO)
X_train, y_train = X_all[:split], y_all[:split]
X_test,  y_test  = X_all[split:], y_all[split:]
# ===== Estimate gamma using "median heuristic" (on training subsample)=====
# Take subsample, calculate median distance mdist between samples; gamma ≈ 1 / (2 * mdist^2)
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_train), size=min(SUBSAMPLE_FOR_GAMMA, len(X_train)), replace=False)
Xs = X_train[idx]
# Standardizing before estimating gamma is more stable
Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-8)
# Approximate estimation: use median distance from each point to sample mean instead of pairwise median distance (faster)
mu = Xs.mean(axis=0, keepdims=True)
d = np.sqrt(((Xs - mu)**2).sum(axis=1))
mdist = np.median(d) + 1e-8
gamma = 1.0 / (2.0 * (mdist**2))
print(f"[Auto gamma] median-dist={mdist:.4f} -> gamma≈{gamma:.5f}")
# ===== Model: RBFSampler + Ridge (linear closed-form solution, very fast)=====
model = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("rff", RBFSampler(gamma=float(gamma), n_components=N_COMPONENTS, random_state=RANDOM_STATE)),
    ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE))
])
model.fit(X_train, y_train)
yhat_tr = model.predict(X_train).astype(np.float32)
yhat_te = model.predict(X_test).astype(np.float32)
rho_te, p_te = spearmanr(y_test,  yhat_te, nan_policy="omit")
rho_tr, p_tr = spearmanr(y_train, yhat_tr, nan_policy="omit")
alpha = 0.05
# ===== Plot two-panel scatter plots =====
plt.figure(figsize=(12,5))
ax1 = plt.subplot(1,2,1)
ax1.scatter(y_test, yhat_te, s=30)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"predicted {AX_LABEL}")
ax1.set_title("testing experimental others vs predicted", fontsize=14, fontweight="bold")
ax1.text(0.03,0.97, f"correlation: {rho_te:.5f}
p-value: {p_te:.2e}
at alpha-significant level: {alpha}",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
ax2 = plt.subplot(1,2,2)
ax2.scatter(y_train, yhat_tr, s=30)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"predicted {AX_LABEL}")
ax2.set_title("training experimental others vs predicted", fontsize=14, fontweight="bold")
ax2.text(0.03,0.97, f"correlation: {rho_tr:.5f}
p-value: {p_tr:.2e}
at alpha-significant level: {alpha}",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
plt.tight_layout()
out_png = f"FF6_ULTRA_FAST_{TARGET_INDUSTRY}_scatter_2panels.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure:", os.path.abspath(out_png))
