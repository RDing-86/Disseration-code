# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:19:47 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
LSTM (FF-3) Four-Panel Plot —— Replicating "(a)(b)(c)(d)" layout
(a) Test Scatter  (b) Train Scatter  (c) Test Time Series  (d) Train Time Series
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
# ========= Paths (change to your actual files) =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF3_5_CSV    = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
# ========= Configuration =========
TARGET_INDUSTRY = "Cnsmr"  # Options: Cnsmr / Manuf / HiTec / Hlth / Other
AX_LABEL        = "crf"    # Y-axis label for plots (industry excess return)
LOOKBACK        = 20       # LSTM window (can try 20/60/120)
TRAIN_RATIO     = 0.8
BATCH_SIZE      = 256
EPOCHS          = 60
PATIENCE        = 8
RANDOM_STATE    = 42
MISSING_SENTINEL = -99.99
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
# ========= Helper Functions =========
def smart_read_csv(path: str) -> pd.DataFrame:
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    h0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(k in h0 for k in ["Average","Value Weighted","--","—"]) else 0
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
def pick_col(df: pd.DataFrame, candidates):
    key = {str(c).strip().lower().replace(" ","").replace("_",""): c for c in df.columns}
    for n in candidates:
        k = str(n).strip().lower().replace(" ","").replace("_","")
        if k in key: return key[k]
    return None
def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i]); ys.append(y[i])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)
# ========= Data Loading =========
ind = smart_read_csv(INDUSTRY_CSV)
ff  = smart_read_csv(FF3_5_CSV)
col_ind  = pick_col(ind, [TARGET_INDUSTRY, "cnsmr","manuf","hitec","hlth","other"])
col_mktrf= pick_col(ff,  ["Mkt-RF","mktrf","mkt-rf"])
col_smb  = pick_col(ff,  ["SMB"])
col_hml  = pick_col(ff,  ["HML"])
col_rf   = pick_col(ff,  ["RF","riskfree","risk-free"])
if None in [col_ind,col_mktrf,col_smb,col_hml,col_rf]:
    raise ValueError("Column names not matched, please check CSV headers.")
df = ind[["Date", col_ind]].merge(
        ff[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     )
# Convert to numeric + clean anomalies
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target: industry excess return; Features: FF-3
df["target_crf"] = df[col_ind] - df[col_rf]
X_raw = df[[col_mktrf, col_smb, col_hml]].to_numpy(np.float32)
y_raw = df["target_crf"].to_numpy(np.float32)
# ========= Time-ordered split + Standardization (fit only on train) =========
n = len(df)
assert n > LOOKBACK + 50, "Too few samples to construct sequences."
split = int(n * TRAIN_RATIO)
Xtr_raw, Xte_raw = X_raw[:split], X_raw[split:]
ytr_raw, yte_raw = y_raw[:split], y_raw[split:]
x_scaler = StandardScaler().fit(Xtr_raw)
Xtr = x_scaler.transform(Xtr_raw)
Xte = x_scaler.transform(Xte_raw)
y_scaler = StandardScaler().fit(ytr_raw.reshape(-1,1))
ytr = y_scaler.transform(ytr_raw.reshape(-1,1)).ravel()
yte = y_scaler.transform(yte_raw.reshape(-1,1)).ravel()
# Combine then create sequences, then split by boundary
X_all = np.vstack([Xtr, Xte])
y_all = np.concatenate([ytr, yte])
X_seq, y_seq = build_sequences(X_all, y_all, LOOKBACK)
cut = split - LOOKBACK
X_train, y_train = X_seq[:cut], y_seq[:cut]
X_test,  y_test  = X_seq[cut:], y_seq[cut:]
print(f"[INFO] X_train={X_train.shape}, X_test={X_test.shape}")
# ========= LSTM Model =========
model = models.Sequential([
    layers.Input(shape=(LOOKBACK, X_train.shape[-1])),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]
# Only use validation_split if sufficient samples
if len(X_train) >= 300:
    hist = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=cb_list,
        verbose=1
    )
else:
    cb_list = [
        callbacks.EarlyStopping(monitor="loss", patience=max(3, PATIENCE//2), restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]
    hist = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=cb_list,
        verbose=1
    )
# ========= Predict and Inverse Standardize =========
yhat_tr_s = model.predict(X_train, batch_size=BATCH_SIZE).ravel()
yhat_te_s = model.predict(X_test,  batch_size=BATCH_SIZE).ravel()
yhat_tr = y_scaler.inverse_transform(yhat_tr_s.reshape(-1,1)).ravel()
yhat_te = y_scaler.inverse_transform(yhat_te_s.reshape(-1,1)).ravel()
y_tr    = y_scaler.inverse_transform(y_train.reshape(-1,1)).ravel()
y_te    = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
# ========= Evaluation =========
rho_te, p_te = spearmanr(y_te, yhat_te, nan_policy="omit")
rho_tr, p_tr = spearmanr(y_tr, yhat_tr, nan_policy="omit")
print(f"[Spearman] Test rho={rho_te:.4f} (p={p_te:.2e}), Train rho={rho_tr:.4f} (p={p_tr:.2e})")
# ========= Plot Four Panels =========
plt.figure(figsize=(14,10))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_te, yhat_te, s=18)
ax1.set_xlabel("experimental crf")
ax1.set_ylabel("crf predicted")
ax1.set_title("(a) Testing", fontsize=12)
ax1.text(0.03,0.97,
         f"RHO-value: {rho_te:.5f}
PVAL-value: {p_te:.2e}
at ALPH-significant level: 0.05",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_tr, yhat_tr, s=18)
ax2.set_xlabel("experimental crf")
ax2.set_ylabel("crf predicted")
ax2.set_title("(b) Training", fontsize=12)
ax2.text(0.03,0.97,
         f"RHO-value: {rho_tr:.5f}
PVAL-value: {p_tr:.2e}
at ALPH-significant level: 0.05",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
# (c) Testing timeline
ax3 = plt.subplot(2,2,3)
ax3.plot(range(len(y_te)), y_te,      label="experimental crf", linewidth=1.0)
ax3.plot(range(len(y_te)), yhat_te,   label="predicted crf",   linewidth=1.0, alpha=0.9)
ax3.set_xlabel("crf points"); ax3.set_ylabel("crf")
ax3.set_title("(c) Testing", fontsize=12); ax3.legend(); ax3.grid(alpha=0.3)
# (d) Training timeline
ax4 = plt.subplot(2,2,4)
ax4.plot(range(len(y_tr)), y_tr,      label="experimental crf", linewidth=0.9)
ax4.plot(range(len(y_tr)), yhat_tr,   label="predicted crf",   linewidth=0.9, alpha=0.9)
ax4.set_xlabel("crf points"); ax4.set_ylabel("crf")
ax4.set_title("(d) Training", fontsize=12); ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle("LSTM (FF-3) — Cnsmr", fontsize=14)  # Change industry name as needed
plt.tight_layout(rect=[0,0,1,0.97])
out_png = "LSTM_FF3_Cnsmr_4panels.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure:", os.path.abspath(out_png))
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:18:38 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-3 (Mkt-RF, SMB, HML) + LSTM —— Manufacturing Industry Manuf Four-Panel Plot
(a) Test: Actual vs Predicted (Scatter + Correlation)
(b) Train: Actual vs Predicted (Scatter + Correlation)
(c) Test: Time Series Comparison (Actual/Predicted)
(d) Train: Time Series Comparison (Actual/Predicted)
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tensorflow.keras import layers, models, callbacks, optimizers
import tensorflow as tf
# ========= Paths (change to yours) =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF35_CSV     = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
# ========= Configuration =========
TARGET_INDUSTRY = "Manuf"   # This time using Manufacturing Manuf
AX_LABEL = "manurf"         # Y-axis unit: industry excess return crf(Manuf-RF)
LOOKBACK = 20               # LSTM sequence length
TRAIN_RATIO = 0.8           # 80/20 time-ordered split
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 6                # Early stopping
RANDOM_STATE = 42
MISSING_SENTINEL = -99.99
# ========= Tools =========
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
def build_sequences(X, y, lookback):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
# ========= Read and Construct FF-3 =========
ind  = smart_read_csv(INDUSTRY_CSV)
ff35 = smart_read_csv(FF35_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"manuf","manufacturing","制造"])
col_mktrf= pick_col(ff35, ["Mkt-RF","mktrf","mkt-rf"])
col_smb  = pick_col(ff35, ["SMB"])
col_hml  = pick_col(ff35, ["HML"])
col_rf   = pick_col(ff35, ["RF"])
need = [col_ind, col_mktrf, col_smb, col_hml, col_rf]
if any(v is None for v in need):
    raise ValueError("Column names do not match, please check the two CSVs.")
df = ind[["Date", col_ind]].merge(
        ff35[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     )
# Convert to numeric + handle missing values
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target: industry excess return; Features: three factors
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
X_raw = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y_raw = df["target_crf"].to_numpy(dtype=np.float32)
# Time-ordered split
n = len(df); split = int(n * TRAIN_RATIO)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
# Standardization (fit only on training segment to avoid future leakage)
x_scaler = StandardScaler().fit(X_train_raw)
X_train_s = x_scaler.transform(X_train_raw)
X_test_s  = x_scaler.transform(X_test_raw)
y_scaler = StandardScaler().fit(y_train_raw.reshape(-1,1))
y_train_s = y_scaler.transform(y_train_raw.reshape(-1,1)).ravel()
y_test_s  = y_scaler.transform(y_test_raw.reshape(-1,1)).ravel()
# Construct sequences (combine then split)
X_all_s = np.vstack([X_train_s, X_test_s])
y_all_s = np.concatenate([y_train_s, y_test_s])
X_seq, y_seq = build_sequences(X_all_s, y_all_s, LOOKBACK)
cut = split - LOOKBACK
X_train, y_train = X_seq[:cut], y_seq[:cut]
X_test,  y_test  = X_seq[cut:], y_seq[cut:]
# ========= LSTM =========
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
model = models.Sequential([
    layers.Input(shape=(LOOKBACK, X_train.shape[-1])),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
es  = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=False,                 # Do not shuffle for time series
    callbacks=[es, rlr],
    verbose=1
)
# ========= Predict and Inverse Standardize =========
yhat_train_s = model.predict(X_train, batch_size=BATCH_SIZE).ravel()
yhat_test_s  = model.predict(X_test,  batch_size=BATCH_SIZE).ravel()
yhat_train = y_scaler.inverse_transform(yhat_train_s.reshape(-1,1)).ravel()
yhat_test  = y_scaler.inverse_transform(yhat_test_s.reshape(-1,1)).ravel()
y_train_o  = y_scaler.inverse_transform(y_train.reshape(-1,1)).ravel()
y_test_o   = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
# Correlation (Spearman)
rho_te, p_te = spearmanr(y_test_o,  yhat_test,  nan_policy="omit")
rho_tr, p_tr = spearmanr(y_train_o, yhat_train, nan_policy="omit")
# ========= Plot Four Panels =========
plt.figure(figsize=(14,10))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test_o, yhat_test, s=18)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"{AX_LABEL} predicted")
ax1.set_title("(a) Testing", fontsize=12)
ax1.text(0.03,0.97,
         f"RHO-value: {rho_te:.5f}
PVAL-value: {p_te:.2e}
at ALPH-significant level: 0.05",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train_o, yhat_train, s=18)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"{AX_LABEL} predicted")
ax2.set_title("(b) Training", fontsize=12)
ax2.text(0.03,0.97,
         f"RHO-value: {rho_tr:.5f}
PVAL-value: {p_tr:.2e}
at ALPH-significant level: 0.05",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
# Time series index
dates = df["Date"].to_numpy()
train_idx = np.arange(LOOKBACK, int(n*TRAIN_RATIO))
test_idx  = np.arange(int(n*TRAIN_RATIO), n)
# (c) Testing timeline
ax3 = plt.subplot(2,2,3)
ax3.plot(range(len(test_idx)), y_test_o, label="experimental manurf", linewidth=1.0)
ax3.plot(range(len(test_idx)), yhat_test,  label="predicted manurf",   linewidth=1.0, alpha=0.9)
ax3.set_xlabel("manurf points")
ax3.set_ylabel("manurf")
ax3.set_title("(c) Testing", fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)
# (d) Training timeline
ax4 = plt.subplot(2,2,4)
ax4.plot(range(len(train_idx)), y_train_o, label="experimental manurf", linewidth=0.9)
ax4.plot(range(len(train_idx)), yhat_train,  label="predicted manurf",   linewidth=0.9, alpha=0.9)
ax4.set_xlabel("manurf points")
ax4.set_ylabel("manurf")
ax4.set_title("(d) Training", fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)
plt.suptitle(f"LSTM (FF-3) — {TARGET_INDUSTRY}", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.97])
out_png = f"LSTM_FF3_{TARGET_INDUSTRY}_4panels.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print(f"[Summary] Spearman rho Test={rho_te:.4f} (p={p_te:.2e}), Train={rho_tr:.4f} (p={p_tr:.2e})")
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:47:18 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-3 (Mkt-RF, SMB, HML) + LSTM —— HiTec Four-Panel Plot
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tensorflow.keras import layers, models, callbacks, optimizers
import tensorflow as tf
# ========= Paths (change to yours) =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF35_CSV     = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
# ========= Configuration =========
TARGET_INDUSTRY = "HiTec"     # Cnsmr / Manuf / HiTec / Hlth / Other
AX_LABEL = "hitecrf"          # Y-axis label (high-tech industry excess return)
LOOKBACK = 20                 # Sequence length (days)
TRAIN_RATIO = 0.8             # 80/20 time-ordered split
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 6
RANDOM_STATE = 42
MISSING_SENTINEL = -99.99
# ========= Tools =========
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
def build_sequences(X, y, lookback):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
# ========= Read and Construct FF-3 =========
ind  = smart_read_csv(INDUSTRY_CSV)
ff35 = smart_read_csv(FF35_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"hitec","hi-tech","高科"])
col_mktrf= pick_col(ff35, ["Mkt-RF","mktrf","mkt-rf"])
col_smb  = pick_col(ff35, ["SMB"])
col_hml  = pick_col(ff35, ["HML"])
col_rf   = pick_col(ff35, ["RF","riskfree","risk-free"])
need = [col_ind, col_mktrf, col_smb, col_hml, col_rf]
if any(v is None for v in need):
    raise ValueError("Column names do not match, please check the CSV.")
df = ind[["Date", col_ind]].merge(
        ff35[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     )
# Convert to numeric + handle missing values
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
# Target: industry excess return; Features: three factors
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
X_raw = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y_raw = df["target_crf"].to_numpy(dtype=np.float32)
# ========= Time-ordered split =========
n = len(df); split = int(n * TRAIN_RATIO)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
# ========= Standardization (fit only on training segment to avoid leakage) =========
x_scaler = StandardScaler().fit(X_train_raw)
X_train_s = x_scaler.transform(X_train_raw)
X_test_s  = x_scaler.transform(X_test_raw)
y_scaler = StandardScaler().fit(y_train_raw.reshape(-1,1))
y_train_s = y_scaler.transform(y_train_raw.reshape(-1,1)).ravel()
y_test_s  = y_scaler.transform(y_test_raw.reshape(-1,1)).ravel()
# ========= Construct LSTM Sequences =========
X_all_s = np.vstack([X_train_s, X_test_s])
y_all_s = np.concatenate([y_train_s, y_test_s])
X_seq, y_seq = build_sequences(X_all_s, y_all_s, LOOKBACK)
cut = split - LOOKBACK
X_train, y_train = X_seq[:cut], y_seq[:cut]
X_test,  y_test  = X_seq[cut:], y_seq[cut:]
# ========= LSTM Model =========
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
model = models.Sequential([
    layers.Input(shape=(LOOKBACK, X_train.shape[-1])),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
es  = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=False,
    callbacks=[es, rlr],
    verbose=1
)
# ========= Predict and Inverse Standardize to Original Scale =========
yhat_train_s = model.predict(X_train, batch_size=BATCH_SIZE).ravel()
yhat_test_s  = model.predict(X_test,  batch_size=BATCH_SIZE).ravel()
yhat_train = y_scaler.inverse_transform(yhat_train_s.reshape(-1,1)).ravel()
yhat_test  = y_scaler.inverse_transform(yhat_test_s.reshape(-1,1)).ravel()
y_train_o  = y_scaler.inverse_transform(y_train.reshape(-1,1)).ravel()
y_test_o   = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
# ========= Correlation (Spearman) =========
rho_te, p_te = spearmanr(y_test_o,  yhat_test,  nan_policy="omit")
rho_tr, p_tr = spearmanr(y_train_o, yhat_train, nan_policy="omit")
# ========= Four-Panel Plot =========
plt.figure(figsize=(14,10))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test_o, yhat_test, s=18)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"{AX_LABEL} predicted")
ax1.set_title("(a) Testing", fontsize=12)
ax1.text(0.03,0.97,
         f"RHO-value: {rho_te:.5f}
PVAL-value: {p_te:.2e}
at ALPH-significant level: 0.05",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train_o, yhat_train, s=18)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"{AX_LABEL} predicted")
ax2.set_title("(b) Training", fontsize=12)
ax2.text(0.03,0.97,
         f"RHO-value: {rho_tr:.5f}
PVAL-value: {p_tr:.2e}
at ALPH-significant level: 0.05",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
# (c) Testing timeline
ax3 = plt.subplot(2,2,3)
ax3.plot(range(len(y_test_o)), y_test_o, label="experimental hitecrf", linewidth=1.0)
ax3.plot(range(len(y_test_o)), yhat_test,  label="predicted hitecrf", linewidth=1.0, alpha=0.9)
ax3.set_xlabel("hitecrf points")
ax3.set_ylabel("hitecrf")
ax3.set_title("(c) Testing", fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)
# (d) Training timeline
ax4 = plt.subplot(2,2,4)
ax4.plot(range(len(y_train_o)), y_train_o, label="experimental hitecrf", linewidth=0.9)
ax4.plot(range(len(y_train_o)), yhat_train,  label="predicted hitecrf", linewidth=0.9, alpha=0.9)
ax4.set_xlabel("hitecrf points")
ax4.set_ylabel("hitecrf")
ax4.set_title("(d) Training", fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)
plt.suptitle("LSTM (FF-3) — HiTec", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.97])
out_png = "LSTM_FF3_HiTec_4panels.png"
plt.savefig(out_png, dpi=180); plt.show()
print("Saved figure to:", os.path.abspath(out_png))
print(f"[Summary] Spearman rho Test={rho_te:.4f} (p={p_te:.2e}), Train={rho_tr:.4f} (p={p_tr:.2e})")
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:51:02 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-3 (Mkt-RF, SMB, HML) + LSTM —— Health Industry Four-Panel Plot
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tensorflow.keras import layers, models, callbacks, optimizers
import tensorflow as tf
# ========= File Paths (change to your actual paths) =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF35_CSV     = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv" 
# ========= Configuration =========
TARGET_INDUSTRY = "Hlth"      # Hlth = Health Industry
AX_LABEL = "healthrf"         # Y-axis label
LOOKBACK = 20                 # LSTM window length
TRAIN_RATIO = 0.8
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 6
RANDOM_STATE = 42
MISSING_SENTINEL = -99.99
# ========= Data Reading Tools =========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd"}: date_col = c; break
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
def build_sequences(X, y, lookback):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
# ========= Data Preparation =========
ind  = smart_read_csv(INDUSTRY_CSV)
ff35 = smart_read_csv(FF35_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"hlth","health"])
col_mktrf= pick_col(ff35, ["Mkt-RF","mktrf"])
col_smb  = pick_col(ff35, ["SMB"])
col_hml  = pick_col(ff35, ["HML"])
col_rf   = pick_col(ff35, ["RF"])
df = ind[["Date", col_ind]].merge(
        ff35[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     )
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
X_raw = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y_raw = df["target_crf"].to_numpy(dtype=np.float32)
n = len(df); split = int(n * TRAIN_RATIO)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
x_scaler = StandardScaler().fit(X_train_raw)
X_train_s = x_scaler.transform(X_train_raw)
X_test_s  = x_scaler.transform(X_test_raw)
y_scaler = StandardScaler().fit(y_train_raw.reshape(-1,1))
y_train_s = y_scaler.transform(y_train_raw.reshape(-1,1)).ravel()
y_test_s  = y_scaler.transform(y_test_raw.reshape(-1,1)).ravel()
X_all_s = np.vstack([X_train_s, X_test_s])
y_all_s = np.concatenate([y_train_s, y_test_s])
X_seq, y_seq = build_sequences(X_all_s, y_all_s, LOOKBACK)
cut = split - LOOKBACK
X_train, y_train = X_seq[:cut], y_seq[:cut]
X_test,  y_test  = X_seq[cut:], y_seq[cut:]
# ========= LSTM Model =========
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
model = models.Sequential([
    layers.Input(shape=(LOOKBACK, X_train.shape[-1])),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
es  = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=False,
    callbacks=[es, rlr],
    verbose=1
)
# ========= Prediction =========
yhat_train_s = model.predict(X_train, batch_size=BATCH_SIZE).ravel()
yhat_test_s  = model.predict(X_test,  batch_size=BATCH_SIZE).ravel()
yhat_train = y_scaler.inverse_transform(yhat_train_s.reshape(-1,1)).ravel()
yhat_test  = y_scaler.inverse_transform(yhat_test_s.reshape(-1,1)).ravel()
y_train_o  = y_scaler.inverse_transform(y_train.reshape(-1,1)).ravel()
y_test_o   = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
rho_te, p_te = spearmanr(y_test_o,  yhat_test,  nan_policy="omit")
rho_tr, p_tr = spearmanr(y_train_o, yhat_train, nan_policy="omit")
# ========= Four-Panel Plot =========
plt.figure(figsize=(14,10))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test_o, yhat_test, s=18)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"{AX_LABEL} predicted")
ax1.set_title("(a) Testing", fontsize=12)
ax1.text(0.03,0.97,
         f"RHO-value: {rho_te:.5f}
PVAL-value: {p_te:.2e}
at ALPH-significant level: 0.05",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train_o, yhat_train, s=18)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"{AX_LABEL} predicted")
ax2.set_title("(b) Training", fontsize=12)
ax2.text(0.03,0.97,
         f"RHO-value: {rho_tr:.5f}
PVAL-value: {p_tr:.2e}
at ALPH-significant level: 0.05",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
# (c) Testing timeline
ax3 = plt.subplot(2,2,3)
ax3.plot(range(len(y_test_o)), y_test_o, label="experimental healthrf", linewidth=1.0)
ax3.plot(range(len(y_test_o)), yhat_test,  label="predicted healthrf", linewidth=1.0, alpha=0.9)
ax3.set_xlabel("healthrf points")
ax3.set_ylabel("healthrf")
ax3.set_title("(c) Testing", fontsize=12)
ax3.legend(); ax3.grid(alpha=0.3)
# (d) Training timeline
ax4 = plt.subplot(2,2,4)
ax4.plot(range(len(y_train_o)), y_train_o, label="experimental healthrf", linewidth=0.9)
ax4.plot(range(len(y_train_o)), yhat_train,  label="predicted healthrf", linewidth=0.9, alpha=0.9)
ax4.set_xlabel("healthrf points")
ax4.set_ylabel("healthrf")
ax4.set_title("(d) Training", fontsize=12)
ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle("LSTM (FF-3) — Health industry", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:51:02 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-3 (Mkt-RF, SMB, HML) + LSTM —— Health Industry Four-Panel Plot
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tensorflow.keras import layers, models, callbacks, optimizers
import tensorflow as tf
# ========= File Paths (change to your actual paths) =========
INDUSTRY_CSV = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF35_CSV     = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv" 
# ========= Configuration =========
TARGET_INDUSTRY = "Hlth"      # Hlth = Health Industry
AX_LABEL = "healthrf"         # Y-axis label
LOOKBACK = 20                 # LSTM window length
TRAIN_RATIO = 0.8
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 6
RANDOM_STATE = 42
MISSING_SENTINEL = -99.99
# ========= Data Reading Tools =========
def smart_read_csv(path):
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str, encoding_errors="ignore")
    cell0 = "" if pd.isna(peek.iloc[0,0]) else str(peek.iloc[0,0])
    header_row = 1 if any(m in cell0 for m in ["Average","Value Weighted","--","—"]) else 0
    df = pd.read_csv(path, header=header_row, encoding_errors="ignore")
    def norm(s): return str(s).strip().lower().replace(" ","").replace("_","")
    date_col = None
    for c in df.columns:
        if norm(c) in {"date","dates","yyyymmdd"}: date_col = c; break
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
def build_sequences(X, y, lookback):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
# ========= Data Preparation =========
ind  = smart_read_csv(INDUSTRY_CSV)
ff35 = smart_read_csv(FF35_CSV)
col_ind  = pick_col(ind,  [TARGET_INDUSTRY,"hlth","health"])
col_mktrf= pick_col(ff35, ["Mkt-RF","mktrf"])
col_smb  = pick_col(ff35, ["SMB"])
col_hml  = pick_col(ff35, ["HML"])
col_rf   = pick_col(ff35, ["RF"])
df = ind[["Date", col_ind]].merge(
        ff35[["Date", col_mktrf, col_smb, col_hml, col_rf]],
        on="Date", how="inner"
     )
for c in df.columns:
    if c != "Date":
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        df.loc[df[c] <= MISSING_SENTINEL, c] = np.nan
df.dropna(inplace=True)
df["target_crf"] = (df[col_ind] - df[col_rf]).astype(np.float32)
X_raw = df[[col_mktrf, col_smb, col_hml]].to_numpy(dtype=np.float32)
y_raw = df["target_crf"].to_numpy(dtype=np.float32)
n = len(df); split = int(n * TRAIN_RATIO)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
x_scaler = StandardScaler().fit(X_train_raw)
X_train_s = x_scaler.transform(X_train_raw)
X_test_s  = x_scaler.transform(X_test_raw)
y_scaler = StandardScaler().fit(y_train_raw.reshape(-1,1))
y_train_s = y_scaler.transform(y_train_raw.reshape(-1,1)).ravel()
y_test_s  = y_scaler.transform(y_test_raw.reshape(-1,1)).ravel()
X_all_s = np.vstack([X_train_s, X_test_s])
y_all_s = np.concatenate([y_train_s, y_test_s])
X_seq, y_seq = build_sequences(X_all_s, y_all_s, LOOKBACK)
cut = split - LOOKBACK
X_train, y_train = X_seq[:cut], y_seq[:cut]
X_test,  y_test  = X_seq[cut:], y_seq[cut:]
# ========= LSTM Model =========
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
model = models.Sequential([
    layers.Input(shape=(LOOKBACK, X_train.shape[-1])),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
es  = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=False,
    callbacks=[es, rlr],
    verbose=1
)
# ========= Prediction =========
yhat_train_s = model.predict(X_train, batch_size=BATCH_SIZE).ravel()
yhat_test_s  = model.predict(X_test,  batch_size=BATCH_SIZE).ravel()
yhat_train = y_scaler.inverse_transform(yhat_train_s.reshape(-1,1)).ravel()
yhat_test  = y_scaler.inverse_transform(yhat_test_s.reshape(-1,1)).ravel()
y_train_o  = y_scaler.inverse_transform(y_train.reshape(-1,1)).ravel()
y_test_o   = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
rho_te, p_te = spearmanr(y_test_o,  yhat_test,  nan_policy="omit")
rho_tr, p_tr = spearmanr(y_train_o, yhat_train, nan_policy="omit")
# ========= Four-Panel Plot =========
plt.figure(figsize=(14,10))
# (a) Testing scatter
ax1 = plt.subplot(2,2,1)
ax1.scatter(y_test_o, yhat_test, s=18)
ax1.set_xlabel(f"experimental {AX_LABEL}")
ax1.set_ylabel(f"{AX_LABEL} predicted")
ax1.set_title("(a) Testing", fontsize=12)
ax1.text(0.03,0.97,
         f"RHO-value: {rho_te:.5f}
PVAL-value: {p_te:.2e}
at ALPH-significant level: 0.05",
         transform=ax1.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax1.grid(alpha=0.3)
# (b) Training scatter
ax2 = plt.subplot(2,2,2)
ax2.scatter(y_train_o, yhat_train, s=18)
ax2.set_xlabel(f"experimental {AX_LABEL}")
ax2.set_ylabel(f"{AX_LABEL} predicted")
ax2.set_title("(b) Training", fontsize=12)
ax2.text(0.03,0.97,
         f"RHO-value: {rho_tr:.5f}
PVAL-value: {p_tr:.2e}
at ALPH-significant level: 0.05",
         transform=ax2.transAxes, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax2.grid(alpha=0.3)
# (c) Testing timeline
ax3 = plt.subplot(2,2,3)
ax3.plot(range(len(y_test_o)), y_test_o, label="experimental healthrf", linewidth=1.0)
ax3.plot(range(len(y_test_o)), yhat_test,  label="predicted healthrf", linewidth=1.0, alpha=0.9)
ax3.set_xlabel("healthrf points")
ax3.set_ylabel("healthrf")
ax3.set_title("(c) Testing", fontsize=12)
ax3.legend(); ax3.grid(alpha=0.3)
# (d) Training timeline
ax4 = plt.subplot(2,2,4)
ax4.plot(range(len(y_train_o)), y_train_o, label="experimental healthrf", linewidth=0.9)
ax4.plot(range(len(y_train_o)), yhat_train,  label="predicted healthrf", linewidth=0.9, alpha=0.9)
ax4.set_xlabel("healthrf points")
ax4.set_ylabel("healthrf")
ax4.set_title("(d) Training", fontsize=12)
ax4.legend(); ax4.grid(alpha=0.3)
plt.suptitle("LSTM (FF-3) — Health industry", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 18:35:22 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
LSTM (FF-6) Bulletproof Process:
- Read industry daily returns (5 industry columns), five factors (Mkt-RF, SMB, HML, RMW, CMA, RF), momentum Mom
- Align dates, target = industry excess return (industry - RF), features = six factors (including Mom)
- Construct time window sequences, train LSTM, plot actual vs predicted scatter for Testing/Training (with Spearman correlation)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# ========= 0) Paths (please change to your actual paths) =========
IND_PATH = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF5_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
# Select industry: 'Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other'
TARGET_COL = "Cnsmr"
# Hyperparameters (adjustable as needed)
TIME_WINDOW = 20            # LSTM time window length
TEST_RATIO  = 0.2           # Test ratio
EPOCHS      = 60
BATCH_SIZE  = 256
LSTM_UNITS  = 96
DROPOUT     = 0.2
LR          = 1e-3
SEED        = 42
tf.keras.utils.set_random_seed(SEED)
# ========= 1) Utility Functions (Bulletproof Reading) =========
def _clean_header(names):
    """Remove spaces, standardize case style but preserve original case for display"""
    return [str(c).strip().replace('\xa0', ' ') for c in names]
def _try_find(colnames, candidates):
    """Find a field name from a list of possible aliases in colnames, return the first match, otherwise None"""
    low = {c.lower(): c for c in colnames}
    for cand in candidates:
        k = cand.lower().strip()
        if k in low:
            return low[k]
    return None
def read_industries_daily(path):
    """
    Industry CSV (e.g.: first row is description, second row is header; first column is date)
    Target columns typically: Cnsmr, Manuf, HiTec, Hlth, Other
    """
    # Try reading with header from row 2; if fails, fall back to header=0
    for hdr in [1, 0]:
        try:
            df = pd.read_csv(path, header=hdr)
            break
        except Exception:
            continue
    df.columns = _clean_header(df.columns)
    # First column treated as date
    date = df.iloc[:, 0].astype(str).str.strip()
    df = df.iloc[:, 1:].copy()
    df.columns = _clean_header(df.columns)
    # Some files have slight column name variations, perform a loose correction
    rename_map = {}
    for want, aliases in {
        'Cnsmr': ['Cnsmr', 'Cnsum', 'Cnsumr', 'Consumer', 'Cnsmer'],
        'Manuf': ['Manuf', 'Manufacturing', 'Manu'],
        'HiTec': ['HiTec', 'Hitec', 'HiTech', 'HighTech', 'Hi-Tec'],
        'Hlth' : ['Hlth', 'Health', 'Healthcare'],
        'Other': ['Other', 'Othr', 'Others']
    }.items():
        found = _try_find(df.columns, aliases)
        if found and found != want:
            rename_map[found] = want
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    # Keep only the industry columns we need (those that exist)
    keep = [c for c in ['Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other'] if c in df.columns]
    if not keep:
        raise ValueError(f"No industry columns recognized in industry file. Available columns: {df.columns.tolist()}")
    # Convert to numeric
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.insert(0, 'date', date.values[:len(df)])
    df = df.dropna(how='any').set_index('date')
    return df[keep]
def read_ff5_daily(path):
    """
    FF5 CSV: first column date, others: Mkt-RF, SMB, HML, RMW, CMA, RF
    """
    df = pd.read_csv(path)
    df.columns = _clean_header(df.columns)
    date = df.iloc[:, 0].astype(str).str.strip()
    df = df.iloc[:, 1:].copy()
    df.columns = _clean_header(df.columns)
    # Loose matching for key columns
    col_map = {}
    need_map = {
        'Mkt-RF': ['Mkt-RF', 'Mkt_RF', 'MKT-RF', 'MKT_RF', 'Mkt-Rf'],
        'SMB'   : ['SMB'],
        'HML'   : ['HML'],
        'RMW'   : ['RMW'],
        'CMA'   : ['CMA'],
        'RF'    : ['RF', 'Rf', 'R_f', 'Rfree', 'Risk-free']
    }
    for std, aliases in need_map.items():
        found = _try_find(df.columns, aliases)
        if not found:
            raise ValueError(f"Five-factor file missing column: {std} (available columns: {df.columns.tolist()})")
        col_map[found] = std
    df.rename(columns=col_map, inplace=True)
    for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.insert(0, 'date', date.values[:len(df)])
    df = df.dropna(how='any').set_index('date')
    return df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]
def read_mom_daily(path):
    """
    MOM CSV: first column date, column name might be Mom / MOM / Momentum
    """
    df = pd.read_csv(path)
    df.columns = _clean_header(df.columns)
    date = df.iloc[:, 0].astype(str).str.strip()
    df = df.iloc[:, 1:].copy()
    df.columns = _clean_header(df.columns)
    mom_col = _try_find(df.columns, ['Mom', 'MOM', 'Momentum'])
    if not mom_col:
        raise ValueError(f"Momentum file did not find Mom column (available columns: {df.columns.tolist()})")
    df.rename(columns={mom_col: 'Mom'}, inplace=True)
    df['Mom'] = pd.to_numeric(df['Mom'], errors='coerce')
    df.insert(0, 'date', date.values[:len(df)])
    df = df.dropna(how='any').set_index('date')
    return df[['Mom']]
# ========= 2) Read Data and Align =========
ind_df = read_industries_daily(IND_PATH)
ff5_df = read_ff5_daily(FF5_PATH)
mom_df = read_mom_daily(MOM_PATH)
# Merge into 6 factors
fac_df = ff5_df.join(mom_df, how='inner')  # Index is string date
# Target: industry excess return (industry - RF)
if TARGET_COL not in ind_df.columns:
    raise KeyError(f"Industry column {TARGET_COL} does not exist. Options: {ind_df.columns.tolist()}")
y_raw = ind_df[[TARGET_COL]].join(fac_df[['RF']], how='inner')
y_raw['excess'] = y_raw[TARGET_COL] - y_raw['RF']
y = y_raw[['excess']].copy()
# Features: Mkt-RF, SMB, HML, RMW, CMA, Mom
X = fac_df.loc[y.index, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']].copy()
# ========= 3) Standardization and Serialization =========
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X.values)
y_scaled = scaler_y.fit_transform(y.values)
def make_sequences(Xa, ya, win=20):
    Xs, ys = [], []
    for i in range(len(Xa) - win):
        Xs.append(Xa[i:i+win])
        ys.append(ya[i+win])
    return np.asarray(Xs), np.asarray(ys)
Xs, ys = make_sequences(X_scaled, y_scaled, TIME_WINDOW)
# Time-ordered split
n = len(Xs)
split = int(n * (1 - TEST_RATIO))
X_train, X_test = Xs[:split], Xs[split:]
y_train, y_test = ys[:split], ys[split:]
# ========= 4) Modeling and Training =========
model = Sequential([
    LSTM(LSTM_UNITS, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(DROPOUT),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
hist = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
# ========= 5) Prediction and Inverse Standardization =========
y_tr_pred = scaler_y.inverse_transform(model.predict(X_train))
y_te_pred = scaler_y.inverse_transform(model.predict(X_test))
y_tr_true = scaler_y.inverse_transform(y_train)
y_te_true = scaler_y.inverse_transform(y_test)
# ========= 6) Evaluation & Plotting (consistent with example style) =========
rho_te, p_te = spearmanr(y_te_true.ravel(), y_te_pred.ravel())
rho_tr, p_tr = spearmanr(y_tr_true.ravel(), y_tr_pred.ravel())
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# (a) Testing
axes[0].scatter(y_te_true, y_te_pred, s=14, alpha=0.75)
axes[0].set_title("(a) Testing")
axes[0].set_xlabel(f"experimental {TARGET_COL.lower()}")
axes[0].set_ylabel(f"predicted {TARGET_COL.lower()}")
axes[0].text(0.02, 0.98,
             f"correlation: {rho_te:.5f}
p-value: {p_te:.3e}
at alpha-significant level: 0.05",
             transform=axes[0].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
# (b) Training
axes[1].scatter(y_tr_true, y_tr_pred, s=14, alpha=0.75)
axes[1].set_title("(b) Training")
axes[1].set_xlabel(f"experimental {TARGET_COL.lower()}")
axes[1].set_ylabel(f"predicted {TARGET_COL.lower()}")
axes[1].text(0.02, 0.98,
             f"correlation: {rho_tr:.5f}
p-value: {p_tr:.3e}
at alpha-significant level: 0.05",
             transform=axes[1].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
plt.suptitle(f"LSTM (FF-6) — {TARGET_COL} industry (Daily)")
plt.tight_layout()
plt.show()
print("
Data size:")
print("Industry rows: ", len(ind_df), " Five-factor rows: ", len(ff5_df), " Mom rows: ", len(mom_df))
print("Aligned samples: ", len(X), " Training sequences: ", len(X_train), " Test sequences: ", len(X_test))
print("Spearman ρ (test / train): ", f"{rho_te:.3f} / {rho_tr:.3f}")
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:09:09 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
FF-6 (FF5 + Momentum) + LSTM for industry excess returns
Plots: (a) Testing vs Predicted, (b) Training vs Predicted (with Spearman rho & p-value)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# ========= 0) Paths (please change to your actual paths) =========
IND_PATH = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF5_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
TARGET_COL = "Manuf"   # Options: 'Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other'
# ========= 1) Training Settings (adjustable as needed) =========
TIME_WINDOW = 20        # LSTM sequence window length
TEST_RATIO  = 0.2       # Test set ratio (time-ordered)
EPOCHS      = 60
BATCH_SIZE  = 256
LSTM_UNITS  = 96
DROPOUT     = 0.2
LR          = 1e-3
SEED        = 42
tf.keras.utils.set_random_seed(SEED)
# ========= 2) Bulletproof Reading Tools =========
def _clean_header(cols):
    return [str(c).strip().replace('\xa0', ' ') for c in cols]
def _try_find(colnames, aliases):
    low = {c.lower(): c for c in colnames}
    for a in aliases:
        k = a.lower().strip()
        if k in low:
            return low[k]
    return None
def read_industries_daily(path):
    # Some files have a description in the first row, header starts from row 2
    for hdr in [1, 0]:
        try:
            df = pd.read_csv(path, header=hdr)
            break
        except Exception:
            continue
    df.columns = _clean_header(df.columns)
    date = df.iloc[:, 0].astype(str).str.strip()
    df = df.iloc[:, 1:].copy()
    df.columns = _clean_header(df.columns)
    # Loose column name mapping
    rename_map = {}
    for want, aliases in {
        'Cnsmr': ['Cnsmr','Cnsum','Cnsumr','Consumer'],
        'Manuf': ['Manuf','Manufacturing','Manu'],
        'HiTec': ['HiTec','Hitec','HiTech','HighTech','Hi-Tec'],
        'Hlth' : ['Hlth','Health','Healthcare'],
        'Other': ['Other','Others','Othr']
    }.items():
        found = _try_find(df.columns, aliases)
        if found and found != want:
            rename_map[found] = want
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    keep = [c for c in ['Cnsmr','Manuf','HiTec','Hlth','Other'] if c in df.columns]
    if not keep:
        raise ValueError(f"Industry file did not recognize target columns; existing columns: {df.columns.tolist()}")
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.insert(0, 'date', date.values[:len(df)])
    df = df.dropna(how='any').set_index('date')
    return df[keep]
def read_ff5_daily(path):
    df = pd.read_csv(path)
    df.columns = _clean_header(df.columns)
    date = df.iloc[:, 0].astype(str).str.strip()
    df = df.iloc[:, 1:].copy()
    df.columns = _clean_header(df.columns)
    need = {
        'Mkt-RF': ['Mkt-RF','Mkt_RF','MKT-RF','MKT_RF'],
        'SMB'   : ['SMB'],
        'HML'   : ['HML'],
        'RMW'   : ['RMW'],
        'CMA'   : ['CMA'],
        'RF'    : ['RF','Rf','Risk-free','Rfree']
    }
    colmap = {}
    for std, aliases in need.items():
        found = _try_find(df.columns, aliases)
        if not found:
            raise ValueError(f"Five-factor file missing column {std}; existing columns: {df.columns.tolist()}")
        colmap[found] = std
    df.rename(columns=colmap, inplace=True)
    for c in ['Mkt-RF','SMB','HML','RMW','CMA','RF']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.insert(0, 'date', date.values[:len(df)])
    df = df.dropna(how='any').set_index('date')
    return df[['Mkt-RF','SMB','HML','RMW','CMA','RF']]
def read_mom_daily(path):
    df = pd.read_csv(path)
    df.columns = _clean_header(df.columns)
    date = df.iloc[:, 0].astype(str).str.strip()
    df = df.iloc[:, 1:].copy()
    df.columns = _clean_header(df.columns)
    mom_col = _try_find(df.columns, ['Mom','MOM','Momentum'])
    if not mom_col:
        raise ValueError(f"Momentum file did not find Mom column; existing columns: {df.columns.tolist()}")
    df.rename(columns={mom_col:'Mom'}, inplace=True)
    df['Mom'] = pd.to_numeric(df['Mom'], errors='coerce')
    df.insert(0, 'date', date.values[:len(df)])
    df = df.dropna(how='any').set_index('date')
    return df[['Mom']]
# ========= 3) Reading and Alignment =========
ind_df = read_industries_daily(IND_PATH)
ff5_df = read_ff5_daily(FF5_PATH)
mom_df = read_mom_daily(MOM_PATH)
if TARGET_COL not in ind_df.columns:
    raise KeyError(f"{TARGET_COL} not in industry columns; options: {ind_df.columns.tolist()}")
fac_df = ff5_df.join(mom_df, how='inner')  # Index is date string
# Target: industry excess return (industry - RF)
y_raw = ind_df[[TARGET_COL]].join(fac_df[['RF']], how='inner')
y_raw['excess'] = y_raw[TARGET_COL] - y_raw['RF']
y = y_raw[['excess']].copy()
# Features: six factors
X = fac_df.loc[y.index, ['Mkt-RF','SMB','HML','RMW','CMA','Mom']].copy()
# ========= 4) Standardization and Serialization =========
sx, sy = StandardScaler(), StandardScaler()
X_scaled = sx.fit_transform(X.values)
y_scaled = sy.fit_transform(y.values)
def make_sequences(Xa, ya, win=20):
    Xs, ys = [], []
    for i in range(len(Xa) - win):
        Xs.append(Xa[i:i+win])
        ys.append(ya[i+win])
    return np.asarray(Xs), np.asarray(ys)
Xs, ys = make_sequences(X_scaled, y_scaled, TIME_WINDOW)
n = len(Xs)
split = int(n * (1 - TEST_RATIO))
X_train, X_test = Xs[:split], Xs[split:]
y_train, y_test = ys[:split], ys[split:]
# ========= 5) LSTM Model =========
model = Sequential([
    LSTM(LSTM_UNITS, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(DROPOUT),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
hist = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
# ========= 6) Prediction (Inverse Standardization) =========
y_tr_pred = sy.inverse_transform(model.predict(X_train))
y_te_pred = sy.inverse_transform(model.predict(X_test))
y_tr_true = sy.inverse_transform(y_train)
y_te_true = sy.inverse_transform(y_test)
# ========= 7) Evaluation & Plotting =========
rho_te, p_te = spearmanr(y_te_true.ravel(), y_te_pred.ravel())
rho_tr, p_tr = spearmanr(y_tr_true.ravel(), y_tr_pred.ravel())
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# (a) Testing
axes[0].scatter(y_te_true, y_te_pred, s=14, alpha=0.75)
axes[0].set_title("(a) Testing")
axes[0].set_xlabel("experimental manufacturing" if TARGET_COL=="Manuf" else f"experimental {TARGET_COL.lower()}")
axes[0].set_ylabel("predicted manufacturing"   if TARGET_COL=="Manuf" else f"predicted {TARGET_COL.lower()}")
axes[0].text(0.02, 0.98,
             f"correlation: {rho_te:.5f}
"
             f"p-value: {p_te:.3e}
"
             f"at alpha-significant level: 0.05",
             transform=axes[0].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
# (b) Training
axes[1].scatter(y_tr_true, y_tr_pred, s=14, alpha=0.75)
axes[1].set_title("(b) Training")
axes[1].set_xlabel("experimental manufacturing" if TARGET_COL=="Manuf" else f"experimental {TARGET_COL.lower()}")
axes[1].set_ylabel("predicted manufacturing"   if TARGET_COL=="Manuf" else f"predicted {TARGET_COL.lower()}")
axes[1].text(0.02, 0.98,
             f"correlation: {rho_tr:.5f}
"
             f"p-value: {p_tr:.3e}
"
             f"at alpha-significant level: 0.05",
             transform=axes[1].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
plt.suptitle(f"LSTM (FF-6) — {TARGET_COL} industry (Daily)")
plt.tight_layout()
plt.show()
print("
Summary")
print("Samples (aligned):", len(X))
print("Train/Test sequences:", len(X_train), "/", len(X_test))
print(f"Spearman rho (test/train): {rho_te:.3f} / {rho_tr:.3f}")
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:20:21 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
LSTM (FF-6 = FF5 + Momentum) → HiTec excess return
Outputs two scatter plots just like the sample:
(a) testing experimental hi-tech vs predicted
(b) training experimental hi-tech vs predicted
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# =============== File Paths (please change to your actual paths) ===============
IND_PATH = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF5_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
TARGET_COL = "HiTec"       # High-tech industry; can be changed to Cnsmr / Manuf / Hlth / Other
TIME_WINDOW = 20           # Sequence window
TEST_RATIO  = 0.2
EPOCHS      = 60
BATCH_SIZE  = 256
LSTM_UNITS  = 96
DROPOUT     = 0.2
LR          = 1e-3
SEED        = 42
tf.keras.utils.set_random_seed(SEED)
# ------------------ Bulletproof Reading ------------------
def _clean(cols): return [str(c).strip().replace('\xa0',' ') for c in cols]
def _find(cols, aliases):
    low = {c.lower():c for c in cols}
    for a in aliases:
        if a.lower() in low: return low[a.lower()]
    return None
def read_industries(path):
    df = pd.read_csv(path, header=None)  # Some files have a description in the first row
    # Find the header row: the row containing any industry column name
    header_row = None
    for i in range(min(5, len(df))):
        row = _clean(df.iloc[i].tolist())
        if any(x.lower() in [*map(str.lower, row)] for x in ['Cnsmr','Manuf','HiTec','Hlth','Other']):
            header_row = i; break
    df = pd.read_csv(path, header=header_row)
    df.columns = _clean(df.columns)
    date = df.iloc[:,0].astype(str).str.strip()
    df = df.iloc[:,1:].copy(); df.columns = _clean(df.columns)
    ren = {}
    for want, aliases in {
        'Cnsmr':['Cnsmr','Cnsum','Consumer'],
        'Manuf':['Manuf','Manufacturing','Manu'],
        'HiTec':['HiTec','HiTech','Hitec','HighTech','Hi-Tec'],
        'Hlth' :['Hlth','Health','Healthcare'],
        'Other':['Other','Others','Othr']
    }.items():
        f = _find(df.columns, aliases)
        if f and f != want: ren[f] = want
    if ren: df.rename(columns=ren, inplace=True)
    keep = [c for c in ['Cnsmr','Manuf','HiTec','Hlth','Other'] if c in df.columns]
    for c in keep: df[c] = pd.to_numeric(df[c], errors='coerce')
    df.insert(0,'date', date.values[:len(df)])
    return df.dropna().set_index('date')[keep]
def read_ff5(path):
    df = pd.read_csv(path); df.columns=_clean(df.columns)
    date = df.iloc[:,0].astype(str).str.strip()
    df = df.iloc[:,1:]; df.columns=_clean(df.columns)
    need = {'Mkt-RF':['Mkt-RF','MKT-RF','Mkt_RF'],
            'SMB':['SMB'],'HML':['HML'],'RMW':['RMW'],'CMA':['CMA'],'RF':['RF','Rf']}
    colmap={}
    for std, als in need.items():
        f=_find(df.columns, als)
        if not f: raise ValueError(f"FF5 missing column {std}; available: {df.columns.tolist()}")
        colmap[f]=std
    df.rename(columns=colmap, inplace=True)
    for c in colmap.values(): df[c]=pd.to_numeric(df[c], errors='coerce')
    df.insert(0,'date',date.values[:len(df)])
    return df.dropna().set_index('date')[['Mkt-RF','SMB','HML','RMW','CMA','RF']]
def read_mom(path):
    df = pd.read_csv(path); df.columns=_clean(df.columns)
    date = df.iloc[:,0].astype(str).str.strip()
    df = df.iloc[:,1:]; df.columns=_clean(df.columns)
    mom = _find(df.columns, ['Mom','MOM','Momentum'])
    if not mom: raise ValueError(f"Did not find Mom column; available: {df.columns.tolist()}")
    df.rename(columns={mom:'Mom'}, inplace=True)
    df['Mom']=pd.to_numeric(df['Mom'], errors='coerce')
    df.insert(0,'date',date.values[:len(df)])
    return df.dropna().set_index('date')[['Mom']]
# Read and Align
ind = read_industries(IND_PATH)
ff5 = read_ff5(FF5_PATH)
mom = read_mom(MOM_PATH)
factors = ff5.join(mom, how='inner')
y_df = ind[[TARGET_COL]].join(factors[['RF']], how='inner')
y_df['excess'] = y_df[TARGET_COL] - y_df['RF']     # Industry excess return
y = y_df[['excess']]
X = factors.loc[y.index, ['Mkt-RF','SMB','HML','RMW','CMA','Mom']]
# ------------------ Standardization + Serialization ------------------
sx, sy = StandardScaler(), StandardScaler()
X_s = sx.fit_transform(X.values)
y_s = sy.fit_transform(y.values)
def make_seq(Xa, ya, w):
    Xs, ys = [], []
    for i in range(len(Xa)-w):
        Xs.append(Xa[i:i+w]); ys.append(ya[i+w])
    return np.asarray(Xs), np.asarray(ys)
W = TIME_WINDOW
Xs, ys = make_seq(X_s, y_s, W)
split = int(len(Xs) * (1-TEST_RATIO))
X_tr, X_te = Xs[:split], Xs[split:]
y_tr, y_te = ys[:split], ys[split:]
# ------------------ LSTM ------------------
model = Sequential([
    LSTM(LSTM_UNITS, input_shape=(X_tr.shape[1], X_tr.shape[2])),
    Dropout(DROPOUT),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
cb = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=0.1, callbacks=cb, verbose=1)
# Inverse Standardize Predictions
ytr_pred = sy.inverse_transform(model.predict(X_tr))
yte_pred = sy.inverse_transform(model.predict(X_te))
ytr_true = sy.inverse_transform(y_tr)
yte_true = sy.inverse_transform(y_te)
# Spearman
rho_te, p_te = spearmanr(yte_true.ravel(), yte_pred.ravel())
rho_tr, p_tr = spearmanr(ytr_true.ravel(), ytr_pred.ravel())
# ------------------ Two Scatter Plots ------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(yte_true, yte_pred, s=18, alpha=0.85)
axes[0].set_title("testing experimental hi-tech vs predicted")
axes[0].set_xlabel("experimental hi-tech"); axes[0].set_ylabel("predicted hi-tech")
axes[0].text(0.02, 0.98,
             f"correlation: {rho_te:.5f}
"
             f"p-value: {p_te:.3e}
"
             f"at alpha-significant level: 0.05",
             transform=axes[0].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
axes[1].scatter(ytr_true, ytr_pred, s=18, alpha=0.85)
axes[1].set_title("training experimental hi-tech vs predicted")
axes[1].set_xlabel("experimental hi-tech"); axes[1].set_ylabel("predicted hi-tech")
axes[1].text(0.02, 0.98,
             f"correlation: {rho_tr:.5f}
"
             f"p-value: {p_tr:.3e}
"
             f"at alpha-significant level: 0.05",
             transform=axes[1].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
plt.tight_layout(); plt.show()
print("
Aligned samples:", len(X))
print("Train/Test sequences:", len(X_tr), "/", len(X_te))
print(f"Spearman rho (test/train): {rho_te:.3f} / {rho_tr:.3f}")
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:30:01 2025
@author: support huawei
"""
# -*- coding: utf-8 -*-
"""
LSTM (FF-6 = FF5 + Momentum) → Health excess return
Outputs two scatter plots like the sample:
(a) testing experimental health vs predicted
(b) training experimental health vs predicted
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# =============== File Paths (replace with your actual paths) ===============
IND_PATH = r"C:\Users\support huawei\OneDrive\桌面\5_Industry_Portfolios_daily_CSV\5_Industry_Portfolios(Daily).csv"
FF5_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Research_Data_5_Factors_2x3_daily_CSV\F-F_Research_Data_5_Factors_2x3(daily).csv"
MOM_PATH = r"C:\Users\support huawei\OneDrive\桌面\F-F_Momentum_Factor_daily_CSV\F-F_Momentum_Factor(daily).csv"
TARGET_COL = "Hlth"        # ✅ Switched to Health Industry
TIME_WINDOW = 20
TEST_RATIO  = 0.2
EPOCHS      = 60
BATCH_SIZE  = 256
LSTM_UNITS  = 96
DROPOUT     = 0.2
LR          = 1e-3
SEED        = 42
tf.keras.utils.set_random_seed(SEED)
# ------------------ Utility Functions ------------------
def _clean(cols): return [str(c).strip().replace('\xa0',' ') for c in cols]
def _find(cols, aliases):
    low = {c.lower():c for c in cols}
    for a in aliases:
        if a.lower() in low: return low[a.lower()]
    return None
def read_industries(path):
    df = pd.read_csv(path, header=None)
    header_row = None
    for i in range(min(5, len(df))):
        row = _clean(df.iloc[i].tolist())
        if any(x.lower() in [*map(str.lower, row)] for x in ['Cnsmr','Manuf','HiTec','Hlth','Other']):
            header_row = i; break
    df = pd.read_csv(path, header=header_row)
    df.columns = _clean(df.columns)
    date = df.iloc[:,0].astype(str).str.strip()
    df = df.iloc[:,1:].copy(); df.columns = _clean(df.columns)
    ren = {}
    for want, aliases in {
        'Cnsmr':['Cnsmr','Cnsum','Consumer'],
        'Manuf':['Manuf','Manufacturing'],
        'HiTec':['HiTec','HiTech','Hitec'],
        'Hlth':['Hlth','Health','Healthcare'],
        'Other':['Other','Others']
    }.items():
        f = _find(df.columns, aliases)
        if f and f != want: ren[f] = want
    if ren: df.rename(columns=ren, inplace=True)
    keep = [c for c in ['Cnsmr','Manuf','HiTec','Hlth','Other'] if c in df.columns]
    for c in keep: df[c] = pd.to_numeric(df[c], errors='coerce')
    df.insert(0,'date', date.values[:len(df)])
    return df.dropna().set_index('date')[keep]
def read_ff5(path):
    df = pd.read_csv(path); df.columns=_clean(df.columns)
    date = df.iloc[:,0].astype(str).str.strip()
    df = df.iloc[:,1:]; df.columns=_clean(df.columns)
    need = {'Mkt-RF':['Mkt-RF','MKT-RF'],
            'SMB':['SMB'],'HML':['HML'],'RMW':['RMW'],'CMA':['CMA'],'RF':['RF','Rf']}
    colmap={}
    for std, als in need.items():
        f=_find(df.columns, als)
        if not f: raise ValueError(f"FF5 missing column {std}; available: {df.columns.tolist()}")
        colmap[f]=std
    df.rename(columns=colmap, inplace=True)
    for c in colmap.values(): df[c]=pd.to_numeric(df[c], errors='coerce')
    df.insert(0,'date',date.values[:len(df)])
    return df.dropna().set_index('date')[['Mkt-RF','SMB','HML','RMW','CMA','RF']]
def read_mom(path):
    df = pd.read_csv(path); df.columns=_clean(df.columns)
    date = df.iloc[:,0].astype(str).str.strip()
    df = df.iloc[:,1:]; df.columns=_clean(df.columns)
    mom = _find(df.columns, ['Mom','MOM','Momentum'])
    if not mom: raise ValueError(f"Did not find Mom column; available: {df.columns.tolist()}")
    df.rename(columns={mom:'Mom'}, inplace=True)
    df['Mom']=pd.to_numeric(df['Mom'], errors='coerce')
    df.insert(0,'date',date.values[:len(df)])
    return df.dropna().set_index('date')[['Mom']]
# ------------------ Read and Align ------------------
ind = read_industries(IND_PATH)
ff5 = read_ff5(FF5_PATH)
mom = read_mom(MOM_PATH)
factors = ff5.join(mom, how='inner')
y_df = ind[[TARGET_COL]].join(factors[['RF']], how='inner')
y_df['excess'] = y_df[TARGET_COL] - y_df['RF']
y = y_df[['excess']]
X = factors.loc[y.index, ['Mkt-RF','SMB','HML','RMW','CMA','Mom']]
# ------------------ Standardization + Serialization ------------------
sx, sy = StandardScaler(), StandardScaler()
X_s = sx.fit_transform(X.values)
y_s = sy.fit_transform(y.values)
def make_seq(Xa, ya, w):
    Xs, ys = [], []
    for i in range(len(Xa)-w):
        Xs.append(Xa[i:i+w]); ys.append(ya[i+w])
    return np.asarray(Xs), np.asarray(ys)
W = TIME_WINDOW
Xs, ys = make_seq(X_s, y_s, W)
split = int(len(Xs) * (1-TEST_RATIO))
X_tr, X_te = Xs[:split], Xs[split:]
y_tr, y_te = ys[:split], ys[split:]
# ------------------ LSTM ------------------
model = Sequential([
    LSTM(LSTM_UNITS, input_shape=(X_tr.shape[1], X_tr.shape[2])),
    Dropout(DROPOUT),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
cb = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=0.1, callbacks=cb, verbose=1)
# ------------------ Prediction and Inverse Standardization ------------------
ytr_pred = sy.inverse_transform(model.predict(X_tr))
yte_pred = sy.inverse_transform(model.predict(X_te))
ytr_true = sy.inverse_transform(y_tr)
yte_true = sy.inverse_transform(y_te)
# Spearman
rho_te, p_te = spearmanr(yte_true.ravel(), yte_pred.ravel())
rho_tr, p_tr = spearmanr(ytr_true.ravel(), ytr_pred.ravel())
# ------------------ Plotting ------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(yte_true, yte_pred, s=18, alpha=0.85)
axes[0].set_title("testing experimental health vs predicted")
axes[0].set_xlabel("experimental health"); axes[0].set_ylabel("predicted health")
axes[0].text(0.02, 0.98,
             f"correlation: {rho_te:.5f}
"
             f"p-value: {p_te:.3e}
"
             f"at alpha-significant level: 0.05",
             transform=axes[0].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
axes[1].scatter(ytr_true, ytr_pred, s=18, alpha=0.85)
axes[1].set_title("training experimental health vs predicted")
axes[1].set_xlabel("experimental health"); axes[1].set_ylabel("predicted health")
axes[1].text(0.02, 0.98,
             f"correlation: {rho_tr:.5f}
"
             f"p-value: {p_tr:.3e}
"
             f"at alpha-significant level: 0.05",
             transform=axes[1].transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
plt.tight_layout(); plt.show()
print("
Aligned samples:", len(X))
print("Train/Test sequences:", len(X_tr), "/", len(X_te))
print(f"Spearman rho (test/train): {rho_te:.3f} / {rho_tr:.3f}")