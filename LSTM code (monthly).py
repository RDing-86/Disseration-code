# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 00:41:16 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 17:48:44 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM（FF3）→ 4联图 + 指标（兼容老版本 scikit-learn：手动计算RMSE）
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models

# ========= 改这里 =========
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
INDUSTRY      = "Cnsmr"   # 'Cnsmr','Manuf','HiTec','Hlth','Other'
SEQ_LEN       = 12        # 窗口长度（月）
TEST_SIZE     = 0.30      # 测试集占比（按时间末尾切）
EPOCHS        = 300
BATCH_SIZE    = 32
PATIENCE      = 20
TITLE_MAP     = {"Cnsmr":"Consumption","Manuf":"Manufacturing",
                 "HiTec":"Hi-Tech","Hlth":"Health","Other":"Other"}
OUT_PNG       = f"lstm_ff3_{INDUSTRY.lower()}_4panel.png"

np.random.seed(42)
tf.random.set_seed(42)

# ========= 工具函数 =========
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML","Mom"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df = df.drop(columns=[first])
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df = df.drop(columns=[first]).sort_values("Date").reset_index(drop=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df

def make_sequences(X, y, L):
    Xs, Ys = [], []
    for i in range(L, len(X)):
        Xs.append(X[i-L:i])
        Ys.append(y[i])
    return np.asarray(Xs), np.asarray(Ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ========= 读数 & 对齐 =========
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益：行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns:
        df[c] = df[c].astype(float) - df["RF"].astype(float)

# FF3 特征与目标
X_raw = df[["Mkt-RF","SMB","HML"]].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 & 仅用训练统计量做标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]

# ========= 小型 LSTM =========
def build_model(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
hist = model.fit(
    X_tr, y_tr, validation_split=0.2,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[es], verbose=0
)

# 预测 & 指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()

rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF3-LSTM] {TITLE_MAP.get(INDUSTRY, INDUSTRY)} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# ========= 画四联图 =========
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
title_top = f"3-Factor (LSTM) — {TITLE_MAP.get(INDUSTRY, INDUSTRY)}"
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(title_top, y=0.98, fontsize=13)

# (a) 测试散点
ax = axes[0,0]
ax.scatter(y_te, yhat_te, s=22)
ax.set_xlabel("experimental " + INDUSTRY.lower())
ax.set_ylabel("predicted "   + INDUSTRY.lower())
ax.set_title("(a) Testing")
ax.text(0.03, 0.97, f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) 训练散点
ax = axes[0,1]
ax.scatter(y_tr, yhat_tr, s=18)
ax.set_xlabel("experimental " + INDUSTRY.lower())
ax.set_ylabel("predicted "   + INDUSTRY.lower())
ax.set_title("(b) Training")
ax.text(0.03, 0.97, f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (c) 测试时序
ax = axes[1,0]
ax.plot(range(len(y_te)), y_te,  marker='o', ms=2.5, lw=0.9, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_te)), yhat_te,marker='o', ms=2.5, lw=0.9, label="predicted "+INDUSTRY.lower())
ax.set_title("(c) Testing"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

# (d) 训练时序
ax = axes[1,1]
ax.plot(range(len(y_tr)), y_tr,  marker='o', ms=2.2, lw=0.8, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_tr)), yhat_tr,marker='o', ms=2.2, lw=0.8, label="predicted "+INDUSTRY.lower())
ax.set_title("(d) Training"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT_PNG, dpi=300)
plt.show()
print("图已保存：", os.path.abspath(OUT_PNG))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 00:12:24 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF3) → 4联图 + 指标（制造业 Manuf；兼容老环境，手动算RMSE）
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models

# ========= 修改这里 =========
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
INDUSTRY      = "Manuf"   # 'Cnsmr','Manuf','HiTec','Hlth','Other'
SEQ_LEN       = 12        # 用过去12个月预测下个月
TEST_SIZE     = 0.30      # 时间末尾30%做测试
EPOCHS        = 300
BATCH_SIZE    = 32
PATIENCE      = 20
TITLE_MAP     = {"Cnsmr":"Consumption","Manuf":"Manufacturing",
                 "HiTec":"Hi-Tech","Hlth":"Health","Other":"Other"}
OUT_PNG       = f"lstm_ff3_{INDUSTRY.lower()}_4panel.png"

np.random.seed(42)
tf.random.set_seed(42)

# ========= 工具 =========
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML","Mom"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i])
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ========= 读取 & 对齐 =========
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益：行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns: df[c] = df[c].astype(float) - df["RF"].astype(float)

# FF3 特征 & 目标
X_raw = df[["Mkt-RF","SMB","HML"]].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分+仅用训练统计量标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]

# ========= LSTM =========
def build_model(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                             restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2,
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()
rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF3-LSTM] {TITLE_MAP.get(INDUSTRY, INDUSTRY)} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# ========= 画四联图（与示例风格一致） =========
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(f"3-Factor (LSTM) — {TITLE_MAP.get(INDUSTRY, INDUSTRY)}", y=0.98, fontsize=13)

# (a) 测试散点
ax = axes[0,0]
ax.scatter(y_te, yhat_te, s=22)
ax.set_xlabel("experimental " + INDUSTRY.lower()); ax.set_ylabel("predicted " + INDUSTRY.lower())
ax.set_title("(a) Testing")
ax.text(0.03, 0.97, f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) 训练散点
ax = axes[0,1]
ax.scatter(y_tr, yhat_tr, s=18)
ax.set_xlabel("experimental " + INDUSTRY.lower()); ax.set_ylabel("predicted " + INDUSTRY.lower())
ax.set_title("(b) Training")
ax.text(0.03, 0.97, f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (c) 测试时序
ax = axes[1,0]
ax.plot(range(len(y_te)), y_te,  marker='o', ms=2.5, lw=0.9, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_te)), yhat_te, marker='o', ms=2.5, lw=0.9, label="predicted "+INDUSTRY.lower())
ax.set_title("(c) Testing"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

# (d) 训练时序
ax = axes[1,1]
ax.plot(range(len(y_tr)), y_tr,  marker='o', ms=2.2, lw=0.8, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_tr)), yhat_tr, marker='o', ms=2.2, lw=0.8, label="predicted "+INDUSTRY.lower())
ax.set_title("(d) Training"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("图已保存：", os.path.abspath(OUT_PNG))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 00:23:48 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF3) for HiTec — 4-panel figure + metrics + predictions CSV
兼容老环境：手动计算 RMSE（不依赖 sklearn 的 squared=False）
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models

# ============== 配置 ==============
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
INDUSTRY      = "HiTec"     # 可改为: 'Cnsmr','Manuf','HiTec','Hlth','Other'
SEQ_LEN       = 12          # 用过去 12 个月预测下一月
TEST_SIZE     = 0.30        # 按时间末尾切分测试集比例
EPOCHS        = 300
BATCH_SIZE    = 32
PATIENCE      = 20          # EarlyStopping
TITLE_MAP     = {"Cnsmr":"Consumption","Manuf":"Manufacturing",
                 "HiTec":"Hi-Tech","Hlth":"Health","Other":"Other"}

OUT_PNG = f"lstm_ff3_{INDUSTRY.lower()}_4panel.png"
OUT_CSV = f"lstm_ff3_{INDUSTRY.lower()}_predictions.csv"

np.random.seed(42)
tf.random.set_seed(42)

# ============== 工具函数 ==============
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML","Mom"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    # 统一列名
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i])
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ============== 读数与对齐 ==============
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益：行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns: df[c] = df[c].astype(float) - df["RF"].astype(float)

# FF3 特征与目标（HiTec）
X_raw = df[["Mkt-RF","SMB","HML"]].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 + 仅用训练统计量标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
dates = df["Date"].iloc[SEQ_LEN:].reset_index(drop=True)  # 与序列后的 y 对齐

adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]
dates_tr, dates_te = dates[:adj_split], dates[adj_split:]

# ============== LSTM 小模型 ==============
def build_model(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                             restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2,
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()

rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF3-LSTM] {TITLE_MAP.get(INDUSTRY, INDUSTRY)} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# 保存预测CSV
pred_df = pd.DataFrame({
    "Date": pd.concat([dates_tr, dates_te], axis=0).values,
    "Set":  ["Train"]*len(y_tr) + ["Test"]*len(y_te),
    "Actual": np.concatenate([y_tr, y_te]),
    "Pred":   np.concatenate([yhat_tr, yhat_te]),
})
pred_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("预测CSV已保存：", os.path.abspath(OUT_CSV))

# ============== 四联图（与论文风格一致） ==============
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(f"3-Factor (LSTM) — {TITLE_MAP.get(INDUSTRY, INDUSTRY)}", y=0.98, fontsize=13)

# (a) 测试散点
ax = axes[0,0]
ax.scatter(y_te, yhat_te, s=22)
ax.set_xlabel("experimental " + INDUSTRY.lower()); ax.set_ylabel("predicted " + INDUSTRY.lower())
ax.set_title("(a) Testing")
ax.text(0.03, 0.97, f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) 训练散点
ax = axes[0,1]
ax.scatter(y_tr, yhat_tr, s=18)
ax.set_xlabel("experimental " + INDUSTRY.lower()); ax.set_ylabel("predicted " + INDUSTRY.lower())
ax.set_title("(b) Training")
ax.text(0.03, 0.97, f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (c) 测试时序
ax = axes[1,0]
ax.plot(range(len(y_te)), y_te,  marker='o', ms=2.5, lw=0.9, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_te)), yhat_te, marker='o', ms=2.5, lw=0.9, label="predicted "+INDUSTRY.lower())
ax.set_title("(c) Testing"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

# (d) 训练时序
ax = axes[1,1]
ax.plot(range(len(y_tr)), y_tr,  marker='o', ms=2.2, lw=0.8, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_tr)), yhat_tr, marker='o', ms=2.2, lw=0.8, label="predicted "+INDUSTRY.lower())
ax.set_title("(d) Training"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("四联图已保存：", os.path.abspath(OUT_PNG))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 00:23:48 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF3) for HiTec — 4-panel figure + metrics + predictions CSV
兼容老环境：手动计算 RMSE（不依赖 sklearn 的 squared=False）
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models

# ============== 配置 ==============
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
INDUSTRY      = "HiTec"     # 可改为: 'Cnsmr','Manuf','HiTec','Hlth','Other'
SEQ_LEN       = 12          # 用过去 12 个月预测下一月
TEST_SIZE     = 0.30        # 按时间末尾切分测试集比例
EPOCHS        = 300
BATCH_SIZE    = 32
PATIENCE      = 20          # EarlyStopping
TITLE_MAP     = {"Cnsmr":"Consumption","Manuf":"Manufacturing",
                 "HiTec":"Hi-Tech","Hlth":"Health","Other":"Other"}

OUT_PNG = f"lstm_ff3_{INDUSTRY.lower()}_4panel.png"
OUT_CSV = f"lstm_ff3_{INDUSTRY.lower()}_predictions.csv"

np.random.seed(42)
tf.random.set_seed(42)

# ============== 工具函数 ==============
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","RF","WML","Mom"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    # 统一列名
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i])
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ============== 读数与对齐 ==============
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益：行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns: df[c] = df[c].astype(float) - df["RF"].astype(float)

# FF3 特征与目标（HiTec）
X_raw = df[["Mkt-RF","SMB","HML"]].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 + 仅用训练统计量标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
dates = df["Date"].iloc[SEQ_LEN:].reset_index(drop=True)  # 与序列后的 y 对齐

adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]
dates_tr, dates_te = dates[:adj_split], dates[adj_split:]

# ============== LSTM 小模型 ==============
def build_model(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                             restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2,
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()

rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF3-LSTM] {TITLE_MAP.get(INDUSTRY, INDUSTRY)} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# 保存预测CSV
pred_df = pd.DataFrame({
    "Date": pd.concat([dates_tr, dates_te], axis=0).values,
    "Set":  ["Train"]*len(y_tr) + ["Test"]*len(y_te),
    "Actual": np.concatenate([y_tr, y_te]),
    "Pred":   np.concatenate([yhat_tr, yhat_te]),
})
pred_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("预测CSV已保存：", os.path.abspath(OUT_CSV))

# ============== 四联图（与论文风格一致） ==============
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(f"3-Factor (LSTM) — {TITLE_MAP.get(INDUSTRY, INDUSTRY)}", y=0.98, fontsize=13)

# (a) 测试散点
ax = axes[0,0]
ax.scatter(y_te, yhat_te, s=22)
ax.set_xlabel("experimental " + INDUSTRY.lower()); ax.set_ylabel("predicted " + INDUSTRY.lower())
ax.set_title("(a) Testing")
ax.text(0.03, 0.97, f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) 训练散点
ax = axes[0,1]
ax.scatter(y_tr, yhat_tr, s=18)
ax.set_xlabel("experimental " + INDUSTRY.lower()); ax.set_ylabel("predicted " + INDUSTRY.lower())
ax.set_title("(b) Training")
ax.text(0.03, 0.97, f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (c) 测试时序
ax = axes[1,0]
ax.plot(range(len(y_te)), y_te,  marker='o', ms=2.5, lw=0.9, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_te)), yhat_te, marker='o', ms=2.5, lw=0.9, label="predicted "+INDUSTRY.lower())
ax.set_title("(c) Testing"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

# (d) 训练时序
ax = axes[1,1]
ax.plot(range(len(y_tr)), y_tr,  marker='o', ms=2.2, lw=0.8, label="experimental "+INDUSTRY.lower())
ax.plot(range(len(y_tr)), yhat_tr, marker='o', ms=2.2, lw=0.8, label="predicted "+INDUSTRY.lower())
ax.set_title("(d) Training"); ax.set_xlabel(INDUSTRY.lower()+" points"); ax.set_ylabel(INDUSTRY.lower())
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("四联图已保存：", os.path.abspath(OUT_PNG))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 00:43:47 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF3) — Other | 4-panel figure + metrics + predictions CSV
兼容老环境：手动计算 RMSE（不依赖 sklearn 的 squared=False）
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models

# ====== 改这里 ======
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"
INDUSTRY      = "Other"     # 这次要画“Other”
SEQ_LEN       = 12          # 用过去12个月预测下一月
TEST_SIZE     = 0.30        # 时间末尾30%做测试
EPOCHS, BATCH_SIZE, PATIENCE = 300, 32, 20
TITLE_MAP = {"Cnsmr":"Consumption","Manuf":"Manufacturing",
             "HiTec":"Hi-Tech","Hlth":"Health","Other":"Other"}
OUT_PNG = f"lstm_ff3_{INDUSTRY.lower()}_4panel.png"
OUT_CSV = f"lstm_ff3_{INDUSTRY.lower()}_predictions.csv"

np.random.seed(42); tf.random.set_seed(42)

# ====== 工具 ======
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RF","RMW","CMA","WML","Mom"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ====== 读取 & 对齐 ======
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益：行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns: df[c] = df[c].astype(float) - df["RF"].astype(float)

# FF3 特征与目标（Other）
X_raw = df[["Mkt-RF","SMB","HML"]].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 + 仅用训练统计量标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
dates = df["Date"].iloc[SEQ_LEN:].reset_index(drop=True)

adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]
dates_tr, dates_te = dates[:adj_split], dates[adj_split:]

# ====== LSTM ======
def build_model(shape):
    m = tf.keras.Sequential([
        layers.Input(shape=shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()
rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF3-LSTM] {TITLE_MAP['Other']} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# 保存预测CSV
pred_df = pd.DataFrame({
    "Date": pd.concat([dates_tr, dates_te]).values,
    "Set":  ["Train"]*len(y_tr) + ["Test"]*len(y_te),
    "Actual": np.concatenate([y_tr, y_te]),
    "Pred":   np.concatenate([yhat_tr, yhat_te]),
})
pred_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("预测CSV已保存：", os.path.abspath(OUT_CSV))

# ====== 四联图（与论文风格一致） ======
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(f"3-Factor (LSTM) — {TITLE_MAP['Other']}", y=0.98, fontsize=13)

# (a) 测试散点
ax = axes[0,0]
ax.scatter(y_te, yhat_te, s=22)
ax.set_xlabel("experimental otherrf"); ax.set_ylabel("predicted otherrf")
ax.set_title("(a) Testing")
ax.text(0.03, 0.97, f"RHO-value: {rho_te:.5f}\nPVAL-value: {p_te:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) 训练散点
ax = axes[0,1]
ax.scatter(y_tr, yhat_tr, s=18)
ax.set_xlabel("experimental otherrf"); ax.set_ylabel("predicted otherrf")
ax.set_title("(b) Training")
ax.text(0.03, 0.97, f"RHO-value: {rho_tr:.5f}\nPVAL-value: {p_tr:.2e}\nat ALPH-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (c) 测试时序
ax = axes[1,0]
ax.plot(range(len(y_te)), y_te,  marker='o', ms=2.5, lw=0.9, label="experimental otherrf")
ax.plot(range(len(y_te)), yhat_te, marker='o', ms=2.5, lw=0.9, label="predicted otherrf")
ax.set_title("(c) Testing"); ax.set_xlabel("otherrf points"); ax.set_ylabel("otherrf")
ax.legend(loc="upper right", fontsize=8)

# (d) 训练时序
ax = axes[1,1]
ax.plot(range(len(y_tr)), y_tr,  marker='o', ms=2.2, lw=0.8, label="experimental otherrf")
ax.plot(range(len(y_tr)), yhat_tr, marker='o', ms=2.2, lw=0.8, label="predicted otherrf")
ax.set_title("(d) Training"); ax.set_xlabel("otherrf points"); ax.set_ylabel("otherrf")
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout(rect=[0,0,1,0.96])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("四联图已保存：", os.path.abspath(OUT_PNG))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 01:06:18 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF6) — Two-panel scatter only (Testing & Training)
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models

# ====== 路径与参数 ======
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"

INDUSTRY  = "Cnsmr"     # 行业：'Cnsmr','Manuf','HiTec','Hlth','Other'
SEQ_LEN   = 12          # 用过去12个月预测下一月
TEST_SIZE = 0.30        # 时间末尾 30% 作为测试集

EPOCHS, BATCH_SIZE, PATIENCE = 300, 32, 20
TITLE = {"Cnsmr":"Consumption","Manuf":"Manufacturing","HiTec":"Hi-Tech",
         "Hlth":"Health","Other":"Other"}[INDUSTRY]
AXLAB = {"Cnsmr":"cons.","Manuf":"manufacturing","HiTec":"hi-tech",
         "Hlth":"health","Other":"others"}[INDUSTRY]
OUT_PNG = f"lstm_ff6_{INDUSTRY.lower()}_2panel.png"

np.random.seed(42); tf.random.set_seed(42)

# ====== 工具 ======
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","RF"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    # 统一列名
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))  # 手算 RMSE，兼容老版本 sklearn
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ====== 读取与对齐 ======
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益：行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns: df[c] = df[c].astype(float) - df["RF"].astype(float)

# FF6 特征与目标
ff6_cols = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
X_raw = df[ff6_cols].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 + 仅用训练统计量做标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
dates = df["Date"].iloc[SEQ_LEN:].reset_index(drop=True)
adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]

# ====== LSTM ======
def build_model(input_shape):
    m = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()
rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF6-LSTM] {TITLE} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# ====== 两联图（只画散点） ======
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
fig.suptitle(f"6-Factor (LSTM) — {TITLE}", y=0.98, fontsize=13)

# (a) Testing
ax = axes[0]
ax.scatter(y_te, yhat_te, s=24)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(a) Testing")
ax.text(0.03, 0.97,
        f"correlation: {rho_te:.5f}\np-value: {p_te:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) Training
ax = axes[1]
ax.scatter(y_tr, yhat_tr, s=20)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(b) Training")
ax.text(0.03, 0.97,
        f"correlation: {rho_tr:.5f}\np-value: {p_tr:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

fig.tight_layout(rect=[0,0,1,0.94])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("两联图已保存：", os.path.abspath(OUT_PNG))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:52:28 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF6) — Manufacturing (Manuf) | Two-panel scatter (Testing & Training)
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models

# ========= 路径与参数 =========
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"

INDUSTRY  = "Manuf"   # 本脚本固定制造业两联图；可改为 Cnsmr/HiTec/Hlth/Other
SEQ_LEN   = 12        # 用过去12个月预测下一月
TEST_SIZE = 0.30      # 末尾30%做测试

EPOCHS, BATCH_SIZE, PATIENCE = 300, 32, 20
TITLE = "Manufacturing"
AXLAB = "manufacturing"

OUT_PNG = f"lstm_ff6_{INDUSTRY.lower()}_2panel.png"
OUT_CSV = f"lstm_ff6_{INDUSTRY.lower()}_predictions.csv"

np.random.seed(42); tf.random.set_seed(42)

# ========= 工具 =========
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","RF"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    df.rename(columns={
        "MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
        "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"
    }, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))   # 手算 RMSE
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ========= 读取 & 对齐 =========
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益：行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns:
        df[c] = df[c].astype(float) - df["RF"].astype(float)

# 六因子特征 + 目标（制造业）
ff6_cols = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
X_raw = df[ff6_cols].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 + 仅用训练统计量标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std  = standardize_train_only(X_raw, split)

# 序列化
SEQ_LEN = int(SEQ_LEN)
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
dates = df["Date"].iloc[SEQ_LEN:].reset_index(drop=True)
adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]

# ========= LSTM =========
def build_model(input_shape):
    m = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()
rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF6-LSTM] {TITLE} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# ========= 两联图（只画散点） =========
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
fig.suptitle(f"6-Factor (LSTM) — {TITLE}", y=0.98, fontsize=13)

# (a) Testing
ax = axes[0]
ax.scatter(y_te, yhat_te, s=24)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(a) Testing")
ax.text(0.03, 0.97,
        f"correlation: {rho_te:.5f}\np-value: {p_te:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) Training
ax = axes[1]
ax.scatter(y_tr, yhat_tr, s=20)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(b) Training")
ax.text(0.03, 0.97,
        f"correlation: {rho_tr:.5f}\np-value: {p_tr:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

fig.tight_layout(rect=[0,0,1,0.94])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("两联图已保存：", os.path.abspath(OUT_PNG))

# 同时保存预测 CSV（方便复现表格/统计）
pd.DataFrame({
    "Set": ["Train"]*len(y_tr) + ["Test"]*len(y_te),
    "Actual": np.concatenate([y_tr, y_te]),
    "Pred":   np.concatenate([yhat_tr, yhat_te]),
}).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("预测CSV已保存：", os.path.abspath(OUT_CSV))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:58:51 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF6) — Hi-Tech (HiTec) | Two-panel scatter (Testing & Training)
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

# ---------- 路径 ----------
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"

# ---------- 配置 ----------
INDUSTRY  = "HiTec"     # 这次就是高科技
TITLE     = "Hi-Tech"
AXLAB     = "hi-tech"
SEQ_LEN   = 12
TEST_SIZE = 0.30
EPOCHS, BATCH_SIZE, PATIENCE = 300, 32, 20
OUT_PNG = f"lstm_ff6_{INDUSTRY.lower()}_2panel.png"
OUT_CSV = f"lstm_ff6_{INDUSTRY.lower()}_predictions.csv"

np.random.seed(42); tf.random.set_seed(42)

# ---------- 小工具 ----------
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","RF"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))  # 手算 RMSE（兼容旧版 sklearn）
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ---------- 读取 & 对齐 ----------
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益 = 行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns:
        df[c] = df[c].astype(float) - df["RF"].astype(float)

# 六因子特征与目标（Hi-Tech）
ff6_cols = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
X_raw = df[ff6_cols].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 + 仅用训练统计量标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std  = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
dates = df["Date"].iloc[SEQ_LEN:].reset_index(drop=True)
adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]

# ---------- LSTM ----------
def build_model(input_shape):
    m = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2, epochs=300, batch_size=32,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()
rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF6-LSTM] {TITLE} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# ---------- 两联图（散点） ----------
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
fig.suptitle(f"6-Factor (LSTM) — {TITLE}", y=0.98, fontsize=13)

# (a) Testing
ax = axes[0]
ax.scatter(y_te, yhat_te, s=24)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(a) Testing")
ax.text(0.03, 0.97,
        f"correlation: {rho_te:.5f}\np-value: {p_te:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) Training
ax = axes[1]
ax.scatter(y_tr, yhat_tr, s=20)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(b) Training")
ax.text(0.03, 0.97,
        f"correlation: {rho_tr:.5f}\np-value: {p_tr:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

fig.tight_layout(rect=[0,0,1,0.94])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("两联图已保存：", os.path.abspath(OUT_PNG))

# 也保存预测 CSV（可复现实证表）
pd.DataFrame({
    "Set": ["Train"]*len(y_tr) + ["Test"]*len(y_te),
    "Actual": np.concatenate([y_tr, y_te]),
    "Pred":   np.concatenate([yhat_tr, yhat_te]),
}).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("预测CSV已保存：", os.path.abspath(OUT_CSV))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 11:17:47 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF6) — Health (Hlth) | Two-panel scatter (Testing & Training)
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

# ---------- 文件路径 ----------
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"

# ---------- 配置 ----------
INDUSTRY  = "Hlth"       # 本次：医疗 Health
TITLE     = "Health"
AXLAB     = "health"
SEQ_LEN   = 12           # 用过去12个月预测下一月
TEST_SIZE = 0.30         # 末尾30%做测试
EPOCHS, BATCH_SIZE, PATIENCE = 300, 32, 20

OUT_PNG = f"lstm_ff6_{INDUSTRY.lower()}_2panel.png"
OUT_CSV = f"lstm_ff6_{INDUSTRY.lower()}_predictions.csv"

np.random.seed(42); tf.random.set_seed(42)

# ---------- 工具 ----------
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","RF"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))  # 手算 RMSE（避免 sklearn 版本差异）
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ---------- 读取 & 对齐 ----------
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益 = 行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns:
        df[c] = df[c].astype(float) - df["RF"].astype(float)

# 六因子特征 + 目标（健康 Hlth）
ff6_cols = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
X_raw = df[ff6_cols].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 时间切分 + 仅用训练统计量做标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std  = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
dates = df["Date"].iloc[SEQ_LEN:].reset_index(drop=True)
adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]

# ---------- LSTM ----------
def build_model(input_shape):
    m = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2, epochs=300, batch_size=32,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()
rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF6-LSTM] {TITLE} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# ---------- 两联图（只画散点） ----------
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
fig.suptitle(f"6-Factor (LSTM) — {TITLE}", y=0.98, fontsize=13)

# (a) Testing
ax = axes[0]
ax.scatter(y_te, yhat_te, s=24)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(a) Testing")
ax.text(0.03, 0.97,
        f"correlation: {rho_te:.5f}\np-value: {p_te:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) Training
ax = axes[1]
ax.scatter(y_tr, yhat_tr, s=20)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(b) Training")
ax.text(0.03, 0.97,
        f"correlation: {rho_tr:.5f}\np-value: {p_tr:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

fig.tight_layout(rect=[0,0,1,0.94])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("两联图已保存：", os.path.abspath(OUT_PNG))

# 保存预测 CSV（便于复现表格/统计）
pd.DataFrame({
    "Set": ["Train"]*len(y_tr) + ["Test"]*len(y_te),
    "Actual": np.concatenate([y_tr, y_te]),
    "Pred":   np.concatenate([yhat_tr, yhat_te]),
}).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("预测CSV已保存：", os.path.abspath(OUT_CSV))

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 12:03:16 2025

@author: support huawei
"""

# -*- coding: utf-8 -*-
"""
LSTM (FF6) — Other | Two-panel scatter (Testing & Training)
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

# ---------- 数据路径 ----------
PATH_FACTORS  = r"D:\Europe_6_Factors_CSV\Europe_6_Factors(monthly).csv"
PATH_INDUSTRY = r"D:\5_Industry_Portfolios_CSV\5_Industry_Portfolios.csv"

# ---------- 配置（本次：Other 行业） ----------
INDUSTRY  = "Other"
TITLE     = "Other"
AXLAB     = "others"
SEQ_LEN   = 12
TEST_SIZE = 0.30
EPOCHS, BATCH_SIZE, PATIENCE = 300, 32, 20

OUT_PNG = f"lstm_ff6_{INDUSTRY.lower()}_2panel.png"
OUT_CSV = f"lstm_ff6_{INDUSTRY.lower()}_predictions.csv"

np.random.seed(42); tf.random.set_seed(42)

# ---------- 工具 ----------
def _find_header(path, tokens):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    pat = re.compile("|".join(map(re.escape, tokens)), re.I)
    for i, ln in enumerate(lines):
        if pat.search(ln): return i
    return 0

def read_factors(path):
    h  = _find_header(path, ["Mkt-RF","SMB","HML","RMW","CMA","WML","Mom","RF"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    df.rename(columns={"MKT_RF":"Mkt-RF","Mkt_RF":"Mkt-RF","MKTRF":"Mkt-RF",
                       "R_f":"RF","R_F":"RF","Mom":"WML","MOM":"WML"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def read_industry(path):
    h  = _find_header(path, ["Cnsmr","Manuf","HiTec","Hlth","Health","Other"])
    df = pd.read_csv(path, header=h)
    first = df.columns[0]
    df = df[df[first].astype(str).str.match(r"^\d{6}$", na=False)]
    df["Date"] = pd.to_datetime(df[first].astype(str), format="%Y%m")
    df.drop(columns=[first], inplace=True)
    if "Hlth" not in df.columns and "Health" in df.columns:
        df.rename(columns={"Health":"Hlth"}, inplace=True)
    df.replace(-99.99, np.nan, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def make_sequences(X, y, L):
    xs, ys = [], []
    for i in range(L, len(X)):
        xs.append(X[i-L:i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)

def standardize_train_only(X, split_idx):
    mu = X[:split_idx].mean(axis=0, keepdims=True)
    sd = X[:split_idx].std(axis=0, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def metrics_manual(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))   # 手算 RMSE（兼容不同 sklearn 版本）
    mae  = np.mean(np.abs(y - yhat))
    rho, p = spearmanr(y, yhat)
    return rmse, mae, rho, p

# ---------- 读取与对齐 ----------
fac = read_factors(PATH_FACTORS)
ind = read_industry(PATH_INDUSTRY)
df  = fac.merge(ind, on="Date", how="inner").dropna().reset_index(drop=True)

# 行业超额收益 = 行业 - RF
for c in ["Cnsmr","Manuf","HiTec","Hlth","Other"]:
    if c in df.columns:
        df[c] = df[c].astype(float) - df["RF"].astype(float)

# 六因子特征 + 目标（Other）
ff6_cols = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
X_raw = df[ff6_cols].to_numpy(float)
y_raw = df[INDUSTRY].to_numpy(float)

# 只用训练集统计量做标准化
split = int(round(len(df)*(1-TEST_SIZE)))
X_std  = standardize_train_only(X_raw, split)

# 序列化
X_seq, y_seq = make_sequences(X_std, y_raw, SEQ_LEN)
adj_split = split - SEQ_LEN
X_tr, X_te = X_seq[:adj_split], X_seq[adj_split:]
y_tr, y_te = y_seq[:adj_split], y_seq[adj_split:]

# ---------- LSTM ----------
def build_model(input_shape):
    m = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m

model = build_model(X_tr.shape[1:])
es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
model.fit(X_tr, y_tr, validation_split=0.2, epochs=300, batch_size=32,
          callbacks=[es], verbose=0)

# 预测与指标
yhat_tr = model.predict(X_tr, verbose=0).ravel()
yhat_te = model.predict(X_te, verbose=0).ravel()
rmse_tr, mae_tr, rho_tr, p_tr = metrics_manual(y_tr, yhat_tr)
rmse_te, mae_te, rho_te, p_te = metrics_manual(y_te, yhat_te)

print(f"[FF6-LSTM] {TITLE} | train={len(y_tr)} test={len(y_te)}")
print(f"Training: RMSE={rmse_tr:.3f}  MAE={mae_tr:.3f}  RHO={rho_tr:.3f} (p={p_tr:.2e})")
print(f"Testing : RMSE={rmse_te:.3f}  MAE={mae_te:.3f}  RHO={rho_te:.3f} (p={p_te:.2e})")

# ---------- 两联图（只画散点） ----------
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
fig.suptitle(f"6-Factor (LSTM) — {TITLE}", y=0.98, fontsize=13)

# (a) Testing
ax = axes[0]
ax.scatter(y_te, yhat_te, s=24)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(a) Testing")
ax.text(0.03, 0.97,
        f"correlation: {rho_te:.5f}\np-value: {p_te:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

# (b) Training
ax = axes[1]
ax.scatter(y_tr, yhat_tr, s=20)
ax.set_xlabel(f"experimental {AXLAB}")
ax.set_ylabel(f"predicted {AXLAB}")
ax.set_title("(b) Training")
ax.text(0.03, 0.97,
        f"correlation: {rho_tr:.5f}\np-value: {p_tr:.2e}\nat alpha-significant level: 0.05",
        transform=ax.transAxes, va="top")

fig.tight_layout(rect=[0,0,1,0.94])
plt.savefig(OUT_PNG, dpi=300); plt.show()
print("两联图已保存：", os.path.abspath(OUT_PNG))

# 保存预测 CSV
pd.DataFrame({
    "Set": ["Train"]*len(y_tr) + ["Test"]*len(y_te),
    "Actual": np.concatenate([y_tr, y_te]),
    "Pred":   np.concatenate([yhat_tr, yhat_te]),
}).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("预测CSV已保存：", os.path.abspath(OUT_CSV))
