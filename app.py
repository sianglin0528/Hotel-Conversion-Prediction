# app.py — Minimal, Clean, Two-Model UI
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ----------------- 基本設定 -----------------
st.set_page_config(page_title="Hotel Booking Propensity", layout="centered")
st.title("Hotel Booking Propensity — Minimal UI")

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH  = ROOT / "test.csv"
OUT_DIR    = ROOT / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

TARGET = "Revenue"
ID_COL = "ID"

# ----------------- Sidebar 選項 -----------------
st.sidebar.header("設定")
model_name = st.sidebar.selectbox(
    "選擇模型",
    ["LogisticRegression", "RandomForest"],
    index=0
)
test_size = st.sidebar.slider("驗證集比例", 0.1, 0.3, 0.2, 0.05)
random_state = 42

# ----------------- 載入資料 -----------------
@st.cache_data
def load_raw():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test

train, test = load_raw()
st.subheader("資料概覽")
c1, c2 = st.columns(2)
with c1:
    st.write("train 形狀：", train.shape)
    st.dataframe(train.head(5))
with c2:
    st.write("test 形狀：", test.shape)
    st.dataframe(test.head(5))

# ----------------- 缺失值統計 -----------------
def missing_table(df: pd.DataFrame):
    mc = df.isnull().sum()
    mp = (mc / len(df) * 100).round(3)
    tbl = pd.DataFrame({"缺失數量": mc, "缺失比例(%)": mp})
    return tbl[tbl["缺失數量"] > 0].sort_values("缺失數量", ascending=False)

with st.expander("缺失值統計（train / test）", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(missing_table(train))
    with c2:
        st.dataframe(missing_table(test))

# ----------------- 清理邏輯 -----------------
def clean_data(train: pd.DataFrame, test: pd.DataFrame):
    # train：缺失列少 → 直接丟
    train_clean = train.dropna().copy()

    # test：補值
    test_clean = test.copy()
    num_cols_test = test_clean.select_dtypes(include=["number"]).columns.tolist()
    obj_cols_test = [c for c in test_clean.columns if c not in num_cols_test]
    for c in num_cols_test:
        if test_clean[c].isnull().any():
            test_clean[c] = test_clean[c].fillna(test_clean[c].median())
    for c in obj_cols_test:
        if test_clean[c].isnull().any():
            mode = test_clean[c].mode(dropna=True)
            test_clean[c] = test_clean[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # 保留原始 ID
    test_clean["OriginalID"] = test_clean["ID"]
    return train_clean, test_clean

train_clean, test_clean = clean_data(train, test)

st.subheader("清洗後預覽")
c1, c2 = st.columns(2)
with c1:
    st.caption(f"train_clean（{train.shape} → {train_clean.shape}）")
    st.dataframe(train_clean.head(5))
with c2:
    st.caption("test_clean（保留 OriginalID）")
    st.dataframe(test_clean[["ID", "OriginalID"]].head(5))

# ----------------- 特徵工程 -----------------
feature_cols = [c for c in train_clean.columns if c not in [ID_COL, TARGET]]
X = train_clean[feature_cols].copy()
y = train_clean[TARGET].copy()

# y to 0/1（更保險）
y = y.astype(str).str.strip().str.lower().map(
    {"true":1,"yes":1,"1":1,"false":0,"no":0,"0":0}
)
assert y.notna().all(), "Revenue 轉換 0/1 失敗：發現未知標籤"
y = y.astype(int)

X_test = test_clean[feature_cols].copy()
test_ids_original = test_clean["OriginalID"].copy()

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", make_ohe(), cat_cols),
    ]
)

# ----------------- 切驗證集 -----------------
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# ----------------- 建模 -----------------
if model_name == "LogisticRegression":
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])
else:
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ))
    ])

pipe.fit(X_tr, y_tr)
va_pred = pipe.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, va_pred)

st.subheader("驗證結果")
st.metric("ROC-AUC", f"{auc:.4f}")
st.caption("classification report（threshold=0.5）")
st.text(classification_report(y_va, (va_pred >= 0.5).astype(int)))

# ----------------- 產出 submission -----------------
st.subheader("產出 Submission")
if st.button("一鍵產生 submission.csv"):
    with st.spinner("訓練全資料、產出預測中…"):
        pipe.fit(X, y)
        prob_test = pipe.predict_proba(X_test)[:, 1]
        sub = pd.DataFrame({
            "ID": test_ids_original.values,   # 使用原始 ID
            "Revenue": np.round(prob_test, 2)
        })
        out_path = OUT_DIR / f"submission_{'lr' if model_name=='LogisticRegression' else 'rf'}.csv"
        sub.to_csv(out_path, index=False)
    st.success(f"已輸出：{out_path}")
    st.dataframe(sub.head(10))
