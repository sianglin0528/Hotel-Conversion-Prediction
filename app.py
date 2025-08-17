# app.py — Streamlit Inference App (CRISP-DM friendly, cleaned-data first)
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support
)

# -----------------------
# Config & Constants
# -----------------------
st.set_page_config(page_title="Hotel Conversion Prediction", layout="wide")
st.title("Hotel Conversion Prediction")

TARGET = "Revenue"
ID_COL = "ID"
# 依你專案欄位：Month/OS/Browser/Region/TrafficType/VisitorType/Weekend 為類別
CANDIDATE_CAT_COLS = ["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend"]

# -----------------------
# Utils
# -----------------------
def show_df(df: pd.DataFrame, keep_index: bool = False):
    """統一表格風格：預設隱藏索引（避免左邊出現 0/1/2/3/4 那欄）。"""
    try:
        if keep_index:
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        # 舊版 streamlit 無 hide_index 參數 → 用 reset_index 退而求其次
        st.dataframe(df if keep_index else df.reset_index(drop=True), use_container_width=True)

def make_ohe():
    # sklearn >=1.2 uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def build_preprocess(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", make_ohe(), cat_cols),
        ]
    )

def coerce_target(y: pd.Series) -> pd.Series:
    if y.dtype == "O":
        return y.astype(str).str.lower().map({"true":1,"yes":1,"1":1,"false":0,"no":0,"0":0}).astype(int)
    return y.astype(int)

def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    # 按你的策略：train 缺值極少 → 直接丟列
    return df.dropna().copy()

def fill_test(df: pd.DataFrame) -> pd.DataFrame:
    # 不刪列；數值補中位數、文字補眾數
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if out[c].isnull().any():
            out[c] = out[c].fillna(out[c].median())
    obj_cols = [c for c in out.columns if c not in num_cols]
    for c in obj_cols:
        if out[c].isnull().any():
            mode = out[c].mode(dropna=True)
            out[c] = out[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return out

@st.cache_data(show_spinner=False)
def load_train_csv(path: str = "train.csv"):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_train_clean_csv(path: str = "artifacts/train_clean.csv"):
    # 若不存在會噴錯 → 呼叫端用 try/except
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def train_models(train_df: pd.DataFrame):
    """回傳：feature_cols, cat_cols, num_cols, models(dict), val_metrics(dict)"""
    assert TARGET in train_df.columns, f"訓練資料缺少目標欄位 {TARGET}"
    assert ID_COL in train_df.columns, f"訓練資料缺少欄位 {ID_COL}"

    # 冪等處理：若已是乾淨資料，dropna 不會改變
    train_clean = clean_train(train_df)
    y = coerce_target(train_clean[TARGET].copy())
    feature_cols = [c for c in train_clean.columns if c not in [TARGET, ID_COL]]

    # 推導類別欄位（以你給的清單為優先，fallback 用 dtype）
    cat_cols = [c for c in CANDIDATE_CAT_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocess = build_preprocess(num_cols, cat_cols)

    X = train_clean[feature_cols].copy()
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 兩個模型
    lr = Pipeline(steps=[("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
    rf = Pipeline(steps=[("prep", preprocess), ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])

    models, metrics = {}, {}
    for name, pipe in [("LogisticRegression", lr), ("RandomForest", rf)]:
        pipe.fit(X_tr, y_tr)
        va_proba = pipe.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, va_proba)
        y_hat = (va_proba >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_va, y_hat, average="binary", zero_division=0)
        metrics[name] = {
            "roc_auc": float(auc),
            "precision_pos": float(prec), "recall_pos": float(rec), "f1_pos": float(f1),
            "report": classification_report(y_va, y_hat, output_dict=True),
        }
        models[name] = pipe

    # 以全資料重訓（推論用）
    for name in models:
        models[name].fit(X, y)

    return feature_cols, cat_cols, num_cols, models, metrics

def extract_feature_names(prep: ColumnTransformer, feature_cols):
    try:
        names = prep.get_feature_names_out(feature_cols)
    except TypeError:
        names = prep.get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in names]

def plot_feature_importance(pipe: Pipeline, feature_cols, title="Top Feature Importance", top_n=20):
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    feat_names = extract_feature_names(prep, feature_cols)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_.ravel())
        title = title.replace("Feature Importance", "Top Coefficients (|coef|)")
        label = "|coef|"
    else:
        st.info("此模型不支援 feature importance / coefficients。")
        return
    fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
    top = fi.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel(label)
    fig.tight_layout()
    st.pyplot(fig)
    with st.expander("下載完整重要性明細 CSV"):
        csv_bytes = fi.to_csv(index=False).encode("utf-8")
        st.download_button("Download feature_importance.csv", csv_bytes, file_name="feature_importance.csv", mime="text/csv")

def threshold_report(y_true, proba, thr: float):
    y_hat = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0,1])
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    return cm, prec, rec, f1, classification_report(y_true, y_hat, output_dict=True)

# -----------------------
# Sidebar — Controls
# -----------------------
with st.sidebar:
    st.header(" 設定")
    st.caption("訓練資料：自動偵測 `artifacts/train_clean.csv` → 若無則使用 `train.csv`（內部 dropna）")
    model_choice = st.selectbox("模型", ["RandomForest", "LogisticRegression"], index=0)
    thr = st.slider("Decision Threshold (預設 0.5)", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
    reindex_id = st.checkbox("將上傳的 test.ID 轉為 1..N（並保留 OriginalID）", value=False)
    uploaded = st.file_uploader("上傳 test.csv", type=["csv"])

# -----------------------
# Training Phase (cached)
# -----------------------
# 優先採用 artifacts/train_clean.csv（與 main.py 結果一致），否則退回 raw + dropna()
try:
    raw_df = load_train_csv("train.csv")
except Exception as e:
    st.error(f"讀取 train.csv 失敗：{e}")
    st.stop()

try:
    clean_df = load_train_clean_csv("artifacts/train_clean.csv")
except Exception:
    clean_df = None

train_df_used = clean_df if clean_df is not None else clean_train(raw_df)
feature_cols, cat_cols, num_cols, models, val_metrics = train_models(train_df_used)

# 左上角概覽卡（顯示清洗後列數 + delta）
raw_rows = len(raw_df)
clean_rows = len(train_df_used)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train Rows (clean)", f"{clean_rows:,}", delta=f"-{raw_rows - clean_rows:,}")
c2.metric("Train Cols", f"{train_df_used.shape[1]}")
c3.metric("Model AUC (RF)", f"{val_metrics['RandomForest']['roc_auc']:.3f}")
c4.metric("Model AUC (LR)", f"{val_metrics['LogisticRegression']['roc_auc']:.3f}")

st.caption("Train Rows 來源：" + ("artifacts/train_clean.csv" if clean_df is not None else "由 app 以 dropna() 清理自 train.csv"))

# 顯示驗證成績表
st.subheader("📊 驗證成績（Hold-out 8:2）")
score_tbl = pd.DataFrame([
    {"model": "RandomForest", "roc_auc": val_metrics["RandomForest"]["roc_auc"],
     "precision_pos": val_metrics["RandomForest"]["precision_pos"],
     "recall_pos": val_metrics["RandomForest"]["recall_pos"],
     "f1_pos": val_metrics["RandomForest"]["f1_pos"]},
    {"model": "LogisticRegression", "roc_auc": val_metrics["LogisticRegression"]["roc_auc"],
     "precision_pos": val_metrics["LogisticRegression"]["precision_pos"],
     "recall_pos": val_metrics["LogisticRegression"]["recall_pos"],
     "f1_pos": val_metrics["LogisticRegression"]["f1_pos"]},
]).sort_values("roc_auc", ascending=False)
show_df(score_tbl)

# Feature importance / coefficients
st.subheader("🔎 特徵重要性 / 係數")
plot_feature_importance(models[model_choice], feature_cols, title=f"Top Feature Importance ({model_choice})", top_n=20)

# -----------------------
# Inference on Uploaded Test
# -----------------------
st.subheader("🧪 上傳測試檔預測")
if uploaded is not None:
    try:
        test_df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"讀取上傳檔案失敗：{e}")
        st.stop()

    st.write("**檔案前 5 列**")
    show_df(test_df_raw.head())

    # 基本欄位檢查
    missing_cols = [c for c in feature_cols + [ID_COL] if c not in test_df_raw.columns]
    if missing_cols:
        st.error(f"上傳檔缺少必要欄位：{missing_cols}")
        st.stop()

    # 清理與（可選）重編 ID
    test_clean = fill_test(test_df_raw)
    if reindex_id:
        test_clean = test_clean.copy()
        test_clean["OriginalID"] = test_clean[ID_COL]
        test_clean[ID_COL] = range(1, len(test_clean) + 1)

    # 推論
    X_test = test_clean[feature_cols].copy()
    proba = models[model_choice].predict_proba(X_test)[:, 1]
    preds = (proba >= thr).astype(int)

    # 預測摘要
    st.write("**預測摘要**")
    a1, a2, a3 = st.columns(3)
    a1.metric("Rows", f"{len(test_clean):,}")
    a2.metric("Avg Prob.", f"{proba.mean():.3f}")
    a3.metric("Predicted Positives", int(preds.sum()))

    # 下載 submission
    sub = pd.DataFrame({ID_COL: test_clean[ID_COL].values, "Revenue": np.round(proba, 2)})
    st.download_button(
        "📥 下載 submission.csv",
        data=sub.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )

    # 預覽 Top N
    st.write("**Top 30 機率**（由高到低）")
    top = test_clean[[ID_COL]].copy()
    top["Probability"] = proba
    top30 = top.sort_values("Probability", ascending=False).head(30)
    show_df(top30)

    # 若使用者也提供了標記（少見），可做評估
    if TARGET in test_clean.columns:
        st.info("偵測到上傳檔包含目標欄位 `Revenue`，以下為即時評估：")
        y_true = coerce_target(test_clean[TARGET])
        cm, p, r, f1, rep = threshold_report(y_true, proba, thr)
        st.write(f"ROC-AUC: **{roc_auc_score(y_true, proba):.4f}**")
        # 混淆矩陣與報告：保留索引/欄標籤（這裡不隱藏）
        cm_df = pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])
        show_df(cm_df, keep_index=True)
        rep_df = pd.DataFrame(rep).T
        show_df(rep_df, keep_index=True)
else:
    st.warning("請在左側上傳 `test.csv` 以產生預測與下載 submission。")

st.caption("© 2025 Hotel Conversion Prediction — CRISP-DM pipeline | Logistic Regression & RandomForest | OneHot + StandardScaler")
