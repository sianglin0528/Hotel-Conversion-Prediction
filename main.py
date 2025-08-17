# main.py (merged & tidy)
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ---------- 0) 路徑 ----------
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH  = ROOT / "test.csv"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

# ---------- 1) 讀資料 ----------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(" 讀檔 OK")
print("  - train 形狀：", train.shape)
print("  - test  形狀：", test.shape)

# ---------- 1.1) 清洗前預覽（存檔 + 終端顯示） ----------
print("\n 【清洗前（train）預覽】")
print(train.head(5).to_string(index=False))
print("\n 【清洗前（test）預覽】")
print(test.head(5).to_string(index=False))
(train.head(20)).to_csv(OUT_DIR / "preview_train_before.csv", index=False)
(test.head(20)).to_csv(OUT_DIR / "preview_test_before.csv", index=False)
print(f" 已輸出預覽：{OUT_DIR/'preview_train_before.csv'}, {OUT_DIR/'preview_test_before.csv'}")

# ---------- 2) 缺失值統計（表格版） ----------
def missing_table(df: pd.DataFrame, name: str):
    mc = df.isnull().sum()
    mp = (mc / len(df) * 100).round(3)
    tbl = pd.DataFrame({"缺失數量": mc, "缺失比例(%)": mp})
    tbl = tbl[tbl["缺失數量"] > 0].sort_values("缺失數量", ascending=False)
    print(f"\n 缺失值統計表（{name}）")
    if tbl.empty:
        print("  - 無缺失值")
    else:
        print(tbl)
    print(f"  - 缺失值總數：{int(mc.sum())}")
    print(f"  - 有缺失的列數：{df.isnull().any(axis=1).sum()}")

missing_table(train, "train")
missing_table(test,  "test")



# ---------- 3) 類別欄位分布（快速掃一眼，可留可註解） ----------
categorical_cols = ["Month", "OperatingSystems", "Browser", "Region",
                    "TrafficType", "VisitorType", "Weekend"]
print("\n 類別欄位分布（train，前 10）：")
for col in categorical_cols:
    if col in train.columns:
        print(f"\n- {col}")
        print(train[col].value_counts(dropna=False).head(10))

# ---------- 4) 缺失值處理 ----------
print("\n🧹 清理前 train：", train.shape)
train_clean = train.dropna().copy()  # 你的策略：缺少很少就丟列
print(" 清理後 train：", train_clean.shape)

# test：通常不能丟列 → 補值（守備性）
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

# ---------- 4.x) 統一 ID ----------
# 保留原始 ID，建立新的連號 ID（1..N）
test_clean = test_clean.copy()
test_clean["OriginalID"] = test_clean["ID"]
test_clean["ID"] = range(1, len(test_clean) + 1)

std_test_path = OUT_DIR / "test_standardized.csv"
test_clean.to_csv(std_test_path, index=False)
print(f"\n 已將 test.ID 標準化為 1..N，並輸出：{std_test_path}")
print(test_clean[["ID", "OriginalID"]].head(10).to_string(index=False))


# ---------- 4.1) 清洗後預覽 + 差異摘要 ----------
print("\n 【清洗後（train_clean）預覽】")
print(train_clean.head(5).to_string(index=False))
print("\n 【清洗後（test_clean）預覽】")
print(test_clean.head(5).to_string(index=False))
missing_table(train_clean, "train（清洗後）")
missing_table(test_clean,  "test（補值後）")
rows_dropped = len(train) - len(train_clean)
print(f"\n 清洗成果：train 共移除 {rows_dropped} 列（{len(train)} ➜ {len(train_clean)}）")

# 輸出清理後資料（方便重現 & 對照）
(train_clean.head(20)).to_csv(OUT_DIR / "preview_train_after.csv", index=False)
(test_clean.head(20)).to_csv(OUT_DIR / "preview_test_after.csv", index=False)
train_clean.to_csv(OUT_DIR / "train_clean.csv", index=False)
test_clean.to_csv(OUT_DIR / "test_clean.csv", index=False)
print(f" 已輸出預覽：{OUT_DIR/'preview_train_after.csv'}, {OUT_DIR/'preview_test_after.csv'}")

# ---------- 5) 特徵與目標 ----------
TARGET = "Revenue"
ID_COL = "ID"
assert TARGET in train_clean.columns, "train 應包含 Revenue 欄位"
assert ID_COL in train_clean.columns and ID_COL in test_clean.columns, "train/test 應包含 ID 欄位"

feature_cols = [c for c in train_clean.columns if c not in [ID_COL, TARGET]]
X = train_clean[feature_cols].copy()
y = train_clean[TARGET].copy()

# 轉 0/1（若已是 0/1 不會改變）
if y.dtype == "O":
    y = y.astype(str).str.lower().map({"true":1,"yes":1,"1":1,"false":0,"no":0,"0":0}).astype(int)

X_test = test_clean[feature_cols].copy()
test_ids = test_clean[ID_COL].copy()

# ---------- 6) OneHotEncoder 版本自動相容 + 前處理 ----------
def make_ohe():
    # sklearn >= 1.2 用 sparse_output；更舊版本用 sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

num_cols  = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols  = [c for c in feature_cols if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", make_ohe(), cat_cols),
    ]
)

# ---------- 7) 切驗證集 ----------
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 8) 模型 1：Logistic Regression（baseline） ----------
log_reg = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])
log_reg.fit(X_tr, y_tr)
va_pred_lr = log_reg.predict_proba(X_va)[:, 1]
auc_lr = roc_auc_score(y_va, va_pred_lr)
print(f"\n Logistic Regression ROC-AUC：{auc_lr:.4f}")
print("\n LR Validation report（threshold=0.5）")
print(classification_report(y_va, (va_pred_lr >= 0.5).astype(int)))

# retrain 全資料 + 預測 test（兩位小數）
log_reg.fit(X, y)
prob_test_lr = log_reg.predict_proba(X_test)[:, 1]
sub_lr = pd.DataFrame({
    "ID": test_clean["ID"].values,        # ← 新的連號 ID
    "Revenue": prob_test_lr.round(2)
})
sub_lr.to_csv(OUT_DIR / "submission_logreg.csv", index=False)
print(f" 已輸出：{OUT_DIR/'submission_logreg.csv'}")


# ---------- 9) 模型 2：Random Forest（加分） ----------
rf = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])
rf.fit(X_tr, y_tr)
va_pred_rf = rf.predict_proba(X_va)[:, 1]
auc_rf = roc_auc_score(y_va, va_pred_rf)
print(f"\n RandomForest ROC-AUC：{auc_rf:.4f}")
print("\n RF Validation report（threshold=0.5）")
print(classification_report(y_va, (va_pred_rf >= 0.5).astype(int)))

# retrain 全資料 + 預測 test（兩位小數）
rf.fit(X, y)
prob_test_rf = rf.predict_proba(X_test)[:, 1]
sub_rf = pd.DataFrame({"ID": test_ids, "Revenue": prob_test_rf.round(2)})
sub_rf.to_csv(OUT_DIR / "submission_rf.csv", index=False)
print(f" 已輸出：{OUT_DIR/'submission_rf.csv'}")

# ---------- 10) 小結 ----------
print("兩模型效能比較（ROC-AUC 越高越好）")
print(pd.DataFrame({
    "model": ["LogisticRegression", "RandomForest"],
    "val_roc_auc": [auc_lr, auc_rf]
}).sort_values("val_roc_auc", ascending=False).to_string(index=False))
print("任務完成：請到 artifacts/ 夾看兩個 submission 檔（已四捨五入到小數點後兩位），以及前/後預覽與清洗後資料。")


# ===== 11) Feature Importance 可視化（隨機森林 + 邏輯回歸）=====
import numpy as np
import matplotlib.pyplot as plt

def _clean_feat_names(prep, feature_cols):
    """
    從 ColumnTransformer 取出 OneHot 展開後的欄位名稱，並去掉 'num__' / 'cat__' 前綴
    """
    try:
        names = prep.get_feature_names_out(feature_cols)
    except TypeError:
        names = prep.get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in names]

def plot_top_importances(pipeline, feature_cols, out_path, top_n=15, title="Top Feature Importance"):
    """
    支援樹模型的 feature_importances_，或線性模型的 coef_（取絕對值）。
    會同時輸出 PNG 圖與 CSV 明細。
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    feat_names = _clean_feat_names(prep, feature_cols)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_.ravel())
        title = title.replace("Feature Importance", "Top Coefficients (|coef|)")
    else:
        raise ValueError("此模型沒有 feature_importances_ 或 coef_ 屬性")

    fi = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_sorted = fi.sort_values("importance", ascending=False)

    # 存成 CSV（完整表）
    csv_path = out_path.with_suffix(".csv")
    fi_sorted.to_csv(csv_path, index=False)

    # 取前 top_n 繪圖（水平長條、由小到大畫起比較好看）
    top = fi_sorted.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f" 已輸出圖檔：{out_path}")
    print(f" 已輸出明細：{csv_path}")

# 針對「全資料重訓後」的兩個模型產出圖
plot_top_importances(
    rf, feature_cols,
    OUT_DIR / "feature_importance_rf.png",
    top_n=15,
    title="Top Feature Importance (RandomForest)"
)

plot_top_importances(
    log_reg, feature_cols,
    OUT_DIR / "feature_importance_lr.png",
    top_n=15,
    title="Top Feature Importance (Logistic Regression)"
)

# =========================
# 末尾附錄：計算過程說明 + 指標比較圖（白底）
# 直接貼在 main.py 最底下即可使用；依賴全域的 CONF（兩個模型的 TP/FP/FN/TN）
# =========================
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def _cm_safe_div(a, b):
    return a / b if b else 0.0

def _cm_metrics(TP, FP, FN, TN):
    total       = TP + FP + FN + TN
    accuracy    = _cm_safe_div(TP + TN, total)
    precision   = _cm_safe_div(TP, TP + FP)
    recall      = _cm_safe_div(TP, TP + FN)
    specificity = _cm_safe_div(TN, TN + FP)
    f1          = _cm_safe_div(2 * precision * recall, precision + recall)
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "Total": total
    }

def _print_explain(name, C):
    TP, FP, FN, TN = C["TP"], C["FP"], C["FN"], C["TN"]
    total = TP + FP + FN + TN
    print(f"\n— 計算過程｜{name} —")
    print(f"Accuracy  = (TP+TN)/(TP+FP+FN+TN) = ({TP}+{TN})/({TP}+{FP}+{FN}+{TN}) = {(TP+TN)}/{total} = {_cm_safe_div(TP+TN, total):.3f}")
    print(f"Precision = TP/(TP+FP)            = {TP}/({TP}+{FP}) = {_cm_safe_div(TP, TP+FP):.3f}")
    print(f"Recall    = TP/(TP+FN)            = {TP}/({TP}+{FN}) = {_cm_safe_div(TP, TP+FN):.3f}")
    print(f"F1        = 2PR/(P+R)             = 2×{_cm_safe_div(TP, TP+FP):.3f}×{_cm_safe_div(TP, TP+FN):.3f}/({_cm_safe_div(TP, TP+FP):.3f}+{_cm_safe_div(TP, TP+FN):.3f}) = {_cm_safe_div(2*_cm_safe_div(TP, TP+FP)*_cm_safe_div(TP, TP+FN), _cm_safe_div(TP, TP+FP)+_cm_safe_div(TP, TP+FN)):.3f}")
    print(f"Specificity = TN/(TN+FP)          = {TN}/({TN}+{FP}) = {_cm_safe_div(TN, TN+FP):.3f}")

# 1) 建表 + 列印逐步公式
try:
    _CONF_SRC = CONF  # 期待上方已定義 CONF
except NameError:
    _CONF_SRC = {
        "Model A": {"TP": 853, "FP": 341, "FN": 576, "TN": 7230},
        "Model B": {"TP": 846, "FP": 316, "FN": 583, "TN": 7255},
    }

_rows = []
for _name, _c in _CONF_SRC.items():
    _C = {k.upper(): v for k, v in _c.items()}
    _print_explain(_name, _C)
    _rows.append({"Model": _name, **_cm_metrics(**_C)})

_metrics_df = pd.DataFrame(_rows).set_index("Model")
print("\n=== 指標總表 ===")
print(_metrics_df[["TP","FP","FN","TN","Total","Accuracy","Precision","Recall","F1","Specificity"]].round(3))

# 2) 產生白底比較圖（Accuracy / Precision / Recall / F1 / Specificity）
_ART = Path("artifacts"); _ART.mkdir(exist_ok=True)
plt.style.use("default")
plt.rcParams["figure.facecolor"]  = "white"
plt.rcParams["axes.facecolor"]    = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.sans-serif"]   = ["PingFang TC", "Noto Sans CJK TC", "Heiti TC", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_cols = ["Accuracy","Precision","Recall","F1","Specificity"]
ax = _metrics_df[_cols].plot(kind="bar", figsize=(9.5, 5.2))
ax.set_title("模型指標比較", fontsize=16)
ax.set_ylabel("分數"); ax.set_ylim(0, 1.0); ax.grid(axis="y", color="#eeeeee")
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.3f}", (p.get_x()+p.get_width()/2, h),
                ha="center", va="bottom", fontsize=9,
                xytext=(0,3), textcoords="offset points")
plt.tight_layout()
_out = _ART / "metrics_bar_compare.png"
plt.savefig(_out, dpi=170)
plt.close()
print(f"\n[Saved] 指標比較圖：{_out}")
