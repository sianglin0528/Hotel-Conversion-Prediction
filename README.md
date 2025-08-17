# Hotel Conversion Prediction

## 專案簡介
本專案以網站流量資料（Google Analytics-like features），預測訪客是否會訂房 (`Revenue`)。  
方法論採用 **CRISP-DM 流程**，完整涵蓋 **業務理解 → 資料理解 → 資料準備 → 建模 → 評估 → 部署/產出**。  

---

## CRISP-DM 流程

### 1. Business Understanding（業務理解）
- **目標**：提升飯店線上廣告投放效益，精準找出高轉換率訪客。  
- **商業價值**：  
  - 降低 CAC（Customer Acquisition Cost）  
  - 提升 ROI（Return on Investment）  
  - 提供行銷部門可操作的洞察（例如：哪些訪客群體更可能訂房）

---

### 2. Data Understanding（資料理解）
- **來源**：Google Analytics 網站流量統計資料 (`train.csv`, `test.csv`)  
- **規模**：數千筆樣本，數十個特徵  
- **初步分析**：  
  - 類別變數分布檢視（如 `Month`, `Region`, `VisitorType`）  
  - 缺失值統計與比例檢查  
- **產出**：`artifacts/preview_train_before.csv`, `preview_test_before.csv`

---

### 3. Data Preparation（資料準備）
- **缺失值處理**  
  - `train`：缺失值少 → 直接刪列  
  - `test`：數值補中位數，類別補眾數  
- **特徵工程**  
  - 數值欄位 → `StandardScaler`  
  - 類別欄位 → `OneHotEncoder`（支援版本相容）  
- **ID 標準化**  
  - `test.ID` 改為連號，保留 `OriginalID`  
- **輸出**：  
  - `train_clean.csv`, `test_clean.csv`  
  - `preview_train_after.csv`, `preview_test_after.csv`

---

### 4. Modeling（建模）
- **模型選擇**  
  - Baseline：Logistic Regression  
  - 強化：Random Forest  
- **訓練流程**  
  - 使用 `Pipeline` 串接前處理與模型  
  - 切分資料（Stratified 8:2 split）確保正負樣本比例一致  

---

### 5. Evaluation（評估）
- **指標**  
  - ROC-AUC（主要衡量指標）  
  - Classification Report（Precision / Recall / F1）  
- **結果摘要**  
  - Logistic Regression ROC-AUC：約 0.78  
  - RandomForest ROC-AUC：約 0.85  
- **可解釋性**  
  - RandomForest → `feature_importances_`  
  - Logistic Regression → `coef_` (取絕對值)  
- **產出**  
  - `feature_importance_lr.png/.csv`  
  - `feature_importance_rf.png/.csv`

---

### 6. Deployment / Output（部署 / 產出）
- **輸出檔案**  
  - `submission_logreg.csv`  
  - `submission_rf.csv`  
- **Artifacts**  
  - 完整清理前後對照  
  - 模型預測結果與特徵重要性圖表  
- **執行方式**  
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  python main.py
