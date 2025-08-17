#  Hotel Conversion Prediction

##  專案簡介
本專案利用網站流量資料（Google Analytics-like features），預測訪客是否會訂房 (`Revenue`)。  
方法論採用 **CRISP-DM 流程**，完整涵蓋 **業務理解 → 資料理解 → 資料準備 → 建模 → 評估 → 部署/產出**。  

---

##  CRISP-DM 流程

### 1. Business Understanding（業務理解）
- **挑戰**：廣告成本高，但大部分訪客不會轉換  
- **專案目標**：預測高轉換率訪客，提升廣告投資報酬率  
- **商業價值**：  
  - 降低 CAC (Customer Acquisition Cost)  
  - 提升 ROI (Return on Investment)  
  - 提供行銷團隊數據決策依據  

---

### 2. Data Understanding（資料理解）
- **來源**：Google Analytics-like 流量資料  
- **資料集規模**：  
  - Train：8100 筆 × 19 欄  
  - Test：900 筆 × 18 欄  
- **缺失值狀況**：  
  - Train 缺失值總數：6（涉及 ProductRelated_Duration, OperatingSystems, Browser, Region, TrafficType, VisitorType）  
  - Test：無缺失值  
- **類別欄位分布**：  
  - `Month`：1,2,8,9 四個月占大宗  
  - `OperatingSystems`：以 1/3/5 為主  
  - `VisitorType`：大多為 2（重訪客）  

**Artifacts**  
- `artifacts/preview_train_before.csv`  
- `artifacts/preview_test_before.csv`  

---

### 3. Data Preparation（資料準備）
- **缺失值處理**  
  - Train：移除含缺失列 → 共刪除 3 筆（8100 ➜ 8097）  
  - Test：數值補中位數、類別補眾數  
- **特徵工程**  
  - 數值欄位 → 標準化 (`StandardScaler`)  
  - 類別欄位 → One-Hot 編碼 (`OneHotEncoder`)  
- **ID 標準化**  
  - Test 新增 `OriginalID`，同時生成連號 ID  

 **Artifacts**  
- `artifacts/train_clean.csv`  
- `artifacts/test_clean.csv`  
- `artifacts/preview_train_after.csv`  
- `artifacts/preview_test_after.csv`  

---

### 4. Modeling（建模）
- **模型選擇**  
  - Logistic Regression → Baseline，可解釋  
  - Random Forest → 樹模型，能捕捉非線性關係  
- **流程**  
  - 使用 `Pipeline` 串接前處理  
  - Stratified 8:2 split，確保類別比例一致  

---

### 5. Evaluation（評估）
- **驗證集效能**  

| Model               | ROC-AUC | Accuracy | Precision (1) | Recall (1) | F1-score (1) |
|----------------------|---------|----------|---------------|------------|--------------|
| Logistic Regression | 0.8678  | 0.88     | 0.74          | 0.38       | 0.50         |
| Random Forest       | 0.9175  | 0.90     | 0.72          | 0.60       | 0.65         |

> `1` 代表正樣本（Revenue = 1）  

- **觀察**  
  - Logistic Regression：準確度高，但對「訂房者」的 Recall 偏低（0.38）。  
  - Random Forest：ROC-AUC 更高 (0.9175)，Recall 提升至 0.60，整體效能最佳。  

- **特徵重要性**  
  - RandomForest → `feature_importances_`  
  - LogisticRegression → `coef_` (取絕對值)  

 **Artifacts**  
- `artifacts/feature_importance_lr.png / .csv`  
- `artifacts/feature_importance_rf.png / .csv`  

---

### 6. Deployment / Output（部署與產出）
- **輸出檔案**  
  - `artifacts/submission_logreg.csv`  
  - `artifacts/submission_rf.csv`  
- **完整流程再現**  
  ```bash
  # 建立環境
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate

  # 安裝依賴
  pip install -r requirements.txt

  # 執行
  python main.py


Artifacts

artifacts/ 資料夾包含清理前後對照、submission 檔、特徵重要性圖表

商業應用價值

精準行銷：將廣告資源鎖定在高轉換族群
降低成本：減少無效投放，提高 ROI
洞察驅動：行銷與營運能依據特徵重要性調整策略