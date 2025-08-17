#  Hotel Conversion Prediction

##  專案簡介
本專案利用網站流量資料（Google Analytics），預測訪客是否會訂房 (`Revenue`)。由於行銷預算有限，必須鎖定較可能轉換的訪客，以降低行銷成本並提升投資報酬率。
方法論採用資料科學跨產業標準化的流程 **CRISP-DM 流程**，完整涵蓋 **業務理解 → 資料理解 → 資料準備 → 建模 → 評估 → 部署/產出**。  

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


## 🔥 加分題：模型混淆矩陣比較與評估

### 混淆矩陣

|            | Model A 預測真 | Model A 預測假 | Model B 預測真 | Model B 預測假 |
|------------|----------------|----------------|----------------|----------------|
| **實際真** | 853 (TP)       | 576 (FN)       | 846 (TP)       | 583 (FN)       |
| **實際假** | 341 (FP)       | 7230 (TN)      | 316 (FP)       | 7255 (TN)      |

---

### 評估指標

| 指標       | Model A | Model B | 最佳者 |
|------------|---------|---------|--------|
| Accuracy   | 0.898   | 0.902   | **B** |
| Precision  | 0.714   | 0.728   | **B** |
| Recall     | 0.597   | 0.592   | A |
| F1-score   | 0.650   | 0.653   | **B** |

---

### 分析解釋
- **Model A**：Recall 較高 → 比較不會漏掉潛在的訂房客人。  
- **Model B**：Precision 較高 → 投放廣告時「命中率更高」，能避免浪費廣告費。  

---

### 結論（Business Perspective）
在「行銷經費有限」的飯店情境下，我們更在意 **Precision（精準度）**，因為每一分廣告費都要花在最可能轉換的客人身上。  

雖然 Model A 在 Recall 略佔優勢，但 **Model B 在 Accuracy 與 Precision 上更佳**，因此更符合本專案需求。  

**建議採用 Model B 作為最終投放決策模型。**

