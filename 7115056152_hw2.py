# -*- coding: utf-8 -*-
"""
================================================================================
巨量資料分析與機器學習 - 作業二 (HW2)
主題：基於 CRISP-DM 流程之金郡房價多元線性回歸分析 (Multiple Linear Regression)
資料集：House Sales in King County, USA (Kaggle)
資料集來源連結：https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
================================================================================
CRISP-DM 流程六大階段：
1. 商業理解 (Business Understanding)
2. 資料理解 (Data Understanding)
3. 資料準備 (Data Preparation)
4. 模型建構 (Modeling)
5. 模型評估 (Evaluation)
6. 部署上線 (Deployment)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# 設定中文字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Microsoft JhengHei')

# 確保圖片輸出資料夾存在
FIGURES_DIR = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("作業二：King County 房價多元線性回歸 (Multiple Linear Regression) 完整分析")
print("遵循 CRISP-DM 資料探勘跨行業標準流程")
print("=" * 80)

# ==============================================================================
# Phase 1: Business Understanding (商業理解)
# ==============================================================================
print("\n" + "=" * 50)
print("【Phase 1: Business Understanding 商業理解】")
print("=" * 50)
print("""
1. 專案背景：
   美國華盛頓州金郡 (King County，涵蓋西雅圖市及周邊主要科技都會區) 房地產市場蓬勃發展。
   房價受生活機能、居住空間、建築結構品質、地理交通等多重維度影響。
2. 業務問題 (Business Problem)：
   房屋仲介機構與購屋者迫切需要一套科學、客觀且具備高解釋力的房價公允估價模型，
   以減少人為估價主觀偏差，並精確衡量各維度特徵（如室內坪數、屋況等級、距市中心距離）
   對房價帶來的邊際溢價效應。
3. 機器學習任務 (Data Mining / ML Goal)：
   建立「多元線性回歸模型 (Multiple Linear Regression)」，預測連續型目標變數 `price`，
   嚴格檢驗統計顯著性 ($p < 0.05$)、診斷多重共線性 (VIF)，並提供 95% 信賴區間與預測區間。
4. 成功準則 (Success Criteria)：
   - 模型統計顯著，測試集 $R^2 \ge 0.70$。
   - 解決特徵高度共線性問題 (VIF < 10)。
   - 殘差分佈合理，提供清晰之點估計與不確定性區間估計。
""")

# ==============================================================================
# Phase 2: Data Understanding (資料理解)
# ==============================================================================
print("\n" + "=" * 50)
print("【Phase 2: Data Understanding 資料理解】")
print("=" * 50)

# 2.1 資料載入
data_path = 'kc_house_data.csv'
if not os.path.exists(data_path):
    # 若在當前目錄找不到，嘗試在當前資料夾搜尋
    data_path = os.path.join(os.getcwd(), 'kc_house_data.csv')

df = pd.read_csv(data_path)
print(f"資料集維度：共 {df.shape[0]:,} 筆觀測值，{df.shape[1]} 個欄位。")
print(f"資料欄位清單：\n{list(df.columns)}")

# 2.2 檢查缺失值與資料型態
print("\n[缺失值統計]")
missing_count = df.isnull().sum()
print(missing_count[missing_count > 0] if missing_count.sum() > 0 else "資料集非常完整，無任何缺失值 (Missing Values = 0)！")

# 2.3 目標變數描述性統計
price_stats = df['price'].describe()
print("\n[房價 (price) 敘述性統計]")
print(f"平均房價: ${price_stats['mean']:,.2f}")
print(f"中位數房價: ${price_stats['50%']:,.2f}")
print(f"房價標準差: ${price_stats['std']:,.2f}")
print(f"最低房價: ${price_stats['min']:,.2f}")
print(f"最高房價: ${price_stats['max']:,.2f}")
print(f"房價偏度 (Skewness): {df['price'].skew():.3f} (呈現高度右偏態)")

# 2.4 視覺化 1: 目標變數分佈 (原始分佈 vs 對數轉換)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['price'] / 1e4, bins=50, kde=True, color='royalblue')
plt.title('原始房價分佈 (單位: 萬美元)\n呈現顯著右偏態 (Skewness: {:.2f})'.format(df['price'].skew()), fontsize=12)
plt.xlabel('房屋售價 (萬美元)')
plt.ylabel('頻率')

plt.subplot(1, 2, 2)
sns.histplot(np.log(df['price']), bins=50, kde=True, color='teal')
plt.title('對數轉換後房價分佈 (Log Price)\n趨近常態分佈 (Skewness: {:.2f})'.format(np.log(df['price']).skew()), fontsize=12)
plt.xlabel('Log(房價)')
plt.ylabel('頻率')

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, 'fig1_target_distribution.png')
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig1_path}")

# 2.5 相關係數分析與視覺化
numeric_cols = df.drop(columns=['id', 'date']).columns
correlations = df[numeric_cols].corr()['price'].sort_values(ascending=False)
print("\n[各特徵與房價之皮爾森相關係數排行 (Top 10)]")
print(correlations.head(10))

# 視覺化 2: 相關係數熱力圖
plt.figure(figsize=(12, 10))
top_corr_features = correlations.head(12).index
corr_matrix = df[top_corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True, square=True)
plt.title('房價核心相關特徵熱力圖 (Pearson Correlation Matrix)', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'fig2_correlation_heatmap.png')
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig2_path}")

# 視覺化 3: 地理空間房價分佈圖 (西雅圖都會區經緯度)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['long'], df['lat'], c=df['price']/1e4, cmap='viridis',
                      alpha=0.4, s=12, vmin=20, vmax=150)
cbar = plt.colorbar(scatter)
cbar.set_label('房屋售價 (萬美元)', fontsize=11)
plt.scatter(-122.3321, 47.6062, color='red', marker='*', s=250, label='西雅圖市中心 (Downtown Seattle)')
plt.title('金郡房屋地理分佈與房價熱力圖 (Lat vs. Long)', fontsize=14)
plt.xlabel('經度 (Longitude)')
plt.ylabel('緯度 (Latitude)')
plt.legend(loc='lower left')
plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, 'fig3_geo_distribution.png')
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig3_path}")

# ==============================================================================
# Phase 3: Data Preparation (資料準備)
# ==============================================================================
print("\n" + "=" * 50)
print("【Phase 3: Data Preparation 資料準備與特徵工程】")
print("=" * 50)

# 3.1 異常值檢查與清洗
# 著名異常值：臥室數 33 間 (建物面積僅 1620 平方英尺，為明顯錄入錯誤)
print(f"清洗前資料筆數: {len(df)}")
df_clean = df[df['bedrooms'] < 30].copy()
print(f"排除極端臥室異常值 (bedrooms >= 30) 後筆數: {len(df_clean)}")

# 3.2 特徵工程 (Feature Engineering)
# 1) 銷售年份與屋齡
df_clean['sale_year'] = df_clean['date'].str[:4].astype(int)
df_clean['house_age'] = df_clean['sale_year'] - df_clean['yr_built']

# 2) 是否曾經翻修過 (二元標記 0/1)
df_clean['is_renovated'] = (df_clean['yr_renovated'] > 0).astype(int)

# 3) 房屋與西雅圖市中心 (Tech Hub / Downtown) 之歐氏距離 (地理加權重要特徵)
DOWNTOWN_LAT, DOWNTOWN_LONG = 47.6062, -122.3321
df_clean['dist_to_seattle'] = np.sqrt((df_clean['lat'] - DOWNTOWN_LAT)**2 + (df_clean['long'] - DOWNTOWN_LONG)**2)

print("\n[新衍生特徵說明]")
print("1. house_age: 房屋售出時屋齡 (sale_year - yr_built)")
print("2. is_renovated: 是否曾經翻修 (0: 未翻修, 1: 曾翻修)")
print("3. dist_to_seattle: 距離西雅圖市中心核心商業區之距離")

# 3.3 多重共線性診斷 (VIF 檢定)
# 統計關鍵發現：sqft_living = sqft_above + sqft_basement 為完全線性代數依賴！
diff_sqft = (df_clean['sqft_living'] - (df_clean['sqft_above'] + df_clean['sqft_basement'])).abs().max()
print(f"\n[共線性重要檢驗] |sqft_living - (sqft_above + sqft_basement)| 最大差異: {diff_sqft}")
print("確認 sqft_living = sqft_above + sqft_basement 存在完全多重共線性 (Perfect Multicollinearity)！")

candidate_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                      'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                      'sqft_basement', 'dist_to_seattle', 'lat', 'long',
                      'sqft_living15', 'sqft_lot15', 'house_age', 'is_renovated']

X_vif_test = sm.add_constant(df_clean[candidate_features].astype(float))
vif_df = pd.DataFrame()
vif_df['Feature'] = X_vif_test.columns
vif_df['VIF'] = [variance_inflation_factor(X_vif_test.values, i) for i in range(X_vif_test.shape[1])]
print("\n[特徵篩選前 VIF 數值 (顯示共線性爆炸)]: ")
print(vif_df.sort_values(by='VIF', ascending=False).head(8).to_string(index=False))

# 3.4 特徵選擇 (Feature Selection)
# 排除導致共線性的 sqft_above, sqft_basement，保留總室內面積 sqft_living，並選取具統計顯著之特徵
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                     'waterfront', 'view', 'condition', 'grade',
                     'dist_to_seattle', 'lat', 'long',
                     'sqft_living15', 'sqft_lot15', 'house_age', 'is_renovated']

print(f"\n[特徵選擇結果] 最終精選 {len(selected_features)} 個關鍵特徵：")
print(", ".join(selected_features))

# 3.5 切分訓練集與測試集 (80% Train, 20% Test)
X = df_clean[selected_features]
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n資料集劃分完成：訓練集樣本數 {X_train.shape[0]:,} 筆，測試集樣本數 {X_test.shape[0]:,} 筆。")

# ==============================================================================
# Phase 4: Modeling (模型建構)
# ==============================================================================
print("\n" + "=" * 50)
print("【Phase 4: Modeling 模型建構】")
print("=" * 50)

# 模型 1: 簡單線性回歸 (Simple Linear Regression, Baseline) - 以 sqft_living 預測
slr_model = LinearRegression()
slr_model.fit(X_train[['sqft_living']], y_train)

# 使用 statsmodels 建構單變數回歸 (用於精準信賴區間與預測區間推導)
X_train_slr_sm = sm.add_constant(X_train[['sqft_living']])
X_test_slr_sm = sm.add_constant(X_test[['sqft_living']])
ols_slr = sm.OLS(y_train, X_train_slr_sm).fit()

# 模型 2: 全特徵多元線性回歸 (Full Multiple Linear Regression)
full_features = candidate_features
X_train_full = df_clean.loc[X_train.index, full_features]
X_test_full = df_clean.loc[X_test.index, full_features]
mlr_full = LinearRegression()
mlr_full.fit(X_train_full, y_train)

# 模型 3: 特徵選擇後多元線性回歸 (Feature-Selected MLR with statsmodels)
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)
ols_mlr = sm.OLS(y_train, X_train_sm).fit()

print("\n[Statsmodels OLS 多元回歸摘要 (精選特徵)]")
print(ols_mlr.summary().tables[0])
print(ols_mlr.summary().tables[1])

# 模型 4: 正則化回歸 (Ridge & Lasso Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge(alpha=100.0)
ridge_model.fit(X_train_scaled, y_train)

lasso_model = Lasso(alpha=100.0)
lasso_model.fit(X_train_scaled, y_train)

# 模型 5: 對數轉換多元線性回歸 (Log-Transformed MLR)
y_train_log = np.log(y_train)
ols_log = sm.OLS(y_train_log, X_train_sm).fit()

# 模型 6: 網路主流/最優解法對照 (Random Forest 非線性集成模型)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("\n模型建構全部完成！包含簡單線性回歸、多元回歸、Ridge/Lasso、Log-MLR 與 Random Forest。")

# ==============================================================================
# Phase 5: Evaluation (模型評估)
# ==============================================================================
print("\n" + "=" * 50)
print("【Phase 5: Evaluation 模型評估與視覺化】")
print("=" * 50)

# 5.1 測試集預測
y_pred_slr = slr_model.predict(X_test[['sqft_living']])
y_pred_full = mlr_full.predict(X_test_full)
y_pred_mlr = ols_mlr.predict(X_test_sm)
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_lasso = lasso_model.predict(X_test_scaled)
y_pred_log = np.exp(ols_log.predict(X_test_sm))
y_pred_rf = rf_model.predict(X_test)

# 5.2 評估指標計算函數
def calculate_metrics(name, y_true, y_pred, k):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        '模型名稱 (Model)': name,
        'R2 Score': round(r2, 4),
        'Adjusted R2': round(adj_r2, 4),
        'RMSE ($)': round(rmse, 2),
        'MAE ($)': round(mae, 2),
        'MAPE (%)': round(mape, 2)
    }

eval_list = [
    calculate_metrics('1. 簡單線性回歸 (Baseline: sqft_living)', y_test, y_pred_slr, 1),
    calculate_metrics('2. 全特徵多元回歸 (Full MLR, 18特徵)', y_test, y_pred_full, len(full_features)),
    calculate_metrics('3. 特徵選擇多元回歸 (Selected MLR)', y_test, y_pred_mlr, len(selected_features)),
    calculate_metrics('4. Ridge 脊回歸 (L2 正則化)', y_test, y_pred_ridge, len(selected_features)),
    calculate_metrics('5. Lasso 套索回歸 (L1 正則化)', y_test, y_pred_lasso, len(selected_features)),
    calculate_metrics('6. 對數轉換多元回歸 (Log-Transformed MLR)', y_test, y_pred_log, len(selected_features)),
    calculate_metrics('7. 隨機森林回歸 (Random Forest 網路主流對照)', y_test, y_pred_rf, len(selected_features)),
]

metrics_df = pd.DataFrame(eval_list)
print("\n[模型評估成效對比總表]")
print(metrics_df.to_string(index=False))

# 5.3 預測圖繪製：包含 95% 信賴區間 (CI) 與 95% 預測區間 (PI)
print("\n繪製預測圖與信賴區間/預測區間...")

# 視覺化 4: 單變數線性回歸 + 95% 信賴區間與預測區間
# 產生平滑的 x 軸點
x_vals = np.linspace(df_clean['sqft_living'].min(), df_clean['sqft_living'].max(), 300)
x_vals_sm = sm.add_constant(pd.DataFrame({'sqft_living': x_vals}))
pred_res = ols_slr.get_prediction(x_vals_sm).summary_frame(alpha=0.05)

plt.figure(figsize=(11, 7))
# 抽樣繪製散佈點以保持圖表清爽
sample_indices = np.random.RandomState(42).choice(len(df_clean), size=1200, replace=False)
plt.scatter(df_clean.iloc[sample_indices]['sqft_living'],
            df_clean.iloc[sample_indices]['price'] / 1e4,
            alpha=0.35, color='gray', s=16, label='實際樣本數據 (Sample N=1200)')

# 繪製回歸線與區間
plt.plot(x_vals, pred_res['mean'] / 1e4, color='crimson', lw=2.5, label='回歸擬合線 (Fitted Line)')
plt.fill_between(x_vals, pred_res['mean_ci_lower'] / 1e4, pred_res['mean_ci_upper'] / 1e4,
                 color='blue', alpha=0.35, label='95% 信賴區間 (95% Confidence Interval for Mean)')
plt.plot(x_vals, pred_res['obs_ci_lower'] / 1e4, 'k--', lw=1.5, alpha=0.8,
         label='95% 預測區間 (95% Prediction Interval for Observations)')
plt.plot(x_vals, pred_res['obs_ci_upper'] / 1e4, 'k--', lw=1.5, alpha=0.8)

plt.title('單變數線性回歸擬合圖 (sqft_living vs. 房價)\n具備 95% 信賴區間 (CI) 與 95% 預測區間 (PI)', fontsize=13)
plt.xlabel('室內居住面積 (sqft_living, 平方英尺)')
plt.ylabel('房屋售價 (萬美元)')
plt.ylim(-10, 400)
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, 'fig4_simple_linear_regression_ci_pi.png')
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig4_path}")

# 視覺化 5: 多元線性回歸 實際值 vs. 預測值散佈圖
plt.figure(figsize=(9, 8))
plt.scatter(y_test / 1e4, y_pred_mlr / 1e4, alpha=0.35, color='royalblue', s=20, edgecolors='none')
# 繪製理想完美預測對角線 y = x
max_val = max(y_test.max(), y_pred_mlr.max()) / 1e4
min_val = min(y_test.min(), y_pred_mlr.min()) / 1e4
plt.plot([0, 450], [0, 450], 'r--', lw=2, label='完美預測線 (Ideal: y = ŷ)')

plt.title('多元線性回歸 (Selected MLR)：實際房價 vs. 預測房價\n$R^2 = {:.4f}$, RMSE = ${:,.0f}'.format(
    r2_score(y_test, y_pred_mlr), np.sqrt(mean_squared_error(y_test, y_pred_mlr))), fontsize=13)
plt.xlabel('實際房屋售價 (萬美元)')
plt.ylabel('模型預測售價 (萬美元)')
plt.xlim(0, 450)
plt.ylim(-20, 450)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, 'fig5_mlr_actual_vs_pred.png')
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig5_path}")

# 視覺化 6: 多元線性回歸測試集前 100 筆樣本之 95% 預測區間圖 (Prediction Interval Bounds)
test_predictions = ols_mlr.get_prediction(X_test_sm.iloc[:80]).summary_frame(alpha=0.05)
test_actuals = y_test.iloc[:80].values

plt.figure(figsize=(14, 6))
idx_range = np.arange(len(test_actuals))

# 繪製 95% 預測區間帶
plt.fill_between(idx_range, test_predictions['obs_ci_lower'] / 1e4, test_predictions['obs_ci_upper'] / 1e4,
                 color='lightblue', alpha=0.5, label='95% 個別預測區間 (95% Prediction Interval)')
# 繪製 95% 信賴區間帶
plt.fill_between(idx_range, test_predictions['mean_ci_lower'] / 1e4, test_predictions['mean_ci_upper'] / 1e4,
                 color='orange', alpha=0.5, label='95% 期望值信賴區間 (95% Confidence Interval)')
# 預測值與實際值點
plt.plot(idx_range, test_predictions['mean'] / 1e4, 'b-', lw=1.8, label='模型預測值 (Predicted)')
plt.scatter(idx_range, test_actuals / 1e4, color='crimson', s=35, zorder=5, label='真實售價 (Actual)')

plt.title('多元線性回歸：測試集前 80 筆樣本預測表現與 95% 預測區間範圍', fontsize=13)
plt.xlabel('測試樣本編號 (Sample Index)')
plt.ylabel('房價 (萬美元)')
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, 'fig6_mlr_prediction_interval.png')
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig6_path}")

# 視覺化 7: 殘差診斷圖四合一 (Residual Diagnostics)
residuals = y_train - ols_mlr.fittedvalues
fitted = ols_mlr.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 7.1 Residuals vs. Fitted
axes[0, 0].scatter(fitted / 1e4, residuals / 1e4, alpha=0.3, color='steelblue', s=12)
axes[0, 0].axhline(0, color='red', linestyle='--', lw=1.5)
axes[0, 0].set_title('殘差與擬合值散佈圖 (Residuals vs Fitted)\n檢驗線性度與變異數同質性', fontsize=11)
axes[0, 0].set_xlabel('擬合值 (萬美元)')
axes[0, 0].set_ylabel('殘差 (萬美元)')

# 7.2 Normal Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[0].set_markerfacecolor('steelblue')
axes[0, 1].get_lines()[0].set_markersize(3.5)
axes[0, 1].get_lines()[0].set_alpha(0.3)
axes[0, 1].set_title('常態機率圖 (Normal Q-Q Plot)\n檢驗殘差常態分佈假設', fontsize=11)

# 7.3 Histogram of Residuals
sns.histplot(residuals / 1e4, bins=50, kde=True, ax=axes[1, 0], color='purple')
axes[1, 0].set_title('殘差分佈直方圖 (Residual Distribution)', fontsize=11)
axes[1, 0].set_xlabel('殘差 (萬美元)')
axes[1, 0].set_ylabel('次數')

# 7.4 Scale-Location Plot (Sqrt Standardized Residuals vs Fitted)
std_residuals = residuals / np.std(residuals)
axes[1, 1].scatter(fitted / 1e4, np.sqrt(np.abs(std_residuals)), alpha=0.3, color='forestgreen', s=12)
axes[1, 1].set_title('尺度-定位圖 (Scale-Location Plot)\n檢驗誤差變異同質性 (Homoscedasticity)', fontsize=11)
axes[1, 1].set_xlabel('擬合值 (萬美元)')
axes[1, 1].set_ylabel('$\sqrt{|標準化殘差|}$')

plt.tight_layout()
fig7_path = os.path.join(FIGURES_DIR, 'fig7_residual_diagnostics.png')
plt.savefig(fig7_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig7_path}")

# 視覺化 8: 各模型評估指標橫向對比條形圖
fig, ax1 = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(metrics_df))
width = 0.35

color1 = 'tab:blue'
ax1.set_xlabel('評估模型', fontsize=12)
ax1.set_ylabel('決定係數 ($R^2$)', color=color1, fontsize=12)
rects1 = ax1.bar(x_pos - width/2, metrics_df['R2 Score'], width, label='$R^2$ Score', color='cornflowerblue')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.0)
ax1.set_xticks(x_pos)
short_names = ['簡單回歸', '全特徵MLR', '精選特徵MLR', 'Ridge', 'Lasso', 'Log-MLR', 'Random Forest']
ax1.set_xticklabels(short_names, rotation=20, ha='right', fontsize=11)

# 次座標軸繪製 RMSE
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('RMSE (萬美元)', color=color2, fontsize=12)
rects2 = ax2.bar(x_pos + width/2, metrics_df['RMSE ($)'] / 1e4, width, label='RMSE (萬美元)', color='coral')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(False)

plt.title('各模型評估指標績效對比 ($R^2$ 與 RMSE)', fontsize=14, pad=12)
fig.tight_layout()
fig8_path = os.path.join(FIGURES_DIR, 'fig8_model_comparison.png')
plt.savefig(fig8_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig8_path}")

# 視覺化 9: 特徵重要度 (回歸係數標準化權重)
coef_df = pd.DataFrame({
    'Feature': selected_features,
    'Std_Coef': ols_mlr.params[1:].values * (X_train.std().values / y_train.std())
}).sort_values(by='Std_Coef', key=abs, ascending=True)

plt.figure(figsize=(10, 7))
colors = ['crimson' if c < 0 else 'royalblue' for c in coef_df['Std_Coef']]
plt.barh(coef_df['Feature'], coef_df['Std_Coef'], color=colors, alpha=0.85)
plt.axvline(0, color='black', lw=0.8, linestyle='--')
plt.title('多元線性回歸特徵標準化係數排行 (Standardized Coefficients)\n正值代表正向溢價，負值代表折價效應', fontsize=13)
plt.xlabel(r'標準化回歸係數 ($\beta^*$)')
plt.tight_layout()
fig9_path = os.path.join(FIGURES_DIR, 'fig9_feature_importance.png')
plt.savefig(fig9_path, dpi=300)
plt.close()
print(f"圖表已儲存：{fig9_path}")

# ==============================================================================
# Phase 6: Deployment (部署與商業應用)
# ==============================================================================
print("\n" + "=" * 50)
print("【Phase 6: Deployment 部署評估與單筆房價預測實作】")
print("=" * 50)

def predict_single_house(house_dict):
    """
    接收單一房屋特徵，回傳預測售價及 95% 信賴區間與 95% 預測區間
    """
    # 按照 selected_features 建立 DataFrame
    house_df = pd.DataFrame([house_dict])[selected_features]
    house_sm = sm.add_constant(house_df, has_constant='add')
    pred = ols_mlr.get_prediction(house_sm).summary_frame(alpha=0.05)
    
    mean_val = pred['mean'].values[0]
    ci_low = pred['mean_ci_lower'].values[0]
    ci_high = pred['mean_ci_upper'].values[0]
    pi_low = pred['obs_ci_lower'].values[0]
    pi_high = pred['obs_ci_upper'].values[0]
    
    print("\n--- [金郡房屋即時線上估價結果] ---")
    print(f"房屋基本參數: {house_dict['bedrooms']}房 {house_dict['bathrooms']}衛, 面積: {house_dict['sqft_living']} sqft, 等級: {house_dict['grade']}")
    print(f"★ 模型點估計預測售價: ${mean_val:,.2f}")
    print(f"★ 95% 平均值信賴區間 (CI): [${ci_low:,.2f}, ${ci_high:,.2f}]")
    print(f"★ 95% 個別房價預測區間 (PI): [${pi_low:,.2f}, ${pi_high:,.2f}]")
    return mean_val, (ci_low, ci_high), (pi_low, pi_high)

# 測試一筆典型三房房屋資料
sample_house = {
    'bedrooms': 3,
    'bathrooms': 2.0,
    'sqft_living': 1800,
    'sqft_lot': 6000,
    'floors': 1.5,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'grade': 7,
    'dist_to_seattle': 0.15,
    'lat': 47.55,
    'long': -122.25,
    'sqft_living15': 1750,
    'sqft_lot15': 5800,
    'house_age': 25,
    'is_renovated': 0
}

predict_single_house(sample_house)

print("\n" + "=" * 80)
print("作業二主程式執行完畢！所有統計圖表已儲存於 figures/ 資料夾。")
print("=" * 80)
