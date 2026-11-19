# city-solar-forecast
Monthly solar PV yield forecasting (kWh/kW) and 10 kW payback analysis for (I used Sargodha) using NASA POWER data
---

## 1. Data & Feature Engineering
- **Data Source:** NASA POWER Agroclimatology Daily API (`2001-01-01` to `2026-10-31`, 9,454 daily observations).
- **Target (`y`):** Monthly specific yield (`kWh/kW/month`), converted using a **0.80 Performance Ratio (PR)** and filtered to complete months (`count >= 25` days, excluding partial November 2026).
- **4-Month Forecast Horizon Setup:** Since predicting `December 2026` through `February 2027` from `October 2026` requires a horizon of up to 4 months, autoregressive lags start at `lag4` (`shift(4)`) alongside `lag12`, `lag24`, and leakage-free expanding monthly climatology (`shift(1).expanding().mean()`).
- **Added Feature (`sin_month`, `cos_month`):** Added cyclical month encoding ($\sin(2\pi \cdot \text{month}/12)$ and $\cos(2\pi \cdot \text{month}/12)$) so tree splits treat December (`12`) and January (`1`) as adjacent winter months.

---

## 2. 5-Year Walk-Forward Validation Results (`TimeSeriesSplit`)
Evaluated out-of-sample across **5 consecutive 12-month test folds** (`2021-11` through `2026-10`, 60 test months total):

| Model | Features Used | 5-Fold Mean MAE (`kWh/kW/month`) | Approx. Error (%) |
| :--- | :--- | :---: | :---: |
| **Climatology (Winner)** | Expanding historical calendar-month average (`clim`) | **7.87** | **~6.3%** |
| **XGBoost (Base)** | `month`, `lag4`, `lag12`, `lag24`, `clim` | 7.87 | ~6.3% |
| **XGBoost (+ Cyclical Month)** | Base + `sin_month`, `cos_month` | 7.89 | ~6.3% |
| **Seasonal Naive** | Same month prior year (`lag12`) | 11.05 | ~8.9% |

---

## 3. 3-Month Out-of-Sample Forecast (`Dec 2026 – Feb 2027`)

| Target Month | Forecast (`kWh/kW`) | Expected 10 kW Output (`kWh`) | Selected Method | Generated Date |
| :--- | :---: | :---: | :--- | :--- |
| **2026-12** | **75.69** | 756.9 | Climatology (`2001–2026` mean) | `2026-11-19` |
| **2027-01** | **75.53** | 755.3 | Climatology (`2001–2026` mean) | `2026-11-19` |
| **2027-02** | **88.49** | 884.9 | Climatology (`2001–2026` mean) | `2026-11-19` |

---

## 4. Sargodha 10 kW On-Grid Solar Payback Summary

| Parameter | Value | Basis / Assumptions |
| :--- | :---: | :--- |
| **Location** | Sargodha (`32.0836° N, 72.6711° E`) | NASA POWER `2001–2025` complete years |
| **Average Annual Specific Yield** | **1,496.9 kWh/kW/year** | NASA irradiance × `0.80` Performance Ratio |
| **System Size** | **10 kW** | Standard on-grid net-metered rooftop system |
| **Expected Annual Generation** | **14,969 kWh/year** | `10 kW × 1,496.9 kWh/kW/year` |
| **System Price (Installed)** | **PKR 1,250,000** | ~PKR 125/W turnkey (panels, inverter, green meter) |
| **Grid Import Tariff Offset** | **PKR 65 / kWh** | FESCO domestic slab + taxes/FPA (55% self-consumed = `8,233 kWh`) |
| **Grid Export Rate (Buyback)** | **PKR 32 / kWh** | NEPRA/FESCO net-metering export credit (45% exported = `6,736 kWh`) |
| **Total Annual Savings** | **PKR 750,696 / year** | `(8,233 × PKR 65) + (6,736 × PKR 32)` *(~PKR 50.15/kWh blended)* |
| **Simple Payback Period** | **1.67 years (~20 months)** | `PKR 1,250,000 / PKR 750,696` |

---

## 5. Key Takeaways (Methodology & Model Selection)
- **Why a random split is wrong here:** Randomly shuffling months puts future months into the training set and past months into the test set, leaking future weather regimes backward in time and producing unrealistically low error that fails in deployment. Walk-forward validation (`TimeSeriesSplit`) strictly trains on past years and evaluates on unseen future 12-month blocks.
- **Why Climatology won (and why XGBoost didn't beat it):** At a 4-month forecast horizon, monthly solar generation in Sargodha is governed almost entirely by repeatable seasonal geometry (`clim`), while residual month-to-month anomalies (winter fog or dust storms) are unforecastable weather noise. **Climatology (`7.87` MAE)** beats **Seasonal Naive (`11.05` MAE)** by `3.18 kWh/kW/month` and matches **XGBoost (`7.87` MAE)** with zero tree complexity and zero overfitting risk. Adding cyclical month encoding (`sin_month`, `cos_month`) yielded `7.89` MAE because expanding climatology already captures the full 12-month seasonal curve.

---

## 6. How to Run
```bash
pip install pandas numpy requests scikit-learn xgboost
jupyter notebook sargodha-solar-forecast.ipynb
