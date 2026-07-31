# HDB Resale Price Estimator

Machine learning solution predicting Singapore HDB resale flat prices, built for
**CAI2C08 Machine Learning for Developers** (Temasek Polytechnic, Diploma in Applied AI).

**Live app:** https://timyimldpproject-gxcen3dkye5iqiddzqky7n.streamlit.app/

---

## Problem

HDB resale prices vary widely by location, size, storey and remaining lease, making it
hard for buyers, sellers and agents to judge whether an asking price is fair. This project
trains a regression model on real transaction data and deploys it as a web app that
returns an instant, evidence-based valuation.

## Dataset

Resale Flat Prices Based on Registration Date from Jan 2017 Onwards — Housing &
Development Board, via [data.gov.sg](https://data.gov.sg).

- 236,609 transactions, 11 columns, January 2017 to July 2026
- 236,293 rows after removing 316 exact duplicates
- No missing values

## Results

All figures on the same 47,259 held-out test transactions.

| Stage | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| 1. Baseline (mean prediction) | $150,249 | $191,023 | 0.0000 | 31.82% |
| 2. HistGradientBoosting, baseline features | $72,147 | $88,589 | 0.7849 | 14.37% |
| 3. + feature engineering | $32,453 | $44,739 | 0.9451 | 6.22% |
| **4. + hyperparameter tuning (final)** | **$24,315** | **$34,482** | **0.9674** | **4.66%** |

**83.8% reduction in mean absolute error versus the baseline.** The final model is
typically within 4.66% of the true sale price.

## Approach

**Model selection.** Four algorithms compared against a `DummyRegressor` baseline:
Linear Regression, Random Forest and HistGradientBoosting. HistGradientBoosting won on
all four metrics. Because boosting fits each tree to the residual errors of the previous
trees, it beat Random Forest by more on RMSE than on MAE — it corrects the hard, expensive
cases rather than averaging over them.

**Feature engineering** (the single largest contributor, 55% MAE reduction):

| Feature | Derived from |
|---|---|
| `remaining_lease_months` | `"61 years 04 months"` → 736 |
| `storey_mid` | `"10 TO 12"` → 11.0, preserving floor ordering that one-hot encoding destroyed |
| `months_since_2017` | `"2017-01"` → market-timing index, capturing the post-2020 price climb |
| `flat_age` | age at point of sale rather than an absolute lease commencement year |

Permutation importance confirms three of the top four predictors are engineered features.
The final set uses 59 columns versus 73 for the baseline — better accuracy on fewer features.

**Tuning.** `RandomizedSearchCV`, 8 combinations × 3-fold CV. Best:
`max_iter=1000, max_leaf_nodes=127, learning_rate=0.05`.

## Repository

| File | Purpose |
|---|---|
| `MLDP Program Codes Submission Template.ipynb` | Full analysis: EDA, preparation, modelling, evaluation, iteration |
| `app.py` | Streamlit web application |
| `hdb_resale_model.pkl` | Tuned model plus feature column order and dropdown metadata (6.1 MB) |
| `requirements.txt` | Pinned dependencies |
| `Resaleflatprices...csv` | Source dataset |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads only `hdb_resale_model.pkl`, so the 23 MB CSV is not needed at runtime.

## Limitations

- Estimates market value under prevailing conditions; it is a valuation aid, not a
  forecast of future prices.
- Less reliable for 1-room and multi-generation flats, which together account for fewer
  than 100 of the 236,293 training transactions. The app warns users when these are selected.
- Location is captured at town level; `block` and `street_name` were excluded as too
  high-cardinality (2,773 and 578 unique values).

---

Koh Tim Yi (2503677C) · AY2026/2027 April Semester
