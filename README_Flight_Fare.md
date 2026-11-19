# Flight Fare Prediction

**Notebook:** `FlightFare_FirstPerson.ipynb` · **Stack:** Python, pandas, scikit-learn

## Goal
Predict the price (₹) of a domestic flight in India from airline, route, timing, duration and stops.

## Data
10,683 flights (March–June 2019). Only `Price` is numeric at the start, so dates, times and durations had to be parsed.

## What I did
1. Cleaned the data (one row with missing values dropped) and engineered features from date, departure/arrival time, duration and stops.
2. Explored price by airline, stops and duration; used a log-transformed target.
3. Built a leak-free scikit-learn pipeline (encoding and scaling fitted on training data only).
4. Compared seven models with 5-fold CV and a holdout set: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, plus tuned versions of the last three (randomized search). The winner was chosen on cross-validated MAE, not on the holdout, so the holdout stays an unbiased estimate.

## Results (holdout)
| Model | CV R² | Holdout R² | Holdout MAE (₹) |
|---|---|---|---|
| **Random Forest** | 0.869 | **0.910** | **602** |
| Random Forest (tuned) | 0.871 | 0.910 | 609 |
| Gradient Boosting (tuned) | 0.879 | 0.907 | 663 |
| Decision Tree | 0.812 | 0.860 | 716 |
| Linear Regression | 0.745 | 0.743 | 1,435 |

Best model: **Random Forest**, holdout R² 0.910, RMSE ₹1,384, MAPE 6.8%.

## Conclusion
Tree-based models clearly beat Linear Regression because price depends on interactions (airline × stops × duration). Tuning added almost nothing over the default Random Forest.

## Limitations
LightGBM and XGBoost were optional and skipped (not installed in my run). Tuned CV scores are slightly optimistic because the same data guided the search. The data covers only four months of 2019.
