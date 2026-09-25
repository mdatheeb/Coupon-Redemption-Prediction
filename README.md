# Coupon Redemption Prediction

Predicts whether a customer will redeem a specific coupon, using a real-world relational retail dataset (customer demographics, transaction history, campaign metadata, and coupon-item mappings). Three classification models — Logistic Regression, Random Forest, and Gradient Boosting — were trained and compared, with a Streamlit app to serve predictions interactively.

## Project structure

```
Coupon Redemption Predictor/
├── data/                        # raw CSVs (train, test, campaign_data, coupon_item_mapping,
│                                 #   customer_demographics, customer_transaction_data, item_data)
├── notebook/
│   └── feature_engineering.ipynb   # data cleaning, joins, feature engineering, model training
├── src/
│   └── models/                  # trained, pickled models + fitted scaler
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       ├── gradient_boosting.pkl
│       └── scaler.pkl
├── frontend/
│   └── app.py                   # Streamlit prediction app
├── requirements.txt
└── README.md
```

## Setup

```
pip install -r requirements.txt
```

`scikit-learn` is pinned to `1.6.1` in `requirements.txt` — this must match the version the models in `src/models/` were trained under, or loading the pickled models will fail with a `ModuleNotFoundError: No module named '_loss'` (an internal module that moved between sklearn versions).

## Running the app

From the project root:

```
streamlit run frontend/app.py
```

The app expects `data/` and `src/models/` to exist as siblings of `frontend/` — it uses relative paths (`./data`, `./src/models`) that resolve correctly only when Streamlit is launched from the project root, as shown above.

## How prediction works

The app takes a **customer ID**, **coupon ID**, and **campaign ID**, then follows one of two paths:

- **Path 1 (real customer):** if transaction history exists for the given customer, behavioral features (spend, coupon usage rate, category/brand relevance to the specific coupon) are computed live from that real history, using only transactions before the campaign's start date (no leakage from the future).
- **Path 2 (fallback):** if no transaction history is found, behavioral features fall back to population averages, and the app flags clearly that the prediction reflects "a typical customer" rather than this specific individual.

Only demographic and campaign-level fields (age range, marital status, family size, income bracket, campaign type, campaign duration) are user-editable. Transaction-derived features are always computed, never manually entered, to keep predictions grounded in real behavior rather than arbitrary guesses.

Each model uses its own independently tuned decision threshold (found by sweeping for best F1-score), not a single shared cutoff:

| Model | Tuned threshold |
|---|---|
| Logistic Regression | 0.80 |
| Random Forest | 0.10 |
| Gradient Boosting | 0.09 |

## Feature engineering

Two feature families drive prediction:

- **Transaction-history features** — total spend, average spend, transaction count, coupon usage rate, coupon discount ratio.
- **Coupon-relevance features** — how much a customer's past purchases overlap with the categories/brands a specific coupon covers (`n_category_matches`, `n_brand_matches`, `coupon_relevance_ratio`).

All historical features are computed using only transactions strictly before each campaign's start date, to avoid leaking future information into training.

## Model results

Redemption is rare (~1% of impressions), so models were evaluated on ROC-AUC and PR-AUC rather than accuracy, with each model's threshold tuned independently.

| Model | ROC-AUC | PR-AUC | Best F1 (redemption) | Best threshold |
|---|---|---|---|---|
| Logistic Regression | 0.798 | 0.043 | ~0.11 | ~0.80 |
| **Random Forest** | 0.914 | **0.141** | **0.273** | 0.10 |
| Gradient Boosting | **0.926** | 0.135 | 0.240 | 0.09 |

**Random Forest** is the recommended model for this problem — it has the highest PR-AUC and F1-score, the metrics most sensitive to correctly identifying the rare positive class, despite Gradient Boosting having a marginally higher ROC-AUC.

`coupon_relevance_ratio` and `coupon_usage_rate` consistently ranked as the strongest predictors across all three models; demographic features (age, income, marital status) contributed comparatively little.

## Known limitations

- A single train/test split was used; results carry some variance given only ~146 positive test examples. K-fold cross-validation would give a more robust estimate.
- Path 2's fallback is scoped to customers the organization already has *some* record of failing to match a transaction history — genuinely new customers with zero history remain an open problem outside this app's scope.
- `campaign_type` is anonymized at the source (`X`/`Y`) with no real-world category attached — this is a property of the original dataset, not a simplification in this app.
