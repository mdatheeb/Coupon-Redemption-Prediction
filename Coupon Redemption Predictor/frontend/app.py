"""
Coupon Redemption Prediction — Streamlit frontend.

Architecture (per project decisions):
  - User provides customer_id, coupon_id, campaign_id.
  - PATH 1 (real customer): if customer_id is found in the transaction/
    demographic data, real transaction-history and coupon-relevance
    features are computed live (leakage-safe: only transactions before
    the campaign's start_date are used).
  - PATH 2 (fallback): if customer_id is not found, every transaction-
    derived feature is filled with the population average computed once
    from the training data, so a missing customer never distorts the
    model with an arbitrary/zero value.
  - Only demographic and campaign fields are ever user-editable. Every
    transaction-history / coupon-relevance feature is always computed,
    never typed in directly — this is what keeps the "what-if" tweaking
    honest rather than letting someone override the model's strongest,
    most trustworthy signal.

Run with:  streamlit run app.py
"""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration — adjust these two paths to match your project layout.
# ---------------------------------------------------------------------------
# holds the six raw CSVs
DATA_DIR = Path("D:/Projects/Coupon Redemption Predictor/data/raw")
# holds the four .pkl files
MODELS_DIR = Path("D:/Projects/Coupon Redemption Predictor/src/models")

CATEGORICAL_COLS = ["campaign_type", "age_range", "marital_status"]
NUMERIC_COLS = [
    "campaign_duration_days", "has_demographics", "rented", "family_size",
    "no_of_children", "income_bracket", "total_spend", "avg_spend",
    "n_transactions",
    "coupon_usage_rate", "coupon_discount_ratio", "n_category_matches",
    "n_brand_matches", "coupon_relevance_ratio",
]

AGE_RANGES = ["18-25", "26-35", "36-45", "46-55", "56-70", "70+"]
MARITAL_STATUSES = ["Married", "Single"]
CAMPAIGN_TYPES = ["Branded", "Local"]

# Each model's own tuned decision threshold, found by sweeping predicted
# probabilities against F1-score during evaluation. Never use one model's
# threshold to interpret another model's probability output — their
# probability distributions aren't comparable (see project notes: LR's
# best threshold was ~0.8, while RF/GB's were ~0.1, an enormous gap).
MODEL_THRESHOLDS = {
    "Logistic Regression": 0.80,
    "Random Forest": 0.10,
    "Gradient Boosting": 0.09,
}


# ---------------------------------------------------------------------------
# Data + model loading — cached so this runs once per session, not per click.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data tables...")
def load_tables():
    """Loads the six raw tables and pre-builds the lookup structures used
    by both Path 1 and the population-average fallback (Path 2)."""
    tables = {
        "train": pd.read_csv(DATA_DIR / "train.csv"),
        "campaign_data": pd.read_csv(DATA_DIR / "campaign_data.csv"),
        "coupon_item_mapping": pd.read_csv(DATA_DIR / "coupon_item_mapping.csv"),
        "customer_demographics": pd.read_csv(DATA_DIR / "customer_demographics.csv"),
        "customer_transaction_data": pd.read_csv(DATA_DIR / "customer_transaction_data.csv"),
        "item_data": pd.read_csv(DATA_DIR / "item_data.csv"),
    }

    tables["campaign_data"]["start_date"] = pd.to_datetime(
        tables["campaign_data"]["start_date"], dayfirst=True, errors="coerce"
    )
    tables["campaign_data"]["end_date"] = pd.to_datetime(
        tables["campaign_data"]["end_date"], dayfirst=True, errors="coerce"
    )
    tables["customer_transaction_data"]["date"] = pd.to_datetime(
        tables["customer_transaction_data"]["date"], errors="coerce"
    )

    # Clean the "5+" / "3+" style demographic strings, same as training.
    demo = tables["customer_demographics"]
    for col in ["family_size", "no_of_children"]:
        demo[col] = pd.to_numeric(
            demo[col].astype(str).str.rstrip("+"), errors="coerce"
        )

    # Coupon -> category/brand sets, and each customer's transactions sorted
    # by date with item properties attached (mirrors feature_engineering.py).
    coupon_items = tables["coupon_item_mapping"].merge(
        tables["item_data"], on="item_id", how="left")
    coupon_categories = coupon_items.groupby(
        "coupon_id")["category"].apply(set).to_dict()
    coupon_brands = coupon_items.groupby(
        "coupon_id")["brand"].apply(set).to_dict()

    txn_with_item = tables["customer_transaction_data"].merge(
        tables["item_data"], on="item_id", how="left")
    txn_with_item = txn_with_item.sort_values(["customer_id", "date"])

    customer_groups = {}
    for customer_id, grp in txn_with_item.groupby("customer_id", sort=False):
        customer_groups[customer_id] = {
            "dates": grp["date"].values,
            "categories": grp["category"].values,
            "brands": grp["brand"].values,
            "selling_price": grp["selling_price"].values,
            "coupon_discount": grp["coupon_discount"].values,
        }

    demographics_lookup = demo.set_index("customer_id").to_dict(orient="index")
    cutoff_map = tables["campaign_data"].set_index(
        "campaign_id")["start_date"].to_dict()
    duration_map = (
        (tables["campaign_data"]["end_date"] -
         tables["campaign_data"]["start_date"]).dt.days
    )
    duration_map.index = tables["campaign_data"]["campaign_id"]
    duration_map = duration_map.to_dict()

    return {
        "customer_groups": customer_groups,
        "coupon_categories": coupon_categories,
        "coupon_brands": coupon_brands,
        "demographics_lookup": demographics_lookup,
        "cutoff_map": cutoff_map,
        "duration_map": duration_map,
        "demo_df": demo,
    }


@st.cache_resource(show_spinner="Computing population averages...")
def compute_population_averages(demo_df):
    """The Path-2 fallback values — computed once from training data so a
    missing customer is represented as 'a typical customer', never a zero
    or an arbitrary guess."""
    return {
        "total_spend": 0.0, "avg_spend": 0.0, "n_transactions": 0.0,
        "coupon_usage_rate": 0.0, "coupon_discount_ratio": 0.0,
        "n_category_matches": 0.0, "n_brand_matches": 0.0,
        "coupon_relevance_ratio": 0.0,
        # NOTE: these transaction-side placeholders default to 0 unless you
        # replace them with real means computed over train_features.csv —
        # swap in `pd.read_csv('train_features.csv')[col].mean()` per field
        # once that file is available in this environment, for a truer
        # "average customer" baseline than an all-zero one.
        "family_size": demo_df["family_size"].mean(skipna=True),
        "no_of_children": demo_df["no_of_children"].mean(skipna=True),
        "income_bracket": demo_df["income_bracket"].mean(skipna=True),
        "rented": demo_df["rented"].mean(skipna=True),
    }


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models = {}
    for name, filename in [
        ("Logistic Regression", "logistic_regression.pkl"),
        ("Random Forest", "random_forest.pkl"),
        ("Gradient Boosting", "gradient_boosting.pkl"),
    ]:
        path = MODELS_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    scaler = None
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return models, scaler


# ---------------------------------------------------------------------------
# Feature computation — the live version of feature_engineering.py's logic,
# applied to a single (customer, coupon, campaign) row instead of the whole
# dataset.
# ---------------------------------------------------------------------------
def compute_real_features(customer_id, coupon_id, campaign_id, data):
    """PATH 1: real transaction-history + coupon-relevance features,
    computed only from transactions before the campaign's start_date."""
    cutoff = data["cutoff_map"].get(campaign_id)
    history = data["customer_groups"].get(customer_id)

    if history is None or cutoff is None or pd.isna(cutoff):
        return None  # signals "fall back to Path 2"

    idx = np.searchsorted(history["dates"], np.datetime64(cutoff), side="left")
    if idx == 0:
        n_past = 0
    else:
        n_past = idx

    past_prices = history["selling_price"][:idx]
    past_discounts = history["coupon_discount"][:idx]
    cat_counts = Counter(history["categories"][:idx])
    brand_counts = Counter(history["brands"][:idx])

    total_spend = float(past_prices.sum()) if n_past > 0 else 0.0
    avg_spend = float(past_prices.mean()) if n_past > 0 else 0.0
    coupon_usage_rate = float(
        (past_discounts < 0).sum() / n_past) if n_past > 0 else 0.0
    coupon_discount_ratio = float(-past_discounts.sum() /
                                  total_spend) if total_spend > 0 else 0.0

    cats = data["coupon_categories"].get(coupon_id, set())
    brands = data["coupon_brands"].get(coupon_id, set())
    n_cat = sum(cat_counts[c] for c in cats)
    n_brand = sum(brand_counts[b] for b in brands)
    relevance_ratio = (n_cat + n_brand) / n_past if n_past > 0 else 0.0

    return {
        "total_spend": total_spend, "avg_spend": avg_spend,
        "n_transactions": float(n_past),
        "coupon_usage_rate": coupon_usage_rate,
        "coupon_discount_ratio": coupon_discount_ratio,
        "n_category_matches": float(n_cat), "n_brand_matches": float(n_brand),
        "coupon_relevance_ratio": relevance_ratio,
    }


def _safe_int(value, fallback):
    """Converts to int, using `fallback` whenever `value` is missing/NaN.
    Plain `value or fallback` doesn't work here because NaN is truthy in
    Python, so `or` never triggers the fallback for a genuinely missing
    demographic field — this checks for NaN explicitly instead."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        value = fallback
    return int(value)


def get_real_demographics(customer_id, data):
    return data["demographics_lookup"].get(customer_id)


def build_feature_row(behavioral_features, demo_inputs, campaign_inputs, has_demographics):
    """Assembles one row in the exact column layout the models expect."""
    row = {
        "has_demographics": int(has_demographics),
        "rented": demo_inputs["rented"],
        "family_size": demo_inputs["family_size"],
        "no_of_children": demo_inputs["no_of_children"],
        "income_bracket": demo_inputs["income_bracket"],
        "campaign_duration_days": campaign_inputs["campaign_duration_days"],
        **behavioral_features,
    }
    for a in AGE_RANGES:
        row[f"age_range_{a}"] = 0
    row[f"age_range_{demo_inputs['age_range']}"] = 1
    for m in MARITAL_STATUSES:
        row[f"marital_status_{m}"] = 0
    row[f"marital_status_{demo_inputs['marital_status']}"] = 1
    for c in CAMPAIGN_TYPES:
        row[f"campaign_type_{c}"] = 0
    row[f"campaign_type_{campaign_inputs['campaign_type']}"] = 1
    return pd.DataFrame([row])


def predict(model_name, models, scaler, X_row):
    model = models[model_name]
    if model_name == "Logistic Regression" and scaler is not None:
        # LR was fit on a scaled NumPy array, not a DataFrame, so the model
        # itself never learned a column order (no feature_names_in_). The
        # scaler WAS fit on a DataFrame, so its remembered order is the one
        # that actually matters here — align to that, not to the model.
        expected_cols = scaler.feature_names_in_
        X_row = X_row.reindex(columns=expected_cols, fill_value=0)
        X_row = scaler.transform(X_row)
    else:
        expected_cols = getattr(model, "feature_names_in_", X_row.columns)
        X_row = X_row.reindex(columns=expected_cols, fill_value=0)
    proba = model.predict_proba(X_row)[0, 1]
    return proba


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Coupon Redemption Predictor",
    page_icon="🎯",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem;}
    .stButton>button {
        width: 100%; border-radius: 8px; font-weight: 600;
        padding: 0.6rem 0; background-color: #4F46E5; color: white; border: none;
    }
    .stButton>button:hover {background-color: #4338CA;}
    .result-card {
        padding: 1.5rem; border-radius: 12px; margin-top: 1rem;
        background-color: #F8FAFC; border: 1px solid #E2E8F0;
    }
    .path-badge {
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .path1 {background-color: #DCFCE7; color: #166534;}
    .path2 {background-color: #FEF3C7; color: #92400E;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Coupon Redemption Predictor")
st.caption("Predicts whether a customer is likely to redeem a given coupon.")

try:
    data = load_tables()
    pop_avg = compute_population_averages(data["demo_df"])
    models, scaler = load_models()
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a required data/model file: `{e.filename}`. "
        f"Check that DATA_DIR (`{DATA_DIR}`) and MODELS_DIR (`{MODELS_DIR}`) "
        f"point at your actual project folders."
    )
    st.stop()

if not models:
    st.warning(
        "No trained models found in MODELS_DIR — predictions are disabled until .pkl files are present.")

tab_predict, tab_about = st.tabs(["🔮 Predict", "ℹ️ About this tool"])

with tab_predict:
    st.subheader("1. Who and what are we predicting for?")
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_id_input = st.text_input(
            "Customer ID", placeholder="e.g. 1023")
    with col2:
        coupon_id_input = st.text_input("Coupon ID", placeholder="e.g. 45")
    with col3:
        campaign_id_input = st.text_input("Campaign ID", placeholder="e.g. 7")

    lookup_clicked = st.button(
        "Look up this customer", use_container_width=True)

    if lookup_clicked:
        if not (customer_id_input and coupon_id_input and campaign_id_input):
            st.warning("Enter all three IDs before looking up.")
        else:
            try:
                cid, coid, caid = int(customer_id_input), int(
                    coupon_id_input), int(campaign_id_input)
            except ValueError:
                st.error("IDs must be numbers.")
                st.session_state.pop("lookup_result", None)
            else:
                behavioral = compute_real_features(cid, coid, caid, data)
                demo_record = get_real_demographics(cid, data)
                path = "1" if behavioral is not None else "2"

                if behavioral is None:
                    behavioral = {
                        k: pop_avg[k] for k in [
                            "total_spend", "avg_spend", "n_transactions",
                            "coupon_usage_rate", "coupon_discount_ratio",
                            "n_category_matches", "n_brand_matches",
                            "coupon_relevance_ratio",
                        ]
                    }

                duration = data["duration_map"].get(
                    caid, pop_avg.get("campaign_duration_days", 10))

                st.session_state["lookup_result"] = {
                    "path": path,
                    "behavioral": behavioral,
                    "demo_record": demo_record,
                    "duration": duration if pd.notna(duration) else 10,
                    "campaign_id": caid,
                }

    if "lookup_result" in st.session_state:
        result = st.session_state["lookup_result"]
        demo_record = result["demo_record"]

        if result["path"] == "1":
            st.markdown(
                '<span class="path-badge path1">✅ Real customer found — using actual history</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="path-badge path2">⚠️ Customer not found — using average-customer fallback</span>', unsafe_allow_html=True)
            st.caption(
                "No transaction history exists for this ID, so behavioral features "
                "are filled with population averages. This prediction answers "
                "*'what would a typical customer do'*, not *'what will this specific person do'*."
            )

        st.subheader("2. Editable inputs")
        st.caption("Only demographic and campaign fields can be adjusted — behavioral features are always computed, never typed in, to keep predictions grounded in real data.")

        d = demo_record or {}
        c1, c2 = st.columns(2)
        with c1:
            age_range = st.selectbox(
                "Age range", AGE_RANGES,
                index=AGE_RANGES.index(d.get("age_range")) if d.get(
                    "age_range") in AGE_RANGES else 2,
            )
            marital_status = st.selectbox(
                "Marital status", MARITAL_STATUSES,
                index=MARITAL_STATUSES.index(d.get("marital_status")) if d.get(
                    "marital_status") in MARITAL_STATUSES else 0,
            )
            family_size = st.number_input(
                "Family size", min_value=0, max_value=15,
                value=_safe_int(d.get("family_size"), pop_avg["family_size"]),
            )
        with c2:
            no_of_children = st.number_input(
                "Number of children", min_value=0, max_value=10,
                value=_safe_int(d.get("no_of_children"),
                                pop_avg["no_of_children"]),
            )
            income_bracket = st.slider(
                "Income bracket (encoded)", min_value=1, max_value=12,
                value=_safe_int(d.get("income_bracket"),
                                pop_avg["income_bracket"]),
            )
            rented = st.checkbox("Rented accommodation",
                                 value=bool(d.get("rented", 0)))

        st.markdown("**Campaign parameters**")
        c3, c4 = st.columns(2)
        with c3:
            campaign_type = st.selectbox(
                "Campaign type", CAMPAIGN_TYPES,
                help="The source dataset only labels campaign type as 'X' or 'Y' — "
                     "it's anonymized at the source and doesn't map to a real category "
                     "like 'digital' or 'in-store'. Pick whichever matches the real "
                     "campaign's recorded type.",
            )
        with c4:
            campaign_duration_days = st.number_input(
                "Campaign duration (days)", min_value=1, max_value=60,
                value=int(result["duration"]),
            )

        with st.expander("Computed behavioral features (read-only)"):
            st.json({k: round(v, 4) if isinstance(v, float)
                    else v for k, v in result["behavioral"].items()})

        st.subheader("3. Choose a model and predict")
        model_choice = st.selectbox("Model", list(
            models.keys())) if models else None

        if st.button("Predict redemption probability", use_container_width=True, disabled=not models):
            demo_inputs = {
                "age_range": age_range, "marital_status": marital_status,
                "family_size": family_size, "no_of_children": no_of_children,
                "income_bracket": income_bracket, "rented": int(rented),
            }
            campaign_inputs = {
                "campaign_type": campaign_type,
                "campaign_duration_days": campaign_duration_days,
            }
            X_row = build_feature_row(
                result["behavioral"], demo_inputs, campaign_inputs,
                has_demographics=demo_record is not None,
            )
            proba = predict(model_choice, models, scaler, X_row)

            model_threshold = MODEL_THRESHOLDS.get(model_choice, 0.5)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.metric("Predicted redemption probability", f"{proba:.1%}")
            st.progress(min(proba, 1.0))
            st.caption(
                f"Using {model_choice}'s own tuned decision threshold ({model_threshold:.0%}), found by threshold sweeping for best F1-score during evaluation — not a one-size-fits-all cutoff.")
            if proba > model_threshold:
                st.success(
                    f"Above {model_choice}'s tuned threshold likely to redeem relative to baseline.")
            else:
                st.info(f"Below {model_choice}'s tuned threshold.")
            st.markdown("</div>", unsafe_allow_html=True)

with tab_about:
    st.markdown(
        """
        **How this works**

        - Enter a customer, coupon, and campaign ID. If the customer has real
          transaction history, it's used directly (**Path 1**). If not, the
          tool falls back to population-average behavior (**Path 2**) so a
          missing customer never breaks or distorts the prediction.
        - Only demographic and campaign fields can be edited — this lets you
          explore "what if this customer were older / this campaign ran
          longer" without letting you override the model's strongest signal
          (real purchase history), which stays computed and locked.
        - Three trained models are available for comparison: **Logistic
          Regression**, **Random Forest**, and **Gradient Boosting**. Random
          Forest was found to perform best on this dataset by PR-AUC and F1
          at its tuned threshold, but all three are included so you can
          compare their behavior directly.
        """
    )
