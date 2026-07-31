"""
HDB Resale Price Estimator - Streamlit Web Application
CAI2C08 Machine Learning for Developers | Koh Tim Yi (2503677C)

Loads the tuned HistGradientBoostingRegressor exported from the notebook and turns it
into an interactive valuation tool for buyers, sellers and property agents.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HDB Resale Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79 0%, #2e86ab 100%);
            padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        }
        .main-header h1 { color: #ffffff; margin: 0; font-size: 2.1rem; }
        .main-header p  { color: #d8e8f5; margin: 0.4rem 0 0 0; font-size: 1.02rem; }
        .price-card {
            background: linear-gradient(135deg, #0f5132 0%, #198754 100%);
            padding: 1.8rem; border-radius: 12px; text-align: center; color: #ffffff;
        }
        .price-card .label  { font-size: 1rem; opacity: 0.85; letter-spacing: 0.5px; }
        .price-card .value  { font-size: 3rem; font-weight: 700; margin: 0.3rem 0; }
        .price-card .range  { font-size: 1rem; opacity: 0.9; }
        .metric-box {
            background: #f1f5f9; border-left: 4px solid #2e86ab;
            padding: 0.9rem 1.1rem; border-radius: 6px; margin-bottom: 0.6rem;
        }
        .footer-note { color: #64748b; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load the exported model artifact
# cache_resource keeps the 6 MB model in memory across reruns instead of reloading
# it every time the user moves a slider.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    path = Path(__file__).parent / "hdb_resale_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


artifact = load_artifact()

if artifact is None:
    st.error(
        "**Model file not found.** `hdb_resale_model.pkl` is missing from the application "
        "folder. Run Section 6.4 of the notebook to generate it, then redeploy."
    )
    st.stop()

MODEL = artifact["model"]
FEATURE_COLUMNS = artifact["feature_columns"]
METRICS = artifact["metrics"]
MAE = METRICS["MAE"]

# Flat types with very few training transactions - predictions here are less reliable
RARE_FLAT_TYPES = {"1 ROOM", "MULTI-GENERATION"}
LEASE_TOTAL_MONTHS = 99 * 12  # HDB flats are sold on 99-year leases


# ---------------------------------------------------------------------------
# Feature construction
# The notebook one-hot encoded with pd.get_dummies OUTSIDE a pipeline, so the app must
# rebuild the identical column layout for a single row. We start from an all-zero frame
# using the saved column order, then fill in the values - this guarantees the columns
# line up exactly with what the model was trained on.
# ---------------------------------------------------------------------------
def build_feature_row(town, flat_type, flat_model, storey_mid, floor_area,
                      remaining_lease_months, flat_age, months_since_2017):
    row = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLUMNS)

    row.loc[0, "floor_area_sqm"] = float(floor_area)
    row.loc[0, "storey_mid"] = float(storey_mid)
    row.loc[0, "remaining_lease_months"] = float(remaining_lease_months)
    row.loc[0, "flat_age"] = float(flat_age)
    row.loc[0, "months_since_2017"] = float(months_since_2017)

    for prefix, value in (("town", town), ("flat_type", flat_type), ("flat_model", flat_model)):
        column = f"{prefix}_{value}"
        if column in row.columns:
            row.loc[0, column] = 1.0

    return row


def storey_midpoint(storey_range):
    low, high = storey_range.split(" TO ")
    return (int(low) + int(high)) / 2


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🏠 HDB Resale Price Estimator</h1>
        <p>Data-driven valuations for Singapore HDB resale flats, trained on 236,293
           real transactions from January 2017 to July 2026.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar - model transparency
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this model")
    st.markdown(
        f"""
        **Algorithm**
        HistGradientBoostingRegressor (scikit-learn)

        **Trained on**
        236,293 resale transactions
        Source: data.gov.sg

        **Test set performance** *(47,259 unseen flats)*
        """
    )
    st.markdown(
        f"""
        <div class="metric-box"><b>Typical error (MAE)</b><br>${MAE:,.0f}</div>
        <div class="metric-box"><b>Average % error (MAPE)</b><br>{METRICS['MAPE']:.2f}%</div>
        <div class="metric-box"><b>Variance explained (R²)</b><br>{METRICS['R2']:.4f}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This tool estimates market value under current conditions. It is a valuation "
        "aid, not a forecast of future prices, and should not replace a professional "
        "valuation for financing purposes."
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("Tell us about the flat")

col1, col2, col3 = st.columns(3)

with col1:
    town = st.selectbox("Town", artifact["towns"],
                        index=artifact["towns"].index("ANG MO KIO")
                        if "ANG MO KIO" in artifact["towns"] else 0,
                        help="The HDB town the flat is located in.")
    flat_type = st.selectbox("Flat type", artifact["flat_types"],
                             index=artifact["flat_types"].index("4 ROOM")
                             if "4 ROOM" in artifact["flat_types"] else 0)

with col2:
    flat_model = st.selectbox("Flat model", artifact["flat_models"],
                              index=artifact["flat_models"].index("Model A")
                              if "Model A" in artifact["flat_models"] else 0,
                              help="The HDB design type, shown on the flat's title documents.")
    storey_range = st.selectbox("Storey range", artifact["storey_ranges"],
                                index=3 if len(artifact["storey_ranges"]) > 3 else 0,
                                help="Which band of floors the unit sits on.")

with col3:
    default_area = float(artifact["median_area_by_type"].get(flat_type, 93.0))
    floor_area = st.number_input(
        "Floor area (sqm)",
        min_value=float(artifact["floor_area_min"]),
        max_value=float(artifact["floor_area_max"]),
        value=default_area, step=1.0,
        help=f"Typical {flat_type} flats are around {default_area:.0f} sqm.")
    lease_commence_year = st.number_input(
        "Lease commencement year", min_value=1966, max_value=2026,
        value=1990, step=1,
        help="The year the flat's 99-year lease began.")

# Market timing. Defaults to the most recent month in the training data, i.e. present
# market conditions. Exposed as a slider so users can see how timing affects value.
st.markdown("")
valuation_month_index = st.slider(
    "Valuation point (months since January 2017)",
    min_value=0,
    max_value=int(artifact["max_months_since_2017"]),
    value=int(artifact["max_months_since_2017"]),
    help=("The model learned that HDB prices rose steeply from 2020 onwards. Leave this "
          "at the maximum for a present-day valuation, or drag it back to see what the "
          "same flat would have been worth earlier."),
)

valuation_year = 2017 + valuation_month_index // 12
valuation_month_name = pd.Period(
    f"2017-01", freq="M").asfreq("M") + valuation_month_index
st.caption(f"Valuing as at **{valuation_month_name}**  "
           f"(dataset covers Jan 2017 – {artifact['dataset_last_month']})")


# ---------------------------------------------------------------------------
# Derived features + validation
# ---------------------------------------------------------------------------
flat_age = valuation_year - int(lease_commence_year)
remaining_lease_months = LEASE_TOTAL_MONTHS - (flat_age * 12)
storey_mid = storey_midpoint(storey_range)

errors, warnings = [], []

if flat_age < 0:
    errors.append(
        f"The lease commencement year ({lease_commence_year}) is after the valuation "
        f"year ({valuation_year}). A flat cannot be sold before its lease begins — "
        "please choose an earlier commencement year or a later valuation point."
    )

if remaining_lease_months <= 0:
    errors.append(
        f"This flat's 99-year lease would already have expired by {valuation_year} "
        f"(commenced {lease_commence_year}, age {flat_age} years). HDB flats cannot be "
        "resold after lease expiry."
    )

if flat_type in RARE_FLAT_TYPES:
    warnings.append(
        f"**{flat_type}** flats make up fewer than 100 of the 236,293 training "
        "transactions. This estimate is considerably less reliable than for 3, 4 or "
        "5-room flats — treat it as indicative only."
    )

typical_area = artifact["median_area_by_type"].get(flat_type)
if typical_area and abs(floor_area - typical_area) > 40:
    warnings.append(
        f"{floor_area:.0f} sqm is unusual for a **{flat_type}** flat, which typically "
        f"measures around {typical_area:.0f} sqm. Double-check the floor area, as the "
        "estimate will be less reliable for combinations the model has rarely seen."
    )

for message in errors:
    st.error(message)
for message in warnings:
    st.warning(message)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.markdown("")
predict_clicked = st.button("💰 Estimate resale price", type="primary",
                            use_container_width=True, disabled=bool(errors))

if predict_clicked and not errors:
    try:
        features = build_feature_row(
            town, flat_type, flat_model, storey_mid, floor_area,
            remaining_lease_months, flat_age, valuation_month_index)

        predicted = float(MODEL.predict(features)[0])
        low, high = predicted - MAE, predicted + MAE

        st.markdown(
            f"""
            <div class="price-card">
                <div class="label">ESTIMATED RESALE PRICE</div>
                <div class="value">${predicted:,.0f}</div>
                <div class="range">Typical range: ${low:,.0f} &nbsp;–&nbsp; ${high:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"The range reflects the model's average error of ±${MAE:,.0f} on 47,259 "
            "unseen transactions. Roughly two-thirds of real sale prices fall within it."
        )

        # ---- Context for the estimate -------------------------------------
        st.markdown("### Estimate breakdown")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Price per sqm", f"${predicted / floor_area:,.0f}")
        b2.metric("Flat age at sale", f"{flat_age} yrs")
        b3.metric("Remaining lease",
                  f"{remaining_lease_months // 12} yrs {remaining_lease_months % 12} mth")
        b4.metric("Storey (midpoint)", f"{storey_mid:.0f}")

        # ---- How storey height changes the valuation ----------------------
        # Re-predicts the same flat at every storey band so the user can see the
        # height premium the model learned.
        st.markdown("### How floor level affects this flat's value")
        storey_prices = []
        for band in artifact["storey_ranges"]:
            row = build_feature_row(
                town, flat_type, flat_model, storey_midpoint(band), floor_area,
                remaining_lease_months, flat_age, valuation_month_index)
            storey_prices.append({"Storey range": band,
                                  "Estimated price": float(MODEL.predict(row)[0])})

        storey_df = pd.DataFrame(storey_prices).set_index("Storey range")
        st.bar_chart(storey_df, height=280, color="#2e86ab")
        st.caption(
            "Same flat, same valuation date, only the floor level changes. "
            "Higher floors generally command a premium for view and privacy."
        )

    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        st.error(
            "**Something went wrong generating the estimate.** Please check your inputs "
            f"and try again.\n\nTechnical detail: `{exc}`"
        )

elif not predict_clicked:
    st.info("Fill in the flat's details above, then select **Estimate resale price**.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
    <p class="footer-note">
    Built for CAI2C08 Machine Learning for Developers · Koh Tim Yi (2503677C)<br>
    Model: HistGradientBoostingRegressor, tuned via RandomizedSearchCV
    ({artifact['best_params']}) · Test MAE ${MAE:,.0f} · R² {METRICS['R2']:.4f}<br>
    Data source: Resale Flat Prices, data.gov.sg (Jan 2017 – {artifact['dataset_last_month']})
    </p>
    """,
    unsafe_allow_html=True,
)
