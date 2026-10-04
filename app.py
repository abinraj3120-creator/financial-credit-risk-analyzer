"""Loan Approval Dashboard  ->  run with:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model as ml

GREEN, RED, BLUE = "#2a9d8f", "#e76f51", "#4361ee"
HERE = Path(__file__).parent

st.set_page_config(page_title="Loan Approval Dashboard", page_icon="🏦", layout="wide")

st.markdown(
    """<style>
.hero {background: linear-gradient(120deg,#4361ee 0%,#2a9d8f 100%);
       padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0; opacity: .9;}
div[data-testid="stMetric"] {background: #f6f8fb; border: 1px solid #e6e9f0;
       padding: .8rem 1rem; border-radius: 12px;}
div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] {color: #1f2937;}
</style>
<div class="hero"><h1>🏦 Loan Approval Dashboard</h1>
<p>Explore the data, then check how a machine-learning model scores a new application.</p></div>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data + model loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    return ml.clean_data(raw)


@st.cache_resource(show_spinner="Training model...")
def get_bundle(raw: pd.DataFrame) -> dict:
    return ml.train(raw)


with st.sidebar:
    st.header("📁 Data")
    upload = st.file_uploader("Upload loan CSV (raw or cleaned)", type="csv")
    st.caption("Without an upload, the app looks for `loan_cleaned.csv` or `train.csv` next to app.py.")

def find_csv():
    """Look for a CSV with a Loan_Status column in the app folder and the working folder."""
    folders = [HERE, Path.cwd(), Path("/content")]
    files = []
    for folder in folders:
        if folder.exists():
            files += sorted(folder.glob("*.csv"))
    preferred = [f for f in files if f.name in ("loan_cleaned.csv", "train.csv")]
    for f in preferred + files:
        try:
            if ml.TARGET in pd.read_csv(f, nrows=0).columns:
                return f
        except Exception:
            continue
    return None


if upload is not None:
    raw = read_csv(upload)
else:
    found = find_csv()
    if found is None:
        st.info("No training CSV found. Put your loan CSV (with a Loan_Status column) in the "
                "same folder as app.py, or upload it in the sidebar.")
        st.stop()
    raw = read_csv(found)
    st.sidebar.success(f"Using {found.name}")

if ml.TARGET not in raw.columns:
    st.error(
        f"This file has no '{ml.TARGET}' column, so the model has nothing to learn from. "
        "Upload the training data (or your loan_cleaned.csv from the notebook), not the test file."
    )
    st.write("Columns found:", list(raw.columns))
    st.stop()

data = prepare(raw)
bundle = get_bundle(raw)

# --------------------------------------------------------------------------- #
# Sidebar filters (apply to the Overview and Explore tabs)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🎛️ Filters")
    areas = st.multiselect("Property area", sorted(data["Property_Area"].unique()),
                           default=sorted(data["Property_Area"].unique()))
    edu = st.multiselect("Education", sorted(data["Education"].unique()),
                         default=sorted(data["Education"].unique()))
    credit = st.multiselect("Credit history", sorted(data["Credit_History"].unique()),
                            default=sorted(data["Credit_History"].unique()),
                            format_func=lambda v: "Good (1)" if v == 1 else "Poor (0)")

view = data[
    data["Property_Area"].isin(areas)
    & data["Education"].isin(edu)
    & data["Credit_History"].isin(credit)
].copy()
view["Approved"] = (view[ml.TARGET] == "Y").astype(int)

if view.empty:
    st.warning("No applications match the current filters.")
    st.stop()

tab_over, tab_explore, tab_predict, tab_batch, tab_model = st.tabs(
    ["📊 Overview", "🔍 Explore", "🤖 Predict", "📂 Batch", "📈 Model"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{len(view):,}")
    c2.metric("Approval rate", f"{view['Approved'].mean():.1%}")
    c3.metric("Avg loan amount", f"{view['LoanAmount'].mean():,.0f}k")
    c4.metric("Median total income", f"{view['TotalIncome'].median():,.0f}")

    left, right = st.columns(2)
    split = view[ml.TARGET].map({"Y": "Approved", "N": "Rejected"}).value_counts().reset_index()
    split.columns = ["Status", "Count"]
    fig = px.pie(split, names="Status", values="Count", hole=0.55, title="Approval split",
                 color="Status", color_discrete_map={"Approved": GREEN, "Rejected": RED})
    left.plotly_chart(fig, use_container_width=True)

    rate = view.groupby("Credit_History")["Approved"].mean().reset_index()
    rate["Credit_History"] = rate["Credit_History"].map({1.0: "Good", 0.0: "Poor"})
    fig = px.bar(rate, x="Credit_History", y="Approved", title="Approval rate by credit history",
                 text_auto=".0%", color_discrete_sequence=[BLUE])
    fig.update_yaxes(tickformat=".0%", title="Approval rate")
    fig.update_xaxes(title="")
    right.plotly_chart(fig, use_container_width=True)

    area = view.groupby("Property_Area")["Approved"].mean().reset_index()
    fig = px.bar(area, x="Property_Area", y="Approved", title="Approval rate by property area",
                 text_auto=".0%", color_discrete_sequence=[GREEN])
    fig.update_yaxes(tickformat=".0%", title="Approval rate")
    fig.update_xaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Explore
# --------------------------------------------------------------------------- #
with tab_explore:
    color_map = {"Y": GREEN, "N": RED}
    num_cols = ["ApplicantIncome", "CoapplicantIncome", "TotalIncome", "LoanAmount"]

    a, b = st.columns(2)
    feat = a.selectbox("Numeric feature", num_cols, index=2)
    cat = b.selectbox("Category", ["Education", "Property_Area", "Married", "Self_Employed",
                                   "Gender", "Dependents", "Credit_History"])

    a, b = st.columns(2)
    fig = px.histogram(view, x=feat, color=ml.TARGET, nbins=40, barmode="overlay", opacity=0.7,
                       color_discrete_map=color_map, title=f"Distribution of {feat}")
    a.plotly_chart(fig, use_container_width=True)

    fig = px.box(view, x=ml.TARGET, y=feat, color=ml.TARGET, color_discrete_map=color_map,
                 title=f"{feat} by loan status")
    b.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    rate = view.groupby(cat)["Approved"].mean().reset_index()
    fig = px.bar(rate, x=cat, y="Approved", text_auto=".0%", color_discrete_sequence=[BLUE],
                 title=f"Approval rate by {cat}")
    fig.update_yaxes(tickformat=".0%", title="Approval rate")
    a.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(view, x="TotalIncome", y="LoanAmount", color=ml.TARGET, opacity=0.7,
                     color_discrete_map=color_map, title="Total income vs loan amount")
    b.plotly_chart(fig, use_container_width=True)

    corr = view.select_dtypes(include="number").drop(columns="Approved").corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation heatmap")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show filtered data"):
        shown = view.drop(columns="Approved")
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("Download as CSV", shown.to_csv(index=False).encode(),
                           "loan_filtered.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Check a new application")
    st.caption("An educational demo, not a real lending decision.")

    terms = sorted(data["Loan_Amount_Term"].unique())
    default_term = 360.0 if 360.0 in terms else terms[0]

    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        app_inc = c1.number_input("Applicant income", 0, 100000, 5000, step=500)
        co_inc = c1.number_input("Co-applicant income", 0, 100000, 0, step=500)
        loan_amt = c2.number_input("Loan amount (in thousands)", 1, 700, 130, step=5)
        term = c2.selectbox("Loan term (months)", terms, index=terms.index(default_term),
                            format_func=lambda v: f"{int(v)}")
        credit_in = c3.radio("Credit history meets guidelines?", [1.0, 0.0],
                             format_func=lambda v: "Yes" if v == 1 else "No", horizontal=True)
        deps = c3.selectbox("Dependents", [0, 1, 2, 3], format_func=lambda v: "3+" if v == 3 else str(v))

        c1, c2, c3 = st.columns(3)
        education = c1.selectbox("Education", sorted(data["Education"].unique()))
        self_emp = c2.selectbox("Self employed", sorted(data["Self_Employed"].unique()))
        prop = c3.selectbox("Property area", sorted(data["Property_Area"].unique()))
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    threshold = st.slider("Approval threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Applications scoring above this are marked 'likely approved'.")

    if submitted:
        applicant = {
            "Dependents": deps, "Education": education, "Self_Employed": self_emp,
            "Property_Area": prop, "ApplicantIncome": app_inc, "CoapplicantIncome": co_inc,
            "LoanAmount": loan_amt, "Loan_Amount_Term": term, "Credit_History": credit_in,
            # only used if you remove them from EXCLUDE in model.py
            "Gender": data["Gender"].mode()[0], "Married": data["Married"].mode()[0],
        }
        p = ml.predict_proba(bundle, applicant)

        left, right = st.columns([1, 1])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Approval probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": GREEN if p >= threshold else RED},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}},
        ))
        gauge.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
        left.plotly_chart(gauge, use_container_width=True)

        with right:
            if p >= threshold:
                st.success("✅ Likely approved")
            else:
                st.error("❌ Likely rejected")

            alt_credit = 0.0 if credit_in == 1.0 else 1.0
            p_alt = ml.predict_proba(bundle, {**applicant, "Credit_History": alt_credit})
            st.markdown("**What-if: credit history**")
            wi = pd.DataFrame({
                "Scenario": ["Good credit history", "Poor credit history"],
                "Probability": [p if credit_in == 1.0 else p_alt, p_alt if credit_in == 1.0 else p],
            })
            fig = px.bar(wi, x="Probability", y="Scenario", orientation="h", text_auto=".0%",
                         color_discrete_sequence=[BLUE], range_x=[0, 1])
            fig.update_layout(height=180, margin=dict(t=10, b=10), xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Estimated installment: about {loan_amt * 1000 / term:,.0f} per month "
                       "before interest.")

# --------------------------------------------------------------------------- #
# Batch prediction (test file)
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Score many applications at once")
    st.caption("Upload the test file (no Loan_Status column needed). The model predicts every row.")
    test_file = st.file_uploader("Upload test CSV", type="csv", key="test_csv")
    batch_threshold = st.slider("Approval threshold", 0.1, 0.9, 0.5, 0.05, key="batch_thr")

    if test_file is not None:
        test_df = pd.read_csv(test_file)
        needed = ["Dependents", "Education", "Self_Employed", "Property_Area", "ApplicantIncome",
                  "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
        missing_cols = [c for c in needed if c not in test_df.columns]
        if missing_cols:
            st.error(f"The file is missing these columns: {missing_cols}")
        else:
            result = ml.predict_batch(bundle, test_df, batch_threshold)
            n_ok = int((result["Prediction"] == "Approved").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Applications scored", f"{len(result):,}")
            c2.metric("Predicted approved", f"{n_ok:,}")
            c3.metric("Predicted rejected", f"{len(result) - n_ok:,}")

            fig = px.histogram(result, x="Approval_Probability", color="Prediction", nbins=25,
                               color_discrete_map={"Approved": GREEN, "Rejected": RED},
                               title="Predicted approval probabilities")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("Download predictions", result.to_csv(index=False).encode(),
                               "loan_predictions.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader(f"Selected model: {bundle['model_name']}")
    st.caption(f"Trained on {bundle['n_train']} applications, evaluated on {bundle['n_test']} "
               "held-out ones. Gender and Married are excluded from the features.")

    cols = st.columns(5)
    for col, (name, val) in zip(cols, bundle["metrics"].items()):
        col.metric(name, f"{val:.2f}")

    st.markdown("**Model comparison (5-fold cross-validation on the training set)**")
    st.dataframe(bundle["comparison"], hide_index=True, use_container_width=True)

    a, b = st.columns(2)
    cm = bundle["confusion"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Pred: Rejected", "Pred: Approved"], y=["Actual: Rejected", "Actual: Approved"],
                    title="Confusion matrix (test set)")
    a.plotly_chart(fig, use_container_width=True)

    roc = bundle["roc"]
    fig = go.Figure()
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Model", line=dict(color=BLUE, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="gray"))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    b.plotly_chart(fig, use_container_width=True)

    imp = bundle["importance"]
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[GREEN],
                 title="What drives the prediction (permutation importance, drop in ROC-AUC)")
    st.plotly_chart(fig, use_container_width=True)
