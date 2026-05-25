import streamlit as st
import numpy as np

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #7c3aed
    );
    color: white;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
    margin-top: 10px;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #e2e8f0;
    margin-bottom: 30px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #1f2937
    );
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #06b6d4,
        #3b82f6
    );
    color: white;
    border: none;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

/* RESULT BOX */
.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

/* SURVIVE */
.survive {
    background: linear-gradient(
        90deg,
        #16a34a,
        #22c55e
    );
    color: white;
}

/* NOT SURVIVE */
.not-survive {
    background: linear-gradient(
        90deg,
        #dc2626,
        #ef4444
    );
    color: white;
}

/* METRICS */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* FOOTER */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================

st.markdown(
    '<p class="title">🚢 Titanic Survival Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Artificial Neural Network using Forward & Backpropagation</p>',
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR INPUTS
# ==================================================

st.sidebar.header("Passenger Inputs")

x1 = st.sidebar.slider(
    "Passenger Class (Normalized)",
    0.0, 1.0, 0.20
)

x2 = st.sidebar.slider(
    "Age (Normalized)",
    0.0, 1.0, 0.24
)

x3 = st.sidebar.slider(
    "Fare (Normalized)",
    0.0, 1.0, 0.80
)

target = st.sidebar.selectbox(
    "Actual Survival",
    [0, 1],
    format_func=lambda x: "Not Survived" if x == 0 else "Survived"
)

# ==================================================
# INITIAL WEIGHTS
# ==================================================

# Input -> Hidden
w1, w2, w3 = 0.11, 0.14, 0.17
w4, w5, w6 = 0.21, 0.24, 0.27

# Hidden Biases
bh1, bh2 = 0.1, 0.1

# Hidden -> Output
w7, w8 = 0.31, 0.34

# Output Bias
bo = 0.1

# Learning Rate
lr = 0.1

# ==================================================
# SIGMOID FUNCTION
# ==================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ==================================================
# PREDICTION BUTTON
# ==================================================

if st.button("Predict Survival"):

    # ==============================================
    # FORWARD PROPAGATION
    # ==============================================

    net_h1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + bh1
    net_h2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + bh2

    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)

    net_o1 = (h1 * w7) + (h2 * w8) + bo

    predicted_output = sigmoid(net_o1)

    # ==============================================
    # CLASSIFICATION
    # ==============================================

    prediction = 1 if predicted_output >= 0.5 else 0

    # ==============================================
    # ERROR CALCULATION
    # ==============================================

    mse = 0.5 * ((target - predicted_output) ** 2)

    # ==============================================
    # BACKPROPAGATION
    # ==============================================

    delta_o = (
        (predicted_output - target)
        * predicted_output
        * (1 - predicted_output)
    )

    delta_h1 = (
        h1 * (1 - h1) * (delta_o * w7)
    )

    delta_h2 = (
        h2 * (1 - h2) * (delta_o * w8)
    )

    # ==============================================
    # WEIGHT UPDATE
    # ==============================================

    w7_new = w7 - (lr * delta_o * h1)
    w8_new = w8 - (lr * delta_o * h2)

    # ==============================================
    # RESULT DISPLAY
    # ==============================================

    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown(f"""
        <div class="result-box survive">
        ✅ Passenger Likely to SURVIVE
        <br><br>
        Prediction Score: {predicted_output:.4f}
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-box not-survive">
        ❌ Passenger Likely to NOT SURVIVE
        <br><br>
        Prediction Score: {predicted_output:.4f}
        </div>
        """, unsafe_allow_html=True)

    # ==============================================
    # METRICS
    # ==============================================

    st.subheader("Model Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Output",
            f"{predicted_output:.4f}"
        )

    with col2:
        st.metric(
            "MSE Error",
            f"{mse:.4f}"
        )

    with col3:
        st.metric(
            "Learning Rate",
            lr
        )

    # ==============================================
    # FORWARD DETAILS
    # ==============================================

    with st.expander("View Forward Propagation"):

        st.write(f"Net h1 = {net_h1:.4f}")
        st.write(f"Net h2 = {net_h2:.4f}")

        st.write(f"h1 Output = {h1:.4f}")
        st.write(f"h2 Output = {h2:.4f}")

        st.write(f"Net Output = {net_o1:.4f}")
        st.write(f"Final Prediction = {predicted_output:.4f}")

    # ==============================================
    # BACKPROP DETAILS
    # ==============================================

    with st.expander("View Backpropagation"):

        st.write(f"Output Gradient = {delta_o:.4f}")
        st.write(f"Hidden Gradient h1 = {delta_h1:.4f}")
        st.write(f"Hidden Gradient h2 = {delta_h2:.4f}")

        st.write(f"Updated w7 = {w7_new:.4f}")
        st.write(f"Updated w8 = {w8_new:.4f}")

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption("Built using Streamlit + NumPy")