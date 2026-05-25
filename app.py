import streamlit as st
import numpy as np

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# =========================================
# CUSTOM CSS
# =========================================

# =========================================
# COLORFUL CUSTOM CSS
# =========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #1e3a8a,
        #7c3aed,
        #db2777
    );
    color: white;
}

/* Main Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #ffffff;
    margin-top: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #f1f5f9;
    margin-bottom: 35px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #06b6d4,
        #3b82f6
    );
    color: white;
    border-radius: 12px;
    border: none;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(
        90deg,
        #3b82f6,
        #8b5cf6
    );
}

/* Prediction Result Box */
.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

/* Survive */
.survive {
    background: linear-gradient(
        90deg,
        #16a34a,
        #22c55e
    );
    color: white;
}

/* Not Survive */
.not-survive {
    background: linear-gradient(
        90deg,
        #dc2626,
        #ef4444
    );
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 18px;
    color: white;
    font-weight: bold;
}

/* Text */
html, body, [class*="css"] {
    color: white;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.markdown('<p class="title">🚢 Titanic Survival Prediction</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Artificial Neural Network using Forward & Backpropagation</p>',
    unsafe_allow_html=True
)

# =========================================
# SIDEBAR INPUTS
# =========================================

st.sidebar.header("Passenger Inputs")

x1 = st.sidebar.slider("Pclass (Normalized)", 0.0, 1.0, 0.20)
x2 = st.sidebar.slider("Age (Normalized)", 0.0, 1.0, 0.24)
x3 = st.sidebar.slider("Fare (Normalized)", 0.0, 1.0, 0.80)

target = st.sidebar.selectbox(
    "Actual Survival",
    [0, 1],
    format_func=lambda x: "Not Survived" if x == 0 else "Survived"
)

# =========================================
# INITIAL WEIGHTS
# =========================================

w1, w2, w3 = 0.11, 0.14, 0.17
w4, w5, w6 = 0.21, 0.24, 0.27

bh1, bh2 = 0.1, 0.1

w7, w8 = 0.31, 0.34

bo = 0.1

lr = 0.1

# =========================================
# SIGMOID FUNCTION
# =========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("Predict Survival", use_container_width=True):

    # =====================================
    # FORWARD PROPAGATION
    # =====================================

    net_h1 = (x1*w1) + (x2*w2) + (x3*w3) + bh1
    net_h2 = (x1*w4) + (x2*w5) + (x3*w6) + bh2

    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)

    net_o1 = (h1*w7) + (h2*w8) + bo

    predicted_output = sigmoid(net_o1)

    # =====================================
    # CLASSIFICATION
    # =====================================

    prediction = 1 if predicted_output >= 0.5 else 0

    # =====================================
    # ERROR
    # =====================================

    mse = 0.5 * ((target - predicted_output) ** 2)

    # =====================================
    # BACKPROPAGATION
    # =====================================

    delta_o = (predicted_output - target) * predicted_output * (1 - predicted_output)

    delta_h1 = h1 * (1 - h1) * (delta_o * w7)
    delta_h2 = h2 * (1 - h2) * (delta_o * w8)

    # =====================================
    # UPDATE WEIGHTS
    # =====================================

    w7_new = w7 - (lr * delta_o * h1)
    w8_new = w8 - (lr * delta_o * h2)

    # =====================================
    # RESULT DISPLAY
    # =====================================

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

    # =====================================
    # METRICS
    # =====================================

    st.subheader("Neural Network Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Output", f"{predicted_output:.4f}")

    with col2:
        st.metric("Mean Squared Error", f"{mse:.4f}")

    with col3:
        st.metric("Learning Rate", lr)

    # =====================================
    # HIDDEN LAYER DETAILS
    # =====================================

    with st.expander("View Forward Propagation Details"):

        st.write("### Hidden Layer")

        st.write(f"Net h1 = {net_h1:.4f}")
        st.write(f"Net h2 = {net_h2:.4f}")

        st.write(f"h1 Output = {h1:.4f}")
        st.write(f"h2 Output = {h2:.4f}")

        st.write("### Output Layer")

        st.write(f"Net Output = {net_o1:.4f}")
        st.write(f"Final Prediction = {predicted_output:.4f}")

    # =====================================
    # BACKPROP DETAILS
    # =====================================

    with st.expander("View Backpropagation Details"):

        st.write("### Gradients")

        st.write(f"Output Gradient = {delta_o:.4f}")
        st.write(f"Hidden Gradient h1 = {delta_h1:.4f}")
        st.write(f"Hidden Gradient h2 = {delta_h2:.4f}")

        st.write("### Updated Weights")

        st.write(f"Updated w7 = {w7_new:.4f}")
        st.write(f"Updated w8 = {w8_new:.4f}")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.caption("Built using Streamlit + Artificial Neural Network")