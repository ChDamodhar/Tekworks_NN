import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# Load Model
model = tf.keras.models.load_model("titanic_ann_model.h5")

# Title Section
st.markdown("""
# 🚢 Titanic Survival Prediction System
### Deep Learning Based Passenger Survival Prediction
""")

# Project Description
st.info("""
This application predicts whether a Titanic passenger would survive or not
using an Artificial Neural Network (ANN) developed with TensorFlow and deployed using Streamlit.
""")

# Create Columns
col1, col2 = st.columns(2)

# Input Section
with col1:

    st.subheader("Passenger Details")

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=50.0
    )

# Prediction Section
with col2:

    st.subheader("Prediction Area")

    if st.button("Predict Survival"):

        # Data Preprocessing
        # Example normalization

        pclass_norm = pclass / 3
        age_norm = age / 80
        fare_norm = fare / 500

        # Convert to array
        input_data = np.array([
            [pclass_norm, age_norm, fare_norm]
        ])

        # Prediction
        prediction = model.predict(input_data)

        probability = prediction[0][0]

        # Prediction Logic
        if probability > 0.5:
            result = "✅ Survived"
        else:
            result = "❌ Not Survived"

        # Output Display
        st.success(f"Prediction: {result}")

        st.metric(
            label="Survival Probability",
            value=f"{probability*100:.2f}%"
        )

        confidence = max(probability, 1 - probability)

        st.metric(
            label="Confidence Score",
            value=f"{confidence*100:.2f}%"
        )

        # Visualization
        st.subheader("Prediction Visualization")

        labels = ["Survival", "Non-Survival"]
        values = [probability, 1 - probability]

        fig, ax = plt.subplots()

        ax.bar(labels, values)

        ax.set_ylabel("Probability")

        st.pyplot(fig)