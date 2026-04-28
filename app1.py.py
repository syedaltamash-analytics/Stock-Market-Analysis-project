#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import numpy as np
from keras.models import load_model
import joblib
import plotly.graph_objects as go

# Load the trained model and scaler
model = load_model("/Altamash/Excelr code/PROJECTS/P_481_Stock market Prediction/lstm_model.h5")
scaler = joblib.load("/Altamash/Excelr code/PROJECTS/P_481_Stock market Prediction/scaler.pkl")

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="centered")

# Background Image Setup
def set_background_image(image_url):
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url({image_url});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background_image("https://source.unsplash.com/1600x900/?finance,stocks")

# App Title
st.markdown("<h1 style='text-align: center;'>📈 Stock Market Prediction App</h1>", unsafe_allow_html=True)

# Input Section
st.sidebar.header("Enter Input Data")
close_price = st.sidebar.number_input("📊 Previous Close Price", min_value=0.0, value=100.0, step=0.1)
look_back = 1

# Prediction
if st.button("Predict Future Price"):
    # Prepare data for prediction
    input_data = np.array([close_price]).reshape(-1, 1)
    scaled_data = scaler.transform(input_data)

    # Create LSTM input structure
    X_input = scaled_data.reshape(1, look_back, 1)

    # Predict using the model
    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction)[0, 0]

    # Display Result
    st.markdown(f"""
    <div style="background-color:rgba(255, 255, 255, 0.8);padding:10px;border-radius:10px;text-align:center;">
        <h2>📈 Predicted Stock Price: <span style='color:green;'>${predicted_price:.2f}</span></h2>
    </div>
    """, unsafe_allow_html=True)

    # Plot Results
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_price,
        title={'text': "Predicted Stock Price ($)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, max(predicted_price * 1.5, 200)], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "green"},
            'bgcolor': "lightgray",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ))
    st.plotly_chart(fig)

# Footer
st.markdown("<h4 style='text-align: center;'>🚀 Powered by Machine Learning</h4>", unsafe_allow_html=True)


# In[ ]:





# In[ ]:




