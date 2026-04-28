#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
from keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Load the trained model and scaler
model = load_model("/Altamash/Excelr code/PROJECTS/P_481_Stock market Prediction/lstm_model.h5")
scaler = joblib.load("/Altamash/Excelr code/PROJECTS/P_481_Stock market Prediction/scaler.pkl")

# Streamlit Page Configuration
st.set_page_config(page_title="💹 Stock Market Predictor", page_icon="📊", layout="centered")

# App Header
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color:#2a9d8f; font-size:48px;'>📈 Stock Market Prediction</h1>
        <p style='font-size:18px;color:#264653;'>Powered by LSTM & Keras | Enhanced with Plotly Charts</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for Inputs
st.sidebar.markdown("## 📥 Input Parameters")
close_price = st.sidebar.number_input("Previous Close Price ($)", min_value=0.0, value=100.0, step=0.1)
look_back = 1

# Prediction Trigger
if st.button("🚀 Predict Future Price"):
    # Data preparation
    input_data = np.array([close_price]).reshape(-1, 1)
    scaled_data = scaler.transform(input_data)
    X_input = scaled_data.reshape(1, look_back, 1)

    # Prediction
    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction)[0, 0]

    # Success Message
    st.markdown(
        f"<div style='background-color:#d4edda; padding:10px; border-radius:8px;'>"
        f"<h2 style='color:#155724; text-align:center;'>Predicted Stock Price: ${predicted_price:.2f}</h2>"
        f"</div>", unsafe_allow_html=True)

    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_price,
        title={'text': "Predicted Price ($)", 'font': {'size':20, 'color':'#2a9d8f'}},
        gauge={
            'axis': {'range': [0, max(predicted_price * 1.5, close_price * 1.5, 200)], 'tickcolor':'#264653'},
            'bar': {'color': "#2a9d8f"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, close_price], 'color': '#f8d7da'},
                {'range': [close_price, predicted_price], 'color': '#d1ecf1'}
            ]
        }
    ))
    fig_gauge.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color':'#264653'}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Line Chart: Previous vs Predicted
    df_line = pd.DataFrame({
        'Type': ['Previous Close', 'Predicted'],
        'Price': [close_price, predicted_price]
    })
    fig_line = px.line(
        df_line, x='Type', y='Price', markers=True,
        title='Previous vs Predicted Price',
        color_discrete_sequence=['#2a9d8f']
    )
    fig_line.update_traces(line=dict(width=3), marker=dict(size=10))
    fig_line.update_layout(
        template='plotly_white',
        title_font=dict(size=18, color='#264653'),
        xaxis_title=None,
        yaxis_title='Price ($)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#264653'
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Histogram: Predicted Price Distribution
    hist_data = np.random.normal(loc=predicted_price, scale=abs(predicted_price*0.05), size=300)
    fig_hist = px.histogram(
        hist_data, nbins=20,
        title='Predicted Price Distribution',
        labels={'value': 'Price ($)'},
        color_discrete_sequence=['#e76f51']
    )
    fig_hist.update_layout(
        template='plotly_white',
        title_font=dict(size=18, color='#264653'),
        xaxis_title='Price ($)',
        yaxis_title='Count',
        bargap=0.1,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#264653'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Conclusion Section
    trend = 'upward' if predicted_price > close_price else 'downward'
    conclusion = (
        f"Based on the model's prediction of ${predicted_price:.2f} vs the previous close of ${close_price:.2f},"
        f" the LSTM model indicates a potential {trend} trend."
    )
    st.markdown(
        f"<div style='background-color:#f1f1f1; padding:20px; border-radius:8px;'>"
        f"<h2 style='color:#264653;'>📊 Conclusion</h2>"
        f"<p style='color:#264653; font-size:16px;'>{conclusion}</p>"
        f"</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr style='border-color:#e0e0e0' />
    <div style='text-align: center; margin-top:10px;'>
        <p style='color:#264653;'>✨ Built with ❤️ using Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)


# In[ ]:





# In[ ]:




