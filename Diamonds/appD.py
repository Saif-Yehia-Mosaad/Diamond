import streamlit as st
import pandas as pd
import joblib

# Load Model
try:
    model = joblib.load('diamond_model.pkl')
    scaler = joblib.load('diamond_scaler.pkl')
except:
    st.error("Please run the notebook code above to save the model files first.")
    st.stop()

# Mappings
cut_rev = {1: 'Fair', 2: 'Good', 3: 'Very Good', 4: 'Premium', 5: 'Ideal'}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

st.title("💎 Diamond Quality Predictor")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        carat = st.number_input("Carat", 0.2, 5.0, 0.5)
        color = st.selectbox("Color", list(color_map.keys()))
        clarity = st.selectbox("Clarity", list(clarity_map.keys()))
        price = st.number_input("Price ($)", 300, 20000, 1000)
    with c2:
        depth = st.number_input("Depth", 40.0, 80.0, 61.0)
        table = st.number_input("Table", 40.0, 80.0, 57.0)
        x = st.number_input("Length (x)", 0.0, 15.0, 5.0)
        y = st.number_input("Width (y)", 0.0, 15.0, 5.0)
        z = st.number_input("Depth (z)", 0.0, 10.0, 3.0)
        
    if st.form_submit_button("Predict"):
        data = pd.DataFrame([[carat, color_map[color], clarity_map[clarity], depth, table, price, x, y, z]], 
                            columns=['carat', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z'])
        pred = model.predict(scaler.transform(data))[0]
        st.success(f"Predicted Cut: {cut_rev[pred]}")
        if cut_rev[pred] == 'Ideal': st.balloons()