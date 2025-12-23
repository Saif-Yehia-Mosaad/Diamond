import streamlit as st
import pandas as pd
import numpy as np
import joblib
# cd /d "E:\Downloads\Diamonds Final\Diamonds" && C:\Users\Hi-Tech\AppData\Local\Microsoft\WindowsApps\python3.11.exe -m notebook
# --- 1. Load Models ---
# Load Classification Model (Predicts Cut)
try:
    clf_model = joblib.load('diamond_model.pkl')
    clf_scaler = joblib.load('diamond_scaler.pkl')
except:
    pass 

# Load Regression Model (Predicts Price)
try:
    reg_model = joblib.load('diamond_price_model.pkl')
    reg_scaler = joblib.load('diamond_price_scaler.pkl')
except:
    pass

# --- 2. Mappings ---
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
cut_map_rev = {v: k for k, v in cut_map.items()} # Reverse map for display

# --- 3. App Layout ---
st.set_page_config(page_title="Diamond AI", page_icon="💎", layout="wide")

# Centered Title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("💎  Diamond AI Analysis")
    st.write("Use the tabs below to predict Quality or Price.")

# 4. TABS 
tab1, tab2 = st.tabs(["💍 Predict Cut Quality", "💰 Predict Price"])


# TAB 1: Predict CUT (Classification)

with tab1:
    st.header("Predict Diamond Quality")
    
    with st.form("cut_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            carat_c = st.slider("Carat", 0.1, 5.0, 0.5, key='c1')
            color_c = st.selectbox("Color", list(color_map.keys()), key='c2')
            clarity_c = st.selectbox("Clarity", list(clarity_map.keys()), key='c3')
        with c2:
            depth_c = st.slider("Depth %", 50.0, 80.0, 61.0, key='c4')
            table_c = st.slider("Table %", 50.0, 80.0, 57.0, key='c5')
            price_c = st.number_input("Price ($)", 0, 100000, 1000, step=50, key='c6')
        with c3:
            x_c = st.slider("Length (x)", 0.0, 20.0, 5.0, key='c7')
            y_c = st.slider("Width (y)", 0.0, 60.0, 5.0, key='c8')
            z_c = st.slider("Depth (z)", 0.0, 35.0, 3.0, key='c9')

        submit_cut = st.form_submit_button("Predict Cut Quality", use_container_width=True)

    if submit_cut:
        try:
            row = pd.DataFrame([{
                'carat': carat_c, 'color': color_map[color_c], 'clarity': clarity_map[clarity_c],
                'depth': depth_c, 'table': table_c, 'price': price_c,
                'x': x_c, 'y': y_c, 'z': z_c
            }])
            pred = clf_model.predict(clf_scaler.transform(row))[0]
            st.success(f"Predicted Quality: **{cut_map_rev[pred]}**")
            if cut_map_rev[pred] == 'Ideal': st.balloons()
        except NameError:
            st.error("Model not loaded. Run the Classification training step in notebook.")


# TAB 2: Predict PRICE (Regression)

with tab2:
    st.header("Predict Diamond Price")
    
    with st.form("price_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            carat_p = st.slider("Carat", 0.1, 5.0, 0.5, key='p1')
            color_p = st.selectbox("Color", list(color_map.keys()), key='p2')
            clarity_p = st.selectbox("Clarity", list(clarity_map.keys()), key='p3')
        with c2:
            cut_p = st.selectbox("Cut Quality", list(cut_map.keys()), index=2, key='p4')
            depth_p = st.slider("Depth %", 50.0, 80.0, 61.0, key='p5')
            table_p = st.slider("Table %", 50.0, 80.0, 57.0, key='p6')
        with c3:
            x_p = st.slider("Length (x)", 0.0, 20.0, 5.0, key='p7')
            y_p = st.slider("Width (y)", 0.0, 60.0, 5.0, key='p8')
            z_p = st.slider("Depth (z)", 0.0, 35.0, 3.0, key='p9')

        submit_price = st.form_submit_button("Predict Price", use_container_width=True)

    if submit_price:
        try:
            row_p = pd.DataFrame([{
                'carat': carat_p, 
                'cut': cut_map[cut_p], # New Input
                'color': color_map[color_p], 
                'clarity': clarity_map[clarity_p],
                'depth': depth_p, 'table': table_p, 
                'x': x_p, 'y': y_p, 'z': z_p
            }])
            
            # Predict
            predicted_price = reg_model.predict(reg_scaler.transform(row_p))[0]
            real_price = np.exp(predicted_price)
            
            # Show Result
            st.success(f"Estimated Price: **${predicted_price:,.2f}**")
        except NameError:
            st.error("Price Model not loaded. Run the Regression training step in notebook.")