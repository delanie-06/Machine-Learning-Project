import streamlit as st
import pickle
import pandas as pd
import joblib

# Load the model
with open('model_saved.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
scaler = joblib.load('MinMaxScaler.pkl')

# Streamlit app
st.title("Calorie Content Prediction")

# Input fields
cal_fat = st.number_input("Calories from Fat", min_value=0.0)
total_fat = st.number_input("Total Fat (g)", min_value=0.0)
trans_fat = st.number_input("Trans Fat (g)", min_value=0.0)
sat_fat = st.number_input("Saturated Fat (g)", min_value=0.0)
cholesterol = st.number_input("Cholesterol", min_value=0.0)
sodium = st.number_input("Sodium", min_value=0.0)
total_carb = st.number_input("Total Carb", min_value=0.0)
fiber = st.number_input("Fiber", min_value=0.0)
sugar = st.number_input("Sugar (g)", min_value=0.0)
protein = st.number_input("Protein", min_value=0.0)
vit_a = st.number_input("Vitamin A", min_value=0.0)
vit_c = st.number_input("Vitamin C", min_value=0.0)
calcium = st.number_input("Calcium", min_value=0.0)

# Button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[cal_fat, total_fat, sat_fat, trans_fat, cholesterol, sodium, total_carb, fiber, sugar, protein, vit_a, vit_c, calcium]], 
                              columns=['cal_fat', 'total_fat', 'sat_fat', 'trans_fat', 'cholesterol', 'sodium', 'total_carb', 'fiber', 'sugar', 'protein', 'vit_a', 'vit_c', 'calcium'])
    
    # Make prediction using the loaded model
    try:
        prediction = model.predict(input_data)
        result = float(prediction)
        st.success(f"Final Prediction is {result}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

