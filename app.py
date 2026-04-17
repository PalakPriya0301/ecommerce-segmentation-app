import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("E.csv")

@st.cache_resource
def load_model():
    return joblib.load("persona_model.pkl")

df = load_data()
model = load_model()

# Create a Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Page", ["📊 Dashboard", "🔮 Predict New Customer"])

if app_mode == "📊 Dashboard":
    st.title("E-Commerce Customer Persona Dashboard")
    st.write("Explore the behavioral segments of our customer base.")
    
    selected_persona = st.sidebar.selectbox("Select a Customer Persona to explore:", df['Persona'].unique())
    filtered_df = df[df['Persona'] == selected_persona]
    
    st.subheader(f"Overview: {selected_persona}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(filtered_df))
    col2.metric("Average Spend", f"${filtered_df['Monetary'].mean():.2f}")
    col3.metric("Average Recency", f"{filtered_df['Recency'].mean():.0f} days")
    
    st.write("### Customer Data")
    st.dataframe(filtered_df)

elif app_mode == "🔮 Predict New Customer":
    st.title("Predict Customer Persona")
    st.write("Enter the details of a new customer to instantly predict their segment.")
    
    # Create input sliders for the user
    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("Recency (Days since last purchase)", 1, 100, 20)
        frequency = st.slider("Frequency (Total items purchased)", 1, 50, 10)
    with col2:
        monetary = st.slider("Monetary (Total Spend in $)", 50.0, 3000.0, 500.0)
        age = st.slider("Age", 18, 80, 30)
    
    # Create a predict button
    if st.button("Predict Persona"):
        # Format the input exactly how the model expects it
        input_data = pd.DataFrame([[recency, frequency, monetary, age]], 
                                  columns=['Recency', 'Frequency', 'Monetary', 'Age'])
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        
        st.success(f"🤖 The model predicts this customer is a: **{prediction}**")