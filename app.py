import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # <-- Added for the 3D Visualization
import joblib
import sqlite3

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# 2. Secure Data & Model Loading
@st.cache_data
def load_data():
    # Connect to the local SQLite database
    conn = sqlite3.connect("customers.db")
    query = "SELECT * FROM customer_segments"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    # Load the trained Random Forest predictive model
    return joblib.load("persona_model.pkl")

# Initialize data and model
try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading backend systems: {e}. Please ensure customers.db and persona_model.pkl are in the repository.")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION & LIVE SEARCH
# ---------------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Module", ["📊 Segmentation Dashboard", "🔮 Predict New Customer"])

st.sidebar.markdown("---")
st.sidebar.header("🔍 Individual Customer Lookup")

# Create a number input for IDs
search_id = st.sidebar.number_input("Enter Customer ID", min_value=101, max_value=450, step=1)

if st.sidebar.button("Fetch Live Data"):
    # Connect to your live database
    conn = sqlite3.connect("customers.db")
    
    # Professional SQL query with a filter
    query = f"SELECT * FROM customer_segments WHERE [Customer ID] = {search_id}"
    customer_profile = pd.read_sql_query(query, conn)
    conn.close()
    
    if not customer_profile.empty:
        st.sidebar.success(f"### Profile Found!")
        st.sidebar.dataframe(customer_profile[['Age', 'Gender', 'City', 'Membership Type']].transpose())
        
        # Pull the persona for this specific customer
        persona = customer_profile['Persona'].values[0]
        st.sidebar.info(f"**Segment:** {persona}")
    else:
        st.sidebar.error("Customer ID not found in database.")

# ---------------------------------------------------------
# MODULE 1: DESCRIPTIVE ANALYTICS (DASHBOARD)
# ---------------------------------------------------------
if app_mode == "📊 Segmentation Dashboard":
    st.title("E-Commerce Customer Persona Dashboard")
    st.write("Explore the behavioral segments of our historical customer database.")
    
    # Interactive Filtering
    selected_persona = st.selectbox(
        "Select a Customer Persona to explore:", 
        df['Persona'].unique()
    )
    
    filtered_df = df[df['Persona'] == selected_persona]
    
    # Key Performance Metrics
    st.subheader(f"Overview: {selected_persona}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(filtered_df))
    col2.metric("Average Spend", f"${filtered_df['Monetary'].mean():.2f}")
    col3.metric("Average Recency", f"{filtered_df['Recency'].mean():.0f} days")
    
    st.markdown("---")
    
    # 3D Interactive Cluster Map
    st.write("### 🌐 Mathematical Cluster Separation (3D View)")
    st.write("This 3D model illustrates how the K-Means algorithm partitions customers based on the RFM framework. Rotate to explore.")
    
    fig_3d = px.scatter_3d(
        df, x='Recency', y='Frequency', z='Monetary',
        color='Persona', 
        labels={'Recency': 'Days Since Last Purchase', 'Frequency': 'Total Items', 'Monetary': 'Total Spend ($)'},
        opacity=0.8,
        color_discrete_map={
            "Top-Tier Customers": "#00CC96",      # Vibrant Green
            "Promising Newcomers": "#636EFA",     # Vibrant Blue
            "High-Value Sleepers": "#EF553B"      # Vibrant Red
        }
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# MODULE 2: PREDICTIVE ANALYTICS (MACHINE LEARNING)
# ---------------------------------------------------------
elif app_mode == "🔮 Predict New Customer":
    st.title("Predict Customer Persona")
    st.write("Enter the real-time behavioral metrics of a new customer to instantly predict their segment using our Random Forest classifier.")
    
    st.markdown("---")
    
    # Interactive Input Sliders
    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("Recency (Days since last purchase)", 1, 100, 20)
        frequency = st.slider("Frequency (Total items purchased)", 1, 50, 10)
    with col2:
        monetary = st.slider("Monetary (Total Spend in $)", 50.0, 3000.0, 500.0)
        age = st.slider("Age", 18, 80, 30)
    
    st.markdown("---")
    
    # Prediction Engine
    if st.button("Predict Persona Segment", type="primary"):
        # Format the input exactly how the model expects it
        input_data = pd.DataFrame([[recency, frequency, monetary, age]], 
                                  columns=['Recency', 'Frequency', 'Monetary', 'Age'])
        
        # Execute prediction
        prediction = model.predict(input_data)[0]
        
        # Display Results
        st.success(f"### 🤖 Prediction: **{prediction}**")
        
        # Actionable Business Logic (Strategy Cards)
        if prediction == "High-Value Sleepers":
            st.error("💡 **Action Item: WIN-BACK** \n\nTrigger automated email campaign with a high-value discount coupon. Immediate intervention required to prevent churn.")
        elif prediction == "Promising Newcomers":
            st.info("💡 **Action Item: UPSELL** \n\nEnroll in loyalty onboarding program. Offer personalized bundles to increase their purchasing frequency.")
        elif prediction == "Top-Tier Customers":
            st.success("💡 **Action Item: RETAIN** \n\nSend VIP early-access invites to new product lines. Reward loyalty to maintain high engagement.")