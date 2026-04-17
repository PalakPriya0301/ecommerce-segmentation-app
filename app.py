import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sqlite3

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# 2. Secure Data & Model Loading
@st.cache_data
def load_data():
    # Connect to the local SQLite database instead of a CSV
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

# 3. Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a Module", ["📊 Segmentation Dashboard", "🔮 Predict New Customer"])

# ---------------------------------------------------------
# MODULE 1: DESCRIPTIVE ANALYTICS (DASHBOARD)
# ---------------------------------------------------------
if app_mode == "📊 Segmentation Dashboard":
    st.title("E-Commerce Customer Persona Dashboard")
    st.write("Explore the behavioral segments of our historical customer database.")
    
    # Interactive Filtering
    st.sidebar.header("Filter Options")
    selected_persona = st.sidebar.selectbox(
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
    
    # Data Table
    st.write("### Database Records")
    st.dataframe(filtered_df)
    
    # Visualization
    st.write("### Spend vs. Recency Analysis")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(filtered_df['Recency'], filtered_df['Monetary'], alpha=0.6, color='teal')
    ax.set_xlabel("Recency (Days Since Last Purchase)")
    ax.set_ylabel("Monetary (Total Spend in $)")
    ax.set_title(f"Purchasing Behavior Profile: {selected_persona}")
    
    # Dark theme styling for the chart to match Streamlit's dark mode
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        
    st.pyplot(fig)

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
        # Format the input exactly how the model expects it (matching the training features)
        input_data = pd.DataFrame([[recency, frequency, monetary, age]], 
                                  columns=['Recency', 'Frequency', 'Monetary', 'Age'])
        
        # Execute prediction
        prediction = model.predict(input_data)[0]
        
        # Display Results
        st.success(f"### 🤖 Prediction: **{prediction}**")
        
        # Add a little business logic context for the presentation
        if prediction == "High-Value Sleepers":
            st.warning("💡 **Action Item:** Trigger win-back email campaign with a high-value discount.")
        elif prediction == "Promising Newcomers":
            st.info("💡 **Action Item:** Enroll in loyalty onboarding program to increase frequency.")
        elif prediction == "Top-Tier Customers":
            st.success("💡 **Action Item:** Send VIP early-access invites to retain loyalty.")