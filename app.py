import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. Set the title of your web app
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("E-Commerce Customer Persona Dashboard")
st.write("Explore the behavioral segments of our customer base.")

# 2. Load the data
# Using st.cache_data ensures the data is only loaded once, making the app faster
@st.cache_data
def load_data():
    return pd.read_csv("E.csv")

df = load_data()

# 3. Create a Sidebar for interactive filtering
st.sidebar.header("Filter Options")
# Create a dropdown menu using the unique values from your 'Persona' column
selected_persona = st.sidebar.selectbox(
    "Select a Customer Persona to explore:", 
    df['Persona'].unique()
)

# Filter the dataframe based on the user's selection
filtered_df = df[df['Persona'] == selected_persona]

# 4. Display Metrics
st.subheader(f"Overview: {selected_persona}")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(filtered_df))
col2.metric("Average Spend", f"${filtered_df['Monetary'].mean():.2f}")
col3.metric("Average Recency", f"{filtered_df['Recency'].mean():.0f} days")

# 5. Display the Data
st.write("### Customer Data")
st.dataframe(filtered_df)

# 6. Create a simple visualization
st.write("### Spend vs. Recency Analysis")
fig, ax = plt.subplots()
ax.scatter(filtered_df['Recency'], filtered_df['Monetary'], alpha=0.6, color='teal')
ax.set_xlabel("Recency (Days Since Last Purchase)")
ax.set_ylabel("Monetary (Total Spend)")
ax.set_title(f"Purchasing Behavior of {selected_persona}")
st.pyplot(fig)