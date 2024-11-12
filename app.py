import streamlit as st
import pandas as pd
from recommendation import recommend_with_clustering

# Load the DataFrame with product names
Selected_df = pd.read_csv('selected_df.csv')

# Streamlit App Layout
st.title("Product Recommendation System")

# Select a product from a dropdown
product_name = st.selectbox("Select a Product", Selected_df['Product Name'].unique())

# Number of recommendations
n_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

# Get recommendations when user clicks the button
if st.button("Get Recommendations"):
    recommendations = recommend_with_clustering(product_name, n_recommendations=n_recommendations)
    st.write("Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
