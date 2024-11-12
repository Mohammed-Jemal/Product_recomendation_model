import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack
from recommendation import recommend_with_clustering
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('/content/drive/MyDrive/data.csv')
Selected_df = df[(df['Category'] == 'Toys') & (df['Selling Price'] < 50)]

# Load the trained TF-IDF vectorizer and KMeans model
tfid = joblib.load('tfidf_vectorizer.joblib')
kmeans = joblib.load('kmeans_model.joblib')

# Combine features for content-based filtering
Selected_df['combined'] = Selected_df['Product Specification'] + " " + Selected_df['Technical Details'] + " " + Selected_df['Category']

tfidf_matrix = tfid.transform(Selected_df['combined']).tocsr()

scale = MinMaxScaler()
Selected_df[['Selling Price', 'Shipping Weight']] = scale.fit_transform(Selected_df[['Selling Price', 'Shipping Weight']])

# Combine the normalized numerical features with the TF-IDF matrix
numeric_features = Selected_df[['Selling Price', 'Shipping Weight']].values  
tfidf_matrix = hstack([tfidf_matrix, numeric_features])  # Combine TF-IDF matrix with numerical features

# Predict clusters for Selected_df and add them as a column
Selected_df['cluster'] = kmeans.predict(tfidf_matrix)

# Streamlit App Layout
st.title("Product Recommendation System")

# Select a product from a dropdown
product_name = st.selectbox("Select a Product", Selected_df['Product Name'].unique())

# Number of recommendations
n_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

# Get recommendations when user clicks the button
if st.button("Get Recommendations"):
    recommendations = recommend_with_clustering(product_name, tfidf_matrix, Selected_df, n_recommendations=n_recommendations)
    st.write("Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
