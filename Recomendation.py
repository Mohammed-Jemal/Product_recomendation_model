import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load components
tfid = joblib.load('tfidf_vectorizer.joblib')
kmeans = joblib.load('kmeans_model.joblib')
Selected_df = pd.read_csv('selected_df.csv')

# Define the recommendation function
def recommend_with_clustering(product_name, n_recommendations=10):
    # Find the cluster of the input product
    product_cluster = Selected_df.loc[Selected_df['Product Name'] == product_name, 'cluster'].values[0]
    
    # Filter products in the same cluster
    cluster_products = Selected_df[Selected_df['cluster'] == product_cluster]
    
    # Get the index of the product
    idx = Selected_df[Selected_df['Product Name'] == product_name].index[0]
    
    # Compute similarity within the cluster
    cluster_indices = cluster_products.index
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix[cluster_indices]).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Return the top recommendations
    recommendations = cluster_products.iloc[[cluster_indices[i[0]] for i in sim_scores[1:n_recommendations + 1]]]
    return recommendations['Product Name'].tolist()
