# importing libraries
import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# setting the title of the app
st.title("Customer Segmentation App")

# uploading the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    # reading the dataset
    data = pd.read_csv(uploaded_file)
    
    # displaying the first few rows of the dataset
    st.write("Dataset Preview:")
    st.dataframe(data.head())
    
    # selecting features for clustering
    features = st.multiselect("Select features for clustering", data.columns.tolist())
    
    if len(features) > 0:
        # scaling the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])
        
        # selecting number of clusters
        num_clusters = st.slider("Select number of clusters", 1, 10, 3)
        
        # applying KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data['Cluster'] = kmeans.fit_predict(scaled_data)
        
        # displaying cluster centers
        st.write("Cluster Centers:")
        st.dataframe(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features))
        
        # plotting the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data[features[0]], y=data[features[1]], hue=data['Cluster'], palette='viridis')
        plt.title('Customer Segmentation')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        st.pyplot(plt)
    else:
        st.warning("Please select at least one feature for clustering.")
else:
    st.info("Please upload a CSV file to get started.") 
# end of the app
# This app allows users to upload a dataset, select features for clustering, and visualize the results
