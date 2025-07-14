import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stSelectbox, .stSlider {
        margin-bottom: 20px;
    }
    .cluster-header {
        color: #2c3e50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    url = "Mall_Customers.csv"
    df = pd.read_csv(url)
    df.rename(columns={
        'Annual Income (k$)': 'Income',
        'Spending Score (1-100)': 'SpendingScore'
    }, inplace=True)
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("🛍️ Mall Customer Segmentation")
st.sidebar.markdown("Adjust clustering parameters:")

# Feature selection
features = st.sidebar.multiselect(
    "Select features for clustering:",
    ['Age', 'Income', 'SpendingScore'],
    default=['Income', 'SpendingScore']
)

# Number of clusters
n_clusters = st.sidebar.slider(
    "Number of clusters:",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

# Main app
st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("""
This application helps mall managers understand customer segments based on spending behavior and demographics 
to improve marketing strategies and customer experience.
""")

# Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(df, height=200)

# Data preprocessing
def preprocess_data(df, features):
    # Convert gender to numeric
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Select features and scale
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

# Perform clustering
def perform_clustering(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    
    return clusters, silhouette_avg

# Run clustering when features are selected
if len(features) >= 2:
    X_scaled = preprocess_data(df, features)
    clusters, silhouette_avg = perform_clustering(X_scaled, n_clusters)
    df['Cluster'] = clusters
    
    # Show results
    st.subheader("Clustering Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Silhouette Score:** `{silhouette_avg:.3f}`")
        st.caption("A higher score (closer to 1) indicates better defined clusters.")
        
        # Cluster statistics
        st.markdown("**Cluster Statistics:**")
        cluster_stats = df.groupby('Cluster')[features].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
    
    with col2:
        # Elbow method plot
        st.markdown("**Elbow Method for Optimal Clusters:**")
        distortions = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            distortions.append(kmeans.inertia_)
        
        fig, ax = plt.subplots()
        ax.plot(K_range, distortions, 'bx-')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Distortion')
        ax.set_title('The Elbow Method')
        st.pyplot(fig)
    
    # Cluster visualization
    st.subheader("Cluster Visualization")
    
    if len(features) == 2:
        # 2D plot
        fig = px.scatter(
            df, 
            x=features[0], 
            y=features[1], 
            color='Cluster',
            title=f'Customer Segments (k={n_clusters})',
            hover_data=['Gender', 'Age'],
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 3D plot if 3 features selected
        fig = px.scatter_3d(
            df, 
            x=features[0], 
            y=features[1], 
            z=features[2],
            color='Cluster',
            title=f'Customer Segments (k={n_clusters})',
            hover_data=['Gender', 'Age'],
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        
        with st.expander(f"🔘 Cluster {cluster} - {len(cluster_data)} customers"):
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Average Age", f"{cluster_data['Age'].mean():.1f} years")
            with cols[1]:
                st.metric("Average Income", f"${cluster_data['Income'].mean():.1f}k")
            with cols[2]:
                st.metric("Average Spending Score", f"{cluster_data['SpendingScore'].mean():.1f}/100")
            
            # Gender distribution
            gender_dist = cluster_data['Gender'].value_counts(normalize=True).mul(100)
            gender_dist.index = gender_dist.index.map({0: 'Male', 1: 'Female'})
            
            fig, ax = plt.subplots()
            gender_dist.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('Percentage')
            ax.set_title('Gender Distribution')
            st.pyplot(fig)
            
            st.markdown("**Recommendations:**")
            if cluster_data['SpendingScore'].mean() > 70:
                st.write("💎 **High-value customers**: Target with premium offers and loyalty programs.")
            elif cluster_data['Income'].mean() > 70 and cluster_data['SpendingScore'].mean() < 40:
                st.write("💰 **High-income, low-spending**: Identify why they're not spending and create targeted campaigns.")
            elif cluster_data['Age'].mean() < 30:
                st.write("👩‍🎓 **Young customers**: Engage with social media campaigns and trendy products.")
            else:
                st.write("👔 **General population**: Maintain standard marketing approaches.")
    
    # Download results
    st.download_button(
        label="Download Cluster Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='mall_customer_segments.csv',
        mime='text/csv'
    )
else:
    st.warning("Please select at least 2 features for clustering.")

# Footer
st.markdown("---")
st.markdown("""
**Mall Customer Segmentation App**  
*Helping mall managers understand their customers better*  
[Deploy your own version] | [GitHub Repository]
""")