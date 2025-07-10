import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from PIL import Image
import base64
import io

# =============================================
# APP CONFIGURATION
# =============================================
st.set_page_config(
    page_title="✨ Ultimate Customer Insights",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM STYLING
# =============================================
st.markdown("""
<style>
    /* Main styles */
    .main {
        background-color: #f9f9f9;
    }
    .header {
        font-size: 2.8em;
        color: #2c3e50;
        text-align: center;
        padding: 15px;
        background: linear-gradient(90deg, #ff9a9e 0%, #fad0c4 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    .cluster-box {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #4facfe;
    }
    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #a1c4fd 0%, #c2e9fb 100%);
    }
    /* Metric boxes */
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .profile-img {
        border-radius: 50%;
        border: 3px solid #4facfe;
        margin-bottom: 10px;
        width: 150px;
        height: 150px;
        object-fit: cover;
    }
    .profile-caption {
        font-size: 0.9em;
        text-align: center;
        color: #666;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    .profile-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# UTILITY FUNCTIONS
# =============================================
@st.cache_data
def load_sample_data():
    """Load sample mall customers dataset"""
    data = pd.read_csv('Mall_Customers.csv')
    data = data.rename(columns={
        'CustomerID': 'ID',
        'Annual Income (k$)': 'Income',
        'Spending Score (1-100)': 'SpendingScore'
    })
    return data

def scale_data(data, features):
    """Scale selected features"""
    scaler = StandardScaler()
    X = data[features]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def perform_clustering(X_scaled, n_clusters=5, random_state=42):
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans

def analyze_clusters(data, features, cluster_col='Cluster'):
    """Calculate cluster statistics"""
    stats = data.groupby(cluster_col)[features].agg(['mean', 'std', 'count'])
    sil_score = silhouette_score(data[features], data[cluster_col])
    db_score = davies_bouldin_score(data[features], data[cluster_col])
    return stats, sil_score, db_score

def get_profile_image(gender, age):
    """Get realistic profile image based on gender and age"""
    if gender == 'Male':
        if age < 30:
            return "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&auto=format&fit=crop&w=200&q=80"
        elif age < 50:
            return "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?ixlib=rb-1.2.1&auto=format&fit=crop&w=200&q=80"
        else:
            return "https://images.unsplash.com/photo-1531384441138-2736e62e0919?ixlib=rb-1.2.1&auto=format&fit=crop&w=200&q=80"
    else:
        if age < 30:
            return "https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-1.2.1&auto=format&fit=crop&w=200&q=80"
        elif age < 50:
            return "https://images.unsplash.com/photo-1531123897727-8f129e1688ce?ixlib=rb-1.2.1&auto=format&fit=crop&w=200&q=80"
        else:
            return "https://images.unsplash.com/photo-1551836022-d5d88e9218df?ixlib=rb-1.2.1&auto=format&fit=crop&w=200&q=80"

# =============================================
# MAIN APP FUNCTION
# =============================================
def main():
    # ======================
    # HEADER SECTION
    # ======================
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=100)
    with col2:
        st.markdown('<div class="header">✨ Ultimate Customer Insights</div>', unsafe_allow_html=True)
        st.caption("Advanced customer segmentation for targeted marketing strategies")
    
    st.markdown("---")
    
    # ======================
    # DATA LOADING SECTION
    # ======================
    st.markdown("### 📊 Data Selection")
    data_source = st.radio(
        "Choose your data source:",
        ("Use sample dataset (Mall Customers)", "Upload your own CSV"),
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if data_source == "Use sample dataset (Mall Customers)":
        data = load_sample_data()
        st.success("✅ Sample dataset loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("✅ Your file was uploaded successfully!")
        else:
            st.info("ℹ️ Please upload a CSV file to get started")
            return
    
    # Show data preview
    with st.expander("🔍 Explore Dataset", expanded=True):
        st.dataframe(data.head().style.background_gradient(cmap='Blues'))
        st.write(f"📐 Dataset shape: {data.shape}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📈 Quick Statistics**")
            st.dataframe(data.describe().T.style.background_gradient(cmap='YlOrBr'))
        with col2:
            st.markdown("**🧐 Missing Values**")
            missing = data.isnull().sum().to_frame('Missing Values')
            st.dataframe(missing.style.background_gradient(cmap='Reds'))
    
    st.markdown("---")
    
    # ======================
    # CLUSTERING SETUP
    # ======================
    st.markdown("### ⚙️ Clustering Configuration")
    
    # Feature selection
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for clustering")
        return
        
    features = st.multiselect(
        "Select features for clustering:",
        numeric_cols,
        default=['Income', 'SpendingScore'] if all(x in numeric_cols for x in ['Income', 'SpendingScore']) else numeric_cols[:2]
    )
    
    if len(features) < 2:
        st.warning("⚠️ Please select at least 2 features")
        return
    
    # Clustering parameters
    cols = st.columns(3)
    with cols[0]:
        n_clusters = st.slider("Number of clusters", 2, 10, 5, 
                              help="Choose how many customer segments you want to identify")
    with cols[1]:
        random_state = st.number_input("Random state", 0, 100, 42, 
                                     help="For reproducible results")
    with cols[2]:
        auto_cluster = st.checkbox("Auto-optimize clusters", 
                                 help="Use elbow method to find optimal cluster count")
    
    # Advanced options
    with st.expander("⚡ Advanced Options"):
        scale_data_option = st.checkbox("Scale data (recommended)", True)
        remove_outliers = st.checkbox("Remove outliers (using IQR)")
    
    # ======================
    # CLUSTER ANALYSIS
    # ======================
    if st.button("🚀 Run Cluster Analysis", type="primary"):
        with st.spinner("🔍 Analyzing customer segments..."):
            # Data preprocessing
            if remove_outliers:
                Q1 = data[features].quantile(0.25)
                Q3 = data[features].quantile(0.75)
                IQR = Q3 - Q1
                data = data[(data[features] < (Q1 - 1.5 * IQR)) | (data[features] > (Q3 + 1.5 * IQR)).any(axis=1)]
            
            # Auto cluster detection
            if auto_cluster:
                wcss = []
                sil_scores = []
                max_clusters = min(10, len(data)-1)
                
                for i in range(2, max_clusters+1):
                    kmeans = KMeans(n_clusters=i, random_state=42)
                    kmeans.fit(data[features])
                    wcss.append(kmeans.inertia_)
                    sil_scores.append(silhouette_score(data[features], kmeans.labels_))
                
                optimal_clusters = sil_scores.index(max(sil_scores)) + 2
                n_clusters = optimal_clusters
                
                fig_elbow = px.line(x=range(2, max_clusters+1), y=wcss, 
                                  title="Elbow Method", 
                                  labels={'x':'Number of Clusters', 'y':'Within-Cluster-Sum-of-Squares'})
                fig_elbow.add_vline(x=optimal_clusters, line_dash="dash", line_color="red")
                st.plotly_chart(fig_elbow, use_container_width=True)
                st.success(f"🎯 Optimal cluster count: {optimal_clusters}")
            
            # Scale data
            if scale_data_option:
                X_scaled, scaler = scale_data(data, features)
            else:
                X_scaled = data[features].values
            
            # Perform clustering
            clusters, kmeans = perform_clustering(X_scaled, n_clusters, random_state)
            data['Cluster'] = clusters
            
            # Analyze clusters
            cluster_stats, sil_score, db_score = analyze_clusters(data, features)
            
            # ======================
            # RESULTS DISPLAY
            # ======================
            st.markdown("---")
            st.markdown("## 📊 Clustering Results")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Silhouette Score", f"{sil_score:.2f}", 
                         help="Higher is better (range: -1 to 1)")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Davies-Bouldin Index", f"{db_score:.2f}", 
                         help="Lower is better (0 is best)")
                st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.metric("Total Clusters", n_clusters)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Cluster centers
            st.markdown("#### 📍 Cluster Centers")
            centers = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_) if scale_data_option else kmeans.cluster_centers_,
                columns=features
            )
            centers['Cluster'] = centers.index
            st.dataframe(centers.style.background_gradient(cmap='YlOrBr'))
            
            # ======================
            # VISUALIZATIONS
            # ======================
            st.markdown("#### 📈 Visualizations")
            
            # Interactive naming
            cluster_names = {}
            for i in range(n_clusters):
                cluster_names[i] = st.text_input(f"Name for Cluster {i}", 
                                                f"Segment {i+1}", 
                                                key=f"cluster_name_{i}")
            data['Segment'] = data['Cluster'].map(cluster_names)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["2D Scatter", "3D Scatter", "Distribution", "Profiles"])
            
            with tab1:
                fig = px.scatter(
                    data,
                    x=features[0],
                    y=features[1],
                    color='Segment',
                    title=f"{features[0]} vs {features[1]}",
                    hover_data=data.columns,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if len(features) >= 3:
                    fig3d = px.scatter_3d(
                        data,
                        x=features[0],
                        y=features[1],
                        z=features[2],
                        color='Segment',
                        title="3D Cluster Visualization",
                        hover_data=data.columns
                    )
                    st.plotly_chart(fig3d, use_container_width=True)
                else:
                    st.warning("Select at least 3 features for 3D visualization")
                    st.image("https://cdn-icons-png.flaticon.com/512/427/427735.png", width=200)
            
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(
                        data,
                        x='Segment',
                        title="Customers per Segment",
                        color='Segment',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col2:
                    if 'Gender' in data.columns:
                        fig_pie = px.pie(
                            data,
                            names='Gender',
                            title="Gender Distribution",
                            facet_col='Segment',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab4:
                st.markdown("#### 👥 Customer Profiles")
                
                for cluster in sorted(data['Cluster'].unique()):
                    cluster_data = data[data['Cluster'] == cluster]
                    
                    with st.expander(f"**{cluster_names[cluster]}** - {len(cluster_data)} customers ({len(cluster_data)/len(data)*100:.1f}%)", expanded=True):
                        cols = st.columns([1, 3])
                        
                        with cols[0]:
                            # Display profile image if we have Age and Gender data
                            if all(col in data.columns for col in ['Age', 'Gender']):
                                avg_age = cluster_data['Age'].mean()
                                gender_dist = cluster_data['Gender'].value_counts(normalize=True)
                                primary_gender = gender_dist.idxmax()
                                
                                img_url = get_profile_image(primary_gender, avg_age)
                                st.markdown(f'''
                                    <div class="profile-container">
                                        <img src="{img_url}" class="profile-img">
                                        <div class="profile-caption">
                                            Representative {primary_gender.lower()}, ~{int(avg_age)} years
                                        </div>
                                    </div>
                                ''', unsafe_allow_html=True)
                            else:
                                st.warning("Profile images require 'Age' and 'Gender' columns in your data")
                                st.image("https://cdn-icons-png.flaticon.com/512/3177/3177440.png", width=150)
                        
                        with cols[1]:
                            st.markdown("**Key Characteristics:**")
                            
                            # Dynamic description based on available features
                            desc_parts = []
                            if 'Age' in data.columns:
                                avg_age = cluster_data['Age'].mean()
                                desc_parts.append(f"Average age: {avg_age:.1f} years")
                            
                            if 'Income' in data.columns:
                                avg_income = cluster_data['Income'].mean()
                                desc_parts.append(f"Income: ${avg_income:.1f}k")
                            
                            if 'SpendingScore' in data.columns:
                                avg_spend = cluster_data['SpendingScore'].mean()
                                desc_parts.append(f"Spending score: {avg_spend:.1f}/100")
                            
                            if 'Gender' in data.columns:
                                gender_dist = cluster_data['Gender'].value_counts(normalize=True) * 100
                                gender_desc = " / ".join([f"{k}: {v:.1f}%" for k, v in gender_dist.items()])
                                desc_parts.append(f"Gender: {gender_desc}")
                            
                            st.write(", ".join(desc_parts))
                            
                            # Marketing recommendations
                            st.markdown("**🎯 Marketing Recommendations:**")
                            if 'SpendingScore' in data.columns:
                                avg_spend = cluster_data['SpendingScore'].mean()
                                if avg_spend > 75:
                                    st.write("- Target with premium/loyalty offers")
                                    st.write("- Upsell complementary products")
                                    st.write("- Exclusive early access to new products")
                                elif avg_spend < 40:
                                    st.write("- Win-back campaign needed")
                                    st.write("- Offer discounts to increase engagement")
                                    st.write("- Re-engagement email series")
                                else:
                                    st.write("- General promotions")
                                    st.write("- Cross-sell opportunities")
                                    st.write("- Loyalty program enrollment")
            
            # ======================
            # EXPORT SECTION
            # ======================
            st.markdown("---")
            st.markdown("## 💾 Export Results")
            
            export_format = st.radio("Select export format:", ["CSV", "Excel", "JSON"], horizontal=True)
            
            if export_format == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    data.to_excel(writer, sheet_name='Segmented Data', index=False)
                    cluster_stats.to_excel(writer, sheet_name='Cluster Stats')
                st.download_button(
                    "📊 Download Excel File",
                    data=buffer.getvalue(),
                    file_name="customer_segments.xlsx",
                    mime="application/vnd.ms-excel"
                )
            elif export_format == "JSON":
                st.download_button(
                    "📄 Download JSON",
                    data=data.to_json(orient='records'),
                    file_name="segments.json",
                    mime="application/json"
                )
            else:
                st.download_button(
                    "📋 Download CSV",
                    data=data.to_csv(index=False),
                    file_name="customer_segments.csv",
                    mime="text/csv"
                )

# =============================================
# RUN THE APP
# =============================================
if __name__ == "__main__":
    main()