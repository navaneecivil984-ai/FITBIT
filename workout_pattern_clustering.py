import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Workout Clustering", layout="wide")
st.title("🏋️ Workout Pattern Clustering (Unsupervised Learning)")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\ADMIN\\Downloads\\Fitbit_dataset.csv")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    return df

df = load_data()

# ---------------------------
# PREPROCESSING
# ---------------------------
df_unsup = df.copy()

# Drop target
if 'workout_type' in df_unsup.columns:
    df_unsup = df_unsup.drop(columns=['workout_type'])

# Encode categorical
df_unsup = pd.get_dummies(df_unsup, columns=['gender'], drop_first=True)

# Feature selection (important)
features = [
    'heart_rate',
    'calories_burned_kcal',
    'session_duration_hours',
    'fat_percentage',
    'water_intake_liters',
    'bmi'
]

# ---------------------------
# SCALING
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_unsup)

# ---------------------------
# PCA
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------
# USER INPUT: NUMBER OF CLUSTERS
# ---------------------------
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Select Number of Clusters (K)", 2, 8, 3)

# ---------------------------
# KMEANS
# ---------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# ---------------------------
# SILHOUETTE SCORE
# ---------------------------
score = silhouette_score(X_pca, clusters)

# ---------------------------
# ADD CLUSTERS
# ---------------------------
df_unsup['cluster'] = clusters

# ---------------------------
# DISPLAY METRICS
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("📊 Silhouette Score", f"{score:.3f}")

with col2:
    if score >= 0.15:
        st.success("✅ Valid clustering (≥ 0.15)")
    else:
        st.warning("⚠️ Weak clustering (< 0.15)")

# ---------------------------
# CLUSTER VISUALIZATION
# ---------------------------
st.subheader("📉 PCA Cluster Visualization")

fig, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("Workout Clusters")

st.pyplot(fig)

# ---------------------------
# CLUSTER SIZE
# ---------------------------
st.subheader("📦 Cluster Distribution")
cluster_counts = df_unsup['cluster'].value_counts()
st.bar_chart(cluster_counts)

# ---------------------------
# CLUSTER INTERPRETATION
# ---------------------------
st.subheader("📊 Cluster Mean Analysis")

cluster_summary = df_unsup.groupby('cluster').mean()
st.dataframe(cluster_summary)

