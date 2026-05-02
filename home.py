import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fitness ML App", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Project Overview",
        "📊 Model Performance",
        "🔥 Calorie Predictor",
        "🏋️ Workout Clustering"
    ]
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\ADMIN\\Downloads\\Fitbit_dataset.csv")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    return df

df = load_data()

# =========================================================
# 🏠 PAGE 1: PROJECT OVERVIEW
# =========================================================
if page == "🏠 Project Overview":

    st.title("🏋️ Fitness ML Project")

    st.markdown("""
    ## 📌 Overview

    This project uses Machine Learning to analyze fitness data.

    ### 🔹 Supervised Learning
    - Predict Calories Burned
    - Models:
        - Linear Regression
        - KNN
        - Random Forest
        - Decision Tree
        - XGBoost
        - SVR

    ### 🔹 Unsupervised Learning
    - Workout Pattern Clustering
    - PCA + KMeans

    ### 📊 Metrics
    - Regression → MAE, RMSE, R²
    - Clustering → Silhouette Score (≥ 0.15 valid)

    ### 🎯 Goal
    - Predict calories accurately
    - Discover workout behavior patterns
    """)

    st.success("✅ Project Ready")

# =========================================================
# 📊 PAGE 2: MODEL PERFORMANCE
# =========================================================
elif page == "📊 Model Performance":

    st.title("📊 Model Performance Metrics")

    df_encoded = pd.get_dummies(df, columns=['gender','workout_type'], drop_first=False)

    # Remove outliers
    for col in ['weight_kg','height_m','fat_percentage']:
        Q1 = df_encoded[col].quantile(0.25)
        Q3 = df_encoded[col].quantile(0.75)
        IQR = Q3 - Q1
        df_encoded = df_encoded[
            (df_encoded[col] >= Q1 - 1.5*IQR) &
            (df_encoded[col] <= Q3 + 1.5*IQR)
        ]

    X = df_encoded[['effective_met','base_met','session_duration_hours','weight_kg','bmi','height_m']]
    y = df_encoded['calories_burned_kcal']

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    models = {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "XGBoost": XGBRegressor(),
        "SVR": SVR()
    }

    results = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        results.append({
            "Model": name,
            "R2 Score": r2_score(y_test, pred),
            "RMSE": np.sqrt(metrics.mean_squared_error(y_test, pred))
        })

    df_results = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

    st.dataframe(df_results)
    st.bar_chart(df_results.set_index("Model")["R2 Score"])

    best_model = df_results.iloc[0]["Model"]
    best_score = df_results.iloc[0]["R2 Score"]

    if best_score >= 0.90:
        st.success(f"🏆 {best_model} (Excellent Model ≥ 0.90)")
    elif best_score >= 0.75:
        st.info(f"👍 {best_model} (Good Model)")
    else:
        st.warning("⚠️ Model needs improvement")

# =========================================================
# 🔥 PAGE 3: CALORIE PREDICTOR
# =========================================================
elif page == "🔥 Calorie Predictor":

    st.title("🔥 Calorie Burn Predictor")

    @st.cache_resource
    def load_model():
        model = pickle.load(open("linear_regression_model.pkl","rb"))
        scaler = pickle.load(open("scaler.pkl","rb"))
        features = json.load(open("features.json"))
        return model, scaler, features

    model, scaler, features = load_model()

    st.subheader("Enter Inputs")

    inputs = {
        "effective_met": st.number_input("Effective MET", value=5.0),
        "base_met": st.number_input("Base MET", value=1.0),
        "session_duration_hours": st.number_input("Duration (hrs)", value=1.0),
        "weight_kg": st.number_input("Weight", value=70.0),
        "bmi": st.number_input("BMI", value=24.0),
        "height_m": st.number_input("Height", value=1.7),
    }

    if st.button("Predict Calories"):
        arr = np.array([[inputs[col] for col in features]])
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)

        st.success(f"🔥 Calories Burned: {pred[0]:.2f} kcal")

# =========================================================
# 🏋️ PAGE 4: WORKOUT CLUSTERING
# =========================================================
elif page == "🏋️ Workout Clustering":

    st.title("🏋️ Workout Pattern Clustering")

    df_unsup = df.copy()

    df_unsup = df_unsup.drop(columns=['workout_type'])
    df_unsup = pd.get_dummies(df_unsup, columns=['gender'], drop_first=True)

    selected_features = [
        'heart_rate','calories_burned_kcal',
        'session_duration_hours','fat_percentage',
        'water_intake_liters','bmi'
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
