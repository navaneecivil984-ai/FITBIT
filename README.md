🏋️ Fitness Machine Learning Project

This project applies both Supervised Learning and Unsupervised Learning techniques on fitness workout data to deliver meaningful insights and predictions.

The system is designed to:

🔥 Predict calories burned using regression models
🧠 Discover hidden workout patterns using clustering
📊 Provide an interactive Streamlit dashboard for visualization and user interaction
🚀 Features
🔹 1. Calorie Burn Prediction (Supervised Learning)
Predicts calories burned based on:
MET values
Session duration
Weight, BMI, Height
Models used:
Linear Regression
Ridge Regression (Tuned)
KNN Regressor
Random Forest
Decision Tree
XGBoost
SVR
🔹 2. Model Performance Evaluation
Compares all models using:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score
Automatically identifies the best performing model
🔹 3. Workout Pattern Clustering (Unsupervised Learning)
Identifies hidden workout behavior patterns
Techniques used:
PCA (Dimensionality Reduction)
KMeans Clustering
Hierarchical Clustering
DBSCAN (Density-Based Clustering)
📊 Clustering Evaluation
Silhouette Score
Acceptable threshold: ≥ 0.15
Cluster interpretation:
High intensity workouts
Medium intensity workouts
Low intensity workouts
🧠 Key Concepts Used
Data Cleaning & Preprocessing
Outlier Removal (IQR Method)
One-Hot Encoding
Feature Scaling (StandardScaler)
Feature Selection (Correlation Analysis)
Hyperparameter Tuning (GridSearchCV)
PCA for dimensionality reduction
📁 Project Structure
fitness-ml-project/
│
├── data/
│   └── Fitbit_dataset.csv
│
├── models/
│   ├── linear_regression_model.pkl
│   ├── scaler.pkl
│   └── features.json
│
├── notebooks/
│   ├── supervised_learning.ipynb
│   └── unsupervised_learning.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│   ├── model_performance_results.csv
│   └── clustering_visualizations.png
│
├── requirements.txt
└── README.md
💻 Streamlit App Pages

The application includes 4 interactive pages:

🏠 Project Overview
📊 Model Performance Metrics
🔥 Calorie Burn Predictor
🏋️ Workout Pattern Clustering
▶️ How to Run
pip install -r requirements.txt
streamlit run streamlit_app.py
🎯 Results
Achieved strong regression performance with high R² scores
Successfully identified meaningful workout clusters
Built an end-to-end ML pipeline with real-world applicability
📌 Real-World Applications
Fitness tracking apps
Personalized workout recommendations
Health monitoring systems
Smart wearable analytics
👨‍💻 Author

Developed as part of a Machine Learning project to gain hands-on experience with:

Supervised Learning
Unsupervised Learning
Model Deployment
