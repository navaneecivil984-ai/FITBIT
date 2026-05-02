import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR



# Extract the csv file
import pandas as pd
fitbitdf = pd.read_csv("C:\\Users\\ADMIN\\Downloads\\Fitbit_dataset.csv")


# Renaming the column names in dataframe
fitbitdf.columns = fitbitdf.columns.str.lower()

# Removing special characters from the columns names
# 1. Replace spaces with underscores
fitbitdf.columns = fitbitdf.columns.str.replace(' ', '_')

# 2. Remove all other special characters (keeping the underscores)
fitbitdf.columns = fitbitdf.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)


# Encoding

# 1. Identify the categorical columns to encode
# From your image, these are 'gender' and 'workout_type'
categorical_cols = ['gender', 'workout_type']

# 2. Apply One-Hot Encoding using pd.get_dummies
# drop_first=True is often used to avoid the "dummy variable trap" (multicollinearity)
df_encoded = pd.get_dummies(fitbitdf, columns=categorical_cols, drop_first=False)



# Identify numerical columns to clean
cols = ['weight_kg', 'height_m', 'fat_percentage']

for col in cols:
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Keep only rows within the boundaries
    df_encoded = df_encoded[(df_encoded[col] >= lower) & (df_encoded[col] <= upper)]

# st.success(f"Dataset size after removal: {df_encoded.shape}")

# Selecting Features and Target

feature=df_encoded[[
                    'effective_met',
                    'base_met',
                    #'hr_intensity',
                    'session_duration_hours','weight_kg','bmi','height_m']] #Feature x
target=df_encoded['calories_burned_kcal'] #Target y
# st.info(feature)
# st.info(target)

# Spliting Data

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=4)

#Standard Scaler


# Initialize StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



# Train Linear Regression Model -  1
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)
y_pred_standard = linear_model.predict(x_test_scaled) # Make prediction on test set

# KNN Model -  2


# Train K-Nearest Neighbors Regression
knn_model = KNeighborsRegressor(n_neighbors=5) # You can adjust n_neighbors
knn_model.fit(x_train_scaled, y_train)
y_pred_knn = knn_model.predict(x_test_scaled) # Make prediction on test set

# Random Forest Model - 3


# Train Random Forest Regressor
# You can adjust parameters like n_estimators, random_state, etc.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train_scaled, y_train)
y_pred_rf = rf_model.predict(x_test_scaled) # Make prediction on test set

# Decision Tree Model - 4


# Train Decision Tree Regressor
# You can adjust parameters like random_state, max_depth, etc.
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train_scaled, y_train)
y_pred_dt = dt_model.predict(x_test_scaled) # Make prediction on test set

# XGBoost Regressor - Model 5


# Train XGBoost Regressor
xgb_model = XGBRegressor(random_state=42) # You can adjust parameters here
xgb_model.fit(x_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(x_test_scaled) # Make prediction on test set

# Support Vector Regressor (SVR) - Model 6


# Train SVR Model
# You can adjust parameters like kernel, C, gamma, etc.
# SVR can be computationally intensive on larger datasets, consider using a smaller subset or linear SVR if it's too slow.
svr_model = SVR(kernel='rbf')
svr_model.fit(x_train_scaled, y_train)
y_pred_svr = svr_model.predict(x_test_scaled) # Make prediction on test set

# Page configuration
st.set_page_config(page_title="Model Evaluation", layout="wide")

st.title("📊 Model Performance Metrics")

# 1. Logic to compile results (Using your existing function)
def get_results_df(y_test, predictions):
    results = {
        'Model': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R-squared': []
    }
    
    for model_name, y_pred in predictions.items():
        results['Model'].append(model_name)
        results['MAE'].append(metrics.mean_absolute_error(y_test, y_pred))
        results['MSE'].append(metrics.mean_squared_error(y_test, y_pred))
        results['RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        results['R-squared'].append(r2_score(y_test, y_pred))
    
    return pd.DataFrame(results).sort_values(by='R-squared', ascending=False)

# 2. Mock Data / Integration
# Replace these with your actual y_test and y_pred variables
predictions_map = {
    'Linear Regression': y_pred_standard,
    'KNN Regressor': y_pred_knn,
    'Random Forest Regressor': y_pred_rf,
    'Decision Tree Regressor': y_pred_dt,
    'XGBoost Regressor': y_pred_xgb,
    'Support Vector Regressor': y_pred_svr
}

df_results = get_results_df(y_test, predictions_map)

# 3. Streamlit Display
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Results Table")
    # Displaying with styling to highlight the best R-squared
    st.dataframe(df_results.style.highlight_max(axis=0, subset=['R-squared'], color='#2e7d32'), 
                 use_container_width=True)

with col2:
    st.subheader("R-squared Comparison")
    st.bar_chart(df_results.set_index('Model')['R-squared'])

# 4. Success Message for Best Model
best_model = df_results.iloc[0]['Model']
st.success(f"🏆 **{best_model}** is the best performing model based on R-squared.")
