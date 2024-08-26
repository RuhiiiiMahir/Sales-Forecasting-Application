import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the XGBoost model
model = xgb.Booster()
model.load_model("xgboost_model.json")

# Function to make predictions
def make_prediction(input_data):
    dmatrix = xgb.DMatrix(input_data)
    predictions = model.predict(dmatrix).clip(0, 20)
    return predictions[0]

# Streamlit App
st.title("Item Sales Prediction")

st.write("This app predicts the monthly sales of an item based on various input features using an XGBoost model.")

# Sidebar inputs for features
st.sidebar.header("Input Features")

shop_id = st.sidebar.number_input("Shop ID", min_value=1, max_value=100, value=5)
item_id = st.sidebar.number_input("Item ID", min_value=1, max_value=10000, value=5037)
shop_category = st.sidebar.number_input("Shop Category", min_value=1, max_value=10, value=1)
shop_city = st.sidebar.number_input("Shop City", min_value=1, max_value=10, value=2)
item_category_id = st.sidebar.number_input("Item Category ID", min_value=1, max_value=100, value=3)
category_type = st.sidebar.number_input("Category Type", min_value=1, max_value=10, value=4)
category_subtype = st.sidebar.number_input("Category Subtype", min_value=1, max_value=10, value=5)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
item_cnt_month_lag_1 = st.sidebar.number_input("Item Count Month Lag 1", min_value=0, max_value=100, value=0)
item_cnt_month_lag_2 = st.sidebar.number_input("Item Count Month Lag 2", min_value=0, max_value=100, value=0)
item_cnt_month_lag_3 = st.sidebar.number_input("Item Count Month Lag 3", min_value=0, max_value=100, value=0)
item_cnt_month_lag_4 = st.sidebar.number_input("Item Count Month Lag 4", min_value=0, max_value=100, value=0)
item_cnt_month_lag_5 = st.sidebar.number_input("Item Count Month Lag 5", min_value=0, max_value=100, value=0)
item_cnt_month_lag_12 = st.sidebar.number_input("Item Count Month Lag 12", min_value=0, max_value=100, value=0)
cnt_block_shop_lag_1 = st.sidebar.number_input("Count Block Shop Lag 1", min_value=0, max_value=100, value=0)
cnt_block_shop_lag_2 = st.sidebar.number_input("Count Block Shop Lag 2", min_value=0, max_value=100, value=0)
cnt_block_shop_lag_3 = st.sidebar.number_input("Count Block Shop Lag 3", min_value=0, max_value=100, value=0)
cnt_block_shop_lag_4 = st.sidebar.number_input("Count Block Shop Lag 4", min_value=0, max_value=100, value=0)
cnt_block_shop_lag_5 = st.sidebar.number_input("Count Block Shop Lag 5", min_value=0, max_value=100, value=0)
cnt_block_shop_lag_12 = st.sidebar.number_input("Count Block Shop Lag 12", min_value=0, max_value=100, value=0)
cnt_block_item_lag_1 = st.sidebar.number_input("Count Block Item Lag 1", min_value=0, max_value=100, value=0)
cnt_block_item_lag_2 = st.sidebar.number_input("Count Block Item Lag 2", min_value=0, max_value=100, value=0)
cnt_block_item_lag_3 = st.sidebar.number_input("Count Block Item Lag 3", min_value=0, max_value=100, value=0)
cnt_block_item_lag_4 = st.sidebar.number_input("Count Block Item Lag 4", min_value=0, max_value=100, value=0)
cnt_block_item_lag_5 = st.sidebar.number_input("Count Block Item Lag 5", min_value=0, max_value=100, value=0)
cnt_block_item_lag_12 = st.sidebar.number_input("Count Block Item Lag 12", min_value=0, max_value=100, value=0)
cnt_block_category_lag_1 = st.sidebar.number_input("Count Block Category Lag 1", min_value=0, max_value=100, value=0)
cnt_block_category_lag_2 = st.sidebar.number_input("Count Block Category Lag 2", min_value=0, max_value=100, value=0)
cnt_block_category_lag_3 = st.sidebar.number_input("Count Block Category Lag 3", min_value=0, max_value=100, value=0)
cnt_block_category_lag_4 = st.sidebar.number_input("Count Block Category Lag 4", min_value=0, max_value=100, value=0)
cnt_block_category_lag_5 = st.sidebar.number_input("Count Block Category Lag 5", min_value=0, max_value=100, value=0)
cnt_block_category_lag_12 = st.sidebar.number_input("Count Block Category Lag 12", min_value=0, max_value=100, value=0)

# Create a DataFrame with the input features
input_data = pd.DataFrame({
    'shop_id': [shop_id],
    'item_id': [item_id],
    'shop_category': [shop_category],
    'shop_city': [shop_city],
    'item_category_id': [item_category_id],
    'category_type': [category_type],
    'category_subtype': [category_subtype],
    'month': [month],
    'item_cnt_month_lag_1': [item_cnt_month_lag_1],
    'item_cnt_month_lag_2': [item_cnt_month_lag_2],
    'item_cnt_month_lag_3': [item_cnt_month_lag_3],
    'item_cnt_month_lag_4': [item_cnt_month_lag_4],
    'item_cnt_month_lag_5': [item_cnt_month_lag_5],
    'item_cnt_month_lag_12': [item_cnt_month_lag_12],
    'cnt_block_shop_lag_1': [cnt_block_shop_lag_1],
    'cnt_block_shop_lag_2': [cnt_block_shop_lag_2],
    'cnt_block_shop_lag_3': [cnt_block_shop_lag_3],
    'cnt_block_shop_lag_4': [cnt_block_shop_lag_4],
    'cnt_block_shop_lag_5': [cnt_block_shop_lag_5],
    'cnt_block_shop_lag_12': [cnt_block_shop_lag_12],
    'cnt_block_item_lag_1': [cnt_block_item_lag_1],
    'cnt_block_item_lag_2': [cnt_block_item_lag_2],
    'cnt_block_item_lag_3': [cnt_block_item_lag_3],
    'cnt_block_item_lag_4': [cnt_block_item_lag_4],
    'cnt_block_item_lag_5': [cnt_block_item_lag_5],
    'cnt_block_item_lag_12': [cnt_block_item_lag_12],
    'cnt_block_category_lag_1': [cnt_block_category_lag_1],
    'cnt_block_category_lag_2': [cnt_block_category_lag_2],
    'cnt_block_category_lag_3': [cnt_block_category_lag_3],
    'cnt_block_category_lag_4': [cnt_block_category_lag_4],
    'cnt_block_category_lag_5': [cnt_block_category_lag_5],
    'cnt_block_category_lag_12': [cnt_block_category_lag_12]
})

# Button to make prediction
if st.button("Predict Item Sales"):
    prediction = make_prediction(input_data)
    st.success(f"Predicted Item Sales: {prediction:.2f}")

# Visualizations
st.header("Visualizations")

# Dummy visualization - Feature Importance (Example)
st.subheader("Feature Importance (Example)")
sns.set(style="whitegrid")
fig, ax = plt.subplots()
sns.barplot(x=[0.15, 0.1, 0.05], y=["Feature A", "Feature B", "Feature C"], palette="Blues_d", ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

# Dummy visualization - Sales Over Time (Example)
st.subheader("Sales Over Time (Example)")
fig, ax = plt.subplots()
time_series_data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [20, 30, 25, 35, 45, 40]
})
sns.lineplot(x='Month', y='Sales', data=time_series_data, marker="o", ax=ax)
ax.set_title("Sales Over Time")
st.pyplot(fig)
