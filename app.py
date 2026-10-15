
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="Restaurant Revenue Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# 2. Dark theme styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
[data-testid="stMetric"] {
    background-color: #1B2330;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #303B4B;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Load dataset and model
# -----------------------------
DATASET_FILE = "DATASET_FILE = "revenue_prediction.csv""
MODEL_FILE = "restaurant_revenue_model.pkl"

df = pd.read_csv(DATASET_FILE)
model = joblib.load(MODEL_FILE)

# -----------------------------
# 4. Dashboard title
# -----------------------------

# Dashboard Filters
st.sidebar.header("🔎 Dashboard Filters")

selected_city = st.sidebar.selectbox(
    "Select City",
    ["All"] + sorted(df["City"].unique().tolist())
)

selected_category = st.sidebar.selectbox(
    "Select Restaurant Category",
    ["All"] + sorted(df["Category"].unique().tolist())
)

filtered_df = df.copy()

if selected_city != "All":
    filtered_df = filtered_df[
        filtered_df["City"] == selected_city
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]

st.title("🍽️ Restaurant Revenue Estimation Dashboard")
st.caption("Historical revenue insights | Machine Learning Analytics")

st.divider()

# -----------------------------
# 5. KPI cards
# -----------------------------
total_restaurants = len(df)
total_revenue = df["Revenue"].sum()
average_revenue = df["Revenue"].mean()
median_revenue = df["Revenue"].median()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{total_restaurants:,}")
col2.metric("Total Historical Revenue", f"{total_revenue:,.0f}")
col3.metric("Average Revenue", f"{average_revenue:,.0f}")
col4.metric("Median Revenue", f"{median_revenue:,.0f}")

st.divider()

# -----------------------------
# 6. Revenue distribution
# -----------------------------
st.subheader("Revenue Distribution")

fig_distribution = px.histogram(
    df,
    x="Revenue",
    nbins=30,
    title="Distribution of Historical Restaurant Revenue",
    template="plotly_dark"
)

fig_distribution.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117"
)

st.plotly_chart(fig_distribution, use_container_width=True)

# -----------------------------
# 7. Revenue by city
# -----------------------------
st.subheader("Revenue by City")

city_revenue = (
    df.groupby("City", as_index=False)["Revenue"]
    .mean()
    .sort_values("Revenue", ascending=False)
)

fig_city = px.bar(
    city_revenue,
    x="City",
    y="Revenue",
    title="Average Historical Revenue by City",
    labels={"Revenue": "Average Revenue"},
    template="plotly_dark"
)

fig_city.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117"
)

st.plotly_chart(fig_city, use_container_width=True)

# -----------------------------
# 8. Revenue by category
# -----------------------------
st.subheader("Revenue by Restaurant Category")

category_revenue = (
    df.groupby("Category", as_index=False)["Revenue"]
    .mean()
    .sort_values("Revenue", ascending=False)
)

fig_category = px.bar(
    category_revenue,
    x="Category",
    y="Revenue",
    title="Average Historical Revenue by Category",
    labels={"Revenue": "Average Revenue"},
    template="plotly_dark"
)

fig_category.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117"
)

st.plotly_chart(fig_category, use_container_width=True)

# -----------------------------
# 9. Data preview
# -----------------------------
with st.expander("View Historical Dataset"):
    st.dataframe(df, use_container_width=True)

st.success("Dashboard overview loaded successfully!")














# Machine Learning Revenue Prediction
st.divider()
st.header("🤖 Machine Learning Revenue Prediction")
st.write("Enter restaurant details to estimate its revenue.")

col1, col2, col3 = st.columns(3)

with col1:
    franchise = st.selectbox(
        "Franchise",
        sorted(df["Franchise"].unique())
    )

    city = st.selectbox(
        "City",
        sorted(df["City"].unique())
    )

with col2:
    category = st.selectbox(
        "Restaurant Category",
        sorted(df["Category"].unique())
    )

    no_of_item = st.number_input(
        "Number of Items",
        min_value=0,
        value=int(df["No_Of_Item"].median()),
        step=1
    )

with col3:
    order_placed = st.number_input(
        "Orders Placed",
        min_value=0.0,
        value=float(df["Order_Placed"].median()),
        step=0.5
    )

if st.button("Predict Revenue"):
    input_data = pd.DataFrame([{
        "Franchise": franchise,
        "Category": category,
        "City": city,
        "No_Of_Item": no_of_item,
        "Order_Placed": order_placed
    }])

    predicted_revenue = model.predict(input_data)[0]

    st.subheader("Estimated Restaurant Revenue")
    st.metric(
        "Predicted Revenue",
        f"{predicted_revenue:,.0f}"
    )












    # Model Performance
st.divider()
st.header("📈 Model Performance")
st.write("Evaluation results from the final Ridge Regression model.")

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", "0.8548")
col2.metric("MAE", "354,720.54")
col3.metric("RMSE", "461,063.52")

st.caption(
    "R² measures how well the model explains revenue variation. "
    "MAE shows the average absolute prediction error. "
    "RMSE gives greater weight to larger prediction errors."
)














# Actual Revenue vs Predicted Revenue
st.divider()
st.header("📊 Actual Revenue vs Predicted Revenue")

comparison_df = df.copy()
comparison_features = [
    "Franchise",
    "Category",
    "City",
    "No_Of_Item",
    "Order_Placed"
]

comparison_df["Predicted Revenue"] = model.predict(
    comparison_df[comparison_features]
)

fig_comparison = px.scatter(
    comparison_df,
    x="Revenue",
    y="Predicted Revenue",
    title="Actual vs Predicted Revenue",
    labels={
        "Revenue": "Actual Revenue",
        "Predicted Revenue": "Predicted Revenue"
    },
    template="plotly_dark",
    opacity=0.7
)

fig_comparison.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117"
)

st.plotly_chart(fig_comparison, use_container_width=True)

st.caption(
    "Each point represents a restaurant. Points closer to the "
    "diagonal relationship between actual and predicted values "
    "indicate closer predictions. These are predictions on the "
    "dataset used here, not held-out test results."
)







# 3D Restaurant Revenue Analysis
st.divider()
st.header("🌐 3D Restaurant Revenue Analysis")

fig_3d = px.scatter_3d(
    df,
    x="No_Of_Item",
    y="Order_Placed",
    z="Revenue",
    color="City",
    title="Restaurant Revenue by Items, Orders, and City",
    labels={
        "No_Of_Item": "Number of Items",
        "Order_Placed": "Orders Placed",
        "Revenue": "Actual Revenue",
        "City": "City"
    },
    template="plotly_dark",
    opacity=0.75
)

fig_3d.update_layout(
    paper_bgcolor="#0E1117",
    scene=dict(
        xaxis_title="Number of Items",
        yaxis_title="Orders Placed",
        zaxis_title="Revenue"
    )
)

st.plotly_chart(fig_3d, use_container_width=True)

st.caption(
    "Each point represents a restaurant. Rotate or zoom the chart "
    "to explore the relationship between items, orders, and actual "
    "revenue. Colors distinguish the city codes."
)
