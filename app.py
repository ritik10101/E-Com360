import streamlit as st
import pandas as pd
import joblib

#Abhay

st.set_page_config(page_title="E-Com360 Dashboard", layout="wide")

st.title("📊 E-Com360: Customer RFM & Recommendations")

# Load RFM data
rfm_path = "data/rfm_scores.csv"
rfm_df = pd.read_csv(rfm_path)

st.subheader("🧮 RFM Score Distribution")
st.dataframe(rfm_df.head())

rfm_score_counts = rfm_df['RFM_Score'].value_counts().sort_index()
st.bar_chart(rfm_score_counts)

# Load the collaborative filtering model
model = joblib.load("models/recommendation_model.pkl")

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")
cf_data = df[['CustomerID', 'StockCode', 'TotalPrice']]
cf_data = cf_data.groupby(['CustomerID', 'StockCode']).sum().reset_index()
cf_data.columns = ['userID', 'itemID', 'rating']

st.subheader("🤖 Product Recommendations")

user_ids = cf_data['userID'].unique()
selected_user = st.selectbox("Select a Customer ID", user_ids)

# Recommend top N items
user_items = cf_data[cf_data['userID'] == selected_user]['itemID'].unique()
all_items = cf_data['itemID'].unique()

# Filter items not yet purchased
items_to_predict = [item for item in all_items if item not in user_items]

predictions = [model.predict(selected_user, item) for item in items_to_predict]
top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]

st.write("Top 5 Recommended Products for Customer", selected_user)
st.table(pd.DataFrame({
    "StockCode": [p.iid for p in top_predictions],
    "Estimated Rating": [round(p.est, 2) for p in top_predictions]
}))




#----> E-Com360_Complete_With_Requirements/
#├── app.py                  ← Streamlit Dashboard
#├── ECom360_Starter.py      ← ETL + Model Training
#├── requirements.txt        ← Required Python packages
#├── data/
#│   ├── raw_data.csv        ← Original dataset
#│   ├── cleaned_data.csv    ← Auto-generated
#│   └── rfm_scores.csv      ← Auto-generated
#└── models/
#    └── recommendation_model.pkl  ← Trained CF model
## python -m streamlit run app.py
