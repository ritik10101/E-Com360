import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib

# ------------------ Step 1: Load & Clean Data ------------------
raw_data_path = os.path.join('data', 'raw_data.csv')
df = pd.read_csv(raw_data_path, encoding='ISO-8859-1')
df.dropna(subset=['CustomerID'], inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df = df.dropna(subset=['InvoiceDate'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
cleaned_data_path = os.path.join('data', 'cleaned_data.csv')
df.to_csv(cleaned_data_path, index=False)

# ------------------ Step 2: RFM Analysis ------------------
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Dynamically adjust quantiles
def safe_qcut(series, q, labels):
    unique_vals = series.nunique()
    bins = min(q, unique_vals)
    return pd.qcut(series.rank(method='first'), bins, labels=labels[:bins], duplicates='drop')

rfm['R'] = safe_qcut(rfm['Recency'], 4, [4, 3, 2, 1])
rfm['F'] = safe_qcut(rfm['Frequency'], 4, [1, 2, 3, 4])
rfm['M'] = safe_qcut(rfm['Monetary'], 4, [1, 2, 3, 4])

rfm['RFM_Segment'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1).astype(int)
rfm_scores_path = os.path.join('data', 'rfm_scores.csv')
rfm.to_csv(rfm_scores_path)

# ------------------ Step 3: Recommendation System ------------------
cf_data = df[['CustomerID', 'StockCode', 'TotalPrice']]
cf_data = cf_data.groupby(['CustomerID', 'StockCode']).sum().reset_index()
cf_data.columns = ['userID', 'itemID', 'rating']
reader = Reader(rating_scale=(cf_data['rating'].min(), cf_data['rating'].max()))
data = Dataset.load_from_df(cf_data[['userID', 'itemID', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
print('RMSE:', accuracy.rmse(predictions))
model_path = os.path.join('models', 'recommendation_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print("✅ Data cleaned\n✅ RFM scores calculated\n✅ Recommendation model trained and saved.")
