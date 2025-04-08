import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('../data/online_retail_II.csv', encoding='ISO-8859-1')

print("Raw columns:", df.columns.tolist())
print("Sample InvoiceDate values:", df['InvoiceDate'].head().tolist())

df = df.dropna(subset=['Customer ID'])
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
df['Total_Spending'] = df['Quantity'] * df['Price']

current_date = df['InvoiceDate'].min() + pd.Timedelta(days=180)
tx_6m = df[df['InvoiceDate'] <= current_date].copy()

rfm = tx_6m.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,  
    'ï»¿Invoice': 'nunique', 
    'Total_Spending': 'sum'  
}).rename(columns={
    'InvoiceDate': 'Recency',
    'ï»¿Invoice': 'Frequency',
    'Total_Spending': 'Revenue'
})

rfm['Revenue_per_Transaction'] = rfm['Revenue'] / rfm['Frequency']

next_6m_start = current_date + pd.Timedelta(days=1)
next_6m_end = current_date + pd.Timedelta(days=180)
tx_next_6m = df[(df['InvoiceDate'] >= next_6m_start) & (df['InvoiceDate'] <= next_6m_end)].copy()
m6_revenue = tx_next_6m.groupby('Customer ID')['Total_Spending'].sum().rename('m6_Revenue')

tx_merge = rfm.merge(m6_revenue, on='Customer ID', how='left').fillna(0)
tx_merge['CustomerID'] = tx_merge.index

tx_merge = tx_merge[tx_merge['m6_Revenue'] < tx_merge['m6_Revenue'].quantile(0.99)]

def order_cluster(cluster_field_name, target_field_name, df, ascending=True):
    cluster_means = df.groupby(cluster_field_name)[target_field_name].mean()
    cluster_order = cluster_means.sort_values(ascending=ascending).index
    mapping = {old: new for new, old in enumerate(cluster_order)}
    df[cluster_field_name] = df[cluster_field_name].map(mapping)
    return df

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(tx_merge[['m6_Revenue']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['m6_Revenue']])
tx_merge = order_cluster('LTVCluster', 'm6_Revenue', tx_merge, True)

print("LTV Cluster Stats:")
print(tx_merge.groupby('LTVCluster')['m6_Revenue'].describe())

tx_merge.to_csv('../data/tx_merge_clustered.csv', index=False)
