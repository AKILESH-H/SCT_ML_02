import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
customer_data = pd.read_csv('Mall_Customers.csv')
X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_model = KMeans(n_clusters=5, random_state=42)
clusters = kmeans_model.fit_predict(X_scaled)
customer_data['Cluster_ID'] = clusters
cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
cluster_names = {
    0: 'High Income, Low Spending',
    1: 'Low Income, Low Spending',
    2: 'Low Income, High Spending',
    3: 'Average Income & Spending',
    4: 'High Income, High Spending'
}
customer_data['Customer_Segment'] = customer_data['Cluster_ID'].map(cluster_names)
custom_palette = ['#FF6347', '#3CB371', '#1E90FF', '#FFD700', '#8A2BE2']
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Customer_Segment',
    data=customer_data,
    palette=custom_palette,
    s=70
)
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    color='black',
    marker='X',
    s=250,
    alpha=0.7,
    label='Centroids'
)
plt.legend(
    title='Segments',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=12,        
    title_fontsize=13,  
    markerscale=1.5,    
    borderpad=1.2,      
    labelspacing=1.0    
)
plt.title('Mall Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.tight_layout()
plt.show()
