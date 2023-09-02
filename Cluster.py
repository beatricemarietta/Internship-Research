import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/drive/MyDrive/Data_Cluster.csv')

#Note: Data preprocessing has already been conducted

# Select the numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64'])
# Select the "name" column
name_column = df['Kecamatan']

### K - MEANS CLUSTER ###
#Standardized Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_columns)

df_scaled = pd.DataFrame(scaled_data)
df_scaled.columns = ['Jumlah SD', 'Jumlah SMP', 'Jumlah SMA', 'Jumlah RS Umum', 'Jumlah Posyandu', 'Luas Sawah', 'Luas Area Perkebunan', 'Luas Area Kolam', 'Jumlah Pasar Tradisional']
df_scaled.head()

#KMO Test (Asumption test: sample that represents the population)
!pip install numpy pandas scikit-learn factor-analyzer matplotlib
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df_scaled)
print("KMO for each variable:", kmo_all)
print("Overall KMO:", kmo_model)

#Non Multicollinearity Test
import statsmodels.api as sm
def calculate_vif(data_frame):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data_frame.columns
    vif_data["VIF"] = [sm.OLS(data_frame[col], sm.add_constant(data_frame.drop(columns=[col]))).fit().rsquared for col in data_frame.columns]
    return vif_data
vif_values = calculate_vif(df_scaled)
print(vif_values)

#Optimal Number of Cluster with Elbow Test
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

distortions = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_scaled)
    distortions.append(kmeans.inertia_)
  
plt.figure(figsize=(8, 6))
plt.plot(K_range, distortions, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Distortion (Inertia)')
plt.grid(True)
plt.show()

#K Means Cluster
K = 4
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(df_scaled)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
results_df = pd.DataFrame({
    'Name': name_column,
    'Cluster': cluster_labels
})
print(results_df)

# Access cluster assignments for each data point
print("Cluster Assignments:")
print(cluster_labels)
# Access cluster centers
print("Cluster Centers:")
print(cluster_centers)

df_cluster = df_scaled.assign(Cluster_Labels=cluster_labels)
df_cluster.head()
sns.pairplot(df_cluster, hue='Cluster_Labels')

### HIERARCHIAL CLUSTER METHOD ###
#Agglomerative Method
mergings = linkage(numeric_columns, method="single", metric='euclidean')
dendrogram(mergings)
plt.rcParams['figure.figsize'] = [15,8]
plt.show()

mergings = linkage(numeric_columns, method="complete", metric='euclidean')
dendrogram(mergings)
plt.rcParams['figure.figsize'] = [15,8]
plt.show()

mergings = linkage(numeric_columns, method="average", metric='euclidean')
dendrogram(mergings)
plt.rcParams['figure.figsize'] = [8,15]
plt.show()

mergings = linkage(numeric_columns, method="ward", metric='euclidean')
dendrogram(mergings)
plt.rcParams['figure.figsize'] = [15,8]
plt.show()

