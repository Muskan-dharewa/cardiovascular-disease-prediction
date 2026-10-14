import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Configuration
INPUT_FILE = "data/combined_statslogExcluded.csv"
TARGET_COL = "target"
REGION_COL = "Cluster_region"
SEED = 42

# 1. Load dataset
df = pd.read_csv(INPUT_FILE)
print("Loaded dataset:", df.shape)

# Exclude target from clustering
X = df.drop(columns=[TARGET_COL]).select_dtypes(include=["number"])

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_filled = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# 2. K-Means + PCA — 4 clusters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

n_components = min(6, X_scaled.shape[1])
pca = PCA(n_components=n_components, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
kmeans_labels = kmeans.fit_predict(X_pca)
kmeans_score = silhouette_score(X_pca, kmeans_labels)

print("\n" + "=" * 60)
print("K-Means + PCA: 4 clusters")
print(f"Silhouette score: {kmeans_score:.4f}")
print("Cluster counts:")
print(
    pd.Series(kmeans_labels, name=REGION_COL)
    .value_counts()
    .sort_index()
)

# Keep results in memory only
df_kmeans = df.copy()
df_kmeans[REGION_COL] = kmeans_labels

# 3. Agglomerative — 4 clusters (diagnostic only)
X_agg = X_filled.to_numpy()

labels_4 = AgglomerativeClustering(n_clusters=4).fit_predict(X_agg)
score_4 = silhouette_score(X_agg, labels_4)

print("\n" + "=" * 60)
print("Agglomerative clustering: 4 clusters")
print(f"Silhouette score: {score_4:.4f}")
print("Cluster counts:")
print(
    pd.Series(labels_4, name=REGION_COL)
    .value_counts()
    .sort_index()
)

# 4. Agglomerative — 3 clusters (final dataset in memory)
labels_3 = AgglomerativeClustering(n_clusters=3).fit_predict(X_agg)
score_3 = silhouette_score(X_agg, labels_3)

print("\n" + "=" * 60)
print("Agglomerative clustering: 3 clusters")
print(f"Silhouette score: {score_3:.4f}")
print("Cluster counts:")
print(
    pd.Series(labels_3, name=REGION_COL)
    .value_counts()
    .sort_index()
)

df_agg3 = df.copy()
df_agg3[REGION_COL] = labels_3

print("\nRegional clustering completed.")