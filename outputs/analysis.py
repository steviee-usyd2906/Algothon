"""
Unsupervised learning analysis of instrument price correlations.

Input:  a whitespace-separated text file where the first line is a header
        of instrument tickers, and every following line is one time step
        of prices for all instruments (same column order as the header).

What it does:
  1. Loads prices, computes daily log returns (more stationary than raw
     price levels, so correlations reflect co-movement, not shared trend).
  2. Builds the correlation matrix between all instrument pairs.
  3. Hierarchical clustering (Ward linkage) on 1 - correlation distance,
     to group instruments that move together.
  4. PCA to see how many independent "factors" drive the whole basket,
     and which instruments load most heavily on each factor.
  5. KMeans clustering in PCA space as a second, independent clustering
     view (good sanity check against the hierarchical result).
  6. Saves: correlation heatmap (clustered), dendrogram, PCA scree +
     loadings plot, a CSV of the full correlation matrix, and a CSV of
     the top most-correlated / most-anti-correlated pairs.

Usage:
    python correlation_analysis.py prices.txt
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "prices.txt"
OUT_DIR = "."

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH, sep=r"\s+", header=0)
print(f"Loaded {df.shape[0]} time steps x {df.shape[1]} instruments")

# ---------------------------------------------------------------------
# 2. Log returns (drop first NaN row)
# ---------------------------------------------------------------------
returns = np.log(df / df.shift(1)).dropna()

# ---------------------------------------------------------------------
# 3. Correlation matrix
# ---------------------------------------------------------------------
corr = returns.corr()
corr.to_csv(f"{OUT_DIR}/correlation_matrix.csv")

# Top correlated / anti-correlated pairs (excluding self-pairs & dupes)
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
pairs = pairs.rename("correlation").reset_index()
pairs.columns = ["instrument_1", "instrument_2", "correlation"]
pairs_sorted = pairs.reindex(pairs.correlation.abs().sort_values(ascending=False).index)
pairs_sorted.to_csv(f"{OUT_DIR}/top_pairs.csv", index=False)

print("\nTop 15 strongest relationships (by |correlation|):")
print(pairs_sorted.head(15).to_string(index=False))

# ---------------------------------------------------------------------
# 4. Hierarchical clustering on correlation distance
# ---------------------------------------------------------------------
dist = (1 - corr).copy()
dist_vals = dist.values.copy()
np.fill_diagonal(dist_vals, 0)
condensed = squareform(dist_vals, checks=False)
Z = linkage(condensed, method="average")

# Cluster heatmap (reordered by dendrogram)
g = sns.clustermap(
    corr, row_linkage=Z, col_linkage=Z, cmap="coolwarm", center=0,
    figsize=(14, 14), xticklabels=True, yticklabels=True
)
g.ax_heatmap.set_title("Instrument correlation matrix (hierarchically clustered)", pad=80)
g.savefig(f"{OUT_DIR}/correlation_heatmap_clustered.png", dpi=150, bbox_inches="tight")
plt.close("all")

# Standalone dendrogram
plt.figure(figsize=(16, 6))
dendrogram(Z, labels=corr.columns.tolist(), leaf_rotation=90)
plt.title("Hierarchical clustering of instruments (1 - correlation distance)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dendrogram.png", dpi=150)
plt.close()

# Flat clusters at a chosen distance threshold
n_clusters_target = 8
flat_clusters = fcluster(Z, t=n_clusters_target, criterion="maxclust")
cluster_map = pd.DataFrame({"instrument": corr.columns, "cluster": flat_clusters})
cluster_map = cluster_map.sort_values("cluster")
cluster_map.to_csv(f"{OUT_DIR}/hierarchical_clusters.csv", index=False)

print(f"\nHierarchical clusters (k={n_clusters_target}):")
for c in sorted(cluster_map.cluster.unique()):
    members = cluster_map.loc[cluster_map.cluster == c, "instrument"].tolist()
    print(f"  Cluster {c}: {', '.join(members)}")

# ---------------------------------------------------------------------
# 5. PCA — how many latent factors drive the whole basket?
# ---------------------------------------------------------------------
X = StandardScaler().fit_transform(returns)
pca = PCA()
scores = pca.fit_transform(X)
explained = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained), marker="o")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA scree plot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_scree.png", dpi=150)
plt.close()

print(f"\nPC1 explains {explained[0]*100:.1f}% of variance, "
      f"first 5 PCs explain {explained[:5].sum()*100:.1f}%")

# Which instruments load most heavily on PC1 (the dominant common factor)
loadings = pd.DataFrame(pca.components_[:5].T, index=corr.columns,
                         columns=[f"PC{i+1}" for i in range(5)])
loadings.to_csv(f"{OUT_DIR}/pca_loadings.csv")
top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10)
print("\nInstruments most tied to the dominant common factor (PC1):")
print(loadings.loc[top_pc1.index, "PC1"].to_string())

# ---------------------------------------------------------------------
# 6. KMeans on PCA scores as a second clustering view
# ---------------------------------------------------------------------
k = n_clusters_target
km = KMeans(n_clusters=k, n_init=10, random_state=0)
# Cluster instruments, not time steps: use loadings (PC space per instrument)
inst_coords = loadings.values  # instruments x 5 PCs
km_labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(inst_coords)
kmeans_map = pd.DataFrame({"instrument": corr.columns, "kmeans_cluster": km_labels})
kmeans_map = kmeans_map.sort_values("kmeans_cluster")
kmeans_map.to_csv(f"{OUT_DIR}/kmeans_clusters.csv", index=False)

print(f"\nKMeans clusters (k={k}) on PCA loadings:")
for c in sorted(kmeans_map.kmeans_cluster.unique()):
    members = kmeans_map.loc[kmeans_map.kmeans_cluster == c, "instrument"].tolist()
    print(f"  Cluster {c}: {', '.join(members)}")

print("\nDone. Outputs written to:", OUT_DIR)