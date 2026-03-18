import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def get_cluster_scores(X_scaled: pd.DataFrame, plot: bool = False) -> list[float]:
    sil_scores = []
    K_range = range(3, 8)
    for k in K_range:
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42)
        labels = model.fit_predict(X_scaled)
        dist_matrix = np.zeros((len(X_scaled), len(X_scaled)))
        for i in range(len(X_scaled)):
            for j in range(len(X_scaled)):
                dist_matrix[i, j] = dtw(X_scaled[i], X_scaled[j])
        
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        sil_scores.append(score)
    if plot:
        plt.figure()
        plt.plot(K_range, sil_scores, marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette selection")
        plt.show()
    return sil_scores

def get_cluster_mapping(df_wide: pd.DataFrame, plot: bool = False) -> dict[str, float]:
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(df_wide.values)
    scores = get_cluster_scores(X_scaled, plot)
    K_range = range(3, 8)
    best_k = K_range[np.argmax(scores)]
    model = TimeSeriesKMeans(n_clusters=best_k, metric="dtw", random_state=1337)
    labels = model.fit_predict(X_scaled)

    if plot:
        centroids = model.cluster_centers_
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for cluster_id in range(best_k):
            cluster_series = X_scaled[labels == cluster_id]
            
            for ts in cluster_series:
                axes[cluster_id].plot(ts.ravel(), alpha=0.2)
            
            axes[cluster_id].plot(centroids[cluster_id].ravel(), linewidth=3, color='red')
            axes[cluster_id].set_title(f"Cluster {cluster_id}")
            axes[cluster_id].set_xlabel("Time")
            axes[cluster_id].set_ylabel("Value")
        
        for idx in range(best_k, 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    return {key: val for key, val in zip(df_wide.index, labels)}
