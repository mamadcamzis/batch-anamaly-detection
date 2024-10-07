import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from loguru import logger

from simulate import simulate_ride_data


def cluster_and_label(data, create_and_show_plot=True):
    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    logger.info('Estimated number of clusters: %d' % n_clusters_)
    logger.info('Estimated number of noise points: %d' % n_noise_)
    try:
        silhouette_sc = metrics.silhouette_score(data, labels)
    except Exception as e:
        
        logger.error(f"Error calculating silhouette score: {e}")
    
    logger.info("Silhouette Coefficient: %0.3f" % silhouette_sc)
    run_metadata = {'nClusters': n_clusters_, 'nNoise': n_noise_, 
                    'silhouette': metrics.silhouette_score(data, labels),
                    'labels': labels}

    if create_and_show_plot:
        # Black removed and is used for noise instead.
        plot_cluster_results(data, labels, core_samples_mask, n_clusters_)
    else:
        pass

    return run_metadata


def plot_cluster_results(data, labels,
                        core_samples_mask,
                        n_clusters_):
    # Black removed and is used for noise instead.
    fig = plt.figure(figsize=(10, 10))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '^', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)
       
    plt.xlabel('Standardized Scaled Ride Distance')
    plt.ylabel('Standardized Scaled Ride Time')
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig('cluster_results.png')
    plt.show()
    
    
if __name__ == "__main__":
    # Load ride data
    import os
    file_path = 'taxi-rides.csv'
    if os.path.exists(file_path):
        ride_data = pd.read_csv(file_path)
    else:
        logger.info('Simulating ride data ...   ')
        ride_data = simulate_ride_data()
        ride_data.to_csv(file_path, index=False)    
    print(ride_data.head())
    X  = ride_data[['ride_dists', 'ride_times']]
    logger.info('Cluster and labeling ride data ...')
    # Cluster and label ride data
    cluster_and_label(ride_data)
    