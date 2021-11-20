import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from database import parsed_files, extract_clustering_features, clustering_feature_labels, query_rows
import matplotlib.pyplot as plt

import timeit

# pick a dataset
X, y = extract_clustering_features(parsed_files["player_regular_season.txt"])

# instantiate the sklearn clustering algorithms

KMeans = KMeans(n_clusters=5, random_state=0)

MeanShift = MeanShift()

DBSCAN = DBSCAN()

AggClustering = AgglomerativeClustering(n_clusters=3)

# use the clustering algorithms on the data

clustering = [KMeans]  # , MeanShift, DBSCAN, AggClustering]
predicted_cluster_labels = []

for c in clustering:
    start = timeit.default_timer()
    labels = c.fit_predict(X)
    print('Done in ', timeit.default_timer() - start, ' seconds')
    predicted_cluster_labels.append(labels)


def player_names(cluster_labels, filename='player_regular_season.txt'):
    # names of players in the cluster
    names = []

    def cluster_player_test(row): return row["ilkid"] in cluster_labels

    rows = query_rows(cluster_player_test, parsed_files[filename])

    # read the player names from file
    player_names = np.array([row['firstname'] + ' ' + row['lastname'] for row in rows])

    # remove duplicates as there may be players for more than one season
    for n in player_names:
        player_name = n.upper()
        if player_name not in names:
            names.append(player_name)

    return names


def plot_cluster_averages(averages):
    c1, c2, c3, c4, c5 = (np.array(list(a.values())) for a in averages)
    fig, ax = plt.subplots()
    labels = clustering_feature_labels
    x = np.arange(len(labels))
    width = 0.1
    rects1 = ax.bar(x, c1, width, label="Cluster 1")
    rects2 = ax.bar(x + width, c2, width, label="Cluster 2")
    rects3 = ax.bar(x + 2 * width, c3, width, label="Cluster 3")
    rects4 = ax.bar(x + 3 * width, c4, width, label="Cluster 4")
    rects5 = ax.bar(x + 4 * width, c5, width, label="Cluster 5")
    ax.set_ylabel("Scaled units")
    ax.set_title("Average feature values per cluster.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


clustered_features = []
clustered_labels = []


# output number of features in each cluster, according to each clustering algorithm
for (i, labels) in enumerate(predicted_cluster_labels):  # for each set of predictions
    print(f"\npredictor {i}")
    averages = []
    for cluster_label in [0, 1, 2, 3, 4]:       # for each label
        cluster_features = np.array(X)[labels == cluster_label]
        cluster_labels = np.array(y)[labels == cluster_label]

        if cluster_label in [0, 4]:
            clustered_features.append(cluster_features)
            clustered_labels.append(cluster_labels)

        feature_means = dict(zip(clustering_feature_labels, np.mean(cluster_features, axis=0)))  # axis=0 to take means for each feature across samples
        averages.append(feature_means)
        def cluster_player_test(row): return row["ilkid"] in cluster_labels
        player_info_rows = query_rows(cluster_player_test, parsed_files["players.txt"])
        average_height = np.mean([int(row["h_feet"]) + float(row["h_inches"]) / 12.0 if row["h_feet"] != '' else 6.5 for row in player_info_rows])
        average_weight = np.mean([int(row["weight"]) if row["weight"] != '' else 217 for row in player_info_rows])
        count = len(cluster_features)
        print(f"{count} in cluster {cluster_label}")
        print(f"feature means: {feature_means}")
        print(f"average height (ft) and weight (lbs?): {average_height}, {average_weight}")
    plot_cluster_averages(averages)


# _______________________________________________________________________________


# perform clustering on the players in cluster 1
print('\nRefining cluster 1')
cluster_1_X = clustered_features[0]
cluster_1_y = clustered_labels[0]

predicted_cluster_labels = []
for c in clustering:
    start = timeit.default_timer()
    labels = c.fit_predict(cluster_1_X)
    print('Done in ', timeit.default_timer() - start, ' seconds')
    predicted_cluster_labels.append(labels)


# output number of features in each cluster, according to each clustering algorithm
for (i, labels) in enumerate(predicted_cluster_labels):  # for each set of predictions
    print(f"\npredictor {i}")
    averages = []
    for cluster_label in [0, 1, 2, 3, 4]:       # for each label
        cluster_features = np.array(cluster_1_X)[labels == cluster_label]
        cluster_labels = np.array(cluster_1_y)[labels == cluster_label]
        feature_means = dict(zip(clustering_feature_labels, np.mean(cluster_features, axis=0)))  # axis=0 to take means for each feature across samples
        averages.append(feature_means)
        def cluster_player_test(row): return row["ilkid"] in cluster_labels
        player_info_rows = query_rows(cluster_player_test, parsed_files["players.txt"])

        if cluster_label == 1:
            player_shooting_names = player_names(cluster_labels)

        average_height = np.mean([int(row["h_feet"]) + float(row["h_inches"]) / 12.0 if row["h_feet"] != '' else 6.5 for row in player_info_rows])
        average_weight = np.mean([int(row["weight"]) if row["weight"] != '' else 217 for row in player_info_rows])
        count = len(cluster_features)
        print(f"{count} in cluster {cluster_label}")
        print(f"feature means: {feature_means}")
        print(f"average height (ft) and weight (lbs?): {average_height}, {average_weight}")
    plot_cluster_averages(averages)


textfile = open('best_players_1.txt', 'w')
textfile.write(str(len(player_shooting_names)) + '\n')
for n in player_shooting_names:
    textfile.write(n + '\n')
textfile.close()

# _______________________________________________________________________________

# perform clustering on the players in cluster 5
print('\nRefining cluster 5')
cluster_1_X = clustered_features[1]
cluster_1_y = clustered_labels[1]

predicted_cluster_labels = []
for c in clustering:
    start = timeit.default_timer()
    labels = c.fit_predict(cluster_1_X)
    print('Done in ', timeit.default_timer() - start, ' seconds')
    predicted_cluster_labels.append(labels)


# output number of features in each cluster, according to each clustering algorithm
for (i, labels) in enumerate(predicted_cluster_labels):  # for each set of predictions
    print(f"\npredictor {i}")
    averages = []
    for cluster_label in [0, 1, 2, 3, 4]:       # for each label
        cluster_features = np.array(cluster_1_X)[labels == cluster_label]
        cluster_labels = np.array(cluster_1_y)[labels == cluster_label]
        feature_means = dict(zip(clustering_feature_labels, np.mean(cluster_features, axis=0)))  # axis=0 to take means for each feature across samples
        averages.append(feature_means)
        def cluster_player_test(row): return row["ilkid"] in cluster_labels
        player_info_rows = query_rows(cluster_player_test, parsed_files["players.txt"])

        if cluster_label == 3:
            player_forward_names = player_names(cluster_labels)

        average_height = np.mean([int(row["h_feet"]) + float(row["h_inches"]) / 12.0 if row["h_feet"] != '' else 6.5 for row in player_info_rows])
        average_weight = np.mean([int(row["weight"]) if row["weight"] != '' else 217 for row in player_info_rows])
        count = len(cluster_features)
        print(f"{count} in cluster {cluster_label}")
        print(f"feature means: {feature_means}")
        print(f"average height (ft) and weight (lbs?): {average_height}, {average_weight}")
    plot_cluster_averages(averages)

textfile = open('best_players_2.txt', 'w')
textfile.write(str(len(player_forward_names)) + '\n')
for n in player_forward_names:
    textfile.write(n + '\n')
textfile.close()
