import pyspark
import random
import numpy as np
from pyspark.shell import sc, spark
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score, adjusted_rand_score
import pandas as pd
from pyspark import SparkContext
import math
import psutil


def mapper(cent, point):
    min_dist = calc_dist(point, cent[0])
    closest = 0
    for i in range(1, len(cent)):
        curr_dist = calc_dist(point, cent[i])
        if curr_dist < min_dist:
            min_dist = curr_dist
            closest = i
    return closest


def calc_dist(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))


def reduce(x):
    values = x[1]
    return np.mean(values, axis=0)


def K_MEANS_ALGO(csv_name, K, ct=0.0001, iter=30, exp=10):
    """
    Runs the K-means clustering algorithm on a given dataset.
    """
    # Ensure the correct file extension is included in the path
    data = spark.read.csv(f"{csv_name}.csv", header=True, inferSchema=True)
    rdd = data.rdd.map(lambda x: x[:-1])  # Extract features, ignoring the last column (assumed to be labels)
    scaler = MinMaxScaler()
    rdd = sc.parallelize(scaler.fit_transform(rdd.collect()))

    labels_true = np.array(data.rdd.map(lambda x: x[-1]).collect())

    scores = [[], []]

    for _ in range(exp):
        centroids = rdd.takeSample(False, K)
        centroids_broadcast = sc.broadcast(centroids)

        for _ in range(iter):
            mapped = rdd.map(lambda x: (mapper(centroids_broadcast.value, x), x))
            combined = mapped.groupByKey().mapValues(list)
            reduced = combined.map(lambda x: reduce(x)).collect()

            if all(calc_dist(reduced[i], centroids_broadcast.value[i]) <= ct for i in range(K)):
                break

            centroids = reduced
            centroids_broadcast = sc.broadcast(centroids)

        labels_pred = np.array(mapped.map(lambda x: x[0]).collect())

        scores[0].append(calinski_harabasz_score(rdd.collect(), labels_pred))
        scores[1].append(adjusted_rand_score(labels_true, labels_pred))

    return csv_name, K, (np.mean(scores[0]), np.std(scores[0])), (np.mean(scores[1]), np.std(scores[1]))


if __name__ == '__main__':
    """
        Runs K-means on multiple datasets and saves results.
        """
    names = ["iris", "glass", "parkinsons"]
    K_values = [2, 3, 5, 6, 8]

    df = pd.DataFrame(columns=['Dataset name', 'K', "Average and std CH", "Average and std ARI"])

    for name in names:
        for k in K_values:
            df.loc[len(df)] = K_MEANS_ALGO(name, k)

    df.to_csv('kmeans_results.csv', index=False)
