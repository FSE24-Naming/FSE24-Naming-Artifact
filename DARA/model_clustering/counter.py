    
import json
from loguru import logger


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def counter(eps, cluster, outliers):
    
    num_clusters = 0
    num_groups = 0
    for key in cluster:
        for subkey in cluster[key]:
            num_clusters += len(cluster[key][subkey])
            num_groups += 1

    num_outliers = 0
    for key in outliers:
        num_outliers += len(outliers[key])

    return num_groups, num_clusters, num_outliers


if __name__ == '__main__':
    eps = 0.031
    cluster_path = f'./Results/{eps}_[0, 1, 0.1]/clusters.json'
    outliers_path = f'./Results/{eps}_[0, 1, 0.1]/outliers.json'

    cluster = load_json(cluster_path)
    outliers = load_json(outliers_path)

    num_groups, num_clusters, num_outliers = counter(eps, cluster, outliers)
    logger.info(f"Clusters for eps={eps}: {num_clusters}")
    logger.info(f"Outliers for eps={eps}: {num_outliers}")