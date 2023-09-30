import json
import os
import pickle
import time
import re
import warnings


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import great_circle
from loguru import logger
from matplotlib.lines import Line2D
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

from loguru import logger


# load HF internal vectors
D_PATH = "./vectors/vec_d.pkl"
L_PATH = "./vectors/vec_l.pkl"
P_PATH = "./vectors/vec_p.pkl"

# load external vectors 
ONNX_D_PATH = "./vectors/external/vec_d.pickle"
ONNX_L_PATH = "./vectors/external/vec_l.pickle"
ONNX_P_PATH = "./vectors/external/vec_p.pickle"

def load_vec(path):
    # load pickle file from path
    with open(path, 'rb') as f:
        return pickle.load(f)


def concatenate_vec(dim_vec, layer_vec, param_vec, weights=[0, 1, 0.1], mode='internal'):
    logger.info(f"Concatenating vectors with weights [d, l, p]: {weights}...")
    # concatenate three vectors
    model_vec = {}

    weight_d = weights[0]
    weight_l = weights[1]
    weight_p = weights[2]

    if mode == 'internal':
        for model_arch in dim_vec:
            model_family = re.split('For|Model|LMHead', model_arch)[0]
            if model_family not in model_vec:
                model_vec[model_family] = {}
            models_dim_vec = dim_vec[model_arch]
            models_layer_vec = layer_vec[model_arch]
            models_param_vec = param_vec[model_arch]

            # if model_arch not in model_vec:
            #     model_vec[model_arch] = {}
                
            for model_name in models_dim_vec:
                dim_arr = np.array(models_dim_vec[model_name])
                layer_arr = np.array(models_layer_vec[model_name])
                param_arr = np.array(models_param_vec[model_name])
                model_vec[model_family][model_name] = np.concatenate(
                    (weight_d * dim_arr, weight_l * layer_arr, weight_p * param_arr))
    
    elif mode == 'external':
        for model_family in dim_vec:
            for model_framework in dim_vec[model_family]:
                if model_family not in model_vec:
                    model_vec[model_family] = {}
                models_dim_vec = dim_vec[model_family][model_framework]
                models_layer_vec = layer_vec[model_family][model_framework]
                models_param_vec = param_vec[model_family][model_framework]
                # if model_arch not in model_vec:
                #     model_vec[model_arch] = {})
                for model_name in models_dim_vec:
                    dim_arr = np.array(models_dim_vec[model_name])
                    layer_arr = np.array(models_layer_vec[model_name])
                    param_arr = np.array(models_param_vec[model_name])
                    model_name_full = "/".join([model_framework, model_name])
                    model_vec[model_family][model_name_full] = np.concatenate(
                        (weight_d * dim_arr, weight_l * layer_arr, weight_p * param_arr))
    else:
        Exception("Mode not supported!")
    return model_vec
    


def save_vec(model_vec, name="model_vec"):
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model_vec, f)


def model_clustering(model_vec, eps=0.8, weights=[0, 1, 0.1], mode="internal", plot=False):
    results = {}
    outliers = {}
    # Initialize PCA
    pca = PCA(n_components=2)

    EPS = eps
    if mode == "internal":
        saved_path = f"./Results/internal/{EPS:.3f}_{weights}/"
    elif mode == "external":
        saved_path = f'./Results/external/{EPS:.3f}_{weights}/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    


    for model_family in tqdm(model_vec):

        results[model_family] = {}
        outliers[model_family] = []
        # Create a larger plot
        plt.figure(figsize=(10, 10))
        # if model_arch != "AlbertForMaskedLM":
        #     continue
        # logger.info(f"Clustering {len(model_vec[model_arch])} models in {model_arch}...")
        data = list(model_vec[model_family].values())

        model_names = list(model_vec[model_family].keys())
        if len(data) < 2:
            logger.warning(f"Not enough data to cluster model family {model_family}")
            continue

        # Standardize data to have a mean of ~0 and a variance of 1
        X_std = StandardScaler().fit_transform(data)


        # Check for zero variance features and remove them
        non_zero_var_indices = np.var(X_std, axis=0) != 0
        X_filtered = X_std[:, non_zero_var_indices]

        try:
            # Perform PCA
            data_pca = pca.fit_transform(X_filtered)
            # logger.debug(data_pca)
                

            # Compute the cosine distances
            # cosine_distances = pairwise_distances(data_pca, metric='cosine')
            euclidean_distances = pairwise_distances(data_pca, metric='euclidean')
            # logger.debug(cosine_distances)
            # Perform DBSCAN on the data
            db = DBSCAN(eps=EPS, min_samples=2, metric='precomputed').fit(euclidean_distances)


            labels = db.labels_

            # Identify core samples
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            # Plot result
            unique_labels = set(labels)
            # logger.debug(f"Unique labels: {unique_labels}")
            colors = [plt.cm.Spectral(each)
                    for each in np.linspace(0, 1, len(unique_labels))]

            cluster_names = {}
            # Annotate points for both core and non-core
            for k, col in zip(unique_labels, colors):
                class_member_mask = (labels == k)
                xy = data_pca[class_member_mask]

                if k != -1: 
                    cluster_names[k] = model_names[class_member_mask.tolist().index(True)]
                    centroid = np.mean(xy, axis=0)
                    plt.text(centroid[0]-1, centroid[1]+0.5, cluster_names[k], fontsize=15)

                    
                    # Initialize the list for the cluster if not already done
                    if k not in results[model_family]:
                        results[model_family][str(k)] = []
                    
                    # Save the model names for the cluster
                    for idx in np.where(class_member_mask)[0]:
                        results[model_family][str(k)].append(model_names[idx])


                if k == -1:
                    col = [0, 0, 0, 1]
                    for idx, (x, y) in zip(np.where(class_member_mask)[0], xy):
                        plt.text(x-1, y+0.1, model_names[idx], fontsize=15, color='black')
                        outliers[model_family].append(model_names[idx])
                
                if plot==True:
                    class_member_mask = (labels == k)
                    xy = data_pca[class_member_mask & core_samples_mask]
                    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=20)

                    xy = data_pca[class_member_mask & ~core_samples_mask]
                    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                            markeredgecolor='k', markersize=10)
                    plt.title(
                        f'{model_family}: {n_clusters_} clusters, {n_noise_} outliers', fontsize=30)
                    plt.axis('off')
                    plt.savefig(
                        saved_path+f"{model_family}_{n_clusters_}clusters_{n_noise_}outliers.png", pad_inches=1)
                    plt.close()
        except:
            # logger.warning(f"No variance in model family {model_family}")
            if not results[model_family]:
                results[model_family]['0'] = []
            results[model_family]['0'].extend(model_names)
        # Save 'results' to a JSON file
        with open(saved_path+"clusters.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

        # Save 'outliers' to a JSON file
        with open(saved_path+"outliers.json", "w") as json_file:
            json.dump(outliers, json_file, indent=4)

        
    return


def internal_clustering(load_pkl=False, weights=[0, 1, 0.1]):
    if load_pkl==True:
        if os.path.exists(f"model_vec.pkl"):
            with open(f'model_vec.pkl', 'rb') as f:
                model_vec = pickle.load(f)
    else:
        dim_vec = load_vec(D_PATH)
        layer_vec = load_vec(L_PATH)
        param_vec = load_vec(P_PATH)
        # logger.debug(dim_vec.keys())
        # logger.debug(layer_vec.keys())
        # logger.debug(param_vec.keys())

        model_vec = concatenate_vec(dim_vec, layer_vec, param_vec, weights=weights)
        # logger.debug(model_vec)
        save_vec(model_vec)
        for eps in np.arange(1e-3, 0.1, 1e-3):
            logger.info(f"Clustering model with eps={eps}!")
            model_clustering(model_vec, eps=eps, weights=weights, plot=True)
    # model_clustering_all_3D(model_vec, eps=eps)

def external_clustering(load_pkl=False, weights=[0, 1, 0.1]):
    if load_pkl==True:
        if os.path.exists(f"external_model_vec.pkl"):
            with open(f'external_model_vec.pkl', 'rb') as f:
                model_vec = pickle.load(f)
    else:
        dim_vec = load_vec(ONNX_D_PATH)
        layer_vec = load_vec(ONNX_L_PATH)
        param_vec = load_vec(ONNX_P_PATH)
        # logger.debug(dim_vec.keys())
        # logger.debug(layer_vec.keys())
        # logger.debug(param_vec.keys())

        model_vec = concatenate_vec(dim_vec, layer_vec, param_vec, weights=weights, mode='external')
        # logger.debug(model_vec)
        save_vec(model_vec, name="external_model_vec")
    for eps in np.arange(1e-3, 0.1, 1e-3):
    # eps = 1.0
        logger.info(f"Clustering model with eps={eps}!")
        model_clustering(model_vec, eps=eps, weights=weights, mode="external")
    # model_clustering_all_3D(model_vec, eps=eps)


if __name__ == "__main__":
    # json_files = load_vec(["../comparators/pytorch/ptm_vectors/vec_d.json", "../comparators/pytorch/ptm_vectors/vec_l.json", "../comparators/pytorch/ptm_vectors/vec_p.json"])
    
    internal_clustering(weights=[0, 1, 0])
    external_clustering(weights=[0, 1, 0])
