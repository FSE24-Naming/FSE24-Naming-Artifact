import os
import json
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def counter(eps, cluster, outliers, mode):
    
    
    num_clusters = 0
    num_groups = 0
    if mode == "internal":
        for key in cluster:
            for subkey in cluster[key]:
                num_clusters += len(cluster[key][subkey])
                num_groups += 1

        num_outliers = 0
        for key in outliers:
            num_outliers += len(outliers[key])
    elif mode == "external":
        for key in cluster:
            for subkey in cluster[key]:
                num_clusters += len(cluster[key])
                num_groups += 1
        num_outliers = 0
        for key in outliers:
            num_outliers += len(outliers[key])
    return num_groups, num_clusters, num_outliers


def eps_selection(eps_list, best_eps, mode='internal'):
    nums_groups = []
    nums_outliers = []
    eps_list = []
    # load each file from Results
    if mode == "internal":
        root_path = "./Results/internal/"
    elif mode == "external":
        root_path = "./Results/external/"

    # load all files in each root_path folder
    for saved_path in os.listdir(root_path):
        # logger.info(saved_path)
        cluster_path = root_path + saved_path + '/clusters.json'
        outliers_path = root_path + saved_path + '/outliers.json'
        eps = float(saved_path.split('_')[0])
        eps_list.append(eps)
        cluster = load_json(cluster_path)
        outliers = load_json(outliers_path)

                
        
        num_groups, num_clusters, num_outliers = counter(eps, cluster, outliers, mode=mode)
        # logger.info(f"Clusters for eps={eps}: {num_clusters}")
        # logger.info(f"Outliers for eps={eps}: {num_outliers}")
        
        nums_groups.append(num_groups)
        nums_outliers.append(num_outliers)
        
        # num_wrong_name = 0
        # num_groups, num_clusters, num_outliers = counter(eps, cluster, outliers, mode=mode)
        # # logger.info(f"Clusters for eps={eps}: {num_clusters}")
        # # logger.info(f"Outliers for eps={eps}: {num_outliers}")
        # for key in cluster:
        #     for subkey in cluster[key]:
        #         for model_name in cluster[key][subkey]:
        #             if key.lower() not in model_name.lower():
        #                 num_wrong_name += 1
        #                 # logger.debug(f"{key}: {model_name}")
        # # logger.success(f"Misleading name number: {num_wrong_name}")                    
        # nums_groups.append(num_groups)
        # nums_outliers.append(num_outliers)
    # sort eps_list, outliers and cluster by eps value
    eps_list, nums_groups, nums_outliers = zip(*sorted(zip(eps_list, nums_groups, nums_outliers)))
    plt.plot(eps_list, nums_groups, label='groups')
    plt.plot(eps_list, nums_outliers, label='outliers')

    # Plot the difference between groups and outliers
    # plt.plot(eps_list, np.abs(np.array(nums_groups)-np.array(nums_outliers)), label='Abs(groups-outliers)')
    ################################################################################
    # Add a vertical line at eps=0.031
    plt.axvline(x=best_eps, color='red', linestyle='--', label=f"eps={best_eps}")  # You can adjust color and linestyle as needed

    ################################################################################

    plt.legend()
    # plot_title = 'eps selection'
    # plt.title(plot_title, fontsize=25)
    plt.xlabel('eps values', fontsize=18)
    plt.ylabel('Number of groups/outliers', fontsize=18)

    if mode == "internal":
        plt.savefig('eps_selection_internal.png')
    elif mode == "external":
        plt.savefig('eps_selection_external.png')

    plt.close()
        # logger.debug(num_outliers/(num_clusters+num_outliers))
    return eps_list, nums_groups, nums_outliers


def opt_eps_selection(eps_list, nums_groups, nums_outliers):
    
    gradient = np.gradient(np.array(nums_groups))+np.gradient(np.array(nums_outliers))
    plt.plot(eps_list, gradient)
    plt.savefig('gradient_diffs.png')
    plt.close()
    optimal_idx = np.argmin(np.abs(gradient))
    optimal_eps = eps_list[optimal_idx]
    return optimal_eps


if __name__ == '__main__':
    internal_eps_list =  np.arange(0.001, 0.1, 0.001)
    internal_best_eps = 0.027
    internal_eps_list, in_nums_groups, in_nums_outliers = eps_selection(internal_eps_list, best_eps=internal_best_eps, mode='internal')

    external_eps_list = np.arange(1e-3, 0.1, 1e-3)
    external_best_eps = 0.01
    external_eps_list, ex_nums_groups, ex_nums_outliers = eps_selection(external_eps_list, best_eps=external_best_eps, mode='external')

    
    in_optimal_eps = opt_eps_selection(internal_eps_list, in_nums_groups, in_nums_outliers)
    ex_optimal_eps = opt_eps_selection(external_eps_list, ex_nums_groups, ex_nums_outliers)
    print(f"in_optimal_eps: {in_optimal_eps}")
    print(f"ex_optimal_eps: {ex_optimal_eps}")


    plt.plot(internal_eps_list, in_nums_groups, color='royalblue', label='internal_groups')
    plt.plot(internal_eps_list, in_nums_outliers, color='cornflowerblue', label='internal_outliers')

    plt.plot(external_eps_list, ex_nums_groups, color='lightsalmon', label='external_groups')
    plt.plot(external_eps_list, ex_nums_outliers, color='salmon', label='external_outliers')

    # Plot the difference between groups and outliers
    # plt.plot(eps_list, np.abs(np.array(nums_groups)-np.array(nums_outliers)), label='Abs(groups-outliers)')
    ################################################################################
    plt.axvline(x=in_optimal_eps, color='blue', linestyle='--', label=f"in_opt_eps={internal_best_eps}")
    plt.axvline(x=ex_optimal_eps, color='red', linestyle='--', label=f"ex_opt_eps={external_best_eps}")
    ################################################################################

    plt.legend(fontsize=12, loc='upper right')
    # plot_title = 'eps selection'
    # plt.title(plot_title, fontsize=25)
    plt.xlabel('eps values', fontsize=18)
    plt.ylabel('Number of groups/outliers', fontsize=18)
    plt.savefig('eps_selection.png')
    plt.close()
