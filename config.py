from datetime import datetime
import os
import numpy as np
class MultiClassClassification:

    overlap_assort_seperated = {

        "task": "multiclass",

        "community_sizes": [90, 130, 200, 60], # fixed
        "cluster_sizes": [90, 130, 200, 60], # same as com_size -> overlap
        "m_features": 6,
        "k_clusters": 4,
        "alpha": 2, "beta": 20, "lmbd": .5,

        "between_com_prob_range": (.15, .2),
        "within_com_prob_range": (.4, .5), # within_range > between -> assortative

        "centroid_variance_range": (1, 2),  # spectral detectability of feature cluster
        "within_clust_variance_range": (.5, 1),

        "within_clust_covariance_range": (0, .0), # not important; fixed
        "centroid_covariance_range": (0, 0), # not important; fixed

        "n_targets": 5,
        "degree_importance": 1.5,
        "x_importance": 3,
        "feature_info": "number",  # "number" or "cluster" use x right awy or dummies for cluster
        "community_importance": 1.5, # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1.1,
        "model_error": 5,
    }

def note_config(params: dict, graph_dir: str):
    # Timestamps in filename advised
    ts = datetime.now().strftime("%Y-%m-%d %H:%M").replace(":", "_").replace(" ", "_")

    graph_dir += "_" + ts
    base = r"C:\Users\zogaj\PycharmProjects\MA\SyntheticGraphs"
    subdir_path = os.path.join(base, graph_dir)
    os.makedirs(subdir_path, exist_ok=True)

    full_path = os.path.join(subdir_path, "params.txt")

    with open(full_path, 'w') as file:
        for k, v in params.items():
            file.write(f"{k}: {v}\n")

    print(f"Configuration written to {full_path}")
