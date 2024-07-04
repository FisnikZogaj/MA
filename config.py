from datetime import datetime
import os
import numpy as np
class MultiClassClassification:

    overlap_assort_seperated = {

        "task": "multiclass",

        "community_sizes": [200, 230, 300, 260], # fixed
        "cluster_sizes": [200, 230, 300, 260], # same as com_size -> overlap
        "m_features": 6,
        # "k_clusters": 4,
        "alpha": 2, "beta": 20, "lmbd": .5,

        "between_com_prob_range": (.01, .02),
        "within_com_prob_range": (.5, .6), # within_range > between -> assortative

        "centroid_variance_range": (3, 6),  # spectral detectability of feature cluster
        "within_clust_variance_range": (.5, 1),

        "within_clust_covariance_range": (0, .0), # not important; fixed
        "centroid_covariance_range": (0, 0), # not important; fixed

        "n_targets": 5,
        "degree_importance": 1.5,

        "x_importance": 2,
        "feature_info": "number",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 8.5, # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1.1,
        "model_error": 5,
        "splitweights": [.7, .2, .1]

    }
