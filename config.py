from datetime import datetime
import os
import numpy as np
class Scenarios:

    perfect = {
        # Well seperated by features and community-connectivity (nodes from different communities are not likely, to
        # contribute to nodes from other communities, due to sparse connections). n_com = n_clust = n_target = 4

        "name": "perfect",
        "task": "multiclass",

        "community_sizes": [70, 230, 130, 240, 140, 90], # fixed
        "cluster_sizes": [70, 230, 130, 240, 140, 90], # same as com_size -> overlap
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 2, "beta": 20, "lmbd": .5,

        "between_com_prob_range": (.01, .02),
        "within_com_prob_range": (.5, .6), # within_range > between -> assortative

        "centroid_variance_range": (3, 4),  # spectral detectability of feature cluster
        "within_clust_variance_range": (.5, 1),

        "within_clust_covariance_range": (0, .0), # not important; fixed
        "centroid_covariance_range": (0, 0), # not important; fixed

        "n_targets": 6,
        "degree_importance": 1, # no meaningfull interpretation

        "x_importance": 4.5,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 4.5, # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1,
        "model_error": 3,
        "splitweights": [.7, .2, .1]

    }

    community_relevant = {
        # Well seperated by community-connectivity (nodes from different communities are not likely, to
        # contribute to nodes from other communities, due to sparse connections). n_com = n_clust = n_target = 4

        "name": "community_relevant",
        "task": "multiclass",

        "community_sizes": [150, 150, 150, 150, 150, 150],  # fixed
        "cluster_sizes": [70, 230, 130, 240, 140, 90],  # same as com_size -> overlap
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 2, "beta": 20, "lmbd": .5,

        "between_com_prob_range": (.05, .1),
        "within_com_prob_range": (.5, .6),  # within_range > between -> assortative

        "centroid_variance_range": (2, 4),  # spectral detectability of feature cluster
        "within_clust_variance_range": (1, 2),

        "within_clust_covariance_range": (0, .0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 6,
        "degree_importance": 1,  # no meaningfull interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 6,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 3,
        "splitweights": [.7, .2, .1]

    }

    cluster_relevant = {
        # Well seperated by features (nodes from different communities are likely, to
        # contribute to nodes from other communities, due to inter-community links ). n_com = n_clust = n_target = 4

        "name": "cluster_relevant",
        "task": "multiclass",

        "community_sizes": [70, 230, 130, 240, 140, 90],  # fixed
        "cluster_sizes": [150, 150, 150, 150, 150, 150],  # same as com_size -> overlap
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 2, "beta": 20, "lmbd": .5,

        "between_com_prob_range": (.3, .4),
        "within_com_prob_range": (.3, .4), # heterophillic

        "centroid_variance_range": (4, 6),  # spectral detectability of feature cluster
        "within_clust_variance_range": (.5, 1),

        "within_clust_covariance_range": (0, .0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 6,
        "degree_importance": 1,

        "x_importance": 6,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 1,  # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1,
        "model_error": 3,
        "splitweights": [.7, .2, .1]

    }

    noise = {

            "name": "noise",
            "task": "multiclass",

        "community_sizes": [70, 230, 130, 240, 140, 90],  # fixed
        "cluster_sizes": [150, 150, 150, 150, 150, 150], # same as com_size -> overlap
            "m_features": 6,
            # "k_clusters": 4,
            "alpha": 2, "beta": 20, "lmbd": .5,

            "between_com_prob_range": (.4, .5),
            "within_com_prob_range": (.5, .6), # within_range > between -> assortative

            "centroid_variance_range": (2, 4),  # spectral detectability of feature cluster
            "within_clust_variance_range": (1, 2),

            "within_clust_covariance_range": (0, .0), # not important; fixed
            "centroid_covariance_range": (0, 0), # not important; fixed

            "n_targets": 6,
            "degree_importance": 1.5,

            "x_importance": 1.5,
            "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

            "community_importance": 1.5, # this scale shouldn't be interpreted analog to x and degree
            "community_importance_exponent": 1,
            "model_error": 10,
            "splitweights": [.7, .2, .1]

        }


    hyperparams = {
        # "out_dim": num_targets,
        #"in_dim": num_input_features,
        "weight_decay": 5e-4,
        "hidden_layer1_dim": 16,
        "hidden_layer2_dim": 8,
        "drop_out1": .4,
        "drop_out2": .1,
        "learn_rate": 0.01,
        "attention_heads": 8
    }


    def __init__(self):
        """
        Must have the same order for coherence !!
        Not every scenario must be in the iterable, but what's in the
        iterator determines, what's handled in main.py
        """

        self._elements = [self.perfect,
                          self.noise]
        self.list_of_scenarios = [self.perfect["name"],
                                  self.noise["name"]]
        self._index = 0


    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._elements):
            result = self._elements[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    class Hyperparameters:
        pass
