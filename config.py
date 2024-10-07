from datetime import datetime
import os
import numpy as np

# Note that for Community or cluster relevance, we assign equal sizes accordingly.
class Scenarios:

    # ---- Not in the iterator for Monte-Carlo -----
    # Just for illustrative purposes
    # Well seperated centroids: centroid sigma = 6 & cluster sigma = 1
    # Poorly seperated centroids: centroid sigma = 6 & cluster sigma = 5
    # Homophily: off-diagonal = .1 & diagonal = .5
    # Heterophily: off-diagonal = .5 & diagonal = .25
    # importance: at 6 & unimportance at 1
    # Model error: 2 for perfect, 9 for noise and 3 else
    # 3 and 5 questionable 

    perfect = {

        "name": "perfect",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],  # same as com_size -> overlap
        "m_features": 10,
        # "k_clusters": 4,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.6, .6),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # irrelevant

        "x_importance": 3,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 3,  # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }  # perfect

    community_relevant = {
        # Well seperated by community-connectivity + homophily. But irrelevant features, with poor cluster separation.

        "name": "community_relevant",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.6, .6),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # no meaningfully interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 3,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }  # community relevant

    community_relevant_heterophilic = {

        # Relevant community (in terms of parameter) but heterophilic graph.
        # Features are mildly relevant.

        "name": "community_relevant_heterophilic",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],  # same as com_size -> overlap
        "m_features": 10,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.75, .75),
        "within_com_prob_range": (.025, .025),

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # no meaningfully interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 3,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }  # community_relevant_heterophilic

    cluster_relevant = {
        # Well seperated by features-cluster with relevant parameters.
        # However, no edge clustering and no relevance of community belonging.

        "name": "cluster_relevant",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],  # same as com_size -> overlap
        "m_features": 10,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (15, 15),  # spectral detectability of feature cluster
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 3,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 1,  # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }  # cluster_relevant

    non_seperated_cluster_relevant = {
        # Feature cluster explains target well, but clusters are not well seperated.
        # Communities are well seperated, but not really relevant.

        "name": "non_seperated_cluster_relevant",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],  # same as com_size -> overlap
        "m_features": 10,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),  # heterophilic

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, .0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,

        "x_importance": 3,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 1,  # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }  # non_seperated_cluster_relevant

    noise = {

        "name": "noise",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],  # same as com_size -> overlap
        "m_features": 10,
        # "k_clusters": 4,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.2, .2),
        "within_com_prob_range": (.3, .3),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (8, 8),

        "within_clust_covariance_range": (0, .0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 2,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 2,  # this scale shouldn't be interpreted analog to x and degree
        "community_importance_exponent": 1,
        "model_error": 4,
        "splitweights": [.05, .2, .3]

    }  # noise

    # ------ Hyper parameters across all models -------

    hyperparams = {
        # "out_dim": num_targets,
        # "in_dim": num_input_features,

        "weight_decay": 5e-4,
        "hidden_layer1_dim": 16,
        "drop_out1": .4,
        "learn_rate": 0.01,
        "attention_heads": 8
    }

    def __init__(self):
        """
        Must have the same order for coherence !!
        Not every scenario must be in the iterable, but what's in the
        iterator determines, what's handled in main.py
        After one full iteration it does not revert back to first index.
        """

        self._elements = [self.perfect,
                          self.cluster_relevant,
                          self.community_relevant,
                          self.community_relevant_heterophilic,
                          #self.community_relevant_sparse,
                          self.non_seperated_cluster_relevant,
                          self.noise
            ]

        # Important for what scenario is actually processed !
        self.list_of_scenarios = [self.perfect["name"],
                                  self.cluster_relevant["name"],
                                  self.community_relevant["name"],
                                  self.community_relevant_heterophilic["name"],
                                  #self.community_relevant_sparse["name"],
                                  self.non_seperated_cluster_relevant["name"],
                                  self.noise["name"]
                                  ]
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

class Sparse:
    # ---- Not in the iterator for Monte-Carlo -----
    # Just for illustrative purposes
    # Well seperated centroids: centroid sigma = 6 & cluster sigma = 1
    # Poorly seperated centroids: centroid sigma = 6 & cluster sigma = 5
    # Homophily: off-diagonal = .1 & diagonal = .5
    # Heterophily: off-diagonal = .5 & diagonal = .25
    # importance: at 6 & unimportance at 1
    # Model error: 2 for perfect, 9 for noise and 3 else
    # 3 and 5 questionable

    very_strong = {

        "name": "very_strong",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.01, .01),
        "within_com_prob_range": (.9, .9),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # no meaningfully interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 4,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }

    strong = {

        "name": "strong",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.15, .15),
        "within_com_prob_range": (.7, .7),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # no meaningfully interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 4,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }  # community relevant

    mid = {

        "name": "mid",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.25, .25),
        "within_com_prob_range": (.5, .5),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # no meaningfully interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 4,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }

    weak = {

        "name": "weak",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.6, .6),
        "within_com_prob_range": (.4, .4),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # no meaningfully interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 4,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }

    very_weak = {

        "name": "strong",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        # "k_clusters": 4, determined by "cluster_size"
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.9, .9),  # within_range > between -> assortative

        "centroid_variance_range": (10, 10),  # spectral detectability of feature cluster
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,
        "degree_importance": 0,  # no meaningfully interpretation

        "x_importance": 1,
        "feature_info": "cluster",  # "number" or "cluster" use x right away or dummies for cluster

        "community_importance": 4,  # diag entries
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .2, .3]

    }  # community relevant

    # ------ Hyper parameters across all models -------

    hyperparams = {
        # "out_dim": num_targets,
        # "in_dim": num_input_features,

        "weight_decay": 5e-4,
        "hidden_layer1_dim": 16,
        "drop_out1": .4,
        "learn_rate": 0.01,
        "attention_heads": 8
    }

    def __init__(self):
        """
        Must have the same order for coherence !!
        Not every scenario must be in the iterable, but what's in the
        iterator determines, what's handled in main.py
        After one full iteration it does not revert back to first index.
        """

        self._elements = [self.very_strong,
                          self.strong,
                          self.mid,
                          self.weak,
                          self.very_weak]

        # Important for what scenario is actually processed !
        self.list_of_scenarios = [self.very_strong["name"],
                                  self.strong["name"],
                                  self.mid["name"],
                                  self.weak["name"],
                                  self.very_weak["name"]]
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

class Seperable:

    very_strong = {

        "name": "very_strong",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (15, 15),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 4,
        "feature_info": "cluster",

        "community_importance": 1,
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .15, .3]

    }

    strong = {

        "name": "strong",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 4,
        "feature_info": "cluster",

        "community_importance": 1,
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .15, .3]

    }

    mid = {

        "name": "mid",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (5, 5),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 4,
        "feature_info": "cluster",

        "community_importance": 1,
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .15, .3]

    }

    weak = {

        "name": "weak",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (1, 1),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 4,
        "feature_info": "cluster",

        "community_importance": 1,
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .15, .3]

    }

    very_weak = {

        "name": "very_weak",
        "task": "multiclass",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 6,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (1, 1),
        "within_clust_variance_range": (5, 5),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 4,
        "feature_info": "cluster",

        "community_importance": 1,
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .15, .3]

    }


    hyperparams = {
        # "out_dim": num_targets,
        # "in_dim": num_input_features,

        "weight_decay": 5e-4,
        "hidden_layer1_dim": 16,
        "drop_out1": .4,
        "learn_rate": 0.01,
        "attention_heads": 8
    }

    def __init__(self):
        """
        Must have the same order for coherence !!
        Not every scenario must be in the iterable, but what's in the
        iterator determines, what's handled in main.py
        After one full iteration it does not revert back to first index.
        """

        self._elements = [self.very_strong,
                          self.strong,
                          self.mid,
                          self.weak,
                          self.very_weak]

        # Important for what scenario is actually processed !
        self.list_of_scenarios = [self.very_strong["name"],
                                  self.strong["name"],
                                  self.mid["name"],
                                  self.weak["name"],
                                  self.very_weak["name"]]
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
