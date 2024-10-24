from datetime import datetime
import os
import numpy as np


class Scenarios:

    perfect = {

        "name": "perfect",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.6, .6),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, 0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "x_importance": 3,
        "community_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    community_relevant = {

        "name": "community_relevant",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.7, .7),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),   # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "x_importance": 1,
        "community_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    community_relevant_heterophilic = {

        "name": "community_relevant_heterophilic",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.8, .8),
        "within_com_prob_range": (.02, .02),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "x_importance": 1,
        "community_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    cluster_relevant = {

        "name": "cluster_relevant",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (18, 18),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, 0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "x_importance": 3,
        "community_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    non_seperated_cluster_relevant = {

        "name": "non_seperated_cluster_relevant",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (14, 14),

        "within_clust_covariance_range": (0, .0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "x_importance": 3,
        "community_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    noise = {

        "name": "noise",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.2, .2),
        "within_com_prob_range": (.3, .3),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (8, 8),

        "within_clust_covariance_range": (0, .0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "x_importance": 2,
        "community_importance": 2,
        "model_error": 4,

        "splitweights": [.05, .2, .3]

    }

    def __init__(self):
        """
        Must have the same order for coherence !!
        Not every scenario must be in the iterable, but what's in the
        iterator determines, what's handled in main.py
        After one full iteration it does not revert back to first index.
        """

        self._elements = [self.perfect,
                          self.community_relevant,
                          self.community_relevant_heterophilic,
                          self.cluster_relevant,
                          self.non_seperated_cluster_relevant,
                          self.noise
            ]

        # Important for what scenario is actually processed !
        self.list_of_scenarios = [self.perfect["name"],
                                  self.community_relevant["name"],
                                  self.community_relevant_heterophilic["name"],
                                  self.cluster_relevant["name"],
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

    very_strong = {

        "name": "very_strong",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.01, .01),
        "within_com_prob_range": (.9, .9),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "x_importance": 1,
        "community_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    strong = {

        "name": "strong",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.15, .15),
        "within_com_prob_range": (.7, .7),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,

        "x_importance": 1,
        "community_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }  # community relevant

    mid = {

        "name": "mid",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.25, .25),
        "within_com_prob_range": (.5, .5),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,

        "x_importance": 1,
        "community_importance": 3,  # diag entries
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    weak = {

        "name": "weak",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.3, .3),
        "within_com_prob_range": (.4, .4),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,

        "x_importance": 1,
        "community_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    very_weak = {

        "name": "very_weak",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.5, .5),
        "within_com_prob_range": (.6, .6),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # not important; fixed
        "centroid_covariance_range": (0, 0),  # not important; fixed

        "n_targets": 5,

        "x_importance": 1,
        "community_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }  # community relevant

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
                          self.very_weak
                          ]

        # Important for what scenario is actually processed !
        self.list_of_scenarios = [self.very_strong["name"],
                                  self.strong["name"],
                                  self.mid["name"],
                                  self.weak["name"],
                                  self.very_weak["name"]
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

class Seperable:

    very_strong = {

        "name": "very_strong",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (18, 18),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,

        "x_importance": 3,
        "community_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .15, .3]

    }

    strong = {

        "name": "strong",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (12, 12),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,

        "x_importance": 3,
        "community_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .15, .3]

    }

    mid = {

        "name": "mid",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (6, 6),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,

        "x_importance": 3,
        "community_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .15, .3]

    }

    weak = {

        "name": "weak",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (1, 1),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,
        "degree_importance": 1,

        "x_importance": 3,
        "feature_info": "cluster",

        "community_importance": 1,
        "community_importance_exponent": 1,
        "model_error": 2,
        "splitweights": [.05, .15, .3]

    }

    very_weak = {

        "name": "very_weak",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,
        "alpha": 1, "beta": 1, "lmbd": -1,

        "between_com_prob_range": (.05, .05),
        "within_com_prob_range": (.05, .05),

        "centroid_variance_range": (1, 1),
        "within_clust_variance_range": (5, 5),

        "within_clust_covariance_range": (0, .0),
        "centroid_covariance_range": (0, 0),

        "n_targets": 5,

        "x_importance": 3,
        "community_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .15, .3]

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
