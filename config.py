class Scenarios:

    perfect = {

        "name": "perfect",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,  # deprecated

        "within_com_prob_range": (.6, .6),
        "between_com_prob_range": (.05, .05),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, 0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "community_importance": 3,
        "x_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]  # train - test - validation

    }

    community_relevant = {

        "name": "community_relevant",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,  # deprecated

        "within_com_prob_range": (.7, .7),
        "between_com_prob_range": (.05, .05),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),   # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "community_importance": 3,
        "x_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    community_relevant_heterophilic = {

        "name": "community_relevant_heterophilic",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,  # deprecated

        "within_com_prob_range": (.02, .02),
        "between_com_prob_range": (.8, .8),

        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (10, 10),

        "within_clust_covariance_range": (0, 0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "community_importance": 3,
        "x_importance": 1,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    cluster_relevant = {

        "name": "cluster_relevant",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,   # deprecated

        "within_com_prob_range": (.05, .05),
        "between_com_prob_range": (.05, .05),

        "centroid_variance_range": (18, 18),
        "within_clust_variance_range": (1, 1),

        "within_clust_covariance_range": (0, 0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "community_importance": 1,
        "x_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    non_seperated_cluster_relevant = {

        "name": "non_seperated_cluster_relevant",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,  # deprecated

        "within_com_prob_range": (.05, .05),
        "between_com_prob_range": (.05, .05),


        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (14, 14),

        "within_clust_covariance_range": (0, .0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "community_importance": 1,
        "x_importance": 3,
        "model_error": 2,

        "splitweights": [.05, .2, .3]

    }

    noise = {

        "name": "noise",

        "community_sizes": [600, 600, 600, 600, 600],
        "cluster_sizes": [600, 600, 600, 600, 600],
        "m_features": 10,

        "alpha": 1, "beta": 1, "lmbd": -1,  # deprecated

        "within_com_prob_range": (.3, .3),
        "between_com_prob_range": (.2, .2),


        "centroid_variance_range": (10, 10),
        "within_clust_variance_range": (8, 8),

        "within_clust_covariance_range": (0, .0),  # deprecated
        "centroid_covariance_range": (0, 0),  # deprecated

        "n_targets": 5,

        "community_importance": 2,
        "x_importance": 2,
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
