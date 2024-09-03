import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import gaussian_kde
import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from sklearn.manifold import TSNE
from graspy.simulations import sbm
import random
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

#print(np.__version__) # potential errors with "pybind11" when numpy.__version__ == 2.__

def CramersV(labels_1, labels_2):
    contingency_table = pd.crosstab(pd.Series(labels_1), pd.Series(labels_2))
    chi2, _, _, _ = chi2_contingency(contingency_table)
    phi2 = chi2 / np.sum(contingency_table.values)
    r, k = contingency_table.shape

    return np.sqrt(phi2 / min(k - 1, r - 1))

class ADC_SBM:

    def __init__(self, community_sizes: list, B: np.array):
        """
        :param community_sizes: E.g.: [70, 50, 100]
        :param B: Connection Prob Matrix
        """
        self.Nx = None # Here the NetworkX object is stored
        self.edge_index = None # edge indices [[tensor],[tensor]]

        self.pos_edge_label_index = None # used for link prediction
        self.neg_edge_label_index = None # used for link prediction

        # Split Masks:
        self.train_mask = None
        self.test_mask = None
        self.val_mask = None

        self.degrees = None # degree of each node
        self.task = None # string indicator of the task

        self.community_sizes = community_sizes # list of sizes
        self.n_nodes = sum(community_sizes)

        # labels the communities:
        self.community_labels = np.concatenate([np.full(n, i)
                                                for i, n in enumerate(community_sizes)])
        self.cluster_labels = None # labels the numeric feature cluster (gaussian component)

        # Block prob matrix:
        self.B = B

        # Vector of degree corrections:
        self.dc = None

        # Feature variables:
        self.x = None
        self.x_tsne = None
        self.y = None
        self.y_out_dim = None # either 1 when regression or k labels when not

        self.name = None

    # ------------------------ Instantiate Graph Object -------------------------

    def correct_degree(self, alpha: float, beta: float, lmbd: float = .5, distribution: str = "exp") -> np.array:
        """
        :param alpha: lower bound (or shape parameter for beat-distribution)
        :param beta: upper bound (or shape parameter for beat-distribution)
        :param lmbd: exponential coefficient for shape of the distribution (not needed in case of beta)
        :param distribution: whether to draw from a beta or an exp distribution
        :return: Vector of degree corrections for every node.

        This is fixed for all Graphs and will not be changed within the Monte Carlo Experiments.
        """
        # theta with a distribution between 0 and 1
        assert distribution in ["exp", "beta"], "Distribution must be in [exp, beta]"

        if distribution == "exp":
            degree_corrections = np.random.uniform(alpha, beta, self.n_nodes) ** (-1 / lmbd)

        if distribution == "beta":
            degree_corrections = np.random.beta(alpha, beta, size=self.n_nodes)

        # Block-Wise degree correction:
        for label in np.unique(self.community_labels):  # one label per block
            mask = (self.community_labels == label)
            degree_corrections[mask] = np.sort(degree_corrections[mask])[::-1]  # sort in increasing order
            comm_sum = np.sum(degree_corrections[mask])
            degree_corrections[mask] = degree_corrections[mask] / comm_sum  # Block-wise-normalize

        self.dc = degree_corrections


    def gen_graph(self):
        """
        Generate either DC-SBM or regular SBM from "Graspy" Module, based on availability of degree correction.
        """
        if self.dc is not None:
            dcsbm_graph = sbm(self.community_sizes, self.B, dc=self.dc, loops=False)
            self.Nx = nx.from_numpy_array(dcsbm_graph)
        else:
            sbm_graph = sbm(self.community_sizes, self.B, loops=False)
            self.Nx = nx.from_numpy_array(sbm_graph)

        edge_list = list(self.Nx.edges())
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.degrees = np.array([self.Nx.degree(n) for n in self.Nx.nodes()])

    # ------------------ Set Node features and Targets ------------------------

    def set_x(self, n_c: int, mu: list, sigma: list, w: any) -> np.array:
        """
        :param n_c: number of feature clusters
        :param mu: list of tuples corresponding to the cluster means
        num of features (tuple-length) and number of components (list-length)
        [(1,1),(2,2),(2,3)] -> 3 clusters in 2-dimensional space
        :param sigma: list of Covariance matrices
        :param w: mixture weights for feature cluster sizes
        (either list of probabilities summing up to one, or the actual number of nodes assigned to the clusters)
        :sets: np.Array

        Generate the node feature matrix:
        Add numeric features. Only used within the class. The X values for each component are sorted (asc. order)
         so indexing is straightforward (beneficial for Adj. Matrix?)
        """
        assert len(mu) == len(sigma) == len(w), \
            f"Different dimensions chosen for mu-{len(mu)}-, sigma-{len(sigma)}-, w-{len(w)}-. "

        if np.isclose(sum(w), 1, atol=1.0e-8):
            # w are probs -> Sample from Gaussian-Mixture Model: Additional Monte Carlo Variance!
            component_labels = np.sort(
                np.random.choice(n_c, size=self.n_nodes, p=w)
            )
        else:
            # w are numbers per cluster group, generate by stacking. Correct length if necessary.
            if sum(w) != self.n_nodes:
                print(f"\033[91mWarning: Sum of X-Cluster adjusted by {self.n_nodes-sum(w)}! \033[0m")
                w[0] += (self.n_nodes-sum(w))

            num = np.arange(len(w), dtype=np.int64)
            component_labels = np.repeat(num, w)

        data = np.array([np.random.multivariate_normal(mu[label],
                                                       sigma[label])
                         for label in component_labels])

        self.x = data
        self.cluster_labels = component_labels


    def reduce_dim_x(self, rs=26):
        """
        Used to be able to plot graphs with more than 2 features.
        :param rs: random state
        """
        tsne = TSNE(n_components=2, random_state=rs)
        self.x_tsne = tsne.fit_transform(self.x)


    def set_y(self, task: str, weights: np.array, feature_info="cluster", eps:float=1):
        """
        :param task: ["regression","binary","multiclass"]
        :param weights: array of numbers specifying the importance of each feature
        (order is relevant to match the feature matrix!)
        A vector if not multiclass, else a matrix with m_rows = number of classes, n_col = number of features
        E.g.: weights = np.array([0.5, 1.0, 2.0, 2.0])
        :param feature_info: if "cluster": betas for dummies are generated, else raw coefficients for numeric feature values
        :param eps: Variance of the error component, high variances will lead to heavy Y-mixing between clusters
        :return: targets
        """
        # Assertions are still a bit messy, but they work
        if task == "multiclass":
            assert weights.shape[0] > 1, "Not enough classes"
        elif task == "regression" or task == "binary":
            assert len(weights.shape) == 1, "Weights must be a vector"
        else:
            raise ValueError("Invalid task")

        self.task = task
        scaler = StandardScaler()

        if feature_info == "number": # use x feature as are
            x_continuous = scaler.fit_transform(
                np.concatenate((self.x,
                                self.degrees.reshape(-1, 1)), axis=1)
            )

        elif feature_info == "cluster": # use cluster dummies
            pass
            #x_continuous = np.concatenate(
            #    (scaler.fit_transform(self.degrees.reshape(-1, 1)),  # it's not continuous anymore, but it's just a name
            #    pd.get_dummies(self.cluster_labels).to_numpy(dtype=np.float16)), axis=1
            #)

        else:
            raise ValueError("feature_info must either be 'number' or 'cluster'.")

        feat_mat = np.hstack(
            (pd.get_dummies(self.cluster_labels).to_numpy(dtype=np.float16),
             pd.get_dummies(self.community_labels).to_numpy(dtype=np.float16)
             )
        )

        beta = np.ones(weights.shape) * weights

        if task == "regression":
            error = np.random.normal(0, eps, self.n_nodes)
            self.y = np.dot(feat_mat, beta) + error
            self.y_out_dim = 1

        if task == "binary":
            error = np.random.normal(0, eps, self.n_nodes)
            self.y = ((1 / (1 + np.exp(-np.dot(feat_mat, beta))) + error)
                      > np.random.uniform(size=self.n_nodes))
            self.y_out_dim = 2

        if task == "multiclass":
            # assert
            error = np.random.normal(0, eps, (self.n_nodes, beta.shape[0]))
            logits = np.dot(feat_mat, beta.T) + error
            probabilities = softmax(logits, axis=1)
            self.y = np.argmax(probabilities, axis=1)
            self.y_out_dim = probabilities.shape[1]

    # -------------------- Prepare Data Objects for Training -------------------------------

    def split_data(self, splitweights:list, method:str="runif"):
        """
        Create a boolean tensor mask, to index the nodes for the different sets.
        :param splitweights: [.7,.2,.1] -> 70% train, 20% test and 10% val
        :param method: in ["runif", ...]
        """
        assert np.abs(np.sum(splitweights) - 1) < 1e-8, "Weights dont sum up to 1."

        if method == "runif":
            mv = np.round(self.n_nodes * np.array(splitweights))
            if sum(mv) != self.n_nodes:
                mv[0] += (self.n_nodes-sum(mv))

            # indexing length must match
            rv = np.random.permutation(
                np.concatenate((np.repeat('train', mv[0]),
                                np.repeat('test', mv[1]),
                                np.repeat('val', mv[2]))
                               )
            )

            self.train_mask = torch.tensor(rv == 'train')
            self.test_mask = torch.tensor(rv == 'test')
            self.val_mask = torch.tensor(rv == 'val')
        else:
            raise NotImplementedError


    def set_Data_object(self):
        data = Data(x=torch.tensor(self.x, dtype=torch.float32),
                    edge_index=self.edge_index,  # already a tensor
                    y=torch.tensor(self.y, dtype=torch.int64),
                    train_mask=self.train_mask,
                    test_mask=self.test_mask,
                    val_mask=self.val_mask)

        self.DataObject = data


    def target_edge_counter(self):
        """
        Count how many edges are within and between all targets.
        :return: symmetric square matrix.
        """
        edges = self.Nx.edges(data=True)
        targets = self.y
        nt = self.y_out_dim

        counter = {i: {j: 0 for j in range(0, nt)} for i in range(0, nt)}
        # {0: {0:_,1:_,2:_},1: {0:_,1:_,2:_}, 2: {0:_,1:_,2:_}} example for nt = 3
        for e in edges:
            i, j = (e[0:2])  # e = (1, 114, {'weight': 1.0})
            target_i = targets[i]
            target_j = targets[j] #["target"]
            if target_i == target_j:
                counter[target_i][target_j] += 1
            else:
                counter[target_i][target_j] += 1
                counter[target_j][target_i] += 1

        df = pd.DataFrame(counter)
        return df

    def edge_homophily(self):
        """
        Computes homophilly meassure from Lim et al. (2021)
        :return:
        """
        G = self.Nx
        total_neighbors = np.array([ngbhr[1] for ngbhr in list(G.degree())])
        n = self.n_nodes
        targets = g.y

        same_label_neighbors = np.zeros(n, dtype=int)
        labels = np.array(self.y)

        for node in range(n):
            neighbors = list(G.neighbors(node))
            total_neighbors[node] = len(neighbors)
            same_label_neighbors[node] = sum(
                labels[neighbor] == labels[node] for neighbor in neighbors)

        h_k = np.zeros(self.y_out_dim)
        for tau in range(self.y_out_dim):
            numerator = sum(same_label_neighbors[np.where(targets == tau)])
            denominator = sum(total_neighbors[np.where(targets == tau)])
            h_k[tau] = numerator/denominator
        #return h_k



    def label_correlation(self):
        """
        Computes label correlations of community, feature cluster and targets.
        :return: pandas data.frame.
        """
        labels_1 = self.y
        labels_2 = self.cluster_labels
        labels_3 = self.community_labels

        correlations = pd.DataFrame({

            "Y~F": [normalized_mutual_info_score(labels_1, labels_2),
                    CramersV(labels_1, labels_2),
                    adjusted_rand_score(labels_1, labels_2)],

            "Y~C": [normalized_mutual_info_score(labels_1, labels_3),
                    CramersV(labels_1, labels_3),
                    adjusted_rand_score(labels_1, labels_3)],

            "F~C": [normalized_mutual_info_score(labels_2, labels_3),
                    CramersV(labels_2, labels_3),
                    adjusted_rand_score(labels_2, labels_3)]
        },
            index=["NMI", "CV", "ARI"]
        )

        return correlations


    # ----------------- Plotting Methods -----------------------

    def rich_plot_graph(self, fig_size: tuple, ns: int = 20, wdth: float = .3, alph: float = .1):
        """
        Plot the Graph, with communities/target-class up to 5 categories.
        Embedded into a 2-dimensional t-SNE space.
        Y determines the shape, and community the color.
        """
        #assert self.x_tsne is not None, "call reduce_dim_x first."
        if self.x_tsne is None:
            self.reduce_dim_x()

        target = self.y
        G = self.Nx
        node_idx = list(range(self.n_nodes))

        for i in node_idx:
            G.nodes[i]['pos'] = self.x_tsne[i]
            G.nodes[i]['category'] = self.community_labels[i]
            G.nodes[i]['target'] = target[i]
            # G.nodes[i]['size'] = self.degrees[i]

        pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        color_map = {0: 'purple', 1: 'darkgreen', 2: 'gold', 3: "black", 4: "silver", 5: "darkblue"}
        shape_map = {0: 'x', 1: 'o', 2: "d", 3: "D", 4: "<", 5: ">"}

        # if self.task != "regression":
        plt.figure(figsize=fig_size)
        for shape in shape_map:
            # Could also do sizes
            nodelist = [node for node in G.nodes() if G.nodes[node]['target'] == shape]
            node_colors = [color_map[G.nodes[node]['category']] for node in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_colors, node_size=ns,
                                   node_shape=shape_map[shape])

        nx.draw_networkx_edges(G, pos, edge_color='gray', width=wdth, alpha=alph)

        #plt.title(main)
        plt.axis('off')
        plt.show()


    def purity(self, metric:str = "gini"):
        """
        :param group_by: ["Community", "Feat.Cluster"]
        :param metric: gini or enrtropy
        :param plot_it: colors for plotting
        Plot Grouped Distribution for Target Y
        :return: A dataframe with purity-scores for each group
        """
        #grouped_counts.plot(kind='bar', figsize=fig_size)
        # plt.xlabel(group_by)
        # plt.ylabel('Grouped Frequencies')
        # plt.title(' ')
        # plt.legend(title="Target Label")
        # plt.show()

        df_f = pd.DataFrame({"Feat.Cluster": self.cluster_labels, 'Y': self.y})
        df_c = pd.DataFrame({"Community": self.community_labels, 'Y': self.y})

        grouped_counts_f = df_f.groupby(["Feat.Cluster", 'Y']).size().unstack(fill_value=0)
        grouped_counts_c = df_c.groupby(["Community", 'Y']).size().unstack(fill_value=0)

        gini = lambda x: 1 - sum(x ** 2)
        entropy = lambda x: -sum(x * np.log2(x + 1e-9))
        # lambda x: -np.sum(x * np.log2(x + 1e-9)) if np.all(x > 0) else 0.0

        if metric == "gini":
            pure_df1 = (
                       grouped_counts_f.
                       div(grouped_counts_f.sum(axis=1), axis=0).
                       apply(gini, axis=1)
                    )
            pure_df_2 = (
                       grouped_counts_c.
                       div(grouped_counts_c.sum(axis=1), axis=0).
                       apply(gini, axis=1)
                )
            return {"centroids": pure_df1,
                    "community": pure_df_2}

        if metric == "entropy":
            pure_df_3 = (
                       grouped_counts_f.
                       div(grouped_counts_f.sum(axis=1), axis=0).
                       apply(entropy, axis=1)
                    )
            pure_df_4 = (
                       grouped_counts_c.
                       div(grouped_counts_c.sum(axis=1), axis=0).
                       apply(entropy, axis=1)
                )

            return {"centroids": pure_df_3,
                    "community": pure_df_4}


def getB(m: int, b_range: tuple, w_range: tuple, rs: int = False):
    """
    B is the matrix that is used for the SBM as well as for generating proper covariance matrices.

    :param m: row and col dimension
    :param b_range: between range -> triangular part
    :param w_range: within range -> diagonal part
    :param rs:
    """
    if rs:
        np.random.seed(rs)

    B = np.random.uniform(*b_range, size=(m, m))
    B = np.tril(B) + np.tril(B, -1).T
    diag_elements = np.random.uniform(*w_range, size=m)
    np.fill_diagonal(B, diag_elements)

    return B


def getW(m_targets, n_communities, j_features,
         k_clusters, feature_info: str,
         w_x: float, w_com: float):
    """
    W (Omega) are the weights that are used to determine the importance of Cluster and Community
    Newer Version, that doesn't rely on random chance too much.
    Fills the diagonal by adding number to it.
    """
    # Degree Importance:
    # degree_betas = np.random.normal(loc=0, scale=0, size=(m_targets, 1))  # Deprecated

    # Feature Cluster importance:
    if feature_info == "number":  # Deprecated
        x_betas = np.random.normal(loc=w_x, scale=1, size=(j_features, m_targets)).T

    elif feature_info == "cluster":
        x_betas = np.random.uniform(0, 1, (m_targets, k_clusters))
        np.fill_diagonal(x_betas, x_betas.diagonal() + w_x)
    else:
        raise ValueError(f"feature_info must either be 'number' or 'cluster'. '{feature_info}' provided")

    # Community importance:
    community_betas = np.random.uniform(0, 1, (m_targets, n_communities))
    np.fill_diagonal(community_betas, community_betas.diagonal() + w_com)

    return np.hstack((#degree_betas,
                      x_betas, community_betas))


def from_config(config:dict, rs = 26):
    """
    Generate entire graph from config dictionary.
    :param config: a dictionary with specified args
    :param rs: random_state
    """
    # 0) --------- Set Random State -------------------

    np.random.seed(rs)
    random.seed(rs)
    torch.manual_seed(rs)

    # 1) ----------------- Set Params ----------------- kinda redundant
    community_sizes = config["community_sizes"]
    # n = sum(community_sizes)
    b_communities = len(community_sizes)

    m_features = config["m_features"]
    cluster_sizes = config["cluster_sizes"]
    k_clusters = len(config["cluster_sizes"])
    alpha, beta, lmbd = config["alpha"], config["beta"], config["lmbd"] #fixed anyway

    b_com_r, w_com_r = config["between_com_prob_range"], config["within_com_prob_range"]
    w_clust_v, w_clust_c = config["within_clust_variance_range"], config["within_clust_covariance_range"]

    n_targets = config["n_targets"]

    centroid_variance_range = config["centroid_variance_range"]
    centroid_covariance_range = config["centroid_covariance_range"]

    degree_importance = config["degree_importance"]
    x_importance = config["x_importance"]
    community_importance = config["community_importance"]

    assert sum(community_sizes) == sum(cluster_sizes), "sum(community_sizes) != sum(cluster_sizes)"
    # 3) ----------------- Init Graph -----------------
    B = getB(m=b_communities, b_range=b_com_r, w_range=w_com_r)  # get Connection Matrix
    g = ADC_SBM(community_sizes=community_sizes, B=B)  # instantiate class

    g.correct_degree(alpha=alpha, beta=beta, lmbd=lmbd, distribution="exp")
    g.gen_graph()

    # 4) ------------ Set Node features --------------
    centroids = np.random.multivariate_normal(np.repeat(0, m_features),
                                              getB(m_features,
                                                   centroid_covariance_range,
                                                   centroid_variance_range),
                                              k_clusters)
    g.set_x(n_c=k_clusters,
            mu=[tuple(point) for point in centroids],
            sigma=[getB(m_features,
                        w_clust_c,
                        w_clust_v)
                   for _ in range(k_clusters)],
            w=cluster_sizes,
            )

    # 5) ----------------- Set Node features -----------------
    if config["task"] == "multiclass":  # This will be the case
        omega = getW(m_targets=n_targets, n_communities=b_communities, j_features=m_features,
                     k_clusters=k_clusters, feature_info=config["feature_info"],
                     w_x=x_importance, w_com=community_importance)


    g.set_y(task=config["task"], weights=omega, feature_info="cluster", eps=config["model_error"])

    g.split_data(config["splitweights"])  # Data must be split, before creating the object !
    g.set_Data_object()

    return g

#def synthetic_split(config: dict, splitweights: any, rs:int):
#    """
#    Generate three different graphs with same configs but different sizes for
#    a synthetic train, test, validation split.
#
#    rename nodes, to handle splitting?
#    :param config:
#    :param splitweights:
#    :return:
#    """
#    # rs = 26
#    initial_com_size = config["community_sizes"]
#    initial_clust_size = config["cluster_sizes"]

#    for i, w in enumerate(splitweights):
#        config["community_sizes"] = np.array(config["community_sizes"]) * w
#        config["cluster_sizes"] = np.array(config["cluster_sizes"]) * w

#        if sum(config["community_sizes"]) != sum(config["cluster_sizes"]):
#            config["community_sizes"][0] += (sum(config["cluster_sizes"]) - sum(config["community_sizes"]))

#        Graph = from_config(config, rs=rs+i)

#        config["community_sizes"] = initial_com_size
#        config["cluster_sizes"] = initial_clust_size


if __name__ == "__main__":
    from config import Scenarios

    #g = from_config(Scenarios.community_relevant)
    #g.purity(fig_size=(7,7), group_by="Community", plot_it=False)
    #synthetic_split(MultiClassClassification.perfect_graph, [.7,.2,.1])


    # 1) ----------------- Set Params -----------------
    community_sizes = [100, 100, 100]  # 4 communities; fixed
    n = sum(community_sizes)  # number of nodes (observations)
    b_communities = len(community_sizes)  # number of communities
    m_features = 2  # number of numeric features; fixed
    k_clusters = 3  # number of feature clusters; Overlap Scenario (over, under, match) 3,5,4
    alpha, beta, lmbd = 2, 20, .5  # degree_correction params; fixed
    br, wr = (.05, .05), (.5, .5)  # assortative and dis-assortative

    # 2) ------- Instantiate Class Object (Note: No graspy called yet!) ---------
    B = getB(m=b_communities, b_range=br, w_range=wr)  # get Connection Matrix

    g = ADC_SBM(community_sizes=community_sizes, B=B)  # instantiate class

    # 3) ----------------- Generate the actual Graph -----------------
    # g.correct_degree(alpha=alpha, beta=beta, lmbd=lmbd, distribution="exp")
    g.gen_graph()

    # 4) ----------------- Generate Node Features -----------------
    # Generate "k" centroids with "m" features
    centroids = np.random.multivariate_normal(np.repeat(0, m_features),  # mu
                                         getB(m_features,
                                              (0, 0),  # Covariance
                                              (6, 6)),  # Variance (relevant for cluster separation)
                                         k_clusters)  # n
    # centroids will be an array of size kxm
    # if the centroid variance is low and within-variance high,
    # separation becomes harder, and easier if it's the other way around !

    g.set_x(n_c=k_clusters,  # number of clusters
            mu=[tuple(point) for point in centroids],  # k tuple of coordinates in m-dimensional space
            sigma=[getB(m_features, (0, 0),  # Covariance
                                    (1, 1))  # Variance (relevant for cluster separation)
                   for _ in range(k_clusters)],
            # similar covariance matrix for each centroid
            #w=np.random.dirichlet(np.ones(k_clusters), size=1).flatten()
            w=np.full(k_clusters, n/k_clusters, dtype=np.int64)
            )

    # 5) ----------------- Generate Targets -----------------
    ny = 3  # number of target classes; fixed

    omega = getW(m_targets=ny, n_communities=b_communities, j_features=m_features,
                 k_clusters=k_clusters, feature_info="cluster",
                 w_x=2, w_com=1)

    g.set_y(task="multiclass", weights=omega, feature_info="cluster", eps=.5)
    g.split_data([.7,.2,.1])

    es = g.Nx.edges(data=True)

    print({(n, i) for n, i in g.Nx.neighbors(0)})
