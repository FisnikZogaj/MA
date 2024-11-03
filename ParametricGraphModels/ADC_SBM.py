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
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import StandardScaler

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
        self.Nx = None  # Here the NetworkX object is stored
        self.edge_index = None  # edge indices [[tensor],[tensor]]

        self.pos_edge_label_index = None  # used for link prediction
        self.neg_edge_label_index = None  # used for link prediction

        # Split Masks:
        self.train_mask = None
        self.test_mask = None
        self.val_mask = None

        self.degrees = None  # degree of each node

        self.community_sizes = community_sizes # list of sizes
        self.n_nodes = sum(community_sizes)

        # labels the communities:
        self.community_labels = np.concatenate([np.full(n, i)
                                                for i, n in enumerate(community_sizes)])
        self.cluster_labels = None  # labels the numeric feature cluster (gaussian component)

        # Block prob matrix:
        self.B = B

        # Vector of degree corrections:
        self.dc = None

        # Feature variables:
        self.x = None
        self.x_tsne = None
        self.y = None
        self.y_out_dim = None  # either 1 when regression or k labels when not
        self.m = None

        self.name = None

    # ------------------------ Instantiate Graph Object -------------------------

    def correct_degree(self, alpha: float, beta: float, lmbd: float = .5, distribution: str = "exp") -> np.array:
        """
        :param alpha: lower bound (or shape parameter for beat-distribution)
        :param beta: upper bound (or shape parameter for beat-distribution)
        :param lmbd: exponential coefficient for shape of the distribution (not needed in case of beta)
        :param distribution: whether to draw from a beta or an exp distribution
        :return: Vector of degree corrections for every node.

        Implemented, but not used within the Monte Carlo Experiments.
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

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        self.x = data
        self.cluster_labels = component_labels
        self.m = data.shape[1]


    def set_y(self, weights: np.array, eps:float=1):
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

        feat_mat = np.hstack(
            (pd.get_dummies(self.cluster_labels).to_numpy(dtype=np.float16),
             pd.get_dummies(self.community_labels).to_numpy(dtype=np.float16)
             )
        )

        beta = np.ones(weights.shape) * weights

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
        # assert np.abs(np.sum(splitweights) - 1) < 1e-8, "Weights don't sum up to 1."
        # They don't have to sum up to 1.

        if method == "runif":
            mv = np.round(self.n_nodes * np.array(splitweights))

            rv = np.random.permutation(
                np.concatenate(
                    (np.repeat("None", self.n_nodes-sum(mv)),
                     np.repeat('train', mv[0]),
                     np.repeat('test', mv[1]),  # Maybe swap ...
                     np.repeat('val', mv[2]))
                )
            )
            self.train_mask = torch.tensor(rv == 'train')
            self.test_mask = torch.tensor(rv == 'test')
            self.val_mask = torch.tensor(rv == 'val')
        else:
            raise NotImplementedError


    def set_Data_object(self):
        self.DataObject = Data(x=torch.tensor(self.x, dtype=torch.float32),  # full features
                               edge_index=self.edge_index,  # full Adjacency indices [[1<->2],[1<->6], ...,]
                               y=torch.tensor(self.y, dtype=torch.int64),  # full label vector
                               train_mask=self.train_mask,  # boolean train_mask
                               test_mask=self.test_mask,  # boolean test_mask
                               val_mask=self.val_mask)  # # boolean validation_mask


    # -------------------- Compute metrics for Graph characteristics -------------------------------

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


    def simple_edge_homophily(self):
        G = self.Nx
        labels = np.array(self.y)
        total_neighbors = np.array([degree for _, degree in G.degree()])

        same_label_neighbors = np.zeros(self.n_nodes, dtype=int)

        neighbors_list = [list(G.neighbors(node)) for node in range(self.n_nodes)]

        for node, neighbors in enumerate(neighbors_list):
            same_label_neighbors[node] = np.sum(labels[neighbors] == labels[node])

        return np.sum(same_label_neighbors) / np.sum(total_neighbors)

    def edge_homophily(self):
        """
        Computes homophilly meassure from Lim et al. (2021).
        :return:
        """
        G = self.Nx
        total_neighbors = np.array([tpl[1] for tpl in list(G.degree())])  # tpl: (node, ngbhr)
        n = self.n_nodes
        n_y_k = np.bincount(self.y)  # (_,)
        n_classes = self.y_out_dim #  = tau

        same_label_neighbors = np.zeros(n, dtype=int)
        labels = np.array(self.y)

        for node in range(n):
            neighbors = list(G.neighbors(node))
            # total_neighbors[node] = len(neighbors)
            same_label_neighbors[node] = sum(labels[neighbor] == labels[node] for neighbor in neighbors)

        h_k = np.zeros(n_classes)
        for l in range(n_classes):
            numerator = sum(same_label_neighbors[np.where(labels == l)])
            denominator = sum(total_neighbors[np.where(labels == l)])
            h_k[l] = numerator/denominator  # indexed 0, 1, ..., tau


        h_hat = ((1/(n_classes-1)) *
                 sum(np.maximum(np.zeros(n_classes),
                                h_k - (n_y_k / n))))
        return h_hat


    def manova_x(self):
        """

        :return: Wilks lambda in [0, 1]
        """
        data = np.concatenate((self.x, self.y.reshape(-1, 1)), axis=1)
        df = pd.DataFrame(data, columns=[f"X{i + 1}" for i in range(self.m)] + ['Group'])
        df['Group'] = df['Group'].astype(int)

        formula = " + ".join(df.columns[:-1]) + " ~ " + df.columns[-1]
        manova = MANOVA.from_formula(formula, data=df)
        manova_result = manova.mv_test()

        # manova_result.results['Group']['stat']:
        # Wilks' lambda  0.946569  6, 992.0  9.332549  0.0
        # Pillai's trace 0.053431  6, 992.0  9.332549  0.0
        # ...

        wilks_lambda = manova_result.results['Group']['stat'].iloc[0, 0]
        p_value = manova_result.results['Group']['stat'].iloc[0, -1]

        return np.round(wilks_lambda, 3), np.round(p_value, 3)


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

        },
            index=["NMI", "CV", "ARI"]
        )

        return correlations


    # ----------------- Plotting Methods -----------------------

    def reduce_dim_x(self, rs=26):
        """
        Used to be able to plot graphs with more than 2 features.
        :param rs: random state
        """
        tsne = TSNE(n_components=2, random_state=rs)
        self.x_tsne = tsne.fit_transform(self.x)

        return self.cluster_labels, self.x_tsne

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



def getB(m: int, b_range: tuple, w_range: tuple, rs: int = False):
    """
    B is the matrix that is used for the SBM as well as for generating proper covariance matrices.

    :param m: row and col dimension
    :param b_range: between range -> triangular part
    :param w_range: within range -> diagonal part
    :param rs: random state
    """
    if rs:
        np.random.seed(rs)

    B = np.random.uniform(*b_range, size=(m, m))
    B = np.tril(B) + np.tril(B, -1).T
    diag_elements = np.random.uniform(*w_range, size=m)
    np.fill_diagonal(B, diag_elements)

    return B


def getW(m_targets, n_communities, k_clusters, w_x: float, w_com: float):
    """
    W (Omega) are the weights that are used to determine the importance of Cluster and Community.
    Fills the diagonal by adding number to it.
    """

    # Community importance:
    x_betas = np.random.uniform(0, 1, (m_targets, k_clusters))
    np.fill_diagonal(x_betas, x_betas.diagonal() + w_x)

    # Community importance:
    community_betas = np.random.uniform(0, 1, (m_targets, n_communities))
    np.fill_diagonal(community_betas, community_betas.diagonal() + w_com)

    return np.hstack((x_betas, community_betas))


def from_config(config: dict, rs=26):
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
    b_communities = len(community_sizes)

    m_features = config["m_features"]
    cluster_sizes = config["cluster_sizes"]
    k_clusters = len(config["cluster_sizes"])
    alpha, beta, lmbd = config["alpha"], config["beta"], config["lmbd"]

    b_com_r, w_com_r = config["between_com_prob_range"], config["within_com_prob_range"]
    w_clust_v, w_clust_c = config["within_clust_variance_range"], config["within_clust_covariance_range"]

    n_targets = config["n_targets"]

    centroid_variance_range = config["centroid_variance_range"]
    centroid_covariance_range = config["centroid_covariance_range"]

    x_importance = config["x_importance"]
    community_importance = config["community_importance"]

    assert sum(community_sizes) == sum(cluster_sizes), "sum(community_sizes) != sum(cluster_sizes)"
    # 3) ----------------- Init Graph -----------------
    B = getB(m=b_communities, b_range=b_com_r, w_range=w_com_r)  # get Connection Matrix
    g = ADC_SBM(community_sizes=community_sizes, B=B)  # instantiate class

    #  No degree corrections applied for simulations.
    # g.correct_degree(alpha=alpha, beta=beta, lmbd=lmbd, distribution="exp")
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

    omega = getW(m_targets=n_targets, n_communities=b_communities,
                 k_clusters=k_clusters, w_x=x_importance,
                 w_com=community_importance)


    g.set_y(weights=omega, eps=config["model_error"])

    g.split_data(config["splitweights"])  # Data must be split, before creating the object !
    g.set_Data_object()

    return g



if __name__ == "__main__":
    from config import Scenarios

    g = from_config(Scenarios.community_relevant_heterophilic)


