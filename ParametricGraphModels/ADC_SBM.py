import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkit as nk
from pygsp import graphs as gsp_graphs
import seaborn as sns
import itertools
import random
from scipy.stats import gaussian_kde
import torch
from torch_geometric.data import Data
from graspy.models import DCSBMEstimator
from graspy.simulations import er_np, sbm
# no module from hyppo._utils import gaussian in latent_distribution_test.py only available in version 0.1.3

class Adc_sbm:

    def __init__(self, community_sizes: list, B: np.array):
        """
        :param community_sizes: [70, 50, 100]
        :param B:
        """
        self.nx = None
        self.edge_index = None
        self.pos_edge_label_index = None
        self.neg_edge_label_index = None
        self.degrees = None

        self.community_sizes = community_sizes
        self.n_nodes = sum(community_sizes)
        self.labels = np.concatenate([np.full(n, i) for i, n in enumerate(community_sizes)])

        self.B = B

        self.dc = None
        self.x = None
        self.cluster_labels = None

    def set_x(self, n_c: int, mu: list, sigma: list, w: list, stochastic=True):
        """
        :param stochastic: fixed or stochastic cluster components
        :param n_c: number of cluster components
        :param mu: list of tuples corresponding to
        num of features (tuple-length) and number of components (list-length)
        :param sigma: list of Covariance matrices
        :param w: mixture weights
        :return: None

        Add numeric features. Only used within the class. The X values for each component are sorted (asc. order)
        So indexing is straightforward (beneficial for Adj. Matrix?)
        """
        assert len(mu) == len(sigma) == len(w), "Different dimensions chosen for mu, sigma, w. "

        if stochastic:
            # Sample from Gaussian-Mixture Model:
            component_labels = np.sort(
                np.random.choice(n_c, size=self.n_nodes, p=w)
            )
        else: # Not safe due to rounding !
            num = np.arange(len(w), dtype=np.int64)
            times = np.round(self.n_nodes * np.array(w))
            times = np.array(times, dtype=np.int64)  # 'safe' conversion
            component_labels = np.repeat(num, times)

        data = np.array([np.random.multivariate_normal(mu[label],
                                                       sigma[label])
                         for label in component_labels])

        self.x = data
        self.cluster_labels = component_labels

    def correct_degree(self, alpha: float, beta: float, lmbd: float = .5, distribution: str = "exp"):
        """
        :param alpha: lower (or shape parameter)
        :param beta: upper (or shape parameter)
        :param lmbd: exponential coefficient for shape of
        :param distribution:
        :return:
        """
        # theta with a distribution between 0 and 1
        assert distribution in ["exp", "beta"], "Distribution must be in [exp, beta]"

        if distribution == "exp":
            degree_corrections = np.random.uniform(alpha, beta, self.n_nodes) ** (-1 / lmbd)

        if distribution == "beta":
            degree_corrections = np.random.beta(alpha, beta, size=self.n_nodes)

            # Block-Wise degree correction:
        for label in np.unique(self.labels):  # one label per block
            mask = (self.cluster_labels == label)
            degree_corrections[mask] = np.sort(degree_corrections[mask])[::-1]  # sort in increasing order
            comm_sum = np.sum(degree_corrections[mask])
            degree_corrections[mask] = degree_corrections[mask] / comm_sum  # Block-wise-normalize

        self.dc = degree_corrections

    def gen_graph(self):
        """
        Generate either DC-SBM or SBM based on availability of degree correction.
        """
        if self.dc is not None:
            dcsbm_graph = sbm(community_sizes, B, dc=self.dc, loops=False)
            self.nx = nx.from_numpy_array(dcsbm_graph)
        else:
            sbm_graph = sbm(community_sizes, B, loops=False)
            self.nx = nx.from_numpy_array(sbm_graph)

        edge_list = list(self.nx.edges())
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.degrees = np.array([self.nx.degree(n) for n in self.nx.nodes()])

    def sample_train_edge_labels(self, size: int = None):
        """
        :param size: sample size of pos and neg edges
        Throws an error when not enough negative connections are established.
        """
        assert self.edge_index is not None, "No edge_indices set."

        if not size:
            size = self.n_nodes // 2

        # Positive:
        sidx_pos = np.random.choice(range(self.edge_index.shape[1]), size=size)
        self.pos_edge_label_index = self.edge_index[:, sidx_pos]

        # Negative:
        tuple_pos = set([(int(self.edge_index[0][i]),
                          int(self.edge_index[1][i]))
                         for i in range(self.edge_index.shape[1])])  # Make set of all connected node tuples
        all_permutations = set(list(itertools.combinations(list(range(self.n_nodes)), 2)))

        samp_space = np.array(list(all_permutations - tuple_pos)).T
        # print(samp_space.shape[1] + len(tuple_pos) == len(all_permutations)) # -> True
        assert samp_space.shape[1] >= size, \
            (f"Not enough negative connections to sample from. possible: {samp_space.shape[1]} | "
             f"required:{size}")

        sidx_neg = np.random.choice(range(samp_space.shape[1]), size=size)
        self.neg_edge_label_index = torch.tensor(
            samp_space[:, sidx_neg]
        )

    def set_Data_object(self):
        data = Data(x=torch.tensor(self.x, dtype=torch.float32),
                    edge_index=self.edge_index,  # allready a tensor
                    # y = torch.tensor(self.y, dtype=torch.float32),
                    pos_edge_label_index=self.pos_edge_label_index,  # allready a tensor
                    neg_edge_label_index=self.neg_edge_label_index)  # allready a tensor

        self.DataObject = data

    def plot_data(self, alpha=.9):
        assert self.x is not None and self.cluster_labels is not None, "No data yet created"
        assert 2 <= self.x.shape[1] <= 3, "No plot possible"

        if self.x.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.x[:, 0], self.x[:, 1], self.x[:, 2], c=self.cluster_labels, alpha=.5)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label('Cluster Labels')
            plt.show()

        if self.x.shape[1] == 2:
            plt.scatter(self.x[:, 0], self.x[:, 1], c=self.cluster_labels, alpha=.5)
            plt.show()

    def plot_graph(self, alpha=.9, width=.1):

        if self.x.shape[1] == 2:
            pos = {i: (self.x[i, 0], self.x[i, 1]) for i in range(len(self.x))}
        else:
            pos = nx.spring_layout(self.nx)

        nx.draw(self.nx, pos, node_size=self.degrees, node_color=self.cluster_labels, edge_color='grey',
                font_color='black', width=width, alpha=alpha)
        plt.show()

        # ----------- Density of edges -----------------

        density = gaussian_kde(self.degrees)
        x = np.linspace(min(self.degrees), max(self.degrees), 1000)
        density_values = density(x)

        # Plot the density plot
        plt.figure(figsize=(8, 6))
        plt.plot(x, density_values, color='blue')
        plt.fill_between(x, density_values, alpha=alpha)
        plt.title('Density Plot of Degrees')
        plt.xlabel('Degree')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()


community_sizes = [70, 50, 100]


def setB(m, b, d):
    """
    :param m: number of communities (or feature dimensions)
    :param b: upper bound for cross community probs
    :param d: diag entries (within connection probs)
    :return: Connection prob matrix for communities B (or sigma for clusters)
    """
    B = np.random.uniform(0, b, size=(m, m))
    B = np.tril(B) + np.tril(B, -1).T
    np.fill_diagonal(B, d)

    return B


b_communities = 3
m_features = 2
k_clusters = 4

B = setB(b_communities, .25, .5)  #
g = Adc_sbm(community_sizes=community_sizes, B=B)

data = np.random.multivariate_normal(np.repeat(0, m_features),  # mu
                                     setB(m_features, 20, .5),  # sigma
                                     k_clusters)  # n
means = [tuple(point) for point in data]

g.set_x(n_c=k_clusters,
        mu=means,
        sigma=[setB(m_features, .5, 1) for _ in range(k_clusters)],
        w=np.random.dirichlet(np.ones(k_clusters), size=1).flatten(),
        stochastic=True)

# g.plot_data()
g.correct_degree(alpha=2, beta=1, lmbd=.5, distribution="beta")
g.gen_graph()

g.plot_graph(.9, .02)