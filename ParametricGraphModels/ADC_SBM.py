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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import seaborn as sns
from sklearn.manifold import TSNE
from graspy.simulations import sbm


class Adc_sbm:

    def __init__(self, community_sizes: list, B: np.array):
        """
        :param community_sizes: [70, 50, 100]
        :param B: Connection Prob Matrix
        """
        self.nx = None
        self.edge_index = None
        self.pos_edge_label_index = None
        self.neg_edge_label_index = None
        self.degrees = None
        self.task = None

        self.community_sizes = community_sizes
        self.n_nodes = sum(community_sizes)

        self.community_labels = np.concatenate([np.full(n, i)
                                                for i, n in enumerate(community_sizes)])
        self.cluster_labels = None

        self.B = B

        self.dc = None
        self.x = None
        self.x_tsne = None
        self.y = None

    # ------------------------ Instantiate Graph Object -------------------------

    def gen_graph(self):
        """
        Generate either DC-SBM or SBM based on availability of degree correction.
        """
        if self.dc is not None:
            dcsbm_graph = sbm(self.community_sizes, self.B, dc=self.dc, loops=False)
            self.nx = nx.from_numpy_array(dcsbm_graph)
        else:
            sbm_graph = sbm(self.community_sizes, self.B, loops=False)
            self.nx = nx.from_numpy_array(sbm_graph)

        edge_list = list(self.nx.edges())
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.degrees = np.array([self.nx.degree(n) for n in self.nx.nodes()])


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
        for label in np.unique(self.community_labels):  # one label per block
            mask = (self.community_labels == label)
            degree_corrections[mask] = np.sort(degree_corrections[mask])[::-1]  # sort in increasing order
            comm_sum = np.sum(degree_corrections[mask])
            degree_corrections[mask] = degree_corrections[mask] / comm_sum  # Block-wise-normalize

        self.dc = degree_corrections

    # ------------------ Set Node features and Targets ------------------------

    def set_x(self, n_c: int, mu: list, sigma: list, w: list):
        """
        :param stochastic: fixed or stochastic cluster components
        :param n_c: number of cluster components
        :param mu: list of tuples corresponding to
        num of features (tuple-length) and number of components (list-length)
        :param sigma: list of Covariance matrices
        :param w: mixture weights
        :return: None

        Add numeric features. Only used within the class. The X values for each component are sorted (asc. order) so indexing is straightforward (beneficial for Adj. Matrix?)
        """
        assert len(mu) == len(sigma) == len(w), \
            f"Different dimensions chosen for mu-{len(mu)}-, sigma-{len(sigma)}-, w-{len(w)}-. "

        if np.isclose(sum(w), 1, atol=1.0e-8):
            # w are probs -> Sample from Gaussian-Mixture Model:
            component_labels = np.sort(
                np.random.choice(n_c, size=self.n_nodes, p=w)
            )
        else:
            assert sum(w) == self.n_nodes, \
                f"number of component labels {sum(w)} doesnt match number of nodes{self.n_nodes}"

            num = np.arange(len(w), dtype=np.int64)
            component_labels = np.repeat(num, w)

        data = np.array([np.random.multivariate_normal(mu[label],
                                                       sigma[label])
                         for label in component_labels])

        self.x = data
        self.cluster_labels = component_labels

    def reduce_dim_x(self, rs=26):
        tsne = TSNE(n_components=2, random_state=rs)
        self.x_tsne = tsne.fit_transform(self.x)

    def set_y(self, task: str, weights: np.array, distribution: str = "normal", n_classes: int = 3):
        """
        :param task: ["regression","binary","multiclass"]
        :param weights: array of numbers specifying the importance of each features
        (order is relevant to match the feature matrix!)
        A vector if not multiclass, else a matrix with m_rows = number of classes, n_col = number of features)
        E.g.: weights = np.array([0.5, 1.0, 2.0, 2.0])
        :param distribution: Distribution, from which to draw the parameters from
        :return: targets
        """

        if task == "multiclass":
            assert weights.shape[0] > 1, "Not enough classes"
        elif task == "regression" or task == "binary":
            assert len(weights.shape) == 1, "Weights must be a vector"
        else:
            raise ValueError("Invalid task")

        self.task = task

        scaler = StandardScaler()
        x_continuous = scaler.fit_transform(
            np.concatenate((self.degrees.reshape(-1, 1),
                            self.x), axis=1)
        )

        feat_mat = np.hstack((x_continuous,
                              pd.get_dummies(self.cluster_labels).to_numpy(dtype=np.float16)))

        if distribution == "normal":
            beta = np.random.normal(size=weights.shape) * weights
        if distribution == "uniform":
            beta = np.random.uniform(size=weights.shape) * weights

        if task == "regression":
            self.y = np.dot(feat_mat, beta)

        if task == "binary":
            self.y = 1 / (1 + np.exp(-np.dot(feat_mat, beta))) > np.random.uniform(size=self.n_nodes)

        if task == "multiclass":
            # assert
            logits = np.dot(feat_mat, beta.T)
            probabilities = softmax(logits, axis=1)
            self.y = np.argmax(probabilities, axis=1) + 1  # shift all by one (not important really)

    # -------------------- Prepare Data Objects for Training -------------------------------

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
        data = Data(x=torch.tensor(self.x, dtype=torch.float64),
                    edge_index=self.edge_index,  # allready a tensor
                    y=torch.tensor(self.y, dtype=torch.float64),
                    pos_edge_label_index=self.pos_edge_label_index,  # allready a tensor
                    neg_edge_label_index=self.neg_edge_label_index)  # allready a tensor

        self.DataObject = data

    # ----------------- Plotting Methods -----------------------

    def plot_features(self, alpha=.9):
        assert self.x is not None and self.community_labels is not None, "No data yet created"
        assert 2 <= self.x.shape[1], "No plot possible"

        if self.x.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(self.x[:, 0], self.x[:, 1], self.x[:, 2], c=self.community_labels, alpha=.5)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label('Cluster Labels')
            plt.show()

        if self.x.shape[1] == 2:
            plt.scatter(self.x[:, 0], self.x[:, 1], c=self.community_labels, alpha=.5)
            plt.show()

        if self.x.shape[1] > 3:
            assert self.x_tsne is not None, "call ""reduce_dim_x"" first"

            plt.scatter(self.x_tsne[:, 0], self.x_tsne[:, 1], c=self.community_labels, alpha=.5)
            plt.show()

    def plot_graph(self, alpha=.9, width=.1):

        if self.x.shape[1] == 2:
            pos = {i: (self.x[i, 0], self.x[i, 1]) for i in range(len(self.x))}
        else:
            pos = nx.spring_layout(self.nx)

        nx.draw(self.nx, pos, node_size=self.degrees, node_color=self.cluster_labels,
                edge_color='grey', font_color='black', width=width, alpha=alpha)
        plt.show()

    def plot_edge_density(self, alpha=.9, width=.1):
        density = gaussian_kde(self.degrees)
        x = np.linspace(min(self.degrees), max(self.degrees), 1000)
        density_values = density(x)

        plt.figure(figsize=(8, 6))
        plt.plot(x, density_values, color='blue')
        plt.fill_between(x, density_values, alpha=alpha)
        plt.title('Density Plot of Degrees')
        plt.xlabel('Degree')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

    def characterise(self, group_by: str = "Community", metric: str = "gini", colors: list = None):
        """
        :param group_by: ["Community", "Cluster"]
        :param metric: gini or enrtropy
        :param colors: colors for plotting
        Plot Grouped Distribution for Target Y
        """
        assert group_by in ["Community", "Cluster"], "group_by cluster or community"
        cdict = {"Community": self.community_labels, "Cluster": self.cluster_labels}

        df = pd.DataFrame({group_by: cdict[group_by], 'Y': self.y})

        if not self.task == "regression":
            # Use Characteristics suitable for categorical data
            grouped_counts = df.groupby([group_by, 'Y']).size().unstack(fill_value=0)

            grouped_counts.plot(kind='bar', figsize=(10, 6))
            plt.xlabel(group_by)
            plt.ylabel('Class Frequency')
            plt.title(' ')
            plt.legend(title=group_by)
            plt.show()

            if metric == "gini":
                fun = lambda x: 1 - sum(x ** 2)
            if metric == "entropy":
                fun = lambda x: -sum(x * np.log2(x + 1e-9))

            within_community = grouped_counts
            # within_cluster = grouped_counts.T

            print("Metric: ", metric,
                  (within_community.
                   div(within_community.sum(axis=1), axis=0).
                   apply(fun, axis=1))
                  )
        else:
            unique_groups = df[group_by].unique()

            plt.figure(figsize=(10, 6))
            for color, group in zip(colors, unique_groups):
                sns.kdeplot(data=df[df[group_by] == group], x='Y', label=group,
                            color=color, fill=True, common_norm=False)

            plt.xlabel('Y')
            plt.ylabel('Density')
            # plt.title(' ')
            plt.legend(title=group_by)
            plt.show()

            for group in df[group_by].unique():
                group_values = df[df[group_by] == group]['Y']
                mu = np.mean(group_values)
                sigma = np.std(group_values, ddof=1)
                print(f"Community {group}: μ = {mu:.2f}, σ = {sigma:.2f}")

def setB(m: int, b_range: tuple, w_range: tuple, rs: int = False):
    """
    :param m:
    :param b_range: between range
    :param w_range: within range
    :param rs:
    :return:
    """
    if rs:
        np.random.seed(rs)

    B = np.random.uniform(*b_range, size=(m, m))
    B = np.tril(B) + np.tril(B, -1).T
    diag_elements = np.random.uniform(*w_range, size=m)
    np.fill_diagonal(B, diag_elements)

    return B


if __name__ == "__main__":
    community_sizes = [90, 130, 200, 60]
    n = sum(community_sizes)  # number of nodes (observations)
    b_communities = len(community_sizes)  # number of communities
    m_features = 6  # number of numeric features
    k_clusters = 6  # number of feature clusters

    B = setB(m=b_communities, b_range=(.5, .75), w_range=(0, .5))  # get Connection Matrix

    g = Adc_sbm(community_sizes=community_sizes, B=B)  # instantiate class

    # Generate "k" centroids with "m" features
    centroids = np.random.multivariate_normal(np.repeat(0, m_features),  # mu
                                         setB(m_features,
                                              (0, 2.5),  # Covariance
                                              (0, 15)),  # Variance (relevant for cluster separation
                                         k_clusters)  # n
    # centroids will be an array of size kxm

    g.set_x(n_c=k_clusters,  # number of clusters
            mu=[tuple(point) for point in centroids],  # k tuple of coordinates in m-dimensional space
            sigma=[setB(m_features, (0, .5), (0, 1)) for _ in range(k_clusters)],
            # similar covariance matrix for each centroid
            w=np.random.dirichlet(np.ones(k_clusters), size=1).flatten(),
            # w=[220, 200, 60]
            )

    g.correct_degree(alpha=2, beta=20, lmbd=.5, distribution="exp")
    g.gen_graph()
    # g.plot_graph(.9,.02)
    # g.reduce_dim_x()