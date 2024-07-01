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
import seaborn as sns
from sklearn.manifold import TSNE
from graspy.simulations import sbm

import pickle
import os


class Adc_sbm:

    def __init__(self, community_sizes: list, B: np.array):
        """
        :param community_sizes: E.g.: [70, 50, 100]
        :param B: Connection Prob Matrix
        """
        self.nx = None # Here the NetworkX object is stored
        self.edge_index = None # edge indices [[tensor],[tensor]]

        self.pos_edge_label_index = None # used for link prediction
        self.neg_edge_label_index = None # used for link prediction

        self.train_mask = None
        self.test_mask = None
        self.val_mask = None

        self.degrees = None # degree of each node
        self.task = None # string indicator

        self.community_sizes = community_sizes # list of sizes
        self.n_nodes = sum(community_sizes)

        self.community_labels = np.concatenate([np.full(n, i) # labels the community
                                                for i, n in enumerate(community_sizes)])
        self.cluster_labels = None # labels the numeric feature cluster (gaussian component)

        self.B = B # Block prob matrix

        self.dc = None
        self.x = None
        self.x_tsne = None
        self.y = None
        self.name = None

    # ------------------------ Instantiate Graph Object -------------------------

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

    # ------------------ Set Node features and Targets ------------------------

    def set_x(self, n_c: int, mu: list, sigma: list, w: any):
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

    def set_y(self, task: str, weights: np.array, distribution = None):
        """
        :param task: ["regression","binary","multiclass"]
        :param weights: array of numbers specifying the importance of each feature
        (order is relevant to match the feature matrix!)
        A vector if not multiclass, else a matrix with m_rows = number of classes, n_col = number of features)
        E.g.: weights = np.array([0.5, 1.0, 2.0, 2.0])
        :param distribution: Distribution, from which to draw the parameters from
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
        x_continuous = scaler.fit_transform(
            np.concatenate((self.degrees.reshape(-1, 1),
                            self.x), axis=1)
        )

        feat_mat = np.hstack((x_continuous,
                              pd.get_dummies(self.cluster_labels).to_numpy(dtype=np.float16)))

        if distribution == "normal":
            beta = np.random.normal(size=weights.shape) * weights
        if distribution == "uniform":
            beta = np.random.uniform(low=0, high=1, size=weights.shape) * weights
        if distribution == None:
            beta = np.ones(weights.shape) * weights

        if task == "regression":
            self.y = np.dot(feat_mat, beta)

        if task == "binary":
            self.y = 1 / (1 + np.exp(-np.dot(feat_mat, beta))) > np.random.uniform(size=self.n_nodes)

        if task == "multiclass":
            # assert
            logits = np.dot(feat_mat, beta.T)
            probabilities = softmax(logits, axis=1)
            self.y = np.argmax(probabilities, axis=1)

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


    def split_data(self, splitweights:list, method:str="runif"):
        """
        Create a boolean tensor, to index the nodes for the different sets.
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
                                np.repeat('train', mv[2]))
                               )
            )
            self.train_mask = torch.tensor(rv == 'train')
            self.test_mask = torch.tensor(rv == 'test')
            self.val_mask = torch.tensor(rv == 'val')

    def set_Data_object(self):
        data = Data(x=torch.tensor(self.x, dtype=torch.float32),
                    edge_index=self.edge_index,  # allready a tensor
                    y=torch.tensor(self.y, dtype=torch.int64),
                    train_mask=self.train_mask,
                    test_mask=self.test_mask,
                    val_mask=self.val_mask,
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
        assert group_by in ["Community", "Feat.Cluster"], "group_by cluster or community"
        cdict = {"Community": self.community_labels, "Feat.Cluster": self.cluster_labels}

        df = pd.DataFrame({group_by: cdict[group_by], 'Y': self.y})

        if not self.task == "regression":
            # Use Characteristics suitable for categorical data
            grouped_counts = df.groupby([group_by, 'Y']).size().unstack(fill_value=0)

            grouped_counts.plot(kind='bar', figsize=(10, 6))
            plt.xlabel(group_by)
            plt.ylabel('Grouped Frequency: Target')
            plt.title(' ')
            plt.legend(title="Target Label")
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

def from_config(config:dict):
    """
    Generate entire graph from config dictionary
    :param config: a dictionary with specified args  
    """
    community_sizes = config["community_sizes"]
    n = sum(community_sizes)  
    b_communities = len(community_sizes)
    m_features = config["m_features"]
    k_clusters = config["k_clusters"]
    alpha, beta, lmbd = config["alpha"], config["beta"], config["lmbd"]
    br, wr = config["between_prob_range"], config["within_prob_range"]

    n_targets = config["n_targets"]  # number of target classes; fixed



if __name__ == "__main__":

    # 1) ----------------- Set Params -----------------
    community_sizes = [90, 130, 200, 60] # 4 communities; fixed
    n = sum(community_sizes)  # number of nodes (observations)
    b_communities = len(community_sizes)  # number of communities
    m_features = 6  # number of numeric features; fixed
    k_clusters = 6  # number of feature clusters; Overlap Scenario (over, under, match) 3,5,4
    alpha, beta, lmbd = 2, 20, .5 # degree_correction params; fixed
    br, wr = (.15, .2), (.4, .5) # assortative and dis-assortative

    # 2) ------- Instantiate Class Object (Note: No graspy calles yet!) ---------
    B = setB(m=b_communities, b_range=br, w_range=wr)  # get Connection Matrix
    g = Adc_sbm(community_sizes=community_sizes, B=B)  # instantiate class

    # 3) ----------------- Generate the actual Graph -----------------
    g.correct_degree(alpha=alpha, beta=beta, lmbd=lmbd, distribution="exp")
    g.gen_graph()

    # 4) ----------------- Generate Node Features -----------------
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
    # 4) ----------------- Generate Targets -----------------
    ny = 5  # number of target classes; fixed
    nf = m_features + 1 + k_clusters  # number of features
    # omega = np.repeat(np.random.exponential(size=nf, scale=.5),ny).reshape(ny,-1)
    omega = np.repeat(np.random.uniform(size=nf, low=.5, high=.75), ny).reshape(ny, -1)# ;fixed
    # omega = np.random.exponential(0.5,nf)

    g.set_y(task="multiclass", weights=omega, distribution="normal")
    g.split_data(splitweights=[.7,.2,.1], method="runif")
    g.set_Data_object()
    print(g.DataObject)

    #g.characterise(group_by="Community", # Feat.Cluster
     #              colors=sns.color_palette('husl', k_clusters))

    with open(r'C:\Users\zogaj\PycharmProjects\MA\SyntheticGraphs\g1.pkl', 'wb') as f:
        pickle.dump(g, f)