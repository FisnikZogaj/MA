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
import random

import pickle
import os


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
        Generate either DC-SBM or SBM from Graspy Module based on availability of degree correction.
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

    def set_x(self, n_c: int, mu: list, sigma: list, w: any):
        """
        :param stochastic: fixed or stochastic cluster components
        :param n_c: number of cluster components
        :param mu: list of tuples corresponding to
        num of features (tuple-length) and number of components (list-length)
        :param sigma: list of Covariance matrices
        :param w: mixture weights
        :return: None

        Add numeric features. Only used within the class. The X values for each component are sorted (asc. order)
         so indexing is straightforward (beneficial for Adj. Matrix?)
        """
        assert len(mu) == len(sigma) == len(w), \
            f"Different dimensions chosen for mu-{len(mu)}-, sigma-{len(sigma)}-, w-{len(w)}-. "

        if np.isclose(sum(w), 1, atol=1.0e-8):
            # w are probs -> Sample from Gaussian-Mixture Model:
            component_labels = np.sort(
                np.random.choice(n_c, size=self.n_nodes, p=w)
            )
        else:
            # w are numbers per cluster group, generate by stacking.
            if sum(w) != self.n_nodes:
                w[0] += (self.n_nodes-sum(w)) # Correct length if necessary

            num = np.arange(len(w), dtype=np.int64)
            component_labels = np.repeat(num, w) #stack label[i] w[i] times for each i

        data = np.array([np.random.multivariate_normal(mu[label],
                                                       sigma[label])
                         for label in component_labels])

        self.x = data
        self.cluster_labels = component_labels


    def reduce_dim_x(self, rs=26):
        tsne = TSNE(n_components=2, random_state=rs)
        self.x_tsne = tsne.fit_transform(self.x)


    def set_y(self, task: str, weights: np.array, feature_info="number", eps:float=1):
        """
        :param task: ["regression","binary","multiclass"]
        :param weights: array of numbers specifying the importance of each feature
        (order is relevant to match the feature matrix!)
        A vector if not multiclass, else a matrix with m_rows = number of classes, n_col = number of features)
        E.g.: weights = np.array([0.5, 1.0, 2.0, 2.0])
        :param distribution: Distribution, from which to draw the parameters from
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

        if feature_info == "cluster": # use cluster dummies
            x_continuous = np.concatenate(
                scaler.fit_transform(self.degrees.reshape(-1, 1)),  # it's not continuous anymore, but it's just a name
                pd.get_dummies(self.cluster_labels).to_numpy(dtype=np.float16)
            )
        else:
            raise ValueError("feature_info must either be 'number' or 'cluster'.")

        feat_mat = np.hstack(
            (x_continuous,
             pd.get_dummies(self.community_labels).to_numpy(dtype=np.float16)
             )
        )

        beta = np.ones(weights.shape) * weights

        if task == "regression":
            error = np.random.normal(0, eps, self.n_nodes)
            self.y = np.dot(feat_mat, beta) + error

        if task == "binary":
            error = np.random.normal(0, eps, self.n_nodes)
            self.y = ((1 / (1 + np.exp(-np.dot(feat_mat, beta))) + error)
                      > np.random.uniform(size=self.n_nodes))

        if task == "multiclass":
            # assert
            error = np.random.normal(0, eps, (self.n_nodes, beta.shape[0]))
            logits = np.dot(feat_mat, beta.T) + error
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
            scatter = plt.scatter(self.x[:, 0], self.x[:, 1], c=self.community_labels, alpha=alpha)
            legend_labels = list(set(self.community_labels))
            handles = scatter.legend_elements()[0]
            plt.legend(handles, legend_labels, title="Community Labels")
            plt.show()

        if self.x.shape[1] > 3:
            assert self.x_tsne is not None, "call ""reduce_dim_x"" first"

            scatter = plt.scatter(self.x_tsne[:, 0], self.x_tsne[:, 1], c=self.community_labels, alpha=alpha)
            legend_labels = list(set(self.community_labels))
            handles = scatter.legend_elements()[0]
            plt.legend(handles, legend_labels, title="Community Labels")
            plt.show()


    def plot_graph(self, alpha=.9, width=.1):

        if self.x.shape[1] == 2:
            pos = {i: (self.x[i, 0], self.x[i, 1]) for i in range(len(self.x))}
        else:
            pos = nx.spring_layout(self.Nx)

        nx.draw(self.Nx, pos, node_size=self.degrees, node_color=self.cluster_labels,
                edge_color='grey', font_color='black', width=width, alpha=alpha)
        plt.show()


    def rich_plot_graph(self, ns:int=20, wdth:float=.3, alph:float=.1):
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
        color_map = {0: 'purple', 1: 'darkgreen', 2: 'gold', 3: "black", 4: "silver"}
        shape_map = {0: 'x', 1: 'o', 2: "d", 3: "D", 4: "<"}

        # if self.task != "regression":
        plt.figure(figsize=(10, 10))
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


    def purity(self, group_by: str = "Community", metric: str = "gini", plot_it:bool=False):
        """
        :param group_by: ["Community", "Cluster"]
        :param metric: gini or enrtropy
        :param plot_it: colors for plotting
        Plot Grouped Distribution for Target Y
        :return: A dataframe with purity-scores for each group
        """
        assert group_by in ["Community", "Feat.Cluster"], "group_by 'Feat.Cluster' or 'Community'"
        cdict = {"Community": self.community_labels, "Feat.Cluster": self.cluster_labels}

        df = pd.DataFrame({group_by: cdict[group_by], 'Y': self.y})

        if not self.task == "regression":
            # Use Characteristics suitable for categorical data
            grouped_counts = df.groupby([group_by, 'Y']).size().unstack(fill_value=0)

            if plot_it:
                grouped_counts.plot(kind='bar', figsize=(10, 6))
                plt.xlabel(group_by)
                plt.ylabel('Grouped Frequencies')
                plt.title(' ')
                plt.legend(title="Target Label")
                plt.show()

            if metric == "gini":
                fun = lambda x: 1 - sum(x ** 2)
            if metric == "entropy":
                fun = lambda x: -sum(x * np.log2(x + 1e-9))

            within_c = grouped_counts

            return (within_c.
                   div(within_c.sum(axis=1), axis=0).
                   apply(fun, axis=1)
                    )

       # ------------------ for numeric targets -------------------

        #else:
         #   unique_groups = df[group_by].unique()

          #  plt.figure(figsize=(10, 6))
           # for color, group in zip(colors, unique_groups):
            #    sns.kdeplot(data=df[df[group_by] == group], x='Y', label=group,
             #               color=color, fill=True, common_norm=False)

            #plt.xlabel('Y')
            #plt.ylabel('Density')
            # plt.title(' ')
            #plt.legend(title=group_by)
            #plt.show()

            #for group in df[group_by].unique():
             #   group_values = df[df[group_by] == group]['Y']
              #  mu = np.mean(group_values)
               # sigma = np.std(group_values, ddof=1)
                #print(f"Community {group}: μ = {mu:.2f}, σ = {sigma:.2f}")


def getB(m: int, b_range: tuple, w_range: tuple, rs: int = False):
    """

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

def getW(m_targets, n_communities, j_features, k_clusters, feature_info="number",
         w_degree=0, w_x=4, w_com=1.5, exponent:float=1):
    """
    Generate community beta matrix with row-wise exponential distributed values.

    is relevant.
    :param m_targets:
    :param n_communities:
    :param j_features:
    :param w_degree:
    :param w_x:
    :param exponent:
    :param w_com: if 0 -> communities irrelevant for target
    :return:
    """
    degree_betas = np.random.normal(loc=w_degree, scale=1, size=(m_targets, 1))

    if feature_info == "number":
        x_betas = np.random.normal(loc=w_x, scale=1, size=(j_features, m_targets)).T
    if feature_info == "cluster":
        x_betas = np.power(np.random.exponential(w_x, (k_clusters, m_targets)).T, exponent)
    else:
        raise ValueError("feature_info must either be 'number' or 'cluster'.")

    community_betas = np.power(np.random.exponential(w_com, (n_communities, m_targets)).T, exponent)

    return np.hstack((degree_betas, x_betas, community_betas))

def from_config(config:dict, rs = 26):
    """
    Generate entire graph from config dictionary.
    :param config: a dictionary with specified args
    :param rs: random_state
    """
    # 0) ------- Set Random State and functions--------
    global getB, getW
    np.random.seed(rs)
    random.seed(rs)
    torch.manual_seed(rs)

    # 1) ----------------- Set Params ----------------- kinda redundant
    community_sizes = config["community_sizes"]
    n = sum(community_sizes)  
    b_communities = len(community_sizes)

    m_features = config["m_features"]
    k_clusters = config["k_clusters"]
    alpha, beta, lmbd = config["alpha"], config["beta"], config["lmbd"] #fixed anyway

    b_com_r, w_com_r = config["between_com_prob_range"], config["within_com_prob_range"]
    w_clust_v, w_clust_c = config["within_clust_variance_range"], config["within_clust_covariance_range"]

    n_targets = config["n_targets"]

    centroid_variance_range = config["centroid_variance_range"]
    centroid_covariance_range = config["centroid_covariance_range"]
    cluster_sizes = config["cluster_sizes"]

    degree_importance = config["degree_importance"]
    x_importance = config["x_importance"]
    community_importance = config["community_importance"]

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
    omega = getW(m_targets=n_targets, n_communities=b_communities, j_features=m_features,
                 k_clusters=k_clusters, w_degree=degree_importance, feature_info=config["feature_info"],
                 w_x=x_importance, w_com=community_importance, exponent=1)

    g.set_y(task=config["task"], weights=omega, feature_info="number", eps=config["model_error"])
    g.set_Data_object()

    return g


if __name__ == "__main__":

    # 1) ----------------- Set Params -----------------
    community_sizes = [90, 140, 210, 160] # 4 communities; fixed
    n = sum(community_sizes)  # number of nodes (observations)
    b_communities = len(community_sizes)  # number of communities
    m_features = 6  # number of numeric features; fixed
    k_clusters = 6  # number of feature clusters; Overlap Scenario (over, under, match) 3,5,4
    alpha, beta, lmbd = 2, 20, .5 # degree_correction params; fixed
    br, wr = (.15, .2), (.4, .5) # assortative and dis-assortative

    # 2) ------- Instantiate Class Object (Note: No graspy called yet!) ---------
    B = getB(m=b_communities, b_range=br, w_range=wr)  # get Connection Matrix
    g = ADC_SBM(community_sizes=community_sizes, B=B)  # instantiate class

    # 3) ----------------- Generate the actual Graph -----------------
    g.correct_degree(alpha=alpha, beta=beta, lmbd=lmbd, distribution="exp")
    g.gen_graph()

    # 4) ----------------- Generate Node Features -----------------
    # Generate "k" centroids with "m" features
    centroids = np.random.multivariate_normal(np.repeat(0, m_features),  # mu
                                         getB(m_features,
                                              (0, 0),  # Covariance
                                              (25, 30)),  # Variance (relevant for cluster separation)
                                         k_clusters)  # n
    # centroids will be an array of size kxm
    # if the centroid variance is low and within-variance high,
    # separation becomes harder, and easier if it's the other way around !

    g.set_x(n_c=k_clusters,  # number of clusters
            mu=[tuple(point) for point in centroids],  # k tuple of coordinates in m-dimensional space
            sigma=[getB(m_features, (0, 0), # Covariance
                                    (1, 1.5)) # Variance (relevant for cluster separation)
                   for _ in range(k_clusters)],
            # similar covariance matrix for each centroid
            #w=np.random.dirichlet(np.ones(k_clusters), size=1).flatten()
            w=np.full(k_clusters, n/k_clusters, dtype=np.int64)
            )

    # 5) ----------------- Generate Targets -----------------
    ny = 5  # number of target classes; fixed
    nf = m_features + 1 + b_communities # number of relevant features (community and degree considered!)

    omega = getW(m_targets=ny,n_communities=b_communities,j_features=m_features,
                 k_clusters=k_clusters,feature_info="number",
                 w_degree=0, w_x= 2, w_com=1, exponent=1)
    g.set_y(task="multiclass", weights=omega, distribution="normal", eps=.5)

