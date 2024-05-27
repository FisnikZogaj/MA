import networkx as nx
import matplotlib.pyplot as plt


# Vertex Histogramm: Compare Distribution of given Node features of both graphs

G = nx.Graph()
H = nx.Graph()

G.add_nodes_from([
  ("A", {"color": "blue", "size": 250}),
  ("B", {"color": "yellow", "size": 400}),
  ("C", {"color": "orange", "size": 150}),
  ("D", {"color": "yellow", "size": 400}),
  ("E", {"color": "orange", "size": 150}),
  ("F", {"color": "red", "size": 600})
])

G.add_edges_from([
  ("A", "F"),
  ("F", "E"),
  ("F", "B"),
  ("B", "C"),
  ("E", "C"),
  ("B", "E"),
  ("E", "D")
])

H.add_nodes_from([
  ("A", {"color": "blue", "size": 250}),
  ("B", {"color": "yellow", "size": 400}),
  ("C", {"color": "orange", "size": 150}),
  ("D", {"color": "yellow", "size": 400}),
  ("E", {"color": "orange", "size": 150}),
  ("F", {"color": "red", "size": 600})
])

H.add_edges_from([
  ("A", "F"),
  ("F", "E"),
  ("F", "B"),
  ("D", "C"),
  ("E", "C"),
  ("B", "E"),
  ("E", "D")
])

node_colors = nx.get_node_attributes(G, "color").values()
colors = list(node_colors)
node_sizes = nx.get_node_attributes(G, "size").values()
sizes = list(node_sizes)

edge_indices = [(source, target) for source, target in G.edges]

print(edge_indices)
print(nx.adjacency_matrix(G).toarray())

nx.draw(G, with_labels=True, node_color=colors, node_size=sizes)
plt.show()
nx.draw(H, with_labels=True)
plt.show()


betweenness = nx.betweenness_centrality(G)

for node, centrality in betweenness.items():
    print("Node:", node, "Centrality:", centrality)