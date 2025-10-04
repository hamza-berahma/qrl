import networkx as nx

def get_nsfnet_topology():
    """Returns the NSFNET network graph."""
    G = nx.Graph()
    edges = [
        (1, 2), (1, 3), (1, 8), (2, 3), (2, 4), (3, 6), (4, 5), (5, 6),
        (5, 7), (6, 10), (6, 11), (7, 8), (7, 10), (8, 9), (9, 10),
        (9, 12), (10, 11), (10, 13), (11, 14), (12, 13), (13, 14)
    ]
    G.add_nodes_from(range(1, 15))
    G.add_edges_from(edges)
    return G

def get_grid_topology(n=4, m=4):
    """Returns an n x m grid graph."""
    G = nx.grid_2d_graph(n, m)
    # Relabel nodes to be 1-based integers
    G = nx.convert_node_labels_to_integers(G, first_label=1)
    return G
    
def get_small_world_topology(n=20, k=4, p=0.1):
    """Returns a Watts-Strogatz small-world graph."""
    G = nx.watts_strogatz_graph(n, k, p)
    G = nx.convert_node_labels_to_integers(G, first_label=1)
    return G


# Dictionary to easily access topologies by name
TOPOLOGIES = {
    "nsfnet": get_nsfnet_topology,
    "grid": get_grid_topology,
    "small_world": get_small_world_topology
}
