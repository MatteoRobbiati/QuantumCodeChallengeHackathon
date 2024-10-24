
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def color_graph_by_bitstring(G, bitstring, figname):
    """
    Colors the nodes of a graph G based on a bitstring.
    Ones ('1') in the bitstring are colored red, and zeros ('0') are colored blue.
    
    Args:
        G (networkx.Graph): The graph whose nodes are to be colored.
        bitstring (str): A bitstring representing node colors (length must match number of nodes).
    """
    # Ensure bitstring length matches the number of nodes in the graph
    assert len(bitstring) == G.number_of_nodes(), "Bitstring length must match number of nodes in the graph."

    # Create a color map where '1' is red and '0' is blue
    color_map = ['red' if bit == '1' else 'blue' for bit in bitstring]
    
    # Draw the graph with the assigned colors
    pos = nx.spring_layout(G)  # Layout for visualizing the graph
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=700, font_weight='bold')
    plt.savefig(figname)


def construct_graph(edges):
    """
    Construct graph and adjacency matrix given edges list.
    
    Args: 
        edges: edges.
    
    Return:
        graph, adjacency matrix.
    """
    # Create a graph based on the specified topology
    G = nx.Graph()

    G.add_edges_from(edges)

    num_nodes = G.number_of_nodes()
    num_edges = len(G.edges)

    weights = np.random.uniform(0, 20, num_edges)

    # Initialize adjacency matrix with high weights
    high_weight = 1e8
    adjacency_matrix = np.full((num_nodes, num_nodes), high_weight)

    # Populate the adjacency matrix with weights from the external list
    for idx, (i, j) in enumerate(G.edges()):
        adjacency_matrix[i-1, j-1] = weights[idx]  # Use weights from the external list
        adjacency_matrix[j-1, i-1] = weights[idx]  # Undirected graph: set both (i,j) and (j,i)

    # Optionally set the diagonal to 0 (self-loops with zero weight)
    np.fill_diagonal(adjacency_matrix, 0)

    return G, adjacency_matrix