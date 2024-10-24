import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def color_graph_by_bitstring(G, bitstring, figname, weights):
    """
    Colors the nodes of a graph G based on a bitstring and plots the edge weights on top of the edges.
    Ones ('1') in the bitstring are colored red, and zeros ('0') are colored blue.
    
    Args:
        G (networkx.Graph): The graph whose nodes are to be colored.
        bitstring (str): A bitstring representing node colors (length must match number of nodes).
        figname (str): The filename where the figure will be saved.
    """
    # Ensure bitstring length matches the number of nodes in the graph
    assert len(bitstring) == G.number_of_nodes(), "Bitstring length must match number of nodes in the graph."

    # Create a color map where '1' is red and '0' is blue
    color_map = ['red' if bit == '1' else 'blue' for bit in bitstring]
    
    # Draw the graph with the assigned colors
    pos = nx.spring_layout(G)  # Layout for visualizing the graph
    plt.figure(figsize=(8,6))
    
    # Draw nodes with colors
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=700, font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2.0)
    
    # Ensure edges have 'weight' attributes
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if not edge_labels:  # If no weights exist, set default weights to 1
        edge_labels = {(u, v): weights[u-1][v-1] for u, v in G.edges()}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=12)  # label_pos=0.5 places them at the middle of the edges
    
    # Save the figure
    plt.savefig(figname)
    plt.show()



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

    weights = np.random.uniform(0, 5, num_edges)

    # Initialize adjacency matrix with high weights
    high_weight = 0
    adjacency_matrix = np.full((num_nodes, num_nodes), high_weight)

    # Populate the adjacency matrix with weights from the external list
    for idx, (i, j) in enumerate(G.edges()):
        adjacency_matrix[i-1, j-1] = weights[idx]  # Use weights from the external list
        adjacency_matrix[j-1, i-1] = weights[idx]  # Undirected graph: set both (i,j) and (j,i)

    # Optionally set the diagonal to 0 (self-loops with zero weight)
    np.fill_diagonal(adjacency_matrix, 0)

    return G, adjacency_matrix