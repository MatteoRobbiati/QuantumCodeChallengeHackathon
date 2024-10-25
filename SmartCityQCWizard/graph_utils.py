import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 6), (0, 8), (0, 10), (0, 11), (0, 12),
    (1, 2), (1, 4), (1, 7), (1, 9), (1, 10),
    (2, 3), (2, 4),
    (3, 4),
    (4, 5), (4, 7),
    (5, 6), (5, 7),
    (6, 8), (6, 7),
    (7, 8), (7, 9),
    (8, 9), (8, 10),
    (9, 10),
    (10, 11),
    (11, 12)
]

# Define the graph with 13 nodes and create fixed positions for them
G = nx.Graph()
G.add_nodes_from(range(13))  # Add 13 nodes (from 0 to 12)
# Add any initial edges as needed to construct the layout (these can be temporary)

# Generate fixed positions for 13 nodes and store them in a dictionary
fixed_positions = nx.spring_layout(G, seed=4242)  # Setting a seed for reproducibility

def color_graph_by_bitstring(G, bitstring, figname, weights, positions=fixed_positions):
    """
    Colors the nodes of a graph G based on a bitstring and plots the edge weights on top of the edges.
    Ones ('1') in the bitstring are colored red, and zeros ('0') are colored blue.
    
    Args:
        G (networkx.Graph): The graph whose nodes are to be colored.
        bitstring (str): A bitstring representing node colors (length must match number of nodes).
        figname (str): The filename where the figure will be saved.
        weights (numpy.ndarray): The adjacency matrix weights of the graph.
        positions (dict): Fixed positions for each node to ensure consistent layout.
    """
    # Ensure bitstring length matches the number of nodes in the graph
    assert len(bitstring) == G.number_of_nodes(), "Bitstring length must match number of nodes in the graph."

    # Create a color map where '1' is red and '0' is blue
    color_map = ['red' if bit == '1' else 'blue' for bit in bitstring]
    
    plt.figure(figsize=(8,6))
    
    # Draw nodes with colors using fixed positions
    nx.draw(G, positions, node_color=color_map, with_labels=True, node_size=700, font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, positions, width=2.0)
    
    # Ensure edges have 'weight' attributes
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if not edge_labels:  # If no weights exist, set default weights to 1
        edge_labels = {(u, v): np.round(weights[u][v], decimals=3) for u, v in G.edges()}

    nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, label_pos=0.5, font_size=12)
    
    # Save the figure
    plt.savefig(f"./figures/{figname}.png")



def construct_graph(edges, weights=None):
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

    if weights is not None:
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

    return G, None
