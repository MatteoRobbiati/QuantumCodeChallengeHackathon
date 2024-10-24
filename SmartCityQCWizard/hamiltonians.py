from math import prod
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qibo.symbols import Z, I, X
from qibo import hamiltonians, set_backend
from qibo import Circuit, gates, callbacks
from qibo.models import AdiabaticEvolution


# Traveling Salesman Problem (TSP)

def H0_TSP(nqubits: int, backend: "qibo.Backend"):
    return hamiltonians.SymbolicHamiltonian(
        prod([X(i) for i in range(nqubits)]),
        nqubits=nqubits,
        backend=backend
    )

def H1_TSP(nqubits: int, adj_matrix: "ndarray", backend: "qibo.Backend", lagrange_mul: "ndarray" = None):
    terms = []
    for i in range(nqubits - 1):
        for j in range(i + 1, nqubits):
            terms.append(adj_matrix[i,j] * X(i) * X(j))

    if lagrange_mul is None:
        lagrange_mul = backend.np.ones(nqubits) * 2
    for i in range(nqubits):
        terms.append(
            lagrange_mul[i] *
            sum([(2 - X(i) * X(j)) ** 2 for j in range(nqubits) if j != i])
            )
    return hamiltonians.SymbolicHamiltonian(
        sum(terms),
        nqubits=nqubits,
        backend=backend
    )

# Shortest Path (SP)

def HO_SP(nqubits: int, backend: "qibo.Backend"):
    return H0_TSP(nqubits, backend)


def H1_SP(nqubits: int, start: int, end: int, adj_matrix: "ndarray", backend: "qibo.Backend"):
    HS = hamiltonians.SymbolicHamiltonian(
        (2 * sum([X(start) * X(j) for j in range(nqubits) if j != start]) - 1) ** 2,
        nqubits=nqubits,
        backend=backend
    )

#------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    set_backend("numpy")

    # Create a graph based on the specified topology
    G = nx.Graph()

    # Define edges based on the user's topology
    edges = [
        # (0, 1), (0, 2), (0, 3), (0, 6), (0, 8), (0, 10), (0, 11), (0, 12),
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

    G.add_edges_from(edges)

    num_nodes = G.number_of_nodes()
    num_edges = len(G.edges)

    weights = np.random.uniform(0, 1, num_edges)

    # Initialize adjacency matrix with high weights
    high_weight = float('inf')
    adjacency_matrix = np.full((num_nodes, num_nodes), high_weight)

    # Populate the adjacency matrix with weights from the external list
    for idx, (i, j) in enumerate(G.edges()):
        adjacency_matrix[i-1, j-1] = weights[idx]  # Use weights from the external list
        adjacency_matrix[j-1, i-1] = weights[idx]  # Undirected graph: set both (i,j) and (j,i)

    # Optionally set the diagonal to 0 (self-loops with zero weight)
    np.fill_diagonal(adjacency_matrix, 0)

    # Display the adjacency matrix
    print("Adjacency Matrix:")
    print(adjacency_matrix)

    # Draw the graph to visualize its topology
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G)  
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=10)
    plt.title("Graph Topology")
    plt.savefig("./figures/topology.png")



    # initial hamiltonian
    def construct_H0(graph, node_a, node_b):
        """Initial Hamiltonian"""
        symbolic_ham = Z(node_a)
        symbolic_ham += Z(node_b)
        for i in range(len(graph.nodes)):
            if i not in [node_a, node_b]:
                symbolic_ham -= Z(i)
        return hamiltonians.SymbolicHamiltonian(symbolic_ham)

    # target hamiltonian
    def construct_H1(graph, adjacency_matrix, node_a, node_b, alpha, beta):
        """
        Target Hamiltonian.

        Args:
            graph: the initial graph;
            adjacency_matrix: the cost of edges;
            node_a: starting point;
            node_b: final point;
            alpha: penalty for non-connected nodes;
            beta: penalty for isolated flips;
        """
        # Create the Hamiltonian components
        symb = Z(node_a) + Z(node_b)  # Fix A and B down

        num_nodes = len(graph.nodes)  # Number of nodes in the graph

        # Path penalty: encourage a path between node_a and node_b
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adjacency_matrix[i][j] != float('inf'):
                    # Include penalty for valid path connection
                    symb += adjacency_matrix[i][j] * Z(i) * Z(j)
                else:
                    # Penalize invalid paths (non-existing edges)
                    symb += alpha * Z(i) * Z(j)

        # Isolated flips penalty
        for i in range(num_nodes):
            if i != node_a and i != node_b:
                # Check the degree of the current node i
                degree = sum(1 for j in range(num_nodes) if adjacency_matrix[i][j] != float('inf'))
                if degree < 2:
                    # Penalize isolated nodes or those with only one connection
                    symb += beta * Z(i)  # Apply isolated flip penalty

                # Add terms to allow flipping of the spin
                # This allows other nodes to be flipped up and down
                symb += Z(i)  # This term enables flipping the spin of node i

        # Create the symbolic Hamiltonian
        return hamiltonians.SymbolicHamiltonian(symb)

    node_a = 3
    node_b = 10

    h0 = construct_H0(G, node_a, node_b)

    c = Circuit(len(G.nodes))
    c.add(gates.X(node_a))
    c.add(gates.X(node_b))

    print("Expectation value on the initial circuit state: ", h0.expectation(c().state()))


    h1 = construct_H1(
        graph=G,
        adjacency_matrix=adjacency_matrix,
        node_a=node_a,
        node_b=node_b,
        alpha=10,
        beta=10,
    )

    print("Expectation value on the initial circuit state: ", h1.expectation(c().state()))

    optimal_c = Circuit(len(G.nodes))
    optimal_c.add(gates.X(node_a))
    optimal_c.add(gates.X(node_b))
    optimal_c.add(gates.X(1))
    optimal_c.add(gates.X(2))


    print("Expectation value on the optimal circuit state: ", h1.expectation(optimal_c().state()))

    energy = callbacks.Energy(h1)

    evo = AdiabaticEvolution(
        h0 = h0,
        h1 = h1,
        s = lambda t: t,
        dt = 0.05,
        callbacks=[energy],
    )

    evolved = evo(final_time=10, initial_state=c().state())

    plt.figure(figsize=(8,6))
    plt.plot(energy.results)
    plt.savefig("./figures/energy.png")



