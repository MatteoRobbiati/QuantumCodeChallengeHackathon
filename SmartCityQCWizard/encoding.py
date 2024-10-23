import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qibo.symbols import Z, I, X
from qibo import hamiltonians, set_backend
from qibo import Circuit, gates
from qibo.models import AdiabaticEvolution

set_backend("numpy")

# Create a graph based on the specified topology
G = nx.Graph()

# Define edges based on the user's topology
edges = [
    # (0, 1), (0, 2), (0, 3), (0, 6), (0, 8), (0, 10), (0, 11), (0, 12),
    (1, 2), (1, 4), (1, 7), (1, 9), (1, 10),
    (2, 3), (2, 4),
    # (3, 4),
    # (4, 5), (4, 7),
    # (5, 6), (5, 7),
    # (6, 8), (6, 7),
    # (7, 8), (7, 9),
    # (8, 9), (8, 10),
    # (9, 10),
    # (10, 11),
    # (11, 12)
]

G.add_edges_from(edges)

# Draw the graph to visualize its topology
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

def construct_H1(graph, params, node_a, node_b):
    """Target Hamiltonian"""
    # constraint 1: no isolated flips
    # constraint 2: not to touch a node twice
    # constraint 3: path length
    symbolic_ham = sum((params[i] * (Z(edge[0]) * Z(edge[1]) - 1)) for i, edge in enumerate(graph.edges))
    

node_a = 2
node_b = 7

h0 = construct_H0(G, node_a, node_b)

print("Expectation value on its ground state: ", h0.expectation(h0.ground_state()))

c = Circuit(len(G.nodes)+1)
c.add(gates.X(node_a))
c.add(gates.X(node_b))

print("Expectation value on the circuit state: ", h0.expectation(c().state()))

nedges = len(G.edges)


evo = AdiabaticEvolution(
    h0 = h0,
    h1 = h1,
    s = lambda t: t,
    dt = 0.01,
)

evolved = evo(T)




