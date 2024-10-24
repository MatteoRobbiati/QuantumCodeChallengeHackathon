import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from qibo import set_backend
from qibo import Circuit, gates, callbacks
from qibo.models import AdiabaticEvolution

from graph_utils import construct_graph, color_graph_by_bitstring
from hamiltonians_utils import construct_H0, construct_H1

set_backend("numpy")

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

G, adjacency_matrix = construct_graph(edges=edges)
num_nodes = G.number_of_nodes()
num_edges = len(G.edges)

# Draw the graph to visualize its topology
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G)  
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=10)
plt.title("Graph Topology")
plt.savefig("./figures/topology.png")

h0 = construct_H0(G)
h1 = construct_H1(
    graph=G,
    adjacency_matrix=adjacency_matrix,
)

initial_state = h0.ground_state()

energy = callbacks.Energy(h1)
state = callbacks.State(copy=True)

evo = AdiabaticEvolution(
    h0 = h0,
    h1 = h1,
    s = lambda t: t,
    dt = 0.1,
    callbacks=[energy, state],
)

# initial state
c = Circuit(num_nodes)
for q in range(num_nodes):
    c.add(gates.H(q))
c.add(gates.M(*range(num_nodes)))

# evolve
evolved = evo(final_time=20, initial_state=initial_state)
idx = np.argmin(energy.results)

# energies plot
plt.figure(figsize=(8,6))
plt.plot(energy.results)
plt.vlines(idx, min(energy.results), max(energy.results), color="black", lw=1)
plt.savefig("./figures/energy.png")

c = Circuit(num_nodes)
c.add(gates.M(*range(num_nodes)))

out = c(initial_state=state.results[idx], nshots=10000)
freq = out.frequencies()

max_key = max(freq, key=lambda k: freq[k])
print(max_key, freq[max_key])


color_graph_by_bitstring(G, max_key, figname="final.png")
color_graph_by_bitstring(G, "111111111111", figname="initial.png")

