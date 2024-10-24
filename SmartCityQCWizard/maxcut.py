import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from qibo import set_backend
from qibo import Circuit, gates, callbacks
from qibo.models import AdiabaticEvolution

from graph_utils import construct_graph, color_graph_by_bitstring
from hamiltonians_utils import construct_H0, construct_H1
from prepare_data import zones_data_by_datetime

set_backend("numpy")

# Define edges based on the user's topology
# edges = [
#     # (0, 1), (0, 2), (0, 3), (0, 6), (0, 8), (0, 10), (0, 11), (0, 12),
#     (1, 2), (1, 4), (1, 7), (1, 9), (1, 10),
#     (2, 3), (2, 4),
#     (3, 4),
#     (4, 5), (4, 7),
#     (5, 6), (5, 7),
#     (6, 8), (6, 7),
#     (7, 8), (7, 9),
#     (8, 9), (8, 10),
#     (9, 10),
#     (10, 11),
#     (11, 12)
# ]

edges = [
    (0, 1), (0, 3), (0, 6), (0, 8), (0, 9),
    (1, 2), (1, 3),
    (2, 3),
    (3, 4), (3, 6),
    (4, 5), (4, 6),
    (5, 7), (5, 6),
    (6, 7), (6, 8),
    (7, 8), (7, 9),
    (8, 9),
    (9, 10),
    (10, 11)
]

# load data for a specific datetime
df = pd.read_csv("unique_attendance_15/presenza_15_010824-140824.csv")
datetime = "2024-08-01 23:45:00"

zones_data = zones_data_by_datetime(df=df, datetime=datetime)
print(f"Data zone by zone:\n{zones_data}")

weights = []

for edge in edges:
    weights.append(
        zones_data[str(int(edge[0])).zfill(3)] + zones_data[str(int(edge[1])).zfill(3)]
    )

# rescale into range [0,10]
weights = np.array((weights - min(weights)) / (max(weights) - min(weights))) * 10
print(f"Weights list: {weights}")

G, adjacency_matrix = construct_graph(edges=edges, weights=weights)
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

energy = callbacks.Energy(h1)
state = callbacks.State(copy=True)

evo = AdiabaticEvolution(
    h0 = h0,
    h1 = h1,
    s = lambda t: t,
    dt = 0.01,
    callbacks=[energy, state],
)

# initial state
c = Circuit(num_nodes)
for q in range(num_nodes):
    c.add(gates.H(q))
c.add(gates.M(*range(num_nodes)))

print("GS circuit: ", h0.expectation(c().state()))

inital_time = time.time()
# evolve
evolved = evo(final_time=20, initial_state=c().state())
print(f"\nTotal evolution time: {time.time() - inital_time}")

idx = np.argmin(energy.results)

# energies plot
plt.figure(figsize=(8,6))
plt.plot(energy.results)
plt.vlines(idx, min(energy.results), max(energy.results), color="black", lw=1)
plt.savefig("./figures/energy.png")

c = Circuit(num_nodes)
c.add(gates.M(*range(num_nodes)))

out_best = c(initial_state=state.results[idx], nshots=10000)
freq_best = out_best.frequencies()

out_init = c(initial_state=state.results[0], nshots=10000)
freq_init = out_init.frequencies()

bitstring_best = max(freq_best, key=lambda k: freq_best[k])
bitstring_init = max(freq_init, key=lambda k: freq_init[k])


print(f"bitstring best is {bitstring_best} with frequency {freq_best[bitstring_best]}")
print(f"bitstring init is {bitstring_init} with frequency {freq_best[bitstring_init]}")

# plot_probabilities_from_state(state=state[int(idx)])
# Sort the dictionary by values in descending order and take the top 10 items
sorted_items = sorted(freq_best.items(), key=lambda item: item[1], reverse=True)[:10]
top_10_dict = dict(sorted_items)

print("\n Showing the top states after optimization")
for elem in top_10_dict:
    print(f"bitstring: {elem}, with {top_10_dict[elem]} shots")  

color_graph_by_bitstring(G, bitstring_best, figname="final", weights=adjacency_matrix)
color_graph_by_bitstring(G, bitstring_init, figname="initial", weights=adjacency_matrix)

