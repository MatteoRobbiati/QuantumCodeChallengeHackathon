import time
import argparse
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


def flux_adj_matrix(attendance_t0, attendance_t1, edges):
    matrix = np.zeros((len(attendance_t0), len(attendance_t0)))
    for edge in edges:
        i, j = edge
        di = attendance_t1[i] - attendance_t0[i]
        dj = attendance_t1[j] - attendance_t0[j]
        matrix[i, j] = matrix[j, i] = np.abs(di/dj) + np.abs(dj/di)
    return matrix


def get_zone_data(datetime, dataset):
    # Load dataset and parse datetime
    df = pd.read_csv(dataset)
    datetime = datetime

    # Get zones data for the specified datetime
    zones_data = zones_data_by_datetime(df=df, datetime=datetime)
    print(f"Data zone by zone:\n{zones_data}")
    return zones_data

# Set backend for qibo
set_backend("numpy")
#set_backend("qibojit", platform="cupy")

# Argument parser
parser = argparse.ArgumentParser(description="Run adiabatic evolution with specified dataset and datetime.")
parser.add_argument(
    "--datetime",
    nargs="+",
    type=str,
    default="2024-09-01 23:45:00",
    # required=True,
    help="The datetime string for filtering the dataset, format: 'YYYY-MM-DD HH:MM:SS'."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="./unique_attendance_15/presenza_15_010924-140924.csv",
    # required=True,
    help="Path to the dataset file (CSV format)."
)
args = parser.parse_args()

#breakpoint()
if len(args.datetime) > 2:
    raise RuntimeError

zones_data = [get_zone_data(datetime, args.dataset) for datetime in args.datetime]

# Define edges for the graph
edges = [
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

if len(zones_data) == 1:
    zones_data = zones_data[0]
    # Calculate weights based on zones data
    weights = [zones_data[str(int(edge[0])).zfill(3)] + zones_data[str(int(edge[1])).zfill(3)] for edge in edges]
    weights = np.array(weights) / np.max(weights) * 10

    print(f"Weights list: {weights}")

    # Construct graph and adjacency matrix
    G, adjacency_matrix = construct_graph(edges=edges, weights=weights)

elif len(zones_data) == 2:
    attendances = []
    for z_data in zones_data:
        attendances.append([z_data[zone] for zone in sorted(z_data.keys())])
    adjacency_matrix = flux_adj_matrix(*attendances, edges)
    # Construct graph and adjacency matrix
    G, _ = construct_graph(edges=edges, weights=None)

else:
    raise RuntimeError



num_nodes = G.number_of_nodes()
num_edges = len(G.edges)

# Draw and save the graph topology
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=10)
plt.title("Graph Topology")
plt.savefig("./figures/topology.png")

# Construct initial and target Hamiltonians
h0 = construct_H0(G)
h1 = construct_H1(graph=G, adjacency_matrix=adjacency_matrix)

# Setup callbacks and evolution
energy = callbacks.Energy(h1)
state = callbacks.State(copy=True)
evo = AdiabaticEvolution(h0=h0, h1=h1, s=lambda t: t, dt=0.025, callbacks=[energy, state])

# Initial state setup
c = Circuit(num_nodes)
for q in range(num_nodes):
    c.add(gates.H(q))
c.add(gates.M(*range(num_nodes)))
print("GS circuit: ", h0.expectation(c().state()))

# Run evolution
inital_time = time.time()
evolved = evo(final_time=50, initial_state=c().state())
print(f"\nTotal evolution time: {time.time() - inital_time}")

# Plot and analyze results
idx = np.argmin(energy.results)
plt.figure(figsize=(8,6))
plt.plot(energy.results, color="blue", lw="2")
plt.xlabel("Evolution time")
plt.ylabel("Energy")

# plt.vlines(idx, min(energy.results), max(energy.results), color="black", lw=1)
plt.savefig("./figures/energy.png")

# Find and display best and initial bitstrings
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

# save optimal state
np.save(arr=np.array(state.results[idx]), file=f"""./results/state_{'_'.join(args.datetime)}""")

# Display top states after optimization
top_10_dict = dict(sorted(freq_best.items(), key=lambda item: item[1], reverse=True)[:10])
print("\nShowing the top states after optimization")
for elem in top_10_dict:
    print(f"bitstring: {elem}, with {top_10_dict[elem]} shots")

# Visualize final and initial states on the graph
color_graph_by_bitstring(G, bitstring_best, figname="final", weights=adjacency_matrix)
color_graph_by_bitstring(G, bitstring_init, figname="initial", weights=adjacency_matrix)
