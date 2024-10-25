import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from graph_utils import color_graph_by_bitstring, construct_graph, EDGES
from plotscripts import generate_bitstring_combinations
from prepare_data import zones_data_by_datetime

def get_zone_data(datetime, dataset):
    # Load dataset and parse datetime
    df = pd.read_csv(dataset)
    datetime = datetime

    # Get zones data for the specified datetime
    zones_data = zones_data_by_datetime(df=df, datetime=datetime)
    print(f"Data zone by zone:\n{zones_data}")
    return zones_data

def flux_adj_matrix(attendance_t0, attendance_t1, edges):
    matrix = np.zeros((len(attendance_t0), len(attendance_t0)))
    for edge in edges:
        i, j = edge
        di = attendance_t1[i] - attendance_t0[i]
        dj = attendance_t1[j] - attendance_t0[j]
        matrix[i, j] = matrix[j, i] = np.abs(di/dj) + np.abs(dj/di)
    return matrix


def plot_one_result(index, dates, dataset):
    date1 = dates[index]
    date2 = dates[index+1]

    zones_data = [get_zone_data(datetime, dataset) for datetime in [date1, date2]]

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

    state = np.load(f"./results/presenza_15_150924_300924/state_{dates[i]}_{dates[i+1]}.npy")

    bitstrings = generate_bitstring_combinations(int(np.log2(len(state))))
    probs = (np.abs(state))**2

    top_2_indices = np.argpartition(probs, -2)[-2:]
    top_2_indices = top_2_indices[np.argsort(-probs[top_2_indices])]

    idx = min(top_2_indices)
    mybit = bitstrings[idx]

    color_graph_by_bitstring(
        G=G, 
        bitstring=mybit, 
        figname=f"matteo_graphs/final_{index}.png", 
        weights=adjacency_matrix
    )



dataset = "./unique_attendance_15/presenza_15_150924_300924.csv"
file_path = "./unique_attendance_15/unique_dates_presenza_15_150924_300924.txt"

with open(file_path, 'r') as file:
    dates = [line.strip() for line in file.readlines()]


for i in range(1,50,1):
    plot_one_result(i, dates, dataset)