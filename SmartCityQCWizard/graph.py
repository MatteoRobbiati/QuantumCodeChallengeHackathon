import numpy as np


# Cagliari's topology
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

    

def build_adjacency_matrix(attendance: list) -> np.array:
    """
    Parameters:
        attendance (dict): list of the measurements for a given time:
                     [(attendance) area 0, ... area 12, global]
    """
    if not len(attendance) == 13:
        raise RuntimeError
    matrix = np.ones((12, 12)) * 1e8
    for edge in EDGES:
        i, j = edge[0], edge[1]
        matrix[i, j] = matrix[j, i] = (attendance[i] + attendance[j]) / attendance[-1]
    return matrix

