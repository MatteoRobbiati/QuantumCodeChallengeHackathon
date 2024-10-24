from hamiltonians import H0_TSP, H1_TSP
from graph import build_adjacency_matrix

from qibo import Circuit, gates, callbacks
from qibo.models import AdiabaticEvolution
from qibo.backends import NumpyBackend

from qibojit.backends import CupyBackend, NumbaBackend

import numpy as np
from functools import reduce

def solve_TSP_for_attendance(attendance: list, backend: "qibo.Backend"):

    H0 = H0_TSP(13, backend)
    adj_matrix = build_adjacency_matrix(attendance)
    H1 = H1_TSP(13, adj_matrix, backend)

    print(f"Adjacency Matrix: \n{adj_matrix}")
    print(f"H0: {H0.form}")
    print(f"H1: {H1.form}")
    
    energy = callbacks.Energy(H1)
    evolutor = AdiabaticEvolution(
        h0 = H0,
        h1 = H1,
        s = lambda t: t,
        dt = 0.1,
        callbacks=[energy],
    )

    ground_state = Circuit(1)
    ground_state.add(gates.X(0))
    ground_state.add(gates.H(0))
    ground_state = ground_state().state()
    ground_state = reduce(backend.np.outer, [ground_state for _ in range(13)]).ravel()

    print(f"Ground State: {ground_state}")
    
    evolved = evolutor(final_time=1, initial_state=ground_state)
    

if __name__ == "__main__":

    backend = CupyBackend()
    attendance = backend.cast(np.random.randn(14), backend.precision)
    solve_TSP_for_attendance(attendance, backend=backend)
