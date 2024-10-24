from qibo import hamiltonians
from qibo.symbols import I, X, Z

# initial hamiltonian
def construct_H0(graph):
    """Initial Hamiltonian"""
    symb = sum(X(i) for i in range(0, len(graph.nodes), 1))
    return hamiltonians.SymbolicHamiltonian(symb)

def construct_H1(graph, adjacency_matrix):
    """
    Target Hamiltonian enforcing a chain of down spins connecting A and B.
    
    Args:
        graph: The graph whose nodes represent the spins.
        adjacency_matrix: Adjacency matrix representing the cost of edges.
    """
    symb = sum(Z(i) for i in range(len(graph.nodes)))
    for i in range(len(graph.nodes)):
        for j in range(i+1, len(graph.nodes)):
            symb += Z(i) * Z(j) * adjacency_matrix[i][j]
    
    return hamiltonians.SymbolicHamiltonian(symb)