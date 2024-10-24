import matplotlib.pyplot as plt
import numpy as np

def generate_bitstring_combinations(n):
    """Generate all bitstring combinations given bitstring length `n`."""
    bitstrings = []
    for i in range(2**n):
        bitstrings.append(format(i, f"0{n}b"))
    return bitstrings

def plot_probabilities_from_state(state):
    """Plot amplitudes for a given quantum `state`."""

    bitstring = generate_bitstring_combinations(int(np.log2(len(state))))
    
    fig, ax = plt.subplots(figsize=(10, 5 * 6/8))
    
    ax.set_title('State visualization')
    ax.set_xlabel('States')
    ax.set_ylabel('Probability')
    
    for i, amp in enumerate(state):
        ax.bar(bitstring[i], np.abs(amp)**2, color='#C194D8', edgecolor="black")
        
    plt.xticks(rotation=90)
    plt.savefig("./figures/best_hist.png")