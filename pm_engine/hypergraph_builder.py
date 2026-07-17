import numpy as np
import torch

def build_hypergraph(A, window_size, app_states):
    """
    Constructs a hypergraph incidence matrix H and hyperedge weights W_e.
    A is the 4-channel adjacency matrix: [4, N, N]
      0: Jaccard
      1: Pearson
      2: Mutual Information
      3: Co-occurrence Frequency
    app_states is a list of binary ON/OFF states for each node: [N, W]
    
    Returns:
      H: Incidence matrix [N, N]
      W: Hyperedge weights [N]
    """
    num_nodes = A.shape[1]
    
    H = np.zeros((num_nodes, num_nodes))
    W = np.zeros(num_nodes)
    
    combined_score = 0.5 * A[3] + 0.3 * A[0] + 0.2 * A[2]
    
    # Create exactly one hyperedge per node
    for i in range(num_nodes):
        members = [i]
        for j in range(num_nodes):
            if i != j and combined_score[i, j] > 0.1:
                members.append(j)
                
        for node in members:
            H[node, i] = 1.0
            
        edge_freq = 0.0
        edge_jacc = 0.0
        edge_mi = 0.0
        count = 0
        for u in members:
            for v in members:
                if u != v:
                    edge_freq += A[3, u, v]
                    edge_jacc += A[0, u, v]
                    edge_mi += A[2, u, v]
                    count += 1
                    
        if count > 0:
            edge_freq /= count
            edge_jacc /= count
            edge_mi /= count
            
        W[i] = 0.5 * edge_freq + 0.3 * edge_jacc + 0.2 * edge_mi
        
    return H, W

