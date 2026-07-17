import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGraph(nn.Module):
    """
    Learnable Graph Module for adaptive adjacency learning.
    Combines a fixed graph A_fixed with a learned graph A_learn using a learnable parameter alpha.
    """
    def __init__(self, num_nodes):
        super(LearnableGraph, self).__init__()
        self.num_nodes = num_nodes
        
        # Learnable gate parameter alpha initialized to 0.5
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, Z, A_fixed):
        """
        Z: Node embeddings of shape [Batch, N, D]
        A_fixed: Fixed adjacency matrix (e.g. from Jaccard) of shape [Batch, N, N]
        """
        # A_learn = softmax(ZZ^T)
        # Z shape: [Batch, N, D]
        # Z^T shape: [Batch, D, N]
        scores = torch.matmul(Z, Z.transpose(1, 2)) # [Batch, N, N]
        
        # Apply softmax to get learned adjacency
        A_learn = F.softmax(scores, dim=-1)
        
        # Combine with fixed graph
        # alpha is constrained between 0 and 1
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        
        A_final = alpha_clamped * A_fixed + (1.0 - alpha_clamped) * A_learn
        
        return A_final, A_learn
