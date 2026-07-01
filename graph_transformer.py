import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceEncoder(nn.Module):
    """
    1D CNN to encode an appliance's temporal power and state sequence of length W
    into a continuous node embedding vector of size D.
    """
    def __init__(self, sequence_length=256, in_channels=2, embed_dim=64):
        super(SequenceEncoder, self).__init__()
        
        # 1D Convolutional layers to capture temporal features
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3) # [16, W/2]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)          # [32, W/4]
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)          # [64, W/8]
        
        self.pool = nn.MaxPool1d(kernel_size=2) # Reduces length by half
        
        # Calculate size after convolutions and pooling
        # W=256 -> conv1: 128 -> pool: 64 -> conv2: 32 -> pool: 16 -> conv3: 8 -> final len: 8 * 64 = 512
        conv_output_len = sequence_length // 32
        self.fc = nn.Linear(64 * conv_output_len, embed_dim)
        
    def forward(self, x):
        # Input shape: [Batch * Num_Nodes, W, Channels] -> Transpose to Conv1D format: [Batch * Num_Nodes, Channels, W]
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

class SequenceDecoder(nn.Module):
    """
    Decodes a node embedding vector of size D back into the reconstructed sequence [W, 2].
    """
    def __init__(self, embed_dim=64, sequence_length=256, out_channels=2):
        super(SequenceDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, sequence_length * out_channels)
        
    def forward(self, x):
        # Input shape: [Batch * Num_Nodes, D]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), self.sequence_length, self.out_channels)
        return x

class GraphTransformerLayer(nn.Module):
    """
    Custom Graph Transformer layer executing Multi-Head Self-Attention on nodes,
    incorporating edge weights from the adjacency matrix as an attention gate/bias.
    """
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Normalization and Feed-Forward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h, A):
        # h shape: [Batch, N, D]
        # A shape: [Batch, N, N]
        batch_size, num_nodes, _ = h.shape
        
        # 1. Project Q, K, V and split into heads
        q = self.q_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2) # [B, K, N, D_k]
        k = self.k_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2) # [B, K, N, D_k]
        v = self.v_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2) # [B, K, N, D_k]
        
        # 2. Compute Self-Attention Scores: Q K^T / sqrt(D_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B, K, N, N]
        
        # Softmax over key dimension
        attn_probs = F.softmax(scores, dim=-1) # [B, K, N, N]
        
        # 3. Gate the Attention Probabilities with the Adjacency matrix (Co-occurrence weights)
        # Expand Adjacency matrix to match shape [B, 1, N, N]
        A_expanded = A.unsqueeze(1)
        gated_attn = attn_probs * A_expanded
        
        # Re-normalize to ensure columns sum to 1
        gated_attn = gated_attn / (gated_attn.sum(dim=-1, keepdim=True) + 1e-9)
        gated_attn = self.dropout(gated_attn)
        
        # 4. Context aggregation
        out = torch.matmul(gated_attn, v) # [B, K, N, D_k]
        
        # Concatenate heads and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        out = self.out_proj(out)
        
        # Residual Connection & LayerNorm 1
        h = self.norm1(h + self.dropout(out))
        
        # Feed-Forward Network & LayerNorm 2
        ffn_out = self.ffn(h)
        h = self.norm2(h + self.dropout(ffn_out))
        
        return h

class GraphTransformerAutoencoder(nn.Module):
    """
    Complete Graph Transformer Autoencoder (GTAE) model.
    Encodes timeseries inputs, propagates messages along graph co-occurrences,
    and reconstructs both node features and graph structure.
    """
    def __init__(self, sequence_length=256, num_nodes=9, node_features=2, embed_dim=64, num_heads=4):
        super(GraphTransformerAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.node_features = node_features
        self.embed_dim = embed_dim
        
        # Shared node feature encoder
        self.encoder = SequenceEncoder(sequence_length, node_features, embed_dim)
        
        # New additions: Normalization and learnable node identities
        self.node_embed = nn.Embedding(num_nodes, embed_dim)
        self.enc_norm = nn.LayerNorm(embed_dim)
        
        # Graph Transformer layers
        self.gt1 = GraphTransformerLayer(embed_dim, num_heads)
        self.gt2 = GraphTransformerLayer(embed_dim, num_heads)
        
        # Shared node feature decoder
        self.decoder = SequenceDecoder(embed_dim, sequence_length, node_features)
        
    def encode(self, X, A):
        # X shape: [Batch, N, W, Channels] -> Flatten batch/node dimension for Conv1D encoder
        batch_size = X.size(0)
        X_flat = X.view(batch_size * self.num_nodes, self.sequence_length, self.node_features)
        
        # Encode sequences to initial embeddings: shape [Batch * N, D]
        h = self.encoder(X_flat)
        h = h.view(batch_size, self.num_nodes, self.embed_dim) # Shape: [Batch, N, D]
        
        # Apply LayerNorm and add learnable node embeddings to allow distinguishing appliances when OFF
        h = self.enc_norm(h)
        node_ids = torch.arange(self.num_nodes, device=X.device).unsqueeze(0).expand(batch_size, -1)
        h = h + self.node_embed(node_ids)
        
        # Graph message passing via Graph Transformer Layers
        h = self.gt1(h, A)
        h = self.gt2(h, A)
        return h
        
    def decode(self, h):
        # h shape: [Batch, N, D]
        batch_size = h.size(0)
        
        # 1. Reconstruct Node Sequences: shape [Batch * N, D] -> [Batch * N, W, Channels]
        h_flat = h.view(batch_size * self.num_nodes, self.embed_dim)
        X_recon_flat = self.decoder(h_flat)
        X_recon = X_recon_flat.view(batch_size, self.num_nodes, self.sequence_length, self.node_features)
        
        # 2. Reconstruct Adjacency Matrix A (Inner product decoder): shape [Batch, N, N]
        # Scaled dot product to prevent sigmoid saturation and vanishing gradients
        A_recon = torch.matmul(h, h.transpose(1, 2)) / (self.embed_dim ** 0.5)
        A_recon = torch.sigmoid(A_recon)
        
        return X_recon, A_recon
        
    def forward(self, X, A):
        # Encode
        h = self.encode(X, A)
        # Decode
        X_recon, A_recon = self.decode(h)
        return X_recon, A_recon
