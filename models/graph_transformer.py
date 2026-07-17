import torch
import torch.nn as nn
import torch.nn.functional as F
from learnable_graph import LearnableGraph

class SequenceEncoder(nn.Module):
    """
    1D CNN to encode an appliance's temporal power and state sequence of length W
    into a continuous node embedding vector of size D.
    """
    def __init__(self, sequence_length=256, in_channels=9, embed_dim=64):
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
    Decodes a node embedding vector of size D back into the reconstructed sequence [W, Channels].
    """
    def __init__(self, embed_dim=64, sequence_length=256, out_channels=9):
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
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
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
        batch_size, num_nodes, _ = h.shape
        
        q = self.q_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(scores, dim=-1)
        
        A_expanded = A.unsqueeze(1)
        gated_attn = attn_probs * A_expanded
        
        gated_attn = gated_attn / (gated_attn.sum(dim=-1, keepdim=True) + 1e-9)
        gated_attn = self.dropout(gated_attn)
        
        out = torch.matmul(gated_attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        out = self.out_proj(out)
        
        h = self.norm1(h + self.dropout(out))
        ffn_out = self.ffn(h)
        h = self.norm2(h + self.dropout(ffn_out))
        return h

class HypergraphConv(nn.Module):
    """
    Hypergraph Neural Network layer.
    Propagates messages via incidence matrix H and weights W_e.
    """
    def __init__(self, embed_dim=64):
        super(HypergraphConv, self).__init__()
        self.theta = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, X, H, W_e):
        # X: [Batch, N, D]
        # H: [Batch, N, N]
        # W_e: [Batch, N]
        
        # Node degree D_v and Hyperedge degree D_e
        # D_v = H * W_e
        # D_e = H^T * 1
        
        batch_size, N, _ = H.shape
        
        # Calculate D_e: [Batch, N]
        D_e = H.sum(dim=1)
        D_e_inv = torch.where(D_e > 0, 1.0 / D_e, torch.zeros_like(D_e))
        D_e_inv_diag = torch.diag_embed(D_e_inv) # [Batch, N, N]
        
        # Calculate D_v: [Batch, N]
        D_v = torch.matmul(H, W_e.unsqueeze(-1)).squeeze(-1)
        D_v_inv = torch.where(D_v > 0, 1.0 / torch.sqrt(D_v), torch.zeros_like(D_v))
        D_v_inv_diag = torch.diag_embed(D_v_inv) # [Batch, N, N]
        
        W_e_diag = torch.diag_embed(W_e) # [Batch, N, N]
        
        # Message passing: D_v_inv * H * W_e * D_e_inv * H^T * D_v_inv * X * Theta
        # We can simplify H * W_e * D_e_inv * H^T as Adjacency for hypergraph
        
        term1 = torch.matmul(D_v_inv_diag, H)
        term2 = torch.matmul(term1, W_e_diag)
        term3 = torch.matmul(term2, D_e_inv_diag)
        term4 = torch.matmul(term3, H.transpose(1, 2))
        A_hyper = torch.matmul(term4, D_v_inv_diag) # [Batch, N, N]
        
        X_theta = self.theta(X)
        out = torch.matmul(A_hyper, X_theta)
        return F.relu(out)

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim=64):
        super(AttentionFusion, self).__init__()
        self.w_g = nn.Linear(embed_dim, 1)
        self.w_h = nn.Linear(embed_dim, 1)
        
    def forward(self, Z_graph, Z_hyper):
        # Calculate attention scores for each node
        alpha_g = self.w_g(Z_graph) # [Batch, N, 1]
        alpha_h = self.w_h(Z_hyper) # [Batch, N, 1]
        
        scores = torch.cat([alpha_g, alpha_h], dim=-1) # [Batch, N, 2]
        weights = F.softmax(scores, dim=-1) # [Batch, N, 2]
        
        Z_fusion = weights[:, :, 0:1] * Z_graph + weights[:, :, 1:2] * Z_hyper
        return Z_fusion, weights

class HGLGGTAE(nn.Module):
    """
    Hypergraph Enhanced Learnable Graph Graph Transformer Autoencoder.
    """
    def __init__(self, sequence_length=256, num_nodes=9, node_features=9, embed_dim=64, num_heads=4):
        super(HGLGGTAE, self).__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.node_features = node_features
        self.embed_dim = embed_dim
        
        # Node feature encoder
        self.encoder = SequenceEncoder(sequence_length, node_features, embed_dim)
        
        self.node_embed = nn.Embedding(num_nodes, embed_dim)
        self.enc_norm = nn.LayerNorm(embed_dim)
        
        # Learnable Graph Module
        self.learnable_graph = LearnableGraph(num_nodes)
        
        # Graph Transformer Branch
        self.gt1 = GraphTransformerLayer(embed_dim, num_heads)
        self.gt2 = GraphTransformerLayer(embed_dim, num_heads)
        
        # Hypergraph Branch
        self.hg1 = HypergraphConv(embed_dim)
        self.hg2 = HypergraphConv(embed_dim)
        
        # Attention Fusion
        self.fusion = AttentionFusion(embed_dim)
        
        # Decoder
        self.decoder = SequenceDecoder(embed_dim, sequence_length, node_features)
        
    def encode(self, X, A_fixed, H, W_e):
        batch_size = X.size(0)
        X_flat = X.view(batch_size * self.num_nodes, self.sequence_length, self.node_features)
        
        h = self.encoder(X_flat)
        h = h.view(batch_size, self.num_nodes, self.embed_dim)
        
        h = self.enc_norm(h)
        node_ids = torch.arange(self.num_nodes, device=X.device).unsqueeze(0).expand(batch_size, -1)
        Z_init = h + self.node_embed(node_ids)
        
        # Learnable Graph Adjacency
        A_final, A_learn = self.learnable_graph(Z_init, A_fixed)
        
        # Graph Branch
        Z_graph = self.gt1(Z_init, A_final)
        Z_graph = self.gt2(Z_graph, A_final)
        
        # Hypergraph Branch
        Z_hyper = self.hg1(Z_init, H, W_e)
        Z_hyper = self.hg2(Z_hyper, H, W_e)
        
        # Fusion
        Z_fusion, attn_weights = self.fusion(Z_graph, Z_hyper)
        
        return Z_fusion, A_final, A_learn, attn_weights
        
    def decode(self, Z_fusion):
        batch_size = Z_fusion.size(0)
        
        # Reconstruct Node Sequences
        h_flat = Z_fusion.view(batch_size * self.num_nodes, self.embed_dim)
        X_recon_flat = self.decoder(h_flat)
        X_recon = X_recon_flat.view(batch_size, self.num_nodes, self.sequence_length, self.node_features)
        
        # Reconstruct Graph Adjacency (A_final)
        A_recon = torch.matmul(Z_fusion, Z_fusion.transpose(1, 2)) / (self.embed_dim ** 0.5)
        A_recon = torch.sigmoid(A_recon)
        
        # Reconstruct Hypergraph Incidence (H)
        H_recon = torch.matmul(Z_fusion, Z_fusion.transpose(1, 2)) / (self.embed_dim ** 0.5)
        H_recon = torch.sigmoid(H_recon)
        
        return X_recon, A_recon, H_recon
        
    def forward(self, X, A_fixed, H, W_e):
        Z_fusion, A_final, A_learn, attn_weights = self.encode(X, A_fixed, H, W_e)
        X_recon, A_recon, H_recon = self.decode(Z_fusion)
        return X_recon, A_recon, H_recon, A_final, attn_weights

# Alias for backward compatibility
GraphTransformerAutoencoder = HGLGGTAE
