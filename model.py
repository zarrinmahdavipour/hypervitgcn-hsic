import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class CNN3D(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(64, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

class LightweightViT(nn.Module):
    def __init__(self, in_channels, patch_sizes=[5, 7, 9], num_heads=4, dim=64):
        super(LightweightViT, self).__init__()
        self.patch_sizes = patch_sizes
        self.dim = dim
        self.patch_embed = nn.ModuleList([
            nn.Conv3d(in_channels, dim, kernel_size=(p, p, 3), stride=(p, p, 1), padding=(0, 0, 1))
            for p in patch_sizes
        ])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4),
            num_layers=2
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        embeddings = []
        for embed in self.patch_embed:
            patches = embed(x)
            patches = rearrange(patches, 'b c h w d -> b (h w d) c')
            embeddings.append(patches)
        x = torch.cat(embeddings, dim=1)
        x = self.transformer(x)
        x = self.norm(x)
        return x

class AttentionGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionGCN, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(out_features * 2, 1)
    
    def forward(self, x, adj):
        x = self.fc(x)
        N = x.size(1)
        a_input = torch.cat([x.repeat(1, N, 1), x.repeat(1, 1, N).view(-1, N, N, x.size(-1))], dim=-1)
        e = torch.tanh(self.attention(a_input)).squeeze(-1)
        adj = adj + e
        adj = F.softmax(adj, dim=-1)
        x = torch.matmul(adj, x)
        return x

class HyperViTGCN(nn.Module):
    def __init__(self, in_channels, num_classes, patch_sizes=[5, 7, 9], num_heads=4):
        super(HyperViTGCN, self).__init__()
        self.cnn = CNN3D(in_channels, out_channels=64)
        self.vit = LightweightViT(64, patch_sizes, num_heads, dim=64)
        self.gcn = AttentionGCN(64, 64)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x, adj):
        x = self.cnn(x)
        x = self.vit(x)
        x = self.gcn(x, adj)
        x = self.fc(x.mean(dim=1))
        return x

def create_adjacency_matrix(x, k=10):
    x = x.view(x.size(0), -1)
    cos_sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
    _, indices = torch.topk(cos_sim, k, dim=-1)
    adj = torch.zeros_like(cos_sim)
    adj.scatter_(1, indices, 1)
    return adj