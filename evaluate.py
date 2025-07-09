import torch
from utils import compute_metrics
from model import create_adjacency_matrix

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            adj = create_adjacency_matrix(data, k=10).to(device)
            outputs = model(data, adj)
            all_preds.append(outputs)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return compute_metrics(all_preds, all_labels)