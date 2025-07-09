import torch
import torch.nn as nn
from utils import compute_metrics
from model import create_adjacency_matrix

def train_model(model, train_loader, config, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7 if i < 100 else 1.0 
                                                         for i in range(config['num_classes'])], 
                                                        device=device))
    
    for epoch in range(config['epochs']):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            adj = create_adjacency_matrix(data, k=10).to(device)
            optimizer.zero_grad()
            outputs = model(data, adj)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}")
    return model