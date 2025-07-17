# train.py
import torch
import torch.nn as nn
from utils import compute_metrics
from model import create_adjacency_matrix

def train_model(model, train_loader, config, device, target_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7 if i < 100 else 1.0 
                                                         for i in range(config['num_classes'])], 
                                                        device=device))
    if config.get('use_domain_adaptation', False):
        domain_criterion = nn.BCELoss()
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        if config.get('use_domain_adaptation', False) and target_loader is not None:
            target_iter = iter(target_loader)
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            adj = create_adjacency_matrix(data, k=10).to(device)
            optimizer.zero_grad()
            
            if config.get('use_domain_adaptation', False) and target_loader is not None:
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                target_data = target_data.to(device)
                target_adj = create_adjacency_matrix(target_data, k=10).to(device)
                
                # Forward pass for source (classification + domain)
                outputs, domain_outputs = model(data, adj, return_features=True)
                cls_loss = criterion(outputs, labels)
                
                # Domain labels: 1 for source, 0 for target
                source_domain_labels = torch.ones(data.size(0), 1).to(device)
                target_domain_labels = torch.zeros(target_data.size(0), 1).to(device)
                
                # Forward pass for target (domain only)
                _, target_domain_outputs = model(target_data, target_adj, return_features=True)
                
                # Compute domain loss
                domain_loss_source = domain_criterion(domain_outputs, source_domain_labels)
                domain_loss_target = domain_criterion(target_domain_outputs, target_domain_labels)
                domain_loss = (domain_loss_source + domain_loss_target) * config.get('lambda_adv', 0.1)
                
                # Total loss
                loss = cls_loss + domain_loss
            else:
                outputs = model(data, adj)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model
