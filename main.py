import yaml
import torch
import argparse
from preprocessing import load_data, preprocess_data
from model import HyperViTGCN
from train import train_model
from evaluate import evaluate_model

def main(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess data
    dataset = config['dataset']
    data, labels = load_data(dataset)
    train_loader, test_loader = preprocess_data(data, labels, config)
    
    # Initialize model
    model = HyperViTGCN(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        patch_sizes=config['patch_sizes'],
        num_heads=config['num_heads']
    ).to(device)
    
    # Train model
    model = train_model(model, train_loader, config, device)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    print(f"Results for {dataset}:")
    print(f"OA: {results['OA']:.2f}%, AA: {results['AA']:.2f}%, Kappa: {results['Kappa']:.3f}")
    print(f"SAM: {results['SAM']:.2f}°, F1-Score: {results['F1']:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperViTGCN for Hyperspectral Image Classification")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)