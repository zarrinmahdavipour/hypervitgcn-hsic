# main.py
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
    
    # Load and preprocess source data
    dataset = config['dataset']
    data, labels, _, _ = load_data(dataset)
    train_loader, test_loader = preprocess_data(data, labels, config, is_target=False)
    
    # Initialize model
    model = HyperViTGCN(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        patch_sizes=config['patch_sizes'],
        num_heads=config['num_heads'],
        use_domain_adaptation=config.get('use_domain_adaptation', False)
    ).to(device)
    
    # Train model
    target_loader = None
    if config.get('use_domain_adaptation', False):
        target_dataset = config['target_dataset']
        target_data, target_labels, _, _ = load_data(target_dataset)
        target_loader = preprocess_data(target_data, target_labels, config, is_target=True)
    model = train_model(model, train_loader, config, device, target_loader=target_loader)
    
    # Evaluate model on source dataset
    results = evaluate_model(model, test_loader, device)
    print(f"Results for {dataset} (source):")
    print(f"OA: {results['OA']:.2f}%, AA: {results['AA']:.2f}%, Kappa: {results['Kappa']:.3f}")
    print(f"SAM: {results['SAM']:.2f}°, F1-Score: {results['F1']:.2f}%")
    
    # Evaluate model on target datasets (zero-shot and fine-tuned)
    if config.get('cross_domain', False):
        source_oa = results['OA']  # Store source OA for DSI computation
        for target_dataset in config['target_datasets']:
            target_data, target_labels, _, _ = load_data(target_dataset)
            target_loader = preprocess_data(target_data, target_labels, config, is_target=True)
            
            # Zero-shot evaluation
            results = evaluate_model(model, target_loader, device)
            results['DSI'] = ((source_oa - results['OA']) / source_oa) * 100
            print(f"\nZero-shot results for {target_dataset}:")
            print(f"OA: {results['OA']:.2f}%, AA: {results['AA']:.2f}%, Kappa: {results['Kappa']:.3f}")
            print(f"SAM: {results['SAM']:.2f}°, F1-Score: {results['F1']:.2f}%, DSI: {results['DSI']:.2f}%")
            
            # Fine-tuning
            if config.get('fine_tune', False):
                fine_tune_loader, _ = preprocess_data(target_data, target_labels, config, is_target=False)
                config['epochs'] = 10  # Reduced epochs for fine-tuning
                model_fine_tuned = train_model(model, fine_tune_loader, config, device)
                results = evaluate_model(model_fine_tuned, target_loader, device)
                results['DSI'] = ((source_oa - results['OA']) / source_oa) * 100
                print(f"\nFine-tuned results for {target_dataset}:")
                print(f"OA: {results['OA']:.2f}%, AA: {results['AA']:.2f}%, Kappa: {results['Kappa']:.3f}")
                print(f"SAM: {results['SAM']:.2f}°, F1-Score: {results['F1']:.2f}%, DSI: {results['DSI']:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperViTGCN for Hyperspectral Image Classification")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
