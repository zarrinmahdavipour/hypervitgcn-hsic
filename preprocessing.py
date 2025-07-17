# preprocessing.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DenoisingAutoencoder(nn.Module):
    # Existing code remains unchanged
    def __init__(self, in_channels):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConditionalGAN(nn.Module):
    # Existing code remains unchanged
    def __init__(self, in_channels, num_classes):
        super(ConditionalGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(100 + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_channels * 7 * 7 * 3),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(in_channels * 7 * 7 * 3 + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z, labels):
        x = self.generator(torch.cat([z, labels], dim=-1))
        x = x.view(-1, in_channels, 7, 7, 3)
        return x

def load_data(dataset_name):
    # Existing code remains unchanged
    if dataset_name == 'IndianPines':
        data = scipy.io.loadmat('datasets/IndianPines.mat')['indian_pines_corrected']
        labels = scipy.io.loadmat('datasets/IndianPines.mat')['indian_pines_gt']
        spectral_range = (400, 2500)  # nm
        num_bands = 200
    elif dataset_name == 'PaviaU':
        data = scipy.io.loadmat('datasets/PaviaU.mat')['paviaU']
        labels = scipy.io.loadmat('datasets/PaviaU.mat')['paviaU_gt']
        spectral_range = (430, 860)
        num_bands = 103
    elif dataset_name == 'Houston2013':
        data = scipy.io.loadmat('datasets/Houston2013.mat')['houston2013']
        labels = scipy.io.loadmat('datasets/Houston2013.mat')['houston2013_gt']
        spectral_range = (380, 1050)
        num_bands = 144
    elif dataset_name == 'Botswana':
        data = scipy.io.loadmat('datasets/Botswana.mat')['botswana']
        labels = scipy.io.loadmat('datasets/Botswana.mat')['botswana_gt']
        spectral_range = (400, 2500)
        num_bands = 145
    else:
        raise ValueError("Unknown dataset")
    return data, labels, spectral_range, num_bands

def align_spectral_bands(data, spectral_range, target_bands=103):
    # Align spectral bands to a common range (400--860 nm) and target number of bands
    original_bands = data.shape[-1]
    if original_bands > target_bands:
        # Simple band selection: take evenly spaced bands
        indices = np.linspace(0, original_bands-1, target_bands, dtype=int)
        data = data[..., indices]
    elif original_bands < target_bands:
        # Interpolate to increase bands (simple linear interpolation)
        x = np.linspace(0, 1, original_bands)
        x_new = np.linspace(0, 1, target_bands)
        data = np.apply_along_axis(lambda m: np.interp(x_new, x, m), -1, data)
    return data

def preprocess_data(data, labels, config, is_target=False):
    # Align spectral bands to a common range (400--860 nm, 103 bands for PaviaU compatibility)
    data, labels, spectral_range, num_bands = load_data(config['dataset'] if not is_target else config['target_dataset'])
    data = align_spectral_bands(data, spectral_range, target_bands=103)
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    data = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
    
    if not is_target:
        autoencoder = DenoisingAutoencoder(data.shape[1]).to(config['device'])
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        for _ in range(10):
            noisy_data = data + torch.randn_like(data) * 0.01
            output = autoencoder(noisy_data)
            loss = nn.MSELoss()(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data = autoencoder(data).detach()
        
        minority_classes = np.where(np.bincount(labels.flatten()) < 100)[0]
        gan = ConditionalGAN(data.shape[1], config['num_classes']).to(config['device'])
        for c in minority_classes:
            z = torch.randn(100, 100)
            one_hot = torch.zeros(100, config['num_classes'])
            one_hot[:, c] = 1
            synthetic_data = gan(z, one_hot)
    
    dataset = TensorDataset(data, torch.tensor(labels, dtype=torch.long))
    if is_target:
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
        return loader
    else:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        return train_loader, test_loader
