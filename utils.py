import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from scipy.spatial.distance import cosine

def compute_metrics(preds, labels):
    preds = preds.argmax(dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    OA = accuracy_score(labels, preds) * 100
    AA = np.mean([accuracy_score(labels[labels == c], preds[labels == c]) * 100 
                  for c in np.unique(labels) if np.sum(labels == c) > 0])
    kappa = cohen_kappa_score(labels, preds)
    sam = np.mean([cosine(preds[i], labels[i]) for i in range(len(preds))]) * 180 / np.pi
    f1 = f1_score(labels, preds, average='weighted') * 100
    return {'OA': OA, 'AA': AA, 'Kappa': kappa, 'SAM': sam, 'F1': f1}