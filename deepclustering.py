import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from collections import Counter
import os

# Uncomment if you want to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Load the LSWMD dataset
data = pd.read_pickle("LSWMD.pkl")
print("Original shape:", data.shape)

# Step 2: Define function to extract clean label
def extract_label(label):
    if label is None:
        return None
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if not isinstance(label, list) or len(label) == 0:
        return None
    if label == [[]] or label[0][0] == 'none':
        return None
    return label[0][0]

# Step 3: Apply function to clean failureType column
data['failureType'] = data['failureType'].apply(extract_label)

# Keep all data (labeled and unlabeled)

# Preprocess wafer maps to fixed size
resized_size = 64

def preprocess_wafer_map(wm):
    wm = np.array(wm)
    h, w = wm.shape
    max_dim = max(h, w)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left
    padded = np.pad(wm, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    zoom_factors = (resized_size / max_dim, resized_size / max_dim)
    resized = zoom(padded, zoom_factors, order=0)  # Nearest neighbor for discrete values
    return resized.astype(np.float32) / 2.0  # Normalize to [0, 1]

# Apply preprocessing
print("Preprocessing wafer maps...")
wafer_maps = []
for i in range(len(data)):
    wm = preprocess_wafer_map(data['waferMap'][i])
    wafer_maps.append(wm)
wafer_maps = np.array(wafer_maps)  # (N, 64, 64)
print("Wafer maps shape:", wafer_maps.shape)

# Add channel dimension for CNN (N, 1, 64, 64)
wafer_maps = wafer_maps[:, np.newaxis, :, :]

# Convert to torch tensor
tensors = torch.from_numpy(wafer_maps)

# Create dataset and dataloader for autoencoder training
dataset = TensorDataset(tensors, tensors)  # Input = target for reconstruction
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Define Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128)  # Embedding dimension: 128
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        emb = self.encoder(x)
        out = self.decoder(emb)
        return out

    def get_embedding(self, x):
        return self.encoder(x)

# Initialize model, optimizer, loss
model = ConvAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the autoencoder
epochs = 10  # Adjust as needed (more epochs for better features)
print("Training autoencoder...")
for epoch in range(epochs):
    model.train()
    loss_total = 0
    for batch in dataloader:
        img = batch[0].to(device)
        out = model(img)
        loss = criterion(out, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_total / len(dataloader):.4f}")

# Extract embeddings in batches to save memory
print("Extracting embeddings...")
embeddings = []
with torch.no_grad():
    model.eval()
    for i in range(0, len(tensors), 256):
        batch = tensors[i:i+256].to(device)
        emb = model.get_embedding(batch).cpu().numpy()
        embeddings.append(emb)
embeddings = np.vstack(embeddings)
print("Embeddings shape:", embeddings.shape)

# Perform clustering with KMeans
k = 8  # Number of known failure types
print(f"Performing KMeans clustering with k={k}...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# Map clusters to true labels using labeled data
labeled_mask = data['failureType'].notnull()
labeled_idx = np.where(labeled_mask)[0]
unlabeled_idx = np.where(~labeled_mask)[0]

true_labels = data.loc[labeled_mask, 'failureType']
unique_labels = true_labels.unique()
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_true = true_labels.map(label_to_int).values

cluster_labeled = cluster_labels[labeled_idx]

# Build contingency matrix (clusters x labels)
conf = np.zeros((k, len(unique_labels)))
for c in range(k):
    for l in range(len(unique_labels)):
        conf[c, l] = np.sum((cluster_labeled == c) & (int_true == l))

# Use Hungarian algorithm to optimally assign clusters to labels
row_ind, col_ind = linear_sum_assignment(conf, maximize=True)

# Create cluster to label mapping
cluster_to_label_id = {row: col for row, col in zip(row_ind, col_ind)}
cluster_to_label = {c: unique_labels[cluster_to_label_id.get(c, -1)] if c in cluster_to_label_id else 'Unknown' for c in range(k)}

# Assign labels based on cluster
assigned_labels = [cluster_to_label[cl] for cl in cluster_labels]

# Add to dataframe
data['predicted_failureType'] = assigned_labels

# For labeled data, overwrite predicted with true if you want, but here we keep for comparison
# Evaluate clustering accuracy on labeled data (using optimal assignment)
acc = np.sum(conf[row_ind, col_ind]) / len(int_true)
print(f"Clustering accuracy on labeled data: {acc:.4f}")

# Save the updated dataset
data.to_pickle("labeled_LSWMD.pkl")
print("Done! Updated dataset saved to 'labeled_LSWMD.pkl'. Unlabeled instances are labeled in 'predicted_failureType' column.")