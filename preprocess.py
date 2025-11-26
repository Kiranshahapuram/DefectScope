import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
# Step 4: Show label distribution before filtering
print("\nLabel distribution (including 'none'):")
print(data['failureType'].value_counts(dropna=False))
# Step 5: Filter out unlabeled data (None values)
data = data[data['failureType'].notnull()].reset_index(drop=True)
print("\nShape after removing unlabeled rows:", data.shape)
# Step 6: Show final class counts
print("\nLabel distribution (only labeled samples):")
print(data['failureType'].value_counts())
print(data)
