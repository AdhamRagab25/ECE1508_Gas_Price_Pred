import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Define the custom Dataset class
class GasolineDataset(Dataset):
    def __init__(self, data, seq_length=4):
        """
        Initializes the dataset by creating input sequences and corresponding labels.

        Args:
            data (np.ndarray): Array of shape (num_weeks, num_features).
            seq_length (int): Number of past weeks to use for each input sequence.
        """
        self.seq_length = seq_length
        self.data = data
        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        """
        Creates input-output pairs for the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of input sequences and labels.
        """
        X = []
        y = []
        for i in range(len(self.data) - self.seq_length):
            # Extract the sequence of `seq_length` weeks
            seq = self.data[i:i + self.seq_length]
            X.append(seq)
            # The label is the price at the next week (week 5)
            label = self.data[i + self.seq_length, 0]  # Assuming 'Price' is the first feature
            y.append(label)
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves the input-output pair at the specified index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequence and corresponding label.
        """
        # Convert to torch tensors
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

# Path to your CSV file
csv_file_path = 'Normalized_Weekly_Gasoline_Data__2016-2024_.csv'  # Replace with your actual file path

# Load the dataset using pandas
data_df = pd.read_csv(csv_file_path)

# Display the first few rows to verify
print("First few rows of the dataset:")
print(data_df.head())

# Drop the 'Week Ending' column
data_np = data_df.drop(columns=['Week Ending']).values  # Shape: (431, 20)


print(f"\nDataset shape (weeks, features): {data_np.shape}")  # Expected: (431, 20)

# Split the data into training and testing sets
train_weeks = 3 * 104  # 312 weeks
test_weeks = data_np.shape[0] - train_weeks  # 119 weeks

# Ensure that the dataset has enough weeks
assert train_weeks + test_weeks == data_np.shape[0], "Mismatch in total weeks after splitting."

train_data = data_np[:train_weeks]      # First 312 weeks
test_data = data_np[train_weeks:]       # Remaining 119 weeks

print(f"\nTraining Data Shape: {train_data.shape}")  # Expected: (312, 20)
print(f"Test Data Shape: {test_data.shape}")        # Expected: (119, 20) 

# Split the training data into 3 periods of 104 weeks each
num_periods = 1
weeks_per_period = 312

train_periods = [
    train_data[i * weeks_per_period : (i + 1) * weeks_per_period]
    for i in range(num_periods)
]

# Verify each period's shape
for idx, period in enumerate(train_periods):
    print(f"Training Period {idx+1} Shape: {period.shape}")  # Each should be (104, 20)

# Define sequence length
SEQ_LENGTH = 4  # 4 weeks as input

# Create Dataset instances for each training period
train_datasets = [
    GasolineDataset(period, seq_length=SEQ_LENGTH) for period in train_periods
]

# Verify the number of sequences per period
for idx, dataset in enumerate(train_datasets):
    print(f"Number of Training Samples in Period {idx+1}: {len(dataset)}")  # Each should be 100

# Create Dataset instance for test set
test_dataset = GasolineDataset(test_data, seq_length=SEQ_LENGTH)

print(f"Number of Test Samples: {len(test_dataset)}")  # Expected: 115 (119 -4)

# Create DataLoaders for each training period with batch_size=100
train_loaders = [
    DataLoader(
        dataset, 
        batch_size=308, 
        shuffle=False
    ) 
    for dataset in train_datasets
]

# Create DataLoader for test set with a larger batch size
test_loader = DataLoader(
    test_dataset, 
    batch_size=100,  # Adjust as needed
    shuffle=False,
    drop_last=True
)

# Verify the number of batches
for idx, loader in enumerate(train_loaders):
    print(f"\nTraining Loader {idx+1} Batches: {len(loader)}")  # Each should have 1 batch

print(f"\nTest Loader Batches: {len(test_loader)}")        # Should have 1 batch

# Verify the shape of a batch from each training loader
for idx, loader in enumerate(train_loaders):
    for X_batch, y_batch in loader:
        print(f"\nTraining Loader {idx+1} - Input Batch Shape: {X_batch.shape}")   # (100, 4, 20)
        print(f"Training Loader {idx+1} - Label Batch Shape: {y_batch.shape}")   # (100,)
        break  # Only one batch per loader

# Verify the shape of the test batch
for X_batch, y_batch in test_loader:
    print(f"\nTest Loader - Input Batch Shape: {X_batch.shape}")        # (115, 4, 20)
    print(f"Test Loader - Label Batch Shape: {y_batch.shape}")        # (115,)
    break  # Only one batch
