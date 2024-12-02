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
        self.X, self.y = self.create_sequences(data, seq_length)

    @staticmethod
    def create_sequences(data, seq_length=4):
        """
        Creates input-output pairs for the dataset.

        Args:
            data (np.ndarray): Array of shape (num_weeks, num_features).
            seq_length (int): Number of past weeks to use for each input sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of input sequences and labels.
        """
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])  # Assuming 'Price' is the first feature
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

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
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def load_and_split_data(csv_file_path, seq_length=4, train_weeks=312, val_weeks=52, batch_size=64):
    """
    Loads the CSV data, splits it into training, validation, and test sets, and creates DataLoaders.

    Args:
        csv_file_path (str): Path to the CSV file.
        seq_length (int): Sequence length for input data.
        train_weeks (int): Number of weeks for training.
        val_weeks (int): Number of weeks for validation.
        batch_size (int): Number of samples per batch for training and validation.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Training, validation, and test DataLoaders.
    """
    # Load the dataset using pandas
    data_df = pd.read_csv(csv_file_path)
    
    # Drop the 'Week Ending' column and convert to NumPy array
    data_np = data_df.drop(columns=['Week Ending']).values  # Shape: (431, 20)

    # Verify dataset shape
    print(f"Dataset shape (weeks, features): {data_np.shape}")  # Expected: (431, 20)

    # Split the data into training + validation and testing sets
    train_val_weeks = train_weeks + val_weeks
    train_val_data = data_np[:train_val_weeks]      # First (312 + 52) = 364 weeks
    test_data = data_np[train_val_weeks:]           # Remaining weeks

    # Further split training + validation into training and validation sets
    train_data = train_val_data[:train_weeks]       # First 312 weeks
    val_data = train_val_data[train_weeks:]         # Next 52 weeks

    print(f"Training Data Shape: {train_data.shape}")  # Expected: (312, 20)
    print(f"Validation Data Shape: {val_data.shape}")  # Expected: (52, 20)
    print(f"Test Data Shape: {test_data.shape}")        # Depends on total weeks, initially 431 - 364 = 67 weeks

    # Define sequence length
    SEQ_LENGTH = seq_length  # 4 weeks as input

    # Create Dataset instances
    train_dataset = GasolineDataset(train_data, seq_length=SEQ_LENGTH)
    val_dataset = GasolineDataset(val_data, seq_length=SEQ_LENGTH)
    test_dataset = GasolineDataset(test_data, seq_length=SEQ_LENGTH)

    print(f"Sequence Length:{seq_length}")
    print(f"Number of Training Samples: {len(train_dataset)}")   # 312 - 4 = 308
    print(f"Number of Validation Samples: {len(val_dataset)}")   # 52 - 4 = 48
    print(f"Number of Test Samples: {len(test_dataset)}")         # 67 - 4 = 63

    # Create DataLoaders with configurable batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    
    # Ensure test_loader uses only one batch
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    print(f"Training Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")
    print(f"Test Loader: {len(test_loader)} batch")

    return train_loader, val_loader, test_loader


    