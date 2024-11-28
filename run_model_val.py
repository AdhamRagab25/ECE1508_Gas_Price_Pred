# run_model_val.py

from lstm import LSTM
from train_val import train
from dataset_loader_val import load_and_split_data
import torch.nn as nn
import numpy as np
import itertools

# ================== Configuration ==================

# Hyperparameter Grid
SEQ_LENGTHS = [2,4,8,16]   # List of SEQ_LENGTHs to tune
HDIM_LIST = [10, 15, 20]               # List of Hidden Dimensions to tune
BATCH_SIZES = [4,8,16,32,64]             # List of Batch Sizes to tune
N_RUNS = 50                             # Number of runs per hyperparameter combination
TRAIN_WEEKS = 327                     # Number of weeks for training
VAL_WEEKS = 52                         # Number of weeks for validation
NUM_EPOCHS = 25                        # Number of training epochs
PATIENCE = 3                           # Early Stopping patience

# Path to your CSV file
csv_file_path = 'Normalized_Weekly_Gasoline_Data__2016-2024_Reduced.csv'  # Updated path
#csv_file_path = 'Normalized_Weekly_Gasoline_Data__2016-2024_.csv'

# Initialize a dictionary to store results
# The keys will be tuples of (SEQ_LENGTH, HDim, BATCH_SIZE)
results = {}

# Generate all possible combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(SEQ_LENGTHS, HDIM_LIST, BATCH_SIZES))

print(f"Total hyperparameter combinations to evaluate: {len(hyperparameter_combinations)}")

# Iterate over each hyperparameter combination
for combo in hyperparameter_combinations:
    seq_length, hdim, batch_size = combo
    print(f"\n=== Tuning SEQ_LENGTH = {seq_length}, HDim = {hdim}, BATCH_SIZE = {batch_size} ===")
    
    val_losses = []
    
    # Perform N_RUNS for each hyperparameter combination
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run} for SEQ_LENGTH = {seq_length}, HDim = {hdim}, BATCH_SIZE = {batch_size} ---")
        
        # Load and split the data with the current hyperparameters
        train_loader, val_loader, _ = load_and_split_data(
            csv_file_path,
            seq_length=seq_length,
            train_weeks=TRAIN_WEEKS,
            val_weeks=VAL_WEEKS,
            batch_size=batch_size
        )
        
        # Initialize the model
        model = LSTM(seq_len=seq_length, hidden_dim=hdim)
        
        # Define loss function
        loss_function = nn.HuberLoss()
        
        # Train the model with Early Stopping
        training_losses, validation_losses_run = train(
            model,
            train_loader,
            val_loader,
            loss_function,
            NUM_EPOCHS,
            patience=PATIENCE
        )
        
        # Record the final validation loss
        val_losses.append(validation_losses_run[-1])
    
    # Calculate average validation loss for this hyperparameter combination
    avg_val_loss = np.mean(val_losses)
    
    # Store the averaged results
    results[combo] = {
        'average_validation_loss': avg_val_loss
    }
    
    print(f"\nSEQ_LENGTH = {seq_length}, HDim = {hdim}, BATCH_SIZE = {batch_size}: "
          f"Average Validation Loss = {avg_val_loss:.4f}")

# Determine the best hyperparameter combination based on average validation loss
best_hyperparams = min(results, key=lambda x: results[x]['average_validation_loss'])
best_val_loss = results[best_hyperparams]['average_validation_loss']

# Export the best hyperparameters as module-level variables
best_seq_length, best_hdim, best_batch_size = best_hyperparams

# Print the hyperparameter tuning results
print("\n=== Hyperparameter Tuning Results ===")
for hyperparams, losses in results.items():
    seq_length, hdim, batch_size = hyperparams
    print(f"SEQ_LENGTH={seq_length}, HDim={hdim}, BATCH_SIZE={batch_size}: "
          f"Average Validation Loss={losses['average_validation_loss']:.4f}")

print(f"\nBest Hyperparameters based on Average Validation Loss:")
print(f"SEQ_LENGTH = {best_seq_length}, HDim = {best_hdim}, BATCH_SIZE = {best_batch_size}")
print(f"Average Validation Loss = {best_val_loss:.4f}")