# final_model_shap.py

# Import necessary libraries and modules
from lstm import LSTM
from train_val import train
from dataset_loader_val import load_and_split_data
from test_val import test
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import shap

# ================== Configuration ==================

# Final Training Parameters
NUM_EPOCHS_FINAL = 30      # Number of epochs for final training 
PATIENCE_FINAL = 2         # Early Stopping patience

# Data Parameters
TRAIN_WEEKS = 327          # Number of weeks for training
VAL_WEEKS = 52             # Number of weeks for validation

# Hyperparameters (Set them directly here)
best_seq_length = 1        # Example sequence length
best_hdim =130             # Example hidden dimension
best_batch_size = 128        # Example batch size

# Path to your CSV file
csv_file_path = 'Normalized_Weekly_Gasoline_Data__2016-2024_Reduced.csv'  # Updated path
# csv_file_path = 'Normalized_Weekly_Gasoline_Data__2016-2024_.csv'     # Full dataset path

# ================== Define Feature Names Based on CSV Path ==================

# Define feature names based on the exact CSV file path
if csv_file_path == 'Normalized_Weekly_Gasoline_Data__2016-2024_Reduced.csv':
    # Feature names for reduced dataset
    feature_names = [
        "Price",
        "Taxes",
        "Marketing Margin",
        "Refining Margin",
        "Avg Max Temp (°C)",
        "Avg Min Temp (°C)",
        "Avg Precip (mm)",
        "Total CPI, (seasonally adjusted)",
        "W.BCPI",
        "W.ENER",
        "Fuel Tax Rate"
    ]
elif csv_file_path == 'Normalized_Weekly_Gasoline_Data__2016-2024_.csv':
    # Feature names for full dataset
    feature_names = [
        "Price",
        "Taxes",
        "Marketing Margin",
        "Refining Margin",
        "Avg Max Temp (°C)",
        "Avg Min Temp (°C)",
        "Avg Mean Temp (°C)",
        "Avg Precip (mm)",
        "Total CPI",
        "Total CPI, (seasonally adjusted)",
        "Total CPI, Percentage Change over 1 year ago (unadjusted)",
        "CPI_TRIM",
        "CPI_MEDIAN",
        "CPI_COMMON",
        "CPIX, Percentage Change over 1 year ago (unadjusted)",
        "CPI-XFET, Percentage Change over 1 year ago (unadjusted)",
        "CPIW",
        "W.BCPI",
        "W.ENER",
        "Fuel Tax Rate"
    ]


# ================== Load and Split Data ==================

print(f"\n=== Loading and Splitting Data with SEQ_LENGTH={best_seq_length} ===")
train_loader, val_loader, test_loader = load_and_split_data(
    csv_file_path=csv_file_path,
    seq_length=best_seq_length,
    train_weeks=TRAIN_WEEKS,
    val_weeks=VAL_WEEKS,
    batch_size=best_batch_size
)

# ================== Initialize the Model ==================

print("\n=== Initializing the Final LSTM Model ===")
final_model = LSTM(seq_len=best_seq_length, hidden_dim=best_hdim)

# Define loss function
final_loss_function = nn.HuberLoss()

# ================== Train the Final Model with Early Stopping ==================

print(f"\n=== Training Final Model ===")
final_training_losses, final_validation_losses = train(
    model=final_model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=final_loss_function,
    num_epochs=NUM_EPOCHS_FINAL,
    patience=PATIENCE_FINAL
)

# ================== Evaluate the Model on Test Set ==================

print("\n=== Evaluating Final Model on Test Set ===")
avg_test_loss, test_predictions, test_targets = test(
    model=final_model,
    loss_function=final_loss_function,
    test_loader=test_loader
)
print(f"Final Test Loss: {avg_test_loss:.4f}")

# ================== Denormalize Predictions and Targets ==================

# Given normalization parameters
mean_price = 131.12529
std_price = 25.35632458

# Denormalize predictions and targets
denorm_predictions = test_predictions * std_price + mean_price
denorm_targets = test_targets * std_price + mean_price

# Compute differences

differences = 100*(denorm_predictions - denorm_targets)/denorm_targets

# ================== Compute RMSE ==================

def compute_rmse(predictions, targets):
    """
    Computes the Root Mean Squared Error between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Actual target values.

    Returns:
        float: RMSE value.
    """
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

# Calculate RMSE
rmse = compute_rmse(denorm_predictions, denorm_targets)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# ================== Plot Differences ==================

"""
Plots the denormalized predictions, denormalized targets, and their differences.
"""

plt.figure(figsize=(12, 6))
plt.plot(denorm_targets.numpy(), label='Actual Price', color='blue')
plt.plot(denorm_predictions.numpy(), label='Predicted Price', color='orange')
plt.title('Denormalized Actual vs Predicted Prices and Their Differences')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the differences

plt.figure(figsize=(12, 6))
plt.plot(differences.numpy(), label='Difference (Predicted - Actual)', color='green')
plt.title('Denormalized Actual vs Predicted Prices and Their Differences')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# ================== Plot Training and Validation Losses ==================

def plot_losses(training_losses, validation_losses):
    """
    Plots training and validation losses over epochs.
    """
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, 'b-', label='Training Loss')
    plt.plot(epochs, validation_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the losses
plot_losses(final_training_losses, final_validation_losses)

# ================== Apply SHAP for Explainability ==================

print("\n=== Applying SHAP for Model Explainability ===")

# Set the model to evaluation mode
final_model.eval()

# ------------------ Step 1: Prepare Background for SHAP ------------------
print("\n--- Preparing Background for SHAP ---")
# Creating a zero background with the same shape as one input batch
background = torch.zeros(1, best_seq_length, final_model.input_dim)
print(f"Zero background shape: {background.shape}")

# ------------------ Step 2: Initialize SHAP Explainer ------------------
print("\n--- Initializing SHAP GradientExplainer ---")
explainer = shap.GradientExplainer(final_model, background)

# ------------------ Step 3: Collect All Test Instances ------------------
print("\n--- Collecting All Test Instances ---")
instances = []
with torch.no_grad():
    for inputs, _ in test_loader:
        instances.append(inputs)
instances = torch.cat(instances, dim=0)  # Concatenate all test batches into one tensor
print(f"Instances shape: {instances.shape}")  # Shape: (total_test_samples, seq_length, num_features)

# ------------------ Step 4: Compute SHAP Values ------------------
print("\n--- Computing SHAP Values ---")
shap_values = explainer.shap_values(instances)


# shap_values is a list where each element corresponds to a model output
if isinstance(shap_values, list):
    shap_values = shap_values[0]  # For regression, focus on the single output

# Shape: (total_test_samples, seq_length, num_features)
print(f"SHAP values shape: {shap_values.shape}")

# ------------------ Step 5: Aggregate SHAP Values ------------------
print("\n--- Aggregating SHAP Values for Feature and Time-Step Importance ---")

# Convert shap_values from torch tensor to numpy if necessary
if isinstance(shap_values, torch.Tensor):
    shap_values = shap_values.numpy()

# Aggregate feature importance across time steps for each feature
feature_importance = shap_values.mean(axis=(0, 1))  # Shape: (num_features,)
print(f"Feature Importance Shape: {feature_importance.shape}")

# Aggregate time-step importance across features for each time step
time_step_importance = shap_values.mean(axis=(0, 2))  # Shape: (seq_length,)
print(f"Time-Step Importance Shape: {time_step_importance.shape}")

# ------------------ Step 6: Visualize Feature and Time-Step Importance ------------------

def plot_feature_importance(feature_importance, feature_names):
    """Plots feature importance as a bar chart."""
    feature_importance = feature_importance.flatten()  # Convert to 1D
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importance, align='center', color='skyblue')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel("Mean SHAP Value")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_time_step_importance(time_step_importance, seq_length):
    """Plots time step importance as a line chart."""
    time_step_importance = time_step_importance.flatten()  # Convert to 1D
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, seq_length + 1), time_step_importance, marker='o', linestyle='-', color='green')
    plt.xticks(range(1, seq_length + 1))
    plt.xlabel("Time Step")
    plt.ylabel("Mean SHAP Value")
    plt.title("Time-Step Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot feature importance
plot_feature_importance(feature_importance, feature_names)

# Plot time-step importance
plot_time_step_importance(time_step_importance, best_seq_length)