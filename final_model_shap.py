# final_model_shap.py

# Import necessary libraries and modules
from run_model_val import best_seq_length, best_hdim, best_batch_size, csv_file_path
from lstm import LSTM
from train_val import train
from dataset_loader_val import load_and_split_data
from test_val import test
import torch.nn as nn
import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

# ================== Configuration ==================

# Final Training Parameters
NUM_EPOCHS_FINAL = 30      # Number of epochs for final training
PATIENCE_FINAL = 3         # Early Stopping patience

# Data Parameters
TRAIN_WEEKS = 327          # Number of weeks for training
VAL_WEEKS = 52             # Number of weeks for validation

# SHAP Parameters
BACKGROUND_SIZE = 100      # Number of background samples for SHAP
INSTANCES_TO_EXPLAIN = 52  # Number of test instances to explain

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model.to(device)

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
final_test_loss = test(
    model=final_model,
    loss_function=final_loss_function,
    test_loader=test_loader
)
print(f"Final Test Loss: {final_test_loss:.4f}")

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

# ------------------ Step 1: Prepare Background Data for SHAP ------------------
print(f"\n--- Preparing Background Data (First {BACKGROUND_SIZE} Test Samples) ---")
background = []
with torch.no_grad():
    for i, (inputs, _) in enumerate(test_loader):
        if i >= BACKGROUND_SIZE:
            break
        background.append(inputs)
if not background:
    print("No background samples available. Exiting SHAP explanation.")
    exit()
background = torch.cat(background, dim=0).to(device)  # Shape: (BACKGROUND_SIZE, seq_length, num_features)

print(f"Type of background: {type(background)}")
print(f"Shape of background: {background.shape}")

# ------------------ Step 2: Initialize SHAP Explainer ------------------
print("\n--- Initializing SHAP GradientExplainer ---")
try:
    explainer = shap.GradientExplainer(final_model, background)
except Exception as e:
    print(f"Error initializing SHAP GradientExplainer: {e}")
    exit()

# ------------------ Step 3: Select Instances to Explain ------------------
print(f"\n--- Selecting {INSTANCES_TO_EXPLAIN} Test Instances to Explain ---")
instances = []
with torch.no_grad():
    for i, (inputs, _) in enumerate(test_loader):
        if i >= INSTANCES_TO_EXPLAIN:
            break
        instances.append(inputs)
if not instances:
    print("No instances available for explanation. Exiting SHAP explanation.")
    exit()
instances = torch.cat(instances, dim=0).to(device)  # Shape: (INSTANCES_TO_EXPLAIN, seq_length, num_features)

print(f"Type of instances: {type(instances)}")
print(f"Shape of instances: {instances.shape}")

# ------------------ Step 4: Compute SHAP Values ------------------
print("\n--- Computing SHAP Values ---")
try:
    shap_values = explainer.shap_values(instances)
except Exception as e:
    print(f"Error computing SHAP values: {e}")
    exit()

# shap_values is a list where each element corresponds to a model output
# Since it's regression with a single output, shap_values[0] corresponds to the output
if isinstance(shap_values, list):
    shap_values = shap_values[0]
# Shape: (INSTANCES_TO_EXPLAIN, seq_length, num_features)

print(f"Shape of shap_values: {shap_values.shape}")

# ------------------ Step 5: Aggregate SHAP Values ------------------
print("\n--- Aggregating SHAP Values Across All Test Samples ---")

# Compute the average SHAP values across all samples
average_shap_values = np.mean(shap_values, axis=0)  # Shape: (seq_length, num_features)

print(f"Shape of average_shap_values: {average_shap_values.shape}")

# ------------------ Step 6: Plot Average SHAP Heatmap ------------------
def plot_average_shap_heatmap(avg_shap, feature_names, seq_length):
    """
    Plots the average SHAP heatmap across all test samples.
    """
    plt.figure(figsize=(16, 8))
    plt.imshow(avg_shap, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Average SHAP value')
    plt.xlabel('Features')
    plt.ylabel('Time Steps')
    plt.title('Average SHAP Values Heatmap Across Test Dataset')
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    plt.yticks(ticks=np.arange(seq_length), labels=[f"Step {k+1}" for k in range(seq_length)])
    plt.tight_layout()
    plt.show()

# Hardcoded feature names as provided
feature_names = [
    "Price",
    "Taxes",
    "Marketing Margin",
    "Refining Margin",
    "Avg Max Temp (°C)",
    "Avg Min Temp (°C)",
   # "Avg Mean Temp (°C)",
    "Avg Precip (mm)",
   # "Total CPI",
    "Total CPI, (seasonally adjusted)",
   # "Total CPI, Percentage Change over 1 year ago (unadjusted)",
   # "CPI_TRIM",
    #"CPI_MEDIAN",
   # "CPI_COMMON",
   # "CPIX, Percentage Change over 1 year ago (unadjusted)",
   # "CPI-XFET, Percentage Change over 1 year ago (unadjusted)",
   # "CPIW",
    "W.BCPI",
    "W.ENER",
    "Fuel Tax Rate"
]



# Verify that the number of feature names matches input_dim
assert len(feature_names) == average_shap_values.shape[1], (
    f"Number of feature names ({len(feature_names)}) does not match input_dim ({average_shap_values.shape[1]})"
)

# Plot the average SHAP heatmap
plot_average_shap_heatmap(average_shap_values, feature_names, best_seq_length)