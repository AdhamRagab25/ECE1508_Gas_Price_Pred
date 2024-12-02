# train_val.py

from dataset_loader_val import *
from lstm import *
import torch.optim as optim
import matplotlib.pyplot as plt


def train(model, train_loader, val_loader, loss_function, num_epochs, patience=2):
    """
    Trains the LSTM model with Early Stopping.

    Args:
        model (nn.Module): The LSTM model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_function (nn.Module): Loss function.
        num_epochs (int): Number of epochs to train.
        patience (int, optional): Number of epochs to wait for improvement. Defaults to 5.
        test_loader (DataLoader, optional): DataLoader for test data. Defaults to None.

    Returns:
        Tuple[List[float], List[float]]: Lists of training and validation losses.
    """
    training_losses = []
    validation_losses = []

    optimizer = optim.Adam(model.parameters())

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    final_epoch = 0  # Track the last epoch

    for epoch in range(1, num_epochs + 1):
        final_epoch = epoch  # Update the last epoch
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            labels = labels.unsqueeze(1)  # Ensure labels have shape (batch_size, 1)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                labels = labels.unsqueeze(1)  # Ensure labels have shape (batch_size, 1)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            early_stop = True
            break

    # Print the final epoch's losses
    print(f"Epoch [{final_epoch}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    if not early_stop:
        print("Training completed without early stopping.")

    return training_losses, validation_losses

def plot_losses(training_losses, validation_losses):
    """
    Plots training and validation losses over epochs.

    Args:
        training_losses (List[float]): List of training losses.
        validation_losses (List[float]): List of validation losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss', color='blue')
    plt.plot(validation_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()