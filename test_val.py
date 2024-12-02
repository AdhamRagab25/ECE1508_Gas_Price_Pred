from dataset_loader_val import *
from lstm import *

def test(model, loss_function, test_loader):
    """
    Evaluates the model on the test set and returns the average loss along with predictions and targets.

    Args:
        model (nn.Module): The trained LSTM model.
        loss_function (nn.Module): Loss function.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        tuple:
            float: Average test loss.
            torch.Tensor: Concatenated predictions.
            torch.Tensor: Concatenated actual target values.
    """
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            labels = labels.unsqueeze(1)  # Ensure labels have the same shape as outputs
            loss = loss_function(outputs, labels)
            test_loss += loss.item() * inputs.size(0)  # Accumulate loss weighted by batch size
            
            all_predictions.append(outputs)
            all_targets.append(labels)
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return avg_test_loss, all_predictions, all_targets