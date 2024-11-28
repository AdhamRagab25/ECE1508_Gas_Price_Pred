from dataset_loader_val import *
from lstm import *

def test(model, loss_function, test_loader):
    """
    Evaluates the model on the test set.

    Args:
        model (nn.Module): The trained LSTM model.
        loss_function (nn.Module): Loss function.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        float: Average test loss.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss