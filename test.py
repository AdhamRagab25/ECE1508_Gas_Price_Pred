"""
    Implements the test function to verify the model's performance on the test set.    
"""
from lstm import *

def test(model, loss_function, test_loader):

    # we make sure we are not tracking gradient
    # gradient is used in training, we do not need it for test
    with torch.no_grad():
        risk = 0

        # here we are only evaluating the model
        model.eval()

        # loop over test mini-batches
        for i, data in enumerate(test_loader):
            # Get the inputs and labels
            inputs, label = data
            
            # Reshape the inputs
            inputs = inputs.view(-1, model.seq_len, model.input_dim)

            # Reset initial states
            model.h_state = model.initialize_hidden_state(inputs.size(0))
            
            # Forward pass
            output = model(inputs)
            
            # Compute the loss
            loss = loss_function(output, label)
            
            # Update the test risk
            risk += loss.item()
            
        # average test risk and accuracy over the whole test dataset
        test_risk = risk / len(test_loader)

    return test_risk