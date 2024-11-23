from lstm import *
import torch.optim as optim
import matplotlib.pyplot as plt
from test import *

"""
This module contains the training for the LSTM model for predicting gas prices.

"""

# define loss function as Huber loss
LSTM_LOSS_FUNCTION = nn.HuberLoss()

def train(model, train_loader, test_loader, loss_function, num_epochs):
    optimizer = optim.Adam(model.parameters())
    training_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        
        # Training loop
        for i, data in enumerate(train_loader):
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the training loss
            train_loss += loss.item()
        
        # Store the average training loss for this epoch
        training_losses.append(train_loss / len(train_loader))
        test_losses.append(test(model, loss_function, test_loader))

    
    return training_losses, test_losses

def plot_losses(training_losses, test_losses):
    plt.plot(training_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()
            