from lstm import *
import torch.optim as optim
import matplotlib.pyplot as plt

"""
This module contains the training for the LSTM model for predicting gas prices.

"""

def train(model, train_loader, loss_function, num_epochs):
    optimizer = optim.Adam(model.parameters())
    training_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        
        # Training loop
        for i, data in enumerate(train_loader):
            # Reset initial states
            model.h_state = model.initialize_hidden_state()
            
            # Get the inputs and labels
            inputs, label = data
            
            # Reshape the inputs
            inputs = inputs.view(-1, model.seq_len, model.input_dim)
            
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
        training_losses.append(train_loss / i)
    
    return training_losses

def plot_losses(training_losses):
    plt.plot(training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
            