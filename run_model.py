from lstm import *
from train import *
from dataset_loader import *

# Num epochs
NUM_EPOCHS = 200

# Run the model
model = LSTM(seq_len=SEQ_LENGTH, hidden_dim=20)

for idx, loader in enumerate(train_loaders):
    
    training_losses, test_losses = train(model, loader, test_loader, LSTM_LOSS_FUNCTION, NUM_EPOCHS)
    
    # Plot the losses
    plot_losses(training_losses, test_losses)
    
    

