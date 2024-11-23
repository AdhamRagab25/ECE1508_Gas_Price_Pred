"""
This module contains the implementation of an LSTM model for predicting gas prices.

"""

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, batch_size, seq_len, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        # There are 20 features in the input data
        self.input_dim = 20
        # The output is a single value describing the gas price
        self.output_dim = 1
        
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True)
        
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
    def initialize_hidden_state(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))
        
    def forward(self, x):
        hidden_state = self.initialize_hidden_state()
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        output = self.output_layer(lstm_out[:, -1, :])
        return output