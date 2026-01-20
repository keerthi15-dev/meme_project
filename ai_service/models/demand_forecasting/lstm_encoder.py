"""
LSTM Encoder for Hybrid Demand Forecasting Model
Captures sequential patterns and temporal dependencies in sales data
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    LSTM-based encoder for time series data
    
    Args:
        input_size: Number of input features
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size=10,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            output: LSTM output (batch_size, seq_len, hidden_size)
            hidden: Final hidden state
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        return lstm_out, hidden
    
    def get_output_dim(self):
        """Returns the output dimension"""
        return self.hidden_size
