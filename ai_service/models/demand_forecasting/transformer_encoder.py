"""
Transformer Encoder for Hybrid Demand Forecasting Model
Captures long-range dependencies and attention patterns in sales data
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for time series data
    
    Args:
        input_size: Number of input features
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size=10,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            output: Transformer output (batch_size, seq_len, d_model)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x, mask=mask)
        
        # Layer normalization
        transformer_out = self.layer_norm(transformer_out)
        
        return transformer_out
    
    def get_output_dim(self):
        """Returns the output dimension"""
        return self.d_model
    
    def get_attention_weights(self, x):
        """
        Extract attention weights for visualization
        This is a simplified version - full implementation would require
        modifying the transformer layers to return attention weights
        """
        # For now, return None - can be implemented later for visualization
        return None
