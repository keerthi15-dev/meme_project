"""
Hybrid LSTM-Transformer Model for Demand Forecasting
Combines LSTM and Transformer encoders with attention-based fusion
"""

import torch
import torch.nn as nn
from .lstm_encoder import LSTMEncoder
from .transformer_encoder import TransformerEncoder


class HybridForecaster(nn.Module):
    """
    Hybrid LSTM-Transformer architecture for multi-horizon demand forecasting
    
    Novel Contributions:
    1. Dual-encoder architecture (LSTM + Transformer in parallel)
    2. Attention-based fusion layer
    3. Multi-horizon predictions (7-day and 30-day)
    4. Uncertainty quantification via dropout
    
    Args:
        input_size: Number of input features
        lstm_hidden: LSTM hidden size
        transformer_dim: Transformer model dimension
        lstm_layers: Number of LSTM layers
        transformer_layers: Number of Transformer layers
        nhead: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size=10,
        lstm_hidden=128,
        transformer_dim=256,
        lstm_layers=2,
        transformer_layers=2,
        nhead=4,
        dropout=0.2
    ):
        super(HybridForecaster, self).__init__()
        
        self.input_size = input_size
        self.lstm_hidden = lstm_hidden
        self.transformer_dim = transformer_dim
        
        # LSTM Encoder Branch
        self.lstm_encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # Transformer Encoder Branch
        self.transformer_encoder = TransformerEncoder(
            input_size=input_size,
            d_model=transformer_dim,
            nhead=nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim * 2,
            dropout=dropout * 0.5  # Lower dropout for transformer
        )
        
        # Fusion dimension (concatenate LSTM + Transformer outputs)
        fusion_dim = lstm_hidden + transformer_dim
        
        # Attention-based fusion layer
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim)
        )
        
        # Prediction heads for different horizons
        self.fc_7day = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7)
        )
        
        self.fc_30day = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 30)
        )
        
        # Confidence estimation (for uncertainty quantification)
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # mean and std for confidence intervals
        )
        
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            pred_7day: 7-day forecast
            pred_30day: 30-day forecast
            confidence: Confidence scores (optional)
            attention_weights: Attention weights (optional)
        """
        batch_size = x.size(0)
        
        # LSTM branch
        lstm_out, _ = self.lstm_encoder(x)
        
        # Transformer branch
        transformer_out = self.transformer_encoder(x)
        
        # Concatenate outputs from both branches
        # Take the last time step for both
        lstm_last = lstm_out[:, -1, :]  # (batch_size, lstm_hidden)
        transformer_last = transformer_out[:, -1, :]  # (batch_size, transformer_dim)
        
        # Concatenate
        fused = torch.cat([lstm_last, transformer_last], dim=1)  # (batch_size, fusion_dim)
        
        # Apply attention-based fusion
        # Reshape for attention: (batch_size, 1, fusion_dim)
        fused_expanded = fused.unsqueeze(1)
        
        # Self-attention on fused features
        fused_attended, attention_weights = self.fusion_attention(
            fused_expanded, fused_expanded, fused_expanded
        )
        
        # Squeeze back
        fused_attended = fused_attended.squeeze(1)
        
        # Apply fusion projection
        fused_final = self.fusion_proj(fused_attended)
        
        # Generate predictions for different horizons
        pred_7day = self.fc_7day(fused_final)
        pred_30day = self.fc_30day(fused_final)
        
        # Generate confidence scores
        confidence = self.confidence_head(fused_final)
        
        if return_attention:
            return pred_7day, pred_30day, confidence, attention_weights
        else:
            return pred_7day, pred_30day, confidence
    
    def predict_with_uncertainty(self, x, n_samples=10):
        """
        Generate predictions with uncertainty estimates using Monte Carlo Dropout
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            mean_7day, std_7day: Mean and std for 7-day forecast
            mean_30day, std_30day: Mean and std for 30-day forecast
        """
        self.train()  # Enable dropout
        
        predictions_7day = []
        predictions_30day = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred_7, pred_30, _ = self.forward(x)
                predictions_7day.append(pred_7)
                predictions_30day.append(pred_30)
        
        # Stack predictions
        predictions_7day = torch.stack(predictions_7day)
        predictions_30day = torch.stack(predictions_30day)
        
        # Calculate mean and std
        mean_7day = predictions_7day.mean(dim=0)
        std_7day = predictions_7day.std(dim=0)
        
        mean_30day = predictions_30day.mean(dim=0)
        std_30day = predictions_30day.std(dim=0)
        
        self.eval()  # Disable dropout
        
        return mean_7day, std_7day, mean_30day, std_30day
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
