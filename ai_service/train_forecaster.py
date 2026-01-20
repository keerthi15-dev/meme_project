"""
Training Script for Hybrid LSTM-Transformer Demand Forecasting Model
Optimized for Mac GPU (MPS - Metal Performance Shaders)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from models.demand_forecasting.hybrid_model import HybridForecaster
from models.utils.data_generator import generate_training_data


class ForecastTrainer:
    """Trainer for Hybrid Forecasting Model"""
    
    def __init__(
        self,
        model,
        device='mps',
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        early_stopping_patience=10
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mape': [],
            'val_mape': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        epsilon = 1e-8
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mape_7 = 0
        total_mape_30 = 0
        n_batches = 0
        
        for batch_X, batch_y7, batch_y30 in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y7 = batch_y7.to(self.device)
            batch_y30 = batch_y30.to(self.device)
            
            # Forward pass
            pred_7day, pred_30day, _ = self.model(batch_X)
            
            # Calculate losses
            loss_7day = self.criterion(pred_7day, batch_y7)
            loss_30day = self.criterion(pred_30day, batch_y30)
            
            # Combined loss (weighted)
            loss = 0.6 * loss_7day + 0.4 * loss_30day
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_mape_7 += self.calculate_mape(batch_y7, pred_7day).item()
            total_mape_30 += self.calculate_mape(batch_y30, pred_30day).item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mape = (total_mape_7 + total_mape_30) / (2 * n_batches)
        
        return avg_loss, avg_mape
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mape_7 = 0
        total_mape_30 = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y7, batch_y30 in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y7 = batch_y7.to(self.device)
                batch_y30 = batch_y30.to(self.device)
                
                # Forward pass
                pred_7day, pred_30day, _ = self.model(batch_X)
                
                # Calculate losses
                loss_7day = self.criterion(pred_7day, batch_y7)
                loss_30day = self.criterion(pred_30day, batch_y30)
                loss = 0.6 * loss_7day + 0.4 * loss_30day
                
                # Track metrics
                total_loss += loss.item()
                total_mape_7 += self.calculate_mape(batch_y7, pred_7day).item()
                total_mape_30 += self.calculate_mape(batch_y30, pred_30day).item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mape = (total_mape_7 + total_mape_30) / (2 * n_batches)
        
        return avg_loss, avg_mape
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Training Hybrid LSTM-Transformer Model")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_mape = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_mape = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mape'].append(train_mape)
            self.history['val_mape'].append(val_mape)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train MAPE: {train_mape:.2f}% | Val MAPE: {val_mape:.2f}%")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pth')
                print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save model checkpoint"""
        checkpoint_dir = Path('models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'input_size': self.model.input_size,
                'lstm_hidden': self.model.lstm_hidden,
                'transformer_dim': self.model.transformer_dim
            }
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAPE plot
        ax2.plot(self.history['train_mape'], label='Train MAPE')
        ax2.plot(self.history['val_mape'], label='Val MAPE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAPE (%)')
        ax2.set_title('Training and Validation MAPE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")


def main():
    """Main training function"""
    # Set device (MPS for Mac GPU)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✓ Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✓ Using CUDA GPU")
    else:
        device = 'cpu'
        print("⚠ Using CPU (slower)")
    
    # Generate training data
    print("\nGenerating synthetic MSME sales data...")
    X_train, y_7day_train, y_30day_train, X_val, y_7day_val, y_30day_val, df = generate_training_data(
        n_products=10,
        n_days=365,
        seq_length=60
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_7day_train_tensor = torch.FloatTensor(y_7day_train)
    y_30day_train_tensor = torch.FloatTensor(y_30day_train)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_7day_val_tensor = torch.FloatTensor(y_7day_val)
    y_30day_val_tensor = torch.FloatTensor(y_30day_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_7day_train_tensor, y_30day_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_7day_val_tensor, y_30day_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("\nInitializing Hybrid LSTM-Transformer model...")
    model = HybridForecaster(
        input_size=10,
        lstm_hidden=128,
        transformer_dim=256,
        lstm_layers=2,
        transformer_layers=2,
        nhead=4,
        dropout=0.2
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Initialize trainer
    trainer = ForecastTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        early_stopping_patience=10
    )
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'device': device,
        'n_parameters': model.count_parameters(),
        'best_val_loss': trainer.best_val_loss,
        'final_val_mape': history['val_mape'][-1],
        'n_epochs_trained': len(history['train_loss'])
    }
    
    with open('models/checkpoints/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Training complete! Model saved to models/checkpoints/")


if __name__ == '__main__':
    main()
