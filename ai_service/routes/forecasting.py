"""
FastAPI Route for Demand Forecasting
Serves predictions from trained Hybrid LSTM-Transformer model
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
import json

from models.demand_forecasting.hybrid_model import HybridForecaster

router = APIRouter()

# Global model instance
_model = None
_device = None


class ForecastRequest(BaseModel):
    """Request model for forecasting"""
    product_id: int
    product_name: str
    historical_sales: List[float]  # Last 60 days of sales
    

class ForecastResponse(BaseModel):
    """Response model for forecasting"""
    product_id: int
    product_name: str
    forecast_7day: List[float]
    forecast_30day: List[float]
    confidence_7day_lower: List[float]
    confidence_7day_upper: List[float]
    confidence_30day_lower: List[float]
    confidence_30day_upper: List[float]
    model_info: dict


def load_model():
    """Load trained model from checkpoint"""
    global _model, _device
    
    if _model is not None:
        return _model, _device
    
    # Determine device
    if torch.backends.mps.is_available():
        _device = 'mps'
    elif torch.cuda.is_available():
        _device = 'cuda'
    else:
        _device = 'cpu'
    
    # Load model
    checkpoint_path = Path('models/checkpoints/best_model.pth')
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    # Initialize model
    _model = HybridForecaster(
        input_size=10,
        lstm_hidden=128,
        transformer_dim=256,
        lstm_layers=2,
        transformer_layers=2,
        nhead=4,
        dropout=0.2
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.to(_device)
    _model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path} on device: {_device}")
    
    return _model, _device


def preprocess_input(historical_sales: List[float], product_id: int) -> torch.Tensor:
    """
    Preprocess historical sales data into model input format
    
    Args:
        historical_sales: List of sales values (should be 60 days)
        product_id: Product identifier
        
    Returns:
        Preprocessed tensor ready for model input
    """
    # Ensure we have 60 days
    if len(historical_sales) < 60:
        # Pad with zeros if less than 60
        historical_sales = [0] * (60 - len(historical_sales)) + historical_sales
    elif len(historical_sales) > 60:
        # Take last 60 days
        historical_sales = historical_sales[-60:]
    
    sales = np.array(historical_sales)
    
    # Normalize sales
    sales_min, sales_max = sales.min(), sales.max()
    if sales_max > sales_min:
        sales_norm = (sales - sales_min) / (sales_max - sales_min)
    else:
        sales_norm = sales / (sales_max + 1e-8)
    
    # Create time-based features
    seq_length = 60
    day_of_week = np.arange(seq_length) % 7  # Simplified
    day_of_month = (np.arange(seq_length) % 30) + 1
    month = ((np.arange(seq_length) // 30) % 12) + 1
    
    # Create feature matrix (10 features)
    features = np.stack([
        sales_norm,
        day_of_week / 6.0,
        day_of_month / 31.0,
        month / 12.0,
        np.arange(seq_length) / seq_length,
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.ones(seq_length) * (product_id % 10) / 10
    ], axis=1)
    
    # Convert to tensor and add batch dimension
    tensor = torch.FloatTensor(features).unsqueeze(0)
    
    # Store normalization params for denormalization
    return tensor, sales_min, sales_max


def denormalize_predictions(predictions: torch.Tensor, sales_min: float, sales_max: float) -> List[float]:
    """Denormalize predictions back to original scale"""
    predictions_np = predictions.cpu().numpy()
    
    if sales_max > sales_min:
        denorm = predictions_np * (sales_max - sales_min) + sales_min
    else:
        denorm = predictions_np * (sales_max + 1e-8)
    
    # Ensure non-negative and round
    denorm = np.maximum(denorm, 0)
    
    return denorm.tolist()


@router.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate demand forecast for a product
    
    Args:
        request: ForecastRequest with product info and historical sales
        
    Returns:
        ForecastResponse with 7-day and 30-day predictions + confidence intervals
    """
    try:
        # Load model
        model, device = load_model()
        
        # Preprocess input
        input_tensor, sales_min, sales_max = preprocess_input(
            request.historical_sales,
            request.product_id
        )
        input_tensor = input_tensor.to(device)
        
        # Generate predictions with uncertainty
        with torch.no_grad():
            mean_7day, std_7day, mean_30day, std_30day = model.predict_with_uncertainty(
                input_tensor,
                n_samples=10
            )
        
        # Denormalize predictions
        forecast_7day = denormalize_predictions(mean_7day[0], sales_min, sales_max)
        forecast_30day = denormalize_predictions(mean_30day[0], sales_min, sales_max)
        
        # Calculate confidence intervals (±1 std dev)
        std_7day_denorm = denormalize_predictions(std_7day[0], 0, sales_max - sales_min)
        std_30day_denorm = denormalize_predictions(std_30day[0], 0, sales_max - sales_min)
        
        confidence_7day_lower = [max(0, f - s) for f, s in zip(forecast_7day, std_7day_denorm)]
        confidence_7day_upper = [f + s for f, s in zip(forecast_7day, std_7day_denorm)]
        
        confidence_30day_lower = [max(0, f - s) for f, s in zip(forecast_30day, std_30day_denorm)]
        confidence_30day_upper = [f + s for f, s in zip(forecast_30day, std_30day_denorm)]
        
        # Model info
        model_info = {
            "model_type": "Hybrid LSTM-Transformer",
            "parameters": model.count_parameters(),
            "device": device,
            "input_length": 60,
            "forecast_horizons": [7, 30]
        }
        
        return ForecastResponse(
            product_id=request.product_id,
            product_name=request.product_name,
            forecast_7day=forecast_7day,
            forecast_30day=forecast_30day,
            confidence_7day_lower=confidence_7day_lower,
            confidence_7day_upper=confidence_7day_upper,
            confidence_30day_lower=confidence_30day_lower,
            confidence_30day_upper=confidence_30day_upper,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        model, device = load_model()
        
        # Load training metadata if available
        metadata_path = Path('models/checkpoints/training_metadata.json')
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            "model_type": "Hybrid LSTM-Transformer",
            "architecture": {
                "lstm_hidden": 128,
                "transformer_dim": 256,
                "lstm_layers": 2,
                "transformer_layers": 2,
                "attention_heads": 4
            },
            "parameters": model.count_parameters(),
            "device": device,
            "training_metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model, device = load_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": device
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }
