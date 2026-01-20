"""
Synthetic Data Generator for MSME Demand Forecasting
Generates realistic sales data with trends, seasonality, and noise
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class MSMESalesDataGenerator:
    """
    Generate synthetic MSME sales data with realistic patterns
    
    Features:
    - Trend (growth/decline)
    - Weekly seasonality
    - Monthly seasonality
    - Special events (festivals, promotions)
    - Realistic noise
    """
    
    def __init__(self, n_products=10, n_days=365, seed=42):
        self.n_products = n_products
        self.n_days = n_days
        self.seed = seed
        np.random.seed(seed)
        
        # Product categories for MSMEs
        self.product_categories = [
            'Rice (25kg Bag)', 'Sunflower Oil (1L)', 'Sugar (kg)',
            'Toor Dal (kg)', 'Tea Powder (250g)', 'Wheat Flour (kg)',
            'Milk (1L)', 'Bread', 'Eggs (Dozen)', 'Biscuits'
        ]
        
    def generate_trend(self, n_days, trend_type='growth'):
        """Generate trend component"""
        if trend_type == 'growth':
            return np.linspace(0, 0.3, n_days)
        elif trend_type == 'decline':
            return np.linspace(0, -0.2, n_days)
        else:  # stable
            return np.zeros(n_days)
    
    def generate_weekly_seasonality(self, n_days):
        """Generate weekly seasonality (weekends have higher sales)"""
        weekly_pattern = np.array([0.9, 0.95, 1.0, 1.0, 1.05, 1.15, 1.2])  # Mon-Sun
        return np.tile(weekly_pattern, n_days // 7 + 1)[:n_days]
    
    def generate_monthly_seasonality(self, n_days):
        """Generate monthly seasonality"""
        t = np.arange(n_days)
        return 0.1 * np.sin(2 * np.pi * t / 30) + 1.0
    
    def generate_special_events(self, n_days):
        """Generate special events (festivals, promotions)"""
        events = np.zeros(n_days)
        
        # Add random promotional events (10-15 events per year)
        n_events = np.random.randint(10, 16)
        event_days = np.random.choice(n_days, n_events, replace=False)
        
        for day in event_days:
            # Event lasts 3-5 days with peak on middle day
            duration = np.random.randint(3, 6)
            start = max(0, day - duration // 2)
            end = min(n_days, day + duration // 2 + 1)
            
            for i, d in enumerate(range(start, end)):
                # Gaussian-like peak
                distance_from_peak = abs(d - day)
                boost = 1.5 * np.exp(-distance_from_peak ** 2 / 2)
                events[d] = max(events[d], boost)
        
        return events + 1.0
    
    def generate_noise(self, n_days, noise_level=0.1):
        """Generate realistic noise"""
        return 1.0 + np.random.normal(0, noise_level, n_days)
    
    def generate_product_sales(self, product_id, base_sales=100):
        """Generate sales for a single product"""
        # Determine trend type
        trend_types = ['growth', 'stable', 'decline']
        trend_type = np.random.choice(trend_types, p=[0.5, 0.3, 0.2])
        
        # Generate components
        trend = self.generate_trend(self.n_days, trend_type)
        weekly_season = self.generate_weekly_seasonality(self.n_days)
        monthly_season = self.generate_monthly_seasonality(self.n_days)
        events = self.generate_special_events(self.n_days)
        noise = self.generate_noise(self.n_days)
        
        # Combine all components
        sales = base_sales * (1 + trend) * weekly_season * monthly_season * events * noise
        
        # Ensure non-negative and round to integers
        sales = np.maximum(sales, 0).astype(int)
        
        return sales
    
    def generate_dataset(self):
        """
        Generate complete dataset for all products
        
        Returns:
            DataFrame with columns: date, product_id, product_name, sales
        """
        data = []
        start_date = datetime.now() - timedelta(days=self.n_days)
        
        for product_id in range(self.n_products):
            product_name = self.product_categories[product_id % len(self.product_categories)]
            
            # Base sales varies by product
            base_sales = np.random.randint(50, 200)
            
            # Generate sales
            sales = self.generate_product_sales(product_id, base_sales)
            
            # Create records
            for day in range(self.n_days):
                date = start_date + timedelta(days=day)
                data.append({
                    'date': date,
                    'product_id': product_id,
                    'product_name': product_name,
                    'sales': sales[day],
                    'day_of_week': date.weekday(),
                    'day_of_month': date.day,
                    'month': date.month
                })
        
        df = pd.DataFrame(data)
        return df
    
    def create_sequences(self, df, seq_length=60, forecast_horizon_7=7, forecast_horizon_30=30):
        """
        Create input-output sequences for training
        
        Args:
            df: DataFrame with sales data
            seq_length: Length of input sequence
            forecast_horizon_7: 7-day forecast horizon
            forecast_horizon_30: 30-day forecast horizon
            
        Returns:
            X: Input sequences (n_samples, seq_length, n_features)
            y_7day: 7-day targets (n_samples, 7)
            y_30day: 30-day targets (n_samples, 30)
        """
        X, y_7day, y_30day = [], [], []
        
        for product_id in df['product_id'].unique():
            product_data = df[df['product_id'] == product_id].sort_values('date')
            
            # Extract features
            sales = product_data['sales'].values
            day_of_week = product_data['day_of_week'].values
            day_of_month = product_data['day_of_month'].values
            month = product_data['month'].values
            
            # Create sequences
            for i in range(len(sales) - seq_length - forecast_horizon_30):
                # Input sequence
                seq_sales = sales[i:i+seq_length]
                seq_dow = day_of_week[i:i+seq_length]
                seq_dom = day_of_month[i:i+seq_length]
                seq_month = month[i:i+seq_length]
                
                # Normalize sales (min-max scaling per sequence)
                sales_min, sales_max = seq_sales.min(), seq_sales.max()
                if sales_max > sales_min:
                    seq_sales_norm = (seq_sales - sales_min) / (sales_max - sales_min)
                else:
                    seq_sales_norm = seq_sales / (sales_max + 1e-8)
                
                # Create feature matrix
                features = np.stack([
                    seq_sales_norm,
                    seq_dow / 6.0,  # Normalize to [0, 1]
                    seq_dom / 31.0,
                    seq_month / 12.0,
                    np.arange(seq_length) / seq_length,  # Time index
                    np.sin(2 * np.pi * seq_dow / 7),  # Cyclic encoding
                    np.cos(2 * np.pi * seq_dow / 7),
                    np.sin(2 * np.pi * seq_month / 12),
                    np.cos(2 * np.pi * seq_month / 12),
                    np.ones(seq_length) * product_id / self.n_products  # Product ID
                ], axis=1)
                
                # Target sequences
                target_7day = sales[i+seq_length:i+seq_length+forecast_horizon_7]
                target_30day = sales[i+seq_length:i+seq_length+forecast_horizon_30]
                
                # Normalize targets using same scale as input
                if sales_max > sales_min:
                    target_7day_norm = (target_7day - sales_min) / (sales_max - sales_min)
                    target_30day_norm = (target_30day - sales_min) / (sales_max - sales_min)
                else:
                    target_7day_norm = target_7day / (sales_max + 1e-8)
                    target_30day_norm = target_30day / (sales_max + 1e-8)
                
                X.append(features)
                y_7day.append(target_7day_norm)
                y_30day.append(target_30day_norm)
        
        return np.array(X), np.array(y_7day), np.array(y_30day)


def generate_training_data(n_products=10, n_days=365, seq_length=60):
    """
    Convenience function to generate training data
    
    Returns:
        X_train, y_7day_train, y_30day_train: Training data
        X_val, y_7day_val, y_30day_val: Validation data
        df: Full DataFrame for reference
    """
    generator = MSMESalesDataGenerator(n_products=n_products, n_days=n_days)
    df = generator.generate_dataset()
    
    X, y_7day, y_30day = generator.create_sequences(df, seq_length=seq_length)
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(X))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_7day_train, y_7day_val = y_7day[:split_idx], y_7day[split_idx:]
    y_30day_train, y_30day_val = y_30day[:split_idx], y_30day[split_idx:]
    
    print(f"Generated {len(X)} sequences")
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    print(f"Input shape: {X_train.shape}")
    print(f"7-day target shape: {y_7day_train.shape}")
    print(f"30-day target shape: {y_30day_train.shape}")
    
    return X_train, y_7day_train, y_30day_train, X_val, y_7day_val, y_30day_val, df
