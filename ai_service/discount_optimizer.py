"""
RL-Based Discount Optimization Agent
Uses Q-Learning to determine optimal discount percentages
Based on 2024 research on AI-driven dynamic pricing
"""

import numpy as np
import pickle
import os
from collections import deque
from typing import Dict, Tuple
from datetime import datetime


class DiscountOptimizationAgent:
    """
    Reinforcement Learning agent for discount optimization
    """
    
    def __init__(self):
        self.q_table = {}
        self.actions = [0, 5, 10, 15, 20, 25, 30, 40, 50]  # Discount percentages
        self.lr = 0.15  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
        # Track campaign performance
        self.campaign_history = deque(maxlen=100)
        
        # Model persistence
        self.model_file = "discount_optimization_model.pkl"
        self.load_model()
    
    def get_state(self, product_info: Dict) -> Tuple:
        """
        Create state representation for Q-learning
        
        State = (category, inventory_level, season, demand_level)
        """
        category = product_info.get('category', 'general')
        stock = product_info.get('stock', 50)
        
        # Discretize inventory
        if stock < 20:
            inventory_level = 'low'
        elif stock < 50:
            inventory_level = 'medium'
        else:
            inventory_level = 'high'
        
        # Get season
        season = self._get_season()
        
        # Get demand level (simplified - can be enhanced with real data)
        demand_level = product_info.get('demand', 'medium')
        
        return (category, inventory_level, season, demand_level)
    
    def suggest_discount(self, product_info: Dict) -> Dict:
        """
        Suggest optimal discount using Q-learning
        
        Returns:
            {
                'discount_percentage': 20,
                'original_price': 100,
                'discounted_price': 80,
                'savings': 20,
                'explanation': '...'
            }
        """
        state = self.get_state(product_info)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            discount = np.random.choice(self.actions)  # Explore
        else:
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in self.actions}
            discount = max(self.q_table[state], key=self.q_table[state].get)  # Exploit
        
        # Calculate prices
        base_price = product_info.get('base_price', 100)
        final_price = base_price * (1 - discount / 100)
        
        return {
            'discount_percentage': discount,
            'original_price': base_price,
            'discounted_price': round(final_price, 2),
            'savings': round(base_price - final_price, 2),
            'state': state,
            'explanation': self._explain_discount(discount, state)
        }
    
    def update_from_campaign(
        self,
        state: Tuple,
        discount: int,
        sales: int,
        revenue: float
    ) -> float:
        """
        Online learning from campaign results
        
        Args:
            state: State when discount was applied
            discount: Discount percentage used
            sales: Units sold
            revenue: Total revenue generated
        
        Returns:
            reward: Calculated reward value
        """
        # Calculate reward (sales volume + revenue)
        reward = sales * 10 + revenue * 0.1
        
        # Initialize Q-table for state if needed
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        
        # Q-learning update
        current_q = self.q_table[state].get(discount, 0.0)
        max_next_q = max(self.q_table[state].values()) if self.q_table[state] else 0.0
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][discount] = new_q
        
        # Record campaign
        self.campaign_history.append({
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'discount': discount,
            'sales': sales,
            'revenue': revenue,
            'reward': reward
        })
        
        # Save updated model
        self.save_model()
        
        return reward
    
    def _explain_discount(self, discount: int, state: Tuple) -> str:
        """
        Explain why this discount was chosen
        """
        category, inventory, season, demand = state
        
        reasons = []
        
        if discount >= 30:
            reasons.append("High discount to clear inventory")
        elif discount >= 15:
            reasons.append("Moderate discount to boost sales")
        elif discount == 0:
            reasons.append("No discount - optimal pricing")
        else:
            reasons.append("Small discount for promotion")
        
        if inventory == 'high':
            reasons.append("High stock levels")
        elif inventory == 'low':
            reasons.append("Low stock - maintain margins")
        
        if demand == 'low':
            reasons.append("Low demand - attract customers")
        elif demand == 'high':
            reasons.append("High demand - maximize profit")
        
        if season in ['festival', 'holiday']:
            reasons.append(f"{season.title()} season")
        
        return " | ".join(reasons) if reasons else "Standard pricing strategy"
    
    def _get_season(self) -> str:
        """
        Determine current season
        """
        month = datetime.now().month
        
        # Simplified season detection
        if month in [10, 11]:  # Diwali season
            return 'festival'
        elif month in [12, 1]:  # New Year
            return 'holiday'
        elif month in [6, 7, 8]:  # Monsoon
            return 'monsoon'
        else:
            return 'regular'
    
    def save_model(self):
        """Save Q-table and campaign history"""
        try:
            data = {
                'q_table': self.q_table,
                'campaign_history': list(self.campaign_history)
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving discount model: {e}")
    
    def load_model(self):
        """Load Q-table and campaign history"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.q_table = data.get('q_table', {})
                    history = data.get('campaign_history', [])
                    self.campaign_history = deque(history, maxlen=100)
                print(f"✓ Loaded discount model with {len(self.q_table)} states")
        except Exception as e:
            print(f"Note: Starting with fresh discount model ({e})")
            self.q_table = {}
            self.campaign_history = deque(maxlen=100)


# Singleton instance
discount_optimizer = DiscountOptimizationAgent()
