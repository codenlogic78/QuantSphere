"""
Trading Environment

This module contains the trading environment implementation
for the Quantsphere agent.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from .utils import get_stock_data


class TradingEnvironment:
    """
    Trading Environment for the Quantsphere Agent
    
    Provides a standardized interface for trading operations
    and market simulation.
    """
    
    def __init__(self, data: List[float], initial_balance: float = 10000.0):
        """
        Initialize the trading environment
        
        Args:
            data: Price data for trading
            initial_balance: Starting balance
        """
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.inventory = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial state
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.inventory = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        return self.get_state()
    
    def get_state(self, window_size: int = 10) -> np.ndarray:
        """
        Get current state representation
        
        Args:
            window_size: Size of the state window
            
        Returns:
            State representation
        """
        if self.current_step < window_size:
            # Pad with initial price if not enough data
            state_data = [self.data[0]] * (window_size - self.current_step) + \
                        self.data[:self.current_step + 1]
        else:
            state_data = self.data[self.current_step - window_size + 1:self.current_step + 1]
        
        # Calculate price differences and normalize
        state = []
        for i in range(len(state_data) - 1):
            diff = state_data[i + 1] - state_data[i]
            state.append(self._sigmoid(diff))
        
        return np.array([state])
    
    def _sigmoid(self, x: float) -> float:
        """Apply sigmoid function for normalization"""
        try:
            if x < 0:
                return 1 - 1 / (1 + np.exp(x))
            return 1 / (1 + np.exp(-x))
        except:
            return 0.5
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0=Hold, 1=Buy, 2=Sell)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            return self.get_state(), 0, True, self.get_info()
        
        current_price = self.data[self.current_step]
        reward = 0
        info = {'action': action, 'price': current_price}
        
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.balance -= current_price
                self.inventory.append(current_price)
                info['action_type'] = 'BUY'
                info['shares'] = 1
        elif action == 2:  # Sell
            if len(self.inventory) > 0:
                bought_price = self.inventory.pop(0)
                profit = current_price - bought_price
                self.balance += current_price
                reward = profit
                self.total_trades += 1
                
                if profit > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                info['action_type'] = 'SELL'
                info['profit'] = profit
                info['shares'] = 1
        else:  # Hold
            info['action_type'] = 'HOLD'
        
        self.current_step += 1
        next_state = self.get_state()
        done = self.current_step >= len(self.data) - 1
        
        return next_state, reward, done, info
    
    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value
        
        Returns:
            Total portfolio value
        """
        current_price = self.data[self.current_step] if self.current_step < len(self.data) else self.data[-1]
        return self.balance + len(self.inventory) * current_price
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get environment information
        
        Returns:
            Dictionary with environment info
        """
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'inventory_size': len(self.inventory),
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        }
    
    def render(self, mode: str = 'human') -> str:
        """
        Render the current state
        
        Args:
            mode: Rendering mode
            
        Returns:
            String representation of current state
        """
        info = self.get_info()
        current_price = self.data[self.current_step] if self.current_step < len(self.data) else self.data[-1]
        
        return f"""
        Step: {info['step']}/{len(self.data)-1}
        Price: ${current_price:.2f}
        Balance: ${info['balance']:.2f}
        Inventory: {info['inventory_size']} shares
        Portfolio Value: ${info['portfolio_value']:.2f}
        Total Return: {info['total_return']:.2f}%
        Trades: {info['total_trades']} (Win Rate: {info['win_rate']:.1f}%)
        """
    
    @classmethod
    def from_csv(cls, filepath: str, initial_balance: float = 10000.0) -> 'TradingEnvironment':
        """
        Create environment from CSV file
        
        Args:
            filepath: Path to CSV file
            initial_balance: Starting balance
            
        Returns:
            TradingEnvironment instance
        """
        data = get_stock_data(filepath)
        return cls(data, initial_balance)
    
    def get_trading_history(self) -> List[Dict[str, Any]]:
        """
        Get complete trading history
        
        Returns:
            List of trading actions
        """
        # This would need to be implemented to track all actions
        # For now, return empty list
        return []
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        if self.total_trades < 2:
            return 0.0
        
        # This is a simplified calculation
        # In practice, you'd want to calculate returns over time
        info = self.get_info()
        excess_return = info['total_return'] / 100 - risk_free_rate
        return excess_return  # Simplified - would need proper volatility calculation
    
    def get_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown
        
        Returns:
            Maximum drawdown percentage
        """
        # This would need to track portfolio values over time
        # For now, return 0
        return 0.0
