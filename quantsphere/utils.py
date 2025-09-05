"""
Utility Functions

This module contains utility functions for the Quantsphere trading agent.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def format_currency(price: float) -> str:
    """
    Format price as currency string
    
    Args:
        price: Price value
        
    Returns:
        Formatted currency string
    """
    return f"${abs(price):.2f}"


def format_position(price: float) -> str:
    """
    Format position as currency string with sign
    
    Args:
        price: Position value
        
    Returns:
        Formatted position string
    """
    sign = "-$" if price < 0 else "+$"
    return f"{sign}{abs(price):.2f}"


def get_stock_data(filepath: str) -> List[float]:
    """
    Load stock data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of adjusted closing prices
    """
    try:
        df = pd.read_csv(filepath)
        if 'Adj Close' in df.columns:
            return df['Adj Close'].tolist()
        elif 'Close' in df.columns:
            return df['Close'].tolist()
        else:
            raise ValueError("No 'Adj Close' or 'Close' column found in data")
    except Exception as e:
        logging.error(f"Error loading stock data: {e}")
        raise


def show_train_result(result: Tuple, val_position: float, initial_offset: float):
    """
    Display training results
    
    Args:
        result: Training result tuple (episode, total_episodes, profit, loss)
        val_position: Validation position
        initial_offset: Initial offset value
    """
    episode, total_episodes, profit, loss = result
    
    if val_position == initial_offset or val_position == 0.0:
        logging.info(
            f'Episode {episode}/{total_episodes} - '
            f'Train Position: {format_position(profit)}  '
            f'Val Position: USELESS  '
            f'Train Loss: {loss:.4f}'
        )
    else:
        logging.info(
            f'Episode {episode}/{total_episodes} - '
            f'Train Position: {format_position(profit)}  '
            f'Val Position: {format_position(val_position)}  '
            f'Train Loss: {loss:.4f}'
        )


def show_eval_result(model_name: str, profit: float, initial_offset: float):
    """
    Display evaluation results
    
    Args:
        model_name: Name of the model
        profit: Profit achieved
        initial_offset: Initial offset value
    """
    if profit == initial_offset or profit == 0.0:
        logging.info(f'{model_name}: USELESS\n')
    else:
        logging.info(f'{model_name}: {format_position(profit)}\n')


def switch_k_backend_device():
    """
    Switch Keras backend from GPU to CPU if required
    
    This is useful for training on CPU when using tensorflow-gpu
    """
    try:
        import keras.backend as K
        if K.backend() == "tensorflow":
            logging.debug("Switching to TensorFlow for CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    except ImportError:
        logging.warning("Keras not available, skipping backend configuration")


def create_trading_visualization(
    data: List[float], 
    history: List[Tuple], 
    title: str = "Trading Performance"
) -> plt.Figure:
    """
    Create a visualization of trading performance
    
    Args:
        data: Price data
        history: Trading history (price, action) tuples
        title: Chart title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price data
    ax1.plot(data, label='Price', color='blue', alpha=0.7)
    
    # Mark buy and sell points
    buy_prices = [h[0] for h in history if h[1] == 'BUY']
    sell_prices = [h[0] for h in history if h[1] == 'SELL']
    buy_indices = [i for i, h in enumerate(history) if h[1] == 'BUY']
    sell_indices = [i for i, h in enumerate(history) if h[1] == 'SELL']
    
    if buy_prices:
        ax1.scatter(buy_indices, buy_prices, color='green', marker='^', 
                   s=100, label='Buy', alpha=0.8)
    if sell_prices:
        ax1.scatter(sell_indices, sell_prices, color='red', marker='v', 
                   s=100, label='Sell', alpha=0.8)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot portfolio value
    portfolio_values = []
    portfolio_value = 10000  # Starting value
    inventory = []
    
    for i, (price, action) in enumerate(history):
        if action == 'BUY':
            inventory.append(price)
        elif action == 'SELL' and inventory:
            bought_price = inventory.pop(0)
            portfolio_value += (price - bought_price)
        
        portfolio_values.append(portfolio_value)
    
    ax2.plot(portfolio_values, label='Portfolio Value', color='purple')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_metrics(history: List[Tuple], initial_value: float = 10000) -> dict:
    """
    Calculate trading performance metrics
    
    Args:
        history: Trading history (price, action) tuples
        initial_value: Initial portfolio value
        
    Returns:
        Dictionary of performance metrics
    """
    portfolio_values = []
    portfolio_value = initial_value
    inventory = []
    trades = []
    
    for price, action in history:
        if action == 'BUY':
            inventory.append(price)
        elif action == 'SELL' and inventory:
            bought_price = inventory.pop(0)
            profit = price - bought_price
            portfolio_value += profit
            trades.append(profit)
        
        portfolio_values.append(portfolio_value)
    
    if not portfolio_values:
        return {
            'total_return': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_drawdown': 0
        }
    
    total_return = (portfolio_values[-1] - initial_value) / initial_value * 100
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t > 0])
    losing_trades = len([t for t in trades if t < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_profit = np.mean(trades) if trades else 0
    
    # Calculate maximum drawdown
    peak = initial_value
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return {
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'max_drawdown': max_drawdown
    }


def save_results(results: dict, filepath: str):
    """
    Save results to CSV file
    
    Args:
        results: Results dictionary
        filepath: Output file path
    """
    df = pd.DataFrame([results])
    df.to_csv(filepath, index=False)
    logging.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> dict:
    """
    Load results from CSV file
    
    Args:
        filepath: Input file path
        
    Returns:
        Results dictionary
    """
    df = pd.read_csv(filepath)
    return df.iloc[0].to_dict()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def ensure_directory(directory: str):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
