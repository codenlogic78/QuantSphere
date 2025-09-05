"""
Training Script for Quantsphere Trading Agent

This script trains the Quantsphere trading agent using deep reinforcement learning.
"""

import os
import sys
import logging
import argparse
from typing import List, Tuple
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantsphere import TradingAgent
from quantsphere.utils import (
    get_stock_data, 
    show_train_result, 
    switch_k_backend_device,
    setup_logging,
    ensure_directory
)


def train_agent(
    train_data_path: str,
    val_data_path: str,
    strategy: str = "t-dqn",
    window_size: int = 10,
    batch_size: int = 32,
    episodes: int = 50,
    learning_rate: float = 0.001,
    model_name: str = "model_debug",
    pretrained: bool = False,
    debug: bool = False
) -> None:
    """
    Train the Quantsphere trading agent
    
    Args:
        train_data_path: Path to training data CSV
        val_data_path: Path to validation data CSV
        strategy: DQN strategy to use
        window_size: State window size
        batch_size: Training batch size
        episodes: Number of training episodes
        learning_rate: Learning rate
        model_name: Model name for saving
        pretrained: Whether to load pretrained model
        debug: Whether to use debug logging
    """
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level)
    
    # Switch to CPU if needed
    switch_k_backend_device()
    
    # Ensure models directory exists
    ensure_directory("models")
    
    # Load data
    logging.info(f"Loading training data from {train_data_path}")
    train_data = get_stock_data(train_data_path)
    
    logging.info(f"Loading validation data from {val_data_path}")
    val_data = get_stock_data(val_data_path)
    
    # Calculate initial offset for validation
    initial_offset = val_data[1] - val_data[0] if len(val_data) > 1 else 0
    
    # Initialize agent
    logging.info(f"Initializing agent with strategy: {strategy}")
    agent = TradingAgent(
        state_size=window_size,
        strategy=strategy,
        learning_rate=learning_rate,
        pretrained=pretrained,
        model_name=model_name
    )
    
    # Training loop
    logging.info(f"Starting training for {episodes} episodes")
    for episode in range(1, episodes + 1):
        # Train episode
        train_profit, train_loss = agent.train_episode(train_data, episode, episodes)
        
        # Validate episode
        val_profit, _ = agent.evaluate(val_data, debug=debug)
        
        # Show results
        train_result = (episode, episodes, train_profit, train_loss)
        show_train_result(train_result, val_profit, initial_offset)
        
        # Save model every 10 episodes
        if episode % 10 == 0:
            agent.save_model(episode)
            logging.info(f"Model saved at episode {episode}")
    
    # Save final model
    agent.save_model(episodes)
    logging.info("Training completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train Quantsphere Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py data/GOOG.csv data/GOOG_2018.csv --strategy t-dqn
  python train.py data/AAPL.csv data/AAPL_2018.csv --strategy double-dqn --episodes 100
  python train.py data/TSLA.csv data/TSLA_2018.csv --strategy dqn --window-size 20
        """
    )
    
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("val_data", help="Path to validation data CSV file")
    parser.add_argument(
        "--strategy", 
        choices=["dqn", "t-dqn", "double-dqn"],
        default="t-dqn",
        help="DQN strategy to use (default: t-dqn)"
    )
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=10,
        help="Size of the state window (default: 10)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=50,
        help="Number of training episodes (default: 50)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--model-name", 
        default="model_debug",
        help="Model name for saving (default: model_debug)"
    )
    parser.add_argument(
        "--pretrained", 
        action="store_true",
        help="Load pretrained model"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    try:
        train_agent(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            strategy=args.strategy,
            window_size=args.window_size,
            batch_size=args.batch_size,
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            model_name=args.model_name,
            pretrained=args.pretrained,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
