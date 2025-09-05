"""
Trading Agent Implementation

This module contains the core TradingAgent class that orchestrates
the deep reinforcement learning trading system.
"""

import random
import numpy as np
import tensorflow as tf
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

from .networks import DQNNetwork, TargetNetwork
from .strategies import DQNStrategy, TargetDQNStrategy, DoubleDQNStrategy
from .utils import format_currency, format_position


class TradingAgent:
    """
    Advanced Deep Reinforcement Learning Trading Agent
    
    Implements multiple DQN variants for intelligent trading decisions
    in financial markets.
    """
    
    def __init__(
        self,
        state_size: int = 10,
        strategy: str = "t-dqn",
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        pretrained: bool = False,
        model_name: Optional[str] = None
    ):
        """
        Initialize the Trading Agent
        
        Args:
            state_size: Size of the state representation window
            strategy: DQN strategy to use ('dqn', 't-dqn', 'double-dqn')
            learning_rate: Neural network learning rate
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            memory_size: Size of experience replay buffer
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            pretrained: Whether to load a pretrained model
            model_name: Name of the pretrained model to load
        """
        self.state_size = state_size
        self.action_size = 3  # [Hold, Buy, Sell]
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Trading state
        self.inventory = []
        self.memory = deque(maxlen=memory_size)
        self.first_iter = True
        self.training_step = 0
        
        # Initialize strategy
        self._initialize_strategy()
        
        # Load pretrained model if specified
        if pretrained and model_name:
            self.load_model(model_name)
    
    def _initialize_strategy(self):
        """Initialize the DQN strategy"""
        if self.strategy == "dqn":
            self.strategy_impl = DQNStrategy(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.learning_rate
            )
        elif self.strategy == "t-dqn":
            self.strategy_impl = TargetDQNStrategy(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.learning_rate,
                target_update_freq=self.target_update_freq
            )
        elif self.strategy == "double-dqn":
            self.strategy_impl = DoubleDQNStrategy(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.learning_rate,
                target_update_freq=self.target_update_freq
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_state(self, data: List[float], t: int, n_days: int) -> np.ndarray:
        """
        Get state representation at time t
        
        Args:
            data: Price data
            t: Current time index
            n_days: Number of days for state window
            
        Returns:
            State representation as numpy array
        """
        d = t - n_days + 1
        if d >= 0:
            block = data[d:t + 1]
        else:
            block = [-d * [data[0]]] + data[0:t + 1]
        
        res = []
        for i in range(n_days - 1):
            res.append(self._sigmoid(block[i + 1] - block[i]))
        
        return np.array([res])
    
    def _sigmoid(self, x: float) -> float:
        """Apply sigmoid function for normalization"""
        try:
            if x < 0:
                return 1 - 1 / (1 + np.exp(x))
            return 1 / (1 + np.exp(-x))
        except:
            return 0.5
    
    def act(self, state: np.ndarray, is_eval: bool = False) -> int:
        """
        Select action based on current state
        
        Args:
            state: Current state representation
            is_eval: Whether this is evaluation mode
            
        Returns:
            Action index (0=Hold, 1=Buy, 2=Sell)
        """
        # Random action for exploration during training
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # First iteration always buy
        if self.first_iter:
            self.first_iter = False
            return 1
        
        # Get action from strategy
        return self.strategy_impl.act(state)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self) -> float:
        """
        Train the agent on a batch of experiences
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Train using strategy
        loss = self.strategy_impl.train(batch, self.gamma)
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network if needed
        self.training_step += 1
        if hasattr(self.strategy_impl, 'update_target'):
            self.strategy_impl.update_target(self.training_step)
        
        return loss
    
    def train_episode(self, data: List[float], episode: int, 
                     total_episodes: int) -> Tuple[float, float]:
        """
        Train the agent for one episode
        
        Args:
            data: Training data
            episode: Current episode number
            total_episodes: Total number of episodes
            
        Returns:
            Tuple of (total_profit, average_loss)
        """
        total_profit = 0
        data_length = len(data) - 1
        self.inventory = []
        losses = []
        
        state = self.get_state(data, 0, self.state_size + 1)
        
        for t in range(data_length):
            reward = 0
            next_state = self.get_state(data, t + 1, self.state_size + 1)
            
            # Select action
            action = self.act(state)
            
            # Execute action
            if action == 1:  # Buy
                self.inventory.append(data[t])
            elif action == 2 and len(self.inventory) > 0:  # Sell
                bought_price = self.inventory.pop(0)
                delta = data[t] - bought_price
                reward = delta
                total_profit += delta
            
            # Store experience
            done = (t == data_length - 1)
            self.remember(state, action, reward, next_state, done)
            
            # Train if enough experiences
            if len(self.memory) > self.batch_size:
                loss = self.replay()
                losses.append(loss)
            
            state = next_state
        
        avg_loss = np.mean(losses) if losses else 0.0
        return total_profit, avg_loss
    
    def evaluate(self, data: List[float], debug: bool = False) -> Tuple[float, List]:
        """
        Evaluate the agent on test data
        
        Args:
            data: Test data
            debug: Whether to print debug information
            
        Returns:
            Tuple of (total_profit, trading_history)
        """
        total_profit = 0
        data_length = len(data) - 1
        self.inventory = []
        history = []
        
        state = self.get_state(data, 0, self.state_size + 1)
        
        for t in range(data_length):
            next_state = self.get_state(data, t + 1, self.state_size + 1)
            
            # Select action (no exploration during evaluation)
            action = self.act(state, is_eval=True)
            
            # Execute action
            if action == 1:  # Buy
                self.inventory.append(data[t])
                history.append((data[t], "BUY"))
                if debug:
                    print(f"Buy at: {format_currency(data[t])}")
            elif action == 2 and len(self.inventory) > 0:  # Sell
                bought_price = self.inventory.pop(0)
                delta = data[t] - bought_price
                total_profit += delta
                history.append((data[t], "SELL"))
                if debug:
                    print(f"Sell at: {format_currency(data[t])} | "
                          f"Position: {format_position(delta)}")
            else:
                history.append((data[t], "HOLD"))
            
            state = next_state
        
        return total_profit, history
    
    def save_model(self, episode: int):
        """Save the trained model"""
        self.strategy_impl.save_model(f"models/model_{self.strategy}_{episode}")
    
    def load_model(self, model_name: str):
        """Load a pretrained model"""
        self.strategy_impl.load_model(f"models/{model_name}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "strategy": self.strategy,
            "state_size": self.state_size,
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "training_step": self.training_step
        }
