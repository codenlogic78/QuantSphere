"""
DQN Strategy Implementations

This module contains the different DQN strategy implementations
for the trading agent.
"""

import numpy as np
from typing import List, Tuple
from .networks import DQNNetwork, TargetNetwork, DoubleDQNNetwork


class DQNStrategy:
    """
    Vanilla Deep Q-Network Strategy
    
    Implements the classic DQN algorithm for trading decisions.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """
        Initialize the DQN strategy
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate for the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.network = DQNNetwork(state_size, action_size, learning_rate)
    
    def act(self, state: np.ndarray) -> int:
        """
        Select action based on current state
        
        Args:
            state: Current state representation
            
        Returns:
            Action index
        """
        q_values = self.network.predict(state)
        return np.argmax(q_values[0])
    
    def train(self, batch: List[Tuple], gamma: float) -> float:
        """
        Train the network on a batch of experiences
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
            gamma: Discount factor
            
        Returns:
            Training loss
        """
        states = np.array([e[0][0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3][0] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get current Q-values
        current_q_values = self.network.predict(states)
        
        # Get next Q-values
        next_q_values = self.network.predict(next_states)
        
        # Calculate target Q-values
        targets = current_q_values.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
        
        # Train the network
        loss = self.network.train_step(states, targets)
        return loss
    
    def save_model(self, filepath: str):
        """Save the model"""
        self.network.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load the model"""
        self.network.load_model(filepath)


class TargetDQNStrategy:
    """
    Target DQN Strategy
    
    Implements DQN with a separate target network for stable training.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 target_update_freq: int = 1000):
        """
        Initialize the Target DQN strategy
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate for the network
            target_update_freq: Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.target_update_freq = target_update_freq
        
        self.main_network = DQNNetwork(state_size, action_size, learning_rate)
        self.target_network = TargetNetwork(state_size, action_size, learning_rate)
        
        # Initialize target network with main network weights
        self.target_network.update_from_main(self.main_network)
    
    def act(self, state: np.ndarray) -> int:
        """
        Select action based on current state
        
        Args:
            state: Current state representation
            
        Returns:
            Action index
        """
        q_values = self.main_network.predict(state)
        return np.argmax(q_values[0])
    
    def train(self, batch: List[Tuple], gamma: float) -> float:
        """
        Train the network on a batch of experiences
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
            gamma: Discount factor
            
        Returns:
            Training loss
        """
        states = np.array([e[0][0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3][0] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get current Q-values from main network
        current_q_values = self.main_network.predict(states)
        
        # Get next Q-values from target network
        next_q_values = self.target_network.predict(next_states)
        
        # Calculate target Q-values
        targets = current_q_values.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
        
        # Train the main network
        loss = self.main_network.train_step(states, targets)
        return loss
    
    def update_target(self, training_step: int):
        """
        Update target network if it's time
        
        Args:
            training_step: Current training step
        """
        if training_step % self.target_update_freq == 0:
            self.target_network.update_from_main(self.main_network)
    
    def save_model(self, filepath: str):
        """Save both networks"""
        self.main_network.save_model(f"{filepath}_main")
        self.target_network.save_model(f"{filepath}_target")
    
    def load_model(self, filepath: str):
        """Load both networks"""
        self.main_network.load_model(f"{filepath}_main")
        self.target_network.load_model(f"{filepath}_target")


class DoubleDQNStrategy:
    """
    Double DQN Strategy
    
    Implements Double DQN with separate networks for action selection
    and value estimation to reduce overestimation bias.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 target_update_freq: int = 1000):
        """
        Initialize the Double DQN strategy
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate for the network
            target_update_freq: Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.target_update_freq = target_update_freq
        
        self.main_network = DoubleDQNNetwork(state_size, action_size, learning_rate)
        self.target_network = TargetNetwork(state_size, action_size, learning_rate)
        
        # Initialize target network with main network weights
        self.target_network.update_from_main(self.main_network)
    
    def act(self, state: np.ndarray) -> int:
        """
        Select action based on current state
        
        Args:
            state: Current state representation
            
        Returns:
            Action index
        """
        q_values = self.main_network.predict(state)
        return np.argmax(q_values[0])
    
    def train(self, batch: List[Tuple], gamma: float) -> float:
        """
        Train the network on a batch of experiences
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
            gamma: Discount factor
            
        Returns:
            Training loss
        """
        states = np.array([e[0][0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3][0] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Get current Q-values from main network
        current_q_values = self.main_network.predict(states)
        
        # Use main network to select actions for next states
        next_actions = []
        for i in range(len(batch)):
            if not dones[i]:
                next_action = self.main_network.select_action(next_states[i:i+1])
                next_actions.append(next_action)
            else:
                next_actions.append(0)  # Dummy action for done states
        
        # Get Q-values from target network for selected actions
        next_q_values = self.target_network.predict(next_states)
        
        # Calculate target Q-values using Double DQN
        targets = current_q_values.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + gamma * next_q_values[i][next_actions[i]]
        
        # Train the main network
        loss = self.main_network.train_step(states, targets)
        return loss
    
    def update_target(self, training_step: int):
        """
        Update target network if it's time
        
        Args:
            training_step: Current training step
        """
        if training_step % self.target_update_freq == 0:
            self.target_network.update_from_main(self.main_network)
            self.main_network.update_action_network()
    
    def save_model(self, filepath: str):
        """Save all networks"""
        self.main_network.save_model(f"{filepath}_main")
        self.target_network.save_model(f"{filepath}_target")
    
    def load_model(self, filepath: str):
        """Load all networks"""
        self.main_network.load_model(f"{filepath}_main")
        self.target_network.load_model(f"{filepath}_target")
