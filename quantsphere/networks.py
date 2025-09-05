"""
Neural Network Architectures

This module contains the neural network implementations for the DQN variants.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, clip_delta: float = 1.0) -> tf.Tensor:
    """
    Huber loss function for robust training
    
    Args:
        y_true: True Q-values
        y_pred: Predicted Q-values
        clip_delta: Clipping threshold
        
    Returns:
        Huber loss value
    """
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


class DQNNetwork:
    """
    Deep Q-Network implementation
    
    A neural network that approximates the Q-function for trading decisions.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """
        Initialize the DQN network
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate for the optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
    def _build_model(self) -> keras.Model:
        """Build the neural network architecture"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=self.optimizer,
            loss=huber_loss,
            metrics=['mae']
        )
        
        return model
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for given state
        
        Args:
            state: Input state
            
        Returns:
            Q-values for each action
        """
        return self.model.predict(state, verbose=0)
    
    def train_step(self, states: np.ndarray, targets: np.ndarray) -> float:
        """
        Perform one training step
        
        Args:
            states: Batch of states
            targets: Batch of target Q-values
            
        Returns:
            Training loss
        """
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = huber_loss(targets, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return float(loss)
    
    def get_weights(self) -> list:
        """Get model weights"""
        return self.model.get_weights()
    
    def set_weights(self, weights: list):
        """Set model weights"""
        self.model.set_weights(weights)
    
    def save_model(self, filepath: str):
        """Save the model to file"""
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(
            filepath, 
            custom_objects={'huber_loss': huber_loss}
        )


class TargetNetwork(DQNNetwork):
    """
    Target Network for stable training
    
    A copy of the main network that is updated less frequently
    to provide stable targets during training.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """
        Initialize the target network
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate (not used for target network)
        """
        super().__init__(state_size, action_size, learning_rate)
        # Target network doesn't need optimizer
        self.optimizer = None
    
    def update_from_main(self, main_network: DQNNetwork):
        """
        Update target network weights from main network
        
        Args:
            main_network: The main DQN network
        """
        self.set_weights(main_network.get_weights())
    
    def train_step(self, states: np.ndarray, targets: np.ndarray) -> float:
        """
        Target network doesn't train - just returns 0
        
        Args:
            states: Batch of states (unused)
            targets: Batch of target Q-values (unused)
            
        Returns:
            0 (no training loss)
        """
        return 0.0


class DoubleDQNNetwork(DQNNetwork):
    """
    Double DQN Network
    
    Implements the Double DQN algorithm with separate networks
    for action selection and value estimation.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        """
        Initialize the Double DQN network
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(state_size, action_size, learning_rate)
        # Create a separate network for action selection
        self.action_network = self._build_model()
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using the action network
        
        Args:
            state: Input state
            
        Returns:
            Selected action index
        """
        q_values = self.action_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values using the main network
        
        Args:
            state: Input state
            
        Returns:
            Q-values for each action
        """
        return self.predict(state)
    
    def update_action_network(self):
        """Update action network from main network"""
        self.action_network.set_weights(self.get_weights())
    
    def save_model(self, filepath: str):
        """Save both networks"""
        self.model.save(f"{filepath}_main")
        self.action_network.save(f"{filepath}_action")
    
    def load_model(self, filepath: str):
        """Load both networks"""
        self.model = keras.models.load_model(
            f"{filepath}_main",
            custom_objects={'huber_loss': huber_loss}
        )
        self.action_network = keras.models.load_model(
            f"{filepath}_action",
            custom_objects={'huber_loss': huber_loss}
        )
