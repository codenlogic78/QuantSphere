"""
Quantsphere - Advanced Deep Reinforcement Learning Trading Agent

A sophisticated trading agent that leverages deep reinforcement learning algorithms
to make intelligent trading decisions in financial markets.
"""

__version__ = "1.0.0"
__author__ = "codenlogic78"
__email__ = "codenlogic78@gmail.com"

from .agent import TradingAgent
from .environment import TradingEnvironment
from .networks import DQNNetwork, TargetNetwork
from .strategies import DQNStrategy, TargetDQNStrategy, DoubleDQNStrategy

__all__ = [
    "TradingAgent",
    "TradingEnvironment", 
    "DQNNetwork",
    "TargetNetwork",
    "DQNStrategy",
    "TargetDQNStrategy",
    "DoubleDQNStrategy"
]
