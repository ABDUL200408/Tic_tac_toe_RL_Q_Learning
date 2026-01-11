"""
Q-Learning Agent for Tic-Tac-Toe
"""
import numpy as np
import pickle
import random
from typing import Tuple, Dict, Optional
from collections import defaultdict


class QLearningAgent:
    """Q-Learning agent for Tic-Tac-Toe"""
    
    def __init__(self, 
                 player: int = 1,
                 alpha: float = 0.1, 
                 gamma: float = 0.9, 
                 epsilon: float = 0.1):
        """
        Initialize Q-Learning agent
        
        Args:
            player: Player number (1 or -1)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_history = []
        self.action_history = []
        
    def get_state_key(self, board: np.ndarray, player: int) -> str:
        """
        Get state representation from player's perspective
        
        Args:
            board: Current board state
            player: Current player
            
        Returns:
            String representation of state
        """
        # Normalize board to current player's perspective
        normalized = board * player
        return ''.join(map(str, normalized.flatten()))
    
    def get_action_key(self, action: Tuple[int, int]) -> str:
        """
        Convert action to string key
        
        Args:
            action: Action tuple (row, col)
            
        Returns:
            String representation of action
        """
        return f"{action[0]},{action[1]}"
    
    def choose_action(self, board: np.ndarray, available_actions: list, 
                     training: bool = True) -> Tuple[int, int]:
        """
        Choose an action using epsilon-greedy policy
        
        Args:
            board: Current board state
            available_actions: List of available actions
            training: Whether in training mode
            
        Returns:
            Selected action (row, col)
        """
        if not available_actions:
            return None
        
        # Epsilon-greedy: explore vs exploit
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        
        # Exploit: choose best action based on Q-values
        state_key = self.get_state_key(board, self.player)
        
        best_value = float('-inf')
        best_actions = []
        
        for action in available_actions:
            action_key = self.get_action_key(action)
            q_value = self.q_table[state_key][action_key]
            
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        # If multiple actions have same Q-value, choose randomly
        return random.choice(best_actions)
    
    def start_episode(self) -> None:
        """Start a new episode, clearing history"""
        self.state_history = []
        self.action_history = []
    
    def store_transition(self, state: np.ndarray, action: Tuple[int, int]) -> None:
        """
        Store state-action pair for later updates
        
        Args:
            state: Board state
            action: Action taken
        """
        self.state_history.append(state.copy())
        self.action_history.append(action)
    
    def learn(self, reward: float) -> None:
        """
        Update Q-values based on episode outcome
        
        Args:
            reward: Final reward for the episode
        """
        # Backward update through the episode
        for i in range(len(self.state_history) - 1, -1, -1):
            state = self.state_history[i]
            action = self.action_history[i]
            
            state_key = self.get_state_key(state, self.player)
            action_key = self.get_action_key(action)
            
            # Get current Q-value
            current_q = self.q_table[state_key][action_key]
            
            # Q-learning update
            # For the last state, next_q is 0 (terminal state)
            # For other states, use the Q-value of the next state-action pair
            if i == len(self.state_history) - 1:
                next_q = 0
            else:
                next_state = self.state_history[i + 1]
                next_action = self.action_history[i + 1]
                next_state_key = self.get_state_key(next_state, self.player)
                next_action_key = self.get_action_key(next_action)
                next_q = self.q_table[next_state_key][next_action_key]
            
            # Update Q-value
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_table[state_key][action_key] = new_q
            
            # Decay reward for earlier states
            reward *= self.gamma
    
    def save(self, filename: str) -> None:
        """
        Save Q-table to file
        
        Args:
            filename: Path to save file
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'player': self.player,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f)
        print(f"Agent saved to {filename}")
    
    def load(self, filename: str) -> None:
        """
        Load Q-table from file
        
        Args:
            filename: Path to load file
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
                self.player = data['player']
                self.alpha = data['alpha']
                self.gamma = data['gamma']
                self.epsilon = data['epsilon']
            print(f"Agent loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved agent found at {filename}")
    
    def get_q_table_size(self) -> int:
        """
        Get the number of states in Q-table
        
        Returns:
            Number of states
        """
        return len(self.q_table)
