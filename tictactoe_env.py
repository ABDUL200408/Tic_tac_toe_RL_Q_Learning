"""
Tic-Tac-Toe Environment for 5x5 board
"""
import numpy as np
from typing import Tuple, Optional, List


class TicTacToe:
    """5x5 Tic-Tac-Toe game environment"""
    
    # Constants
    INVALID_MOVE_PENALTY = -10
    
    def __init__(self, board_size: int = 5):
        """
        Initialize the game environment
        
        Args:
            board_size: Size of the board (default: 5 for 5x5)
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.winner = None
        self.game_over = False
        
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state
        
        Returns:
            Initial board state
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        return self.board.copy()
    
    def get_available_actions(self) -> List[Tuple[int, int]]:
        """
        Get list of available moves
        
        Returns:
            List of (row, col) tuples for empty positions
        """
        return list(zip(*np.where(self.board == 0)))
    
    def is_valid_action(self, row: int, col: int) -> bool:
        """
        Check if a move is valid
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if the move is valid, False otherwise
        """
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute a move in the game
        
        Args:
            action: Tuple of (row, col) for the move
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.game_over:
            return self.board.copy(), 0, True, {"winner": self.winner}
        
        row, col = action
        
        if not self.is_valid_action(row, col):
            # Invalid move
            return self.board.copy(), self.INVALID_MOVE_PENALTY, True, {"winner": -self.current_player}
        
        # Make the move
        self.board[row, col] = self.current_player
        
        # Check for winner
        winner = self._check_winner()
        
        if winner is not None:
            self.game_over = True
            self.winner = winner
            if winner == 0:
                # Draw
                reward = 0
            else:
                # Winner gets reward
                reward = 1 if winner == self.current_player else -1
            return self.board.copy(), reward, True, {"winner": winner}
        
        # Check for draw
        if len(self.get_available_actions()) == 0:
            self.game_over = True
            self.winner = 0
            return self.board.copy(), 0, True, {"winner": 0}
        
        # Switch player
        self.current_player = -self.current_player
        
        return self.board.copy(), 0, False, {}
    
    def _check_winner(self) -> Optional[int]:
        """
        Check if there is a winner
        
        Returns:
            Player number (1 or -1) if winner, 0 for draw, None for ongoing game
        """
        # Check rows
        for row in range(self.board_size):
            for col in range(self.board_size - 4):
                sequence = self.board[row, col:col+5]
                if np.all(sequence == 1):
                    return 1
                elif np.all(sequence == -1):
                    return -1
        
        # Check columns
        for col in range(self.board_size):
            for row in range(self.board_size - 4):
                sequence = self.board[row:row+5, col]
                if np.all(sequence == 1):
                    return 1
                elif np.all(sequence == -1):
                    return -1
        
        # Check diagonals (top-left to bottom-right)
        for row in range(self.board_size - 4):
            for col in range(self.board_size - 4):
                sequence = [self.board[row+i, col+i] for i in range(5)]
                if all(s == 1 for s in sequence):
                    return 1
                elif all(s == -1 for s in sequence):
                    return -1
        
        # Check diagonals (top-right to bottom-left)
        for row in range(self.board_size - 4):
            for col in range(4, self.board_size):
                sequence = [self.board[row+i, col-i] for i in range(5)]
                if all(s == 1 for s in sequence):
                    return 1
                elif all(s == -1 for s in sequence):
                    return -1
        
        return None
    
    def render(self) -> None:
        """Print the current board state"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n  " + " ".join([str(i) for i in range(self.board_size)]))
        for i, row in enumerate(self.board):
            print(f"{i} " + " ".join([symbols[cell] for cell in row]))
        print()
    
    def get_state_key(self) -> str:
        """
        Get a string representation of the current state
        
        Returns:
            String representation of the board state
        """
        return ''.join(map(str, self.board.flatten()))
