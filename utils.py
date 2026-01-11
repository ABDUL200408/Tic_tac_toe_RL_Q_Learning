"""
utils.py

Utility functions for Tic-Tac-Toe Reinforcement Learning.

Adds:
- A stronger heuristic opponent policy:
  * win if possible
  * block if necessary
  * centre if available
  * else random
"""

from __future__ import annotations
import numpy as np
import random
import os
from typing import Tuple, Optional, Literal

Action = Tuple[int, int]


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed across numpy, random, and OS-level hashing.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------
# State representation utilities
# ----------------------------------------------------------------------
def encode_state(board: np.ndarray) -> tuple:
    """
    Encode an n x n board as a flat tuple (hashable).
    """
    return tuple(board.flatten())


def decode_state(state_tuple: tuple, board_size: int) -> np.ndarray:
    """
    Decode a flattened tuple back into an n x n board.
    """
    arr = np.array(state_tuple, dtype=int)
    return arr.reshape((board_size, board_size))


flatten = encode_state
unflatten = decode_state


# ----------------------------------------------------------------------
# Logging utility
# ----------------------------------------------------------------------
def log(msg: str, prefix: str = "[INFO]") -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{prefix} {timestamp}: {msg}")


# ----------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------
def safe_mkdir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as ex:
        log(f"Failed to create directory '{path}': {ex}", prefix="[ERROR]")


# ----------------------------------------------------------------------
# Opponent policy (single source of truth)
# ----------------------------------------------------------------------
def select_opponent_action(env, policy: Literal["random", "centre", "heuristic"] = "random") -> Optional[Action]:
    """
    Select an opponent action according to a specified policy.

    Policies:
    - "random": uniform random move
    - "centre": choose centre if available, else random
    - "heuristic": (strong baseline)
         1) win now if possible
         2) block agent win if necessary
         3) centre if available
         4) otherwise random

    Notes:
    - Uses env.get_action_space() and env.step() winner logic indirectly by simulating moves.
    - Assumes env.board is a NumPy array and env.check_winner() exists (true in your env).
    """
    available = env.get_action_space()
    if not available:
        return None

    if policy == "random":
        return random.choice(available)

    # Helper: centre preference
    centre = (env.n // 2, env.n // 2)
    if policy == "centre":
        return centre if centre in available else random.choice(available)

    # -----------------------------
    # Strong heuristic policy
    # -----------------------------
    # opponent mark is env.current_player (should be -1 when opponent acts)
    opp = env.current_player
    agent = -opp

    # 1) Win if possible
    for a in available:
        if _is_winning_move(env, a, opp):
            return a

    # 2) Block agent win if needed
    for a in available:
        if _is_winning_move(env, a, agent):
            return a

    # 3) Centre if available
    if centre in available:
        return centre

    # 4) Random fallback
    return random.choice(available)


def _is_winning_move(env, action: Action, player: int) -> bool:
    """
    Check if placing 'player' mark at 'action' would create a win.
    Does not permanently modify env.board.
    """
    r, c = action
    if env.board[r, c] != 0:
        return False

    # temporarily place
    env.board[r, c] = player
    try:
        winner = env.check_winner()
        return winner == player
    finally:
        env.board[r, c] = 0
