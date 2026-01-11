"""
experiments.py

Facilitates structured experiments (hyperparameter sweeps, ablation studies)
for Q-Learning in Tic-Tac-Toe (n x n board).
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import random
import numpy as np

from agent import QLearningAgent
from environment import TicTacToeEnv
from evaluator import Evaluator

Action = Tuple[int, int]


def _is_winning_move(env: TicTacToeEnv, action: Action, player: int) -> bool:
    r, c = action
    if env.board[r, c] != 0:
        return False

    env.board[r, c] = player
    try:
        return env.check_winner() == player
    finally:
        env.board[r, c] = 0


def _select_opponent_action(env: TicTacToeEnv, policy: str = "random") -> Optional[Action]:
    available = env.get_action_space()
    if not available:
        return None

    if policy == "random":
        return random.choice(available)

    # heuristic: win / block / centre / random
    opp = env.current_player
    agent = -opp

    # 1) win now
    for a in available:
        if _is_winning_move(env, a, opp):
            return a

    # 2) block agent win
    for a in available:
        if _is_winning_move(env, a, agent):
            return a

    # 3) centre
    centre = (env.n // 2, env.n // 2)
    if centre in available:
        return centre

    # 4) random
    return random.choice(available)


def _train_agent(
    config: Dict[str, Any],
    learning_rate: float,
    epsilon_decay: float,
) -> Tuple[QLearningAgent, TicTacToeEnv]:
    board_size = config["board_size"]
    win_condition = config["win_condition"]
    episodes = config["episodes"]

    reward_win = config.get("reward_win", 1.0)
    reward_loss = config.get("reward_loss", -1.0)
    reward_draw = config.get("reward_draw", 0.0)

    actions = [(r, c) for r in range(board_size) for c in range(board_size)]

    env = TicTacToeEnv(
        board_size=board_size,
        win_condition=win_condition,
        reward_win=reward_win,
        reward_loss=reward_loss,
        reward_draw=reward_draw,
    )

    agent = QLearningAgent(
        actions=actions,
        learning_rate=learning_rate,
        discount_factor=config["discount_factor"],
        initial_epsilon=config["epsilon"],
        epsilon_decay=epsilon_decay,
        epsilon_min=config["epsilon_min"],
        board_size=board_size,
    )

    opponent_policy = config.get("opponent_policy", "random")

    for _ in range(episodes):
        state = env.reset()
        done = False

        last_agent_state = None
        last_agent_action = None

        while not done:
            if env.current_player == 1:
                action = agent.choose_action(state)
                if action is None:
                    break

                last_agent_state = state
                last_agent_action = action

                next_state, reward, done, info = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state

            else:
                opp_action = _select_opponent_action(env, opponent_policy)
                if opp_action is None:
                    break

                next_state, _, done, info = env.step(opp_action)

                # Opponent ends the game -> update last agent action with terminal outcome
                if done and last_agent_state is not None and last_agent_action is not None:
                    winner = info.get("winner")
                    if winner == -1:
                        terminal_reward = env.reward_loss
                    else:
                        terminal_reward = env.reward_draw

                    agent.learn(last_agent_state, last_agent_action, terminal_reward, next_state, True)

                state = next_state

    return agent, env


def run_experiments(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = config.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    eval_games = config.get("eval_episodes", 200)
    eval_opponent = config.get("evaluation_opponent", "random")

    results: Dict[str, Any] = {
        "learning_rate_sweep": [],
        "epsilon_decay_sweep": [],
    }

    # Experiment 1: Learning Rate Sweep
    learning_rates: List[float] = [0.05, 0.1, 0.3, 0.5]
    for lr in learning_rates:
        agent, env = _train_agent(config=config, learning_rate=lr, epsilon_decay=config["epsilon_decay"])
        evaluator = Evaluator(env, agent)

        wins, draws, losses, avg_reward = evaluator.evaluate(
            n_games=eval_games,
            opponent_policy=eval_opponent,  # "random" or "heuristic"
            greedy=True,
        )

        results["learning_rate_sweep"].append(
            {"learning_rate": lr, "wins": wins, "draws": draws, "losses": losses, "avg_reward": avg_reward}
        )

    # Experiment 2: Epsilon Decay Sweep
    epsilon_decays: List[float] = [0.99, 0.995, 0.9995]
    for decay in epsilon_decays:
        agent, env = _train_agent(config=config, learning_rate=config["learning_rate"], epsilon_decay=decay)
        evaluator = Evaluator(env, agent)

        wins, draws, losses, avg_reward = evaluator.evaluate(
            n_games=eval_games,
            opponent_policy=eval_opponent,
            greedy=True,
        )

        results["epsilon_decay_sweep"].append(
            {"epsilon_decay": decay, "wins": wins, "draws": draws, "losses": losses, "avg_reward": avg_reward}
        )

    return results
