"""
main.py

Entry point for training + evaluating a Q-Learning Tic-Tac-Toe agent
and running structured experiments.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from environment import TicTacToeEnv
from agent import QLearningAgent
from evaluator import Evaluator
from config import get_config
from experiments import run_experiments
from visualisations import plot_learning_curve, plot_experiment_results, plot_board_template
from utils import set_random_seed

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


def choose_opponent_action(env: TicTacToeEnv, policy: str = "heuristic") -> Optional[Action]:
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


def demo_agent_vs_opponent(env: TicTacToeEnv, agent: QLearningAgent, opponent_policy: str = "random") -> None:
    print("\n===== Demo Game: Agent (X) vs Opponent (O) =====")
    print(f"Board size: {env.n}x{env.n}, win condition: {env.k} in a row")
    print(f"Opponent policy: {opponent_policy}\n")

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        state = env.reset()
        done = False
        player = 1
        move_no = 1

        print("Initial board:")
        env.render()
        print("-" * (2 * env.n - 1))

        while not done:
            if player == 1:
                action = agent.choose_action(state, evaluation=True)
                if action is None:
                    print("\n[Demo] No legal move for agent (board full).")
                    break
                print(f"\nMove {move_no}: Agent (X) plays {action}")
            else:
                action = choose_opponent_action(env, policy=opponent_policy)
                if action is None:
                    print("\n[Demo] No legal move for opponent (board full).")
                    break
                print(f"\nMove {move_no}: Opponent (O) plays {action}")

            state, _, done, info = env.step(action)
            env.render()
            print("-" * (2 * env.n - 1))

            move_no += 1
            player *= -1

        winner = info.get("winner")
        print("\n===== Demo Game Over =====")
        if winner == 1:
            print("Result: Agent (X) wins.")
        elif winner == -1:
            print("Result: Opponent (O) wins.")
        else:
            print("Result: Draw.")
        print("============================================\n")

    finally:
        agent.epsilon = original_epsilon


def main() -> None:
    config = get_config()

    episodes = config["episodes"]
    show_progress = config.get("show_progress", True)

    seed = config.get("random_seed", 42)
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    board_size = config["board_size"]
    win_condition = config["win_condition"]

    reward_win = config.get("reward_win", 1.0)
    reward_loss = config.get("reward_loss", -1.0)
    reward_draw = config.get("reward_draw", 0.0)

    all_actions = [(r, c) for r in range(board_size) for c in range(board_size)]

    env = TicTacToeEnv(
        board_size=board_size,
        win_condition=win_condition,
        reward_win=reward_win,
        reward_loss=reward_loss,
        reward_draw=reward_draw,
    )

    agent = QLearningAgent(
        actions=all_actions,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        initial_epsilon=config["epsilon"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
        board_size=board_size,
    )

    evaluator = Evaluator(env, agent)

    opponent_policy = config.get("opponent_policy", "heuristic")       # training opponent
    eval_opponent = config.get("evaluation_opponent", "random")        # evaluation opponent
    eval_interval = config.get("eval_interval", 500)
    eval_games = config.get("eval_episodes", 200)
    demo_after_training = config.get("demo_after_training", True)

    plot_board_template(board_size, config)

    stats = {"wins": [], "draws": [], "losses": [], "avg_reward": [], "epsilons": [], "episodes": []}

    # ----------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------
    for episode in range(episodes):
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
                opp_action = choose_opponent_action(env, opponent_policy)
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

        # ----------------------------------------------------------------------
        # Periodic evaluation
        # ----------------------------------------------------------------------
        if (episode + 1) % eval_interval == 0 or episode == episodes - 1:
            wins, draws, losses, avg_reward = evaluator.evaluate(
                n_games=eval_games,
                opponent_policy=eval_opponent,  # "random" or "heuristic"
                greedy=True,
            )

            stats["wins"].append(wins)
            stats["draws"].append(draws)
            stats["losses"].append(losses)
            stats["avg_reward"].append(avg_reward)
            stats["epsilons"].append(agent.epsilon)
            stats["episodes"].append(episode + 1)

            if show_progress:
                print(
                    f"[Episode {episode + 1}] "
                    f"Wins={wins}, Draws={draws}, Losses={losses}, "
                    f"AvgReward={avg_reward:.3f}, Epsilon={agent.epsilon:.3f}"
                )

    # Save Q-table + plots
    agent.save(config["q_table_save_path"])
    plot_learning_curve(stats, config)
    print("Learning curve saved to:", config.get("plots_dir", "results/"))

    # Optional interactive play
    if config.get("play_interactive", False):
        evaluator.human_vs_agent()

    # Demo after training
    if demo_after_training:
        demo_agent_vs_opponent(env, agent, opponent_policy=eval_opponent)

    # Experiments
    experiment_results = run_experiments(config)

    plots_dir = config.get("plots_dir", "results/")
    for name, data in experiment_results.items():
        print(f"\n===== Experiment results: {name} =====")
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        plot_experiment_results(data, name, config)
        df.to_csv(f"{plots_dir.rstrip('/')}/{name}.csv", index=False)


if __name__ == "__main__":
    main()
