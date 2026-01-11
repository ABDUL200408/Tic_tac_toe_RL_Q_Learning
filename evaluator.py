"""
evaluator.py

Provides evaluation and analysis of RL agent performance in Tic-Tac-Toe.
"""

from __future__ import annotations
from typing import Tuple, Literal, Optional
import random


Action = Tuple[int, int]


class Evaluator:
    """
    Evaluation helper for a Q-learning Tic-Tac-Toe agent.
    """

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    def evaluate(
        self,
        n_games: int = 100,
        opponent_policy: Literal["random", "heuristic"] = "random",
        greedy: bool = True,
    ) -> Tuple[int, int, int, float]:
        """
        Evaluate the agent over multiple games.

        Returns
        -------
        wins, draws, losses, avg_reward
        avg_reward is from the agent's perspective:
            +reward_win  if agent wins
            +reward_draw if draw
            +reward_loss if agent loses
        """
        wins, draws, losses = 0, 0, 0
        total_reward = 0.0

        original_epsilon = self.agent.epsilon
        if greedy:
            self.agent.epsilon = 0.0

        try:
            for _ in range(n_games):
                state = self.env.reset()
                done = False
                player = 1  # agent considered player 1
                cumulative_game_reward = 0.0

                while not done:
                    if player == 1:
                        action = self.agent.choose_action(state, evaluation=True)
                        if action is None:
                            break
                        next_state, reward, done, info = self.env.step(action)

                        # If terminal happened on agent move, reward already correct.
                        if done:
                            cumulative_game_reward += reward

                        state = next_state
                        player *= -1

                    else:
                        action = self._select_opponent_action(opponent_policy)
                        if action is None:
                            break
                        next_state, _, done, info = self.env.step(action)

                        # Terminal happened on opponent move -> assign outcome from agent perspective
                        if done:
                            winner = info.get("winner")
                            if winner == -1:
                                cumulative_game_reward += self.env.reward_loss
                            else:
                                cumulative_game_reward += self.env.reward_draw

                        state = next_state
                        player *= -1

                winner = info.get("winner")
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1

                total_reward += cumulative_game_reward

        finally:
            self.agent.epsilon = original_epsilon

        avg_reward = total_reward / n_games
        return wins, draws, losses, avg_reward

    def human_vs_agent(self) -> None:
        """
        Human = O (-1), Agent = X (+1)
        """
        print("Play Tic-Tac-Toe against RL Agent (X). You are O (-1).")
        print("Enter moves as: row,col   (e.g. 0,0)")

        state = self.env.reset()
        done = False
        player = 1

        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        try:
            while not done:
                self.env.render()

                if player == 1:
                    action = self.agent.choose_action(state, evaluation=True)
                    print(f"\nAgent (X) chooses: {action}")
                else:
                    action = self._read_human_move()

                state, _, done, info = self.env.step(action)
                player *= -1

            self.env.render()
            winner = info.get("winner")
            print("\nGame Over!")
            if winner == 1:
                print("Agent (X) wins!")
            elif winner == -1:
                print("You (O) win!")
            else:
                print("It's a draw!")

        finally:
            self.agent.epsilon = original_epsilon

    # ------------------------------------------------------------------
    # Stronger heuristic opponent (win / block / centre / random)
    # ------------------------------------------------------------------
    def _select_opponent_action(self, opponent_policy: Literal["random", "heuristic"]) -> Optional[Action]:
        available = self.env.get_action_space()
        if not available:
            return None

        if opponent_policy == "random":
            return random.choice(available)

        # heuristic:
        # 1) win now
        # 2) block agent win
        # 3) centre
        # 4) random
        opp = self.env.current_player         # should be -1 on opponent turn
        agent = -opp

        # 1) Win now
        for a in available:
            if self._is_winning_move(a, opp):
                return a

        # 2) Block agent win
        for a in available:
            if self._is_winning_move(a, agent):
                return a

        # 3) Centre
        centre = (self.env.n // 2, self.env.n // 2)
        if centre in available:
            return centre

        # 4) Random
        return random.choice(available)

    def _is_winning_move(self, action: Action, player: int) -> bool:
        r, c = action
        if self.env.board[r, c] != 0:
            return False

        self.env.board[r, c] = player
        try:
            return self.env.check_winner() == player
        finally:
            self.env.board[r, c] = 0

    def _read_human_move(self) -> Action:
        while True:
            try:
                move = input("Your move (row,col): ")
                r, c = map(int, move.split(","))
                if (r, c) in self.env.get_action_space():
                    return (r, c)
                print("Illegal move. Try again.")
            except Exception:
                print("Invalid input. Use: row,col")
