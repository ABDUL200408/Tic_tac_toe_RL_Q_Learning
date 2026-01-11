"""
Training module for Q-Learning agents
"""
import numpy as np
from tictactoe_env import TicTacToe
from q_learning_agent import QLearningAgent
from typing import Tuple


def train_agents(episodes: int = 50000,
                alpha: float = 0.1,
                gamma: float = 0.9,
                epsilon: float = 0.1,
                save_interval: int = 10000,
                verbose: bool = True) -> Tuple[QLearningAgent, QLearningAgent]:
    """
    Train two Q-Learning agents by self-play
    
    Args:
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        save_interval: Save agents every N episodes
        verbose: Print progress
        
    Returns:
        Tuple of (agent1, agent2)
    """
    env = TicTacToe(board_size=5)
    
    # Create two agents
    agent1 = QLearningAgent(player=1, alpha=alpha, gamma=gamma, epsilon=epsilon)
    agent2 = QLearningAgent(player=-1, alpha=alpha, gamma=gamma, epsilon=epsilon)
    
    # Statistics
    wins_1 = 0
    wins_2 = 0
    draws = 0
    
    print("Starting training...")
    print(f"Episodes: {episodes}, Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        agent1.start_episode()
        agent2.start_episode()
        
        done = False
        current_agent = agent1
        other_agent = agent2
        
        while not done:
            # Get available actions
            available_actions = env.get_available_actions()
            
            # Choose action
            action = current_agent.choose_action(state, available_actions, training=True)
            
            # Store transition
            current_agent.store_transition(state, action)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            
            # Switch agents
            current_agent, other_agent = other_agent, current_agent
        
        # Update statistics
        winner = info.get('winner', 0)
        if winner == 1:
            wins_1 += 1
        elif winner == -1:
            wins_2 += 1
        else:
            draws += 1
        
        # Learn from episode
        if winner == 1:
            agent1.learn(1)    # Winner gets +1
            agent2.learn(-1)   # Loser gets -1
        elif winner == -1:
            agent1.learn(-1)   # Loser gets -1
            agent2.learn(1)    # Winner gets +1
        else:
            agent1.learn(0)    # Draw
            agent2.learn(0)    # Draw
        
        # Print progress
        if verbose and (episode + 1) % save_interval == 0:
            total = episode + 1
            print(f"\nEpisode {total}/{episodes}")
            print(f"  Agent 1 (X) wins: {wins_1} ({100*wins_1/total:.1f}%)")
            print(f"  Agent 2 (O) wins: {wins_2} ({100*wins_2/total:.1f}%)")
            print(f"  Draws: {draws} ({100*draws/total:.1f}%)")
            print(f"  Agent 1 Q-table size: {agent1.get_q_table_size()}")
            print(f"  Agent 2 Q-table size: {agent2.get_q_table_size()}")
            
            # Save agents
            agent1.save(f'agent1_episode_{total}.pkl')
            agent2.save(f'agent2_episode_{total}.pkl')
    
    # Final statistics
    print("\n" + "="*50)
    print("Training completed!")
    print(f"  Agent 1 (X) wins: {wins_1} ({100*wins_1/episodes:.1f}%)")
    print(f"  Agent 2 (O) wins: {wins_2} ({100*wins_2/episodes:.1f}%)")
    print(f"  Draws: {draws} ({100*draws/episodes:.1f}%)")
    print(f"  Agent 1 Q-table size: {agent1.get_q_table_size()}")
    print(f"  Agent 2 Q-table size: {agent2.get_q_table_size()}")
    print("="*50)
    
    # Save final agents
    agent1.save('agent1_final.pkl')
    agent2.save('agent2_final.pkl')
    
    return agent1, agent2


if __name__ == "__main__":
    # Train with default parameters
    train_agents(episodes=50000, verbose=True)
