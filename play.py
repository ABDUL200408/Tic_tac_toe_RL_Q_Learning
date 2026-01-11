"""
Interactive play module - Human vs AI
"""
import numpy as np
from tictactoe_env import TicTacToe
from q_learning_agent import QLearningAgent
import sys


def play_human_vs_ai(agent_file: str = 'agent1_final.pkl', human_first: bool = True):
    """
    Play Tic-Tac-Toe against a trained AI agent
    
    Args:
        agent_file: Path to saved agent file
        human_first: Whether human plays first
    """
    # Initialize environment
    env = TicTacToe(board_size=5)
    
    # Load AI agent
    if human_first:
        human_player = 1
        ai_player = -1
        agent = QLearningAgent(player=ai_player)
    else:
        human_player = -1
        ai_player = 1
        agent = QLearningAgent(player=ai_player)
    
    try:
        agent.load(agent_file)
    except:
        print(f"Warning: Could not load agent from {agent_file}")
        print("Using untrained agent")
    
    print("\n" + "="*50)
    print("5x5 Tic-Tac-Toe - Human vs AI")
    print("="*50)
    print(f"You are playing as {'X' if human_player == 1 else 'O'}")
    print(f"AI is playing as {'X' if ai_player == 1 else 'O'}")
    print("Get 5 in a row to win!")
    print("="*50 + "\n")
    
    # Reset game
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        
        if env.current_player == human_player:
            # Human turn
            print("Your turn!")
            while True:
                try:
                    move_input = input("Enter your move (row col): ").strip()
                    if move_input.lower() in ['q', 'quit', 'exit']:
                        print("Thanks for playing!")
                        return
                    
                    row, col = map(int, move_input.split())
                    
                    if env.is_valid_action(row, col):
                        action = (row, col)
                        break
                    else:
                        print("Invalid move! Try again.")
                except (ValueError, IndexError):
                    print("Invalid input! Enter row and column as two numbers (e.g., '2 3')")
        else:
            # AI turn
            print("AI is thinking...")
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions, training=False)
            print(f"AI plays: {action[0]} {action[1]}")
        
        # Execute move
        state, reward, done, info = env.step(action)
    
    # Game over
    env.render()
    winner = info.get('winner', 0)
    
    print("="*50)
    if winner == human_player:
        print("Congratulations! You won! 🎉")
    elif winner == ai_player:
        print("AI won! Better luck next time! 🤖")
    else:
        print("It's a draw! 🤝")
    print("="*50)


def play_ai_vs_ai(agent1_file: str = 'agent1_final.pkl',
                 agent2_file: str = 'agent2_final.pkl',
                 num_games: int = 10):
    """
    Watch two AI agents play against each other
    
    Args:
        agent1_file: Path to first agent file
        agent2_file: Path to second agent file
        num_games: Number of games to play
    """
    # Initialize environment
    env = TicTacToe(board_size=5)
    
    # Load agents
    agent1 = QLearningAgent(player=1)
    agent2 = QLearningAgent(player=-1)
    
    try:
        agent1.load(agent1_file)
    except:
        print(f"Warning: Could not load agent1 from {agent1_file}")
    
    try:
        agent2.load(agent2_file)
    except:
        print(f"Warning: Could not load agent2 from {agent2_file}")
    
    print("\n" + "="*50)
    print("AI vs AI - Watching trained agents play")
    print("="*50)
    
    wins_1 = 0
    wins_2 = 0
    draws = 0
    
    for game in range(num_games):
        print(f"\n--- Game {game + 1}/{num_games} ---")
        
        # Reset game
        state = env.reset()
        done = False
        
        while not done:
            # Get current agent
            if env.current_player == 1:
                current_agent = agent1
            else:
                current_agent = agent2
            
            # Choose action
            available_actions = env.get_available_actions()
            action = current_agent.choose_action(state, available_actions, training=False)
            
            # Execute move
            state, reward, done, info = env.step(action)
        
        # Show final board
        env.render()
        
        # Update statistics
        winner = info.get('winner', 0)
        if winner == 1:
            wins_1 += 1
            print("Agent 1 (X) wins!")
        elif winner == -1:
            wins_2 += 1
            print("Agent 2 (O) wins!")
        else:
            draws += 1
            print("Draw!")
    
    # Print statistics
    print("\n" + "="*50)
    print("Statistics:")
    print(f"  Agent 1 (X) wins: {wins_1} ({100*wins_1/num_games:.1f}%)")
    print(f"  Agent 2 (O) wins: {wins_2} ({100*wins_2/num_games:.1f}%)")
    print(f"  Draws: {draws} ({100*draws/num_games:.1f}%)")
    print("="*50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'ai-vs-ai':
            play_ai_vs_ai()
        else:
            play_human_vs_ai()
    else:
        play_human_vs_ai()
