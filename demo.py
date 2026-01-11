"""
Example/Demo script for 5x5 Tic-Tac-Toe Q-Learning

This script demonstrates the key features of the implementation.
"""

def demo_environment():
    """Demonstrate the game environment"""
    print("\n" + "="*60)
    print("DEMO 1: Game Environment")
    print("="*60)
    
    from tictactoe_env import TicTacToe
    
    env = TicTacToe(board_size=5)
    print("\n✓ Created 5x5 Tic-Tac-Toe environment")
    
    env.reset()
    print("\n✓ Reset game - empty board:")
    env.render()
    
    # Simulate a winning game
    print("Simulating a horizontal win scenario...")
    moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4)]
    
    for move in moves:
        state, reward, done, info = env.step(move)
        if done:
            print(f"\n✓ Game ended! Winner: {info.get('winner')}")
            env.render()
            break


def demo_agent():
    """Demonstrate the Q-learning agent"""
    print("\n" + "="*60)
    print("DEMO 2: Q-Learning Agent")
    print("="*60)
    
    from q_learning_agent import QLearningAgent
    from tictactoe_env import TicTacToe
    import numpy as np
    
    # Create agent
    agent = QLearningAgent(player=1, epsilon=0.1)
    print("\n✓ Created Q-Learning agent")
    print(f"  - Player: {agent.player}")
    print(f"  - Learning rate (alpha): {agent.alpha}")
    print(f"  - Discount factor (gamma): {agent.gamma}")
    print(f"  - Exploration rate (epsilon): {agent.epsilon}")
    
    # Demonstrate action selection
    env = TicTacToe()
    env.reset()
    available_actions = env.get_available_actions()
    
    print(f"\n✓ Agent can choose from {len(available_actions)} available actions")
    
    action = agent.choose_action(env.board, available_actions, training=True)
    print(f"✓ Agent selected action: {action}")


def demo_quick_training():
    """Demonstrate quick training"""
    print("\n" + "="*60)
    print("DEMO 3: Quick Training (1000 episodes)")
    print("="*60)
    
    from train import train_agents
    
    print("\n⏳ Training two agents through self-play...")
    print("   (This demonstrates the training process)")
    
    agent1, agent2 = train_agents(
        episodes=1000,
        save_interval=500,
        verbose=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"  - Agent 1 learned {agent1.get_q_table_size()} states")
    print(f"  - Agent 2 learned {agent2.get_q_table_size()} states")


def demo_ai_game():
    """Demonstrate AI vs AI game"""
    print("\n" + "="*60)
    print("DEMO 4: AI vs AI Game")
    print("="*60)
    
    from play import play_ai_vs_ai
    
    print("\n⏳ Watching trained agents play...")
    play_ai_vs_ai(num_games=3)


def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        ("Train agents", "python main.py train --episodes 50000"),
        ("Train with custom parameters", "python main.py train --episodes 100000 --alpha 0.15 --epsilon 0.2"),
        ("Play against AI", "python main.py play"),
        ("AI plays first", "python main.py play --ai-first"),
        ("Watch AI vs AI", "python main.py watch --games 10"),
    ]
    
    for desc, cmd in examples:
        print(f"\n• {desc}:")
        print(f"  {cmd}")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("5x5 TIC-TAC-TOE Q-LEARNING - DEMONSTRATION")
    print("="*60)
    
    try:
        demo_environment()
        demo_agent()
        demo_quick_training()
        demo_ai_game()
        show_usage_examples()
        
        print("\n" + "="*60)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now:")
        print("1. Train agents: python main.py train")
        print("2. Play against AI: python main.py play")
        print("3. Watch AI games: python main.py watch")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
