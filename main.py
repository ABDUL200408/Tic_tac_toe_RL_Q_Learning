"""
Main entry point for Tic-Tac-Toe RL Q-Learning
"""
import argparse
from train import train_agents
from play import play_human_vs_ai, play_ai_vs_ai


def main():
    parser = argparse.ArgumentParser(description='5x5 Tic-Tac-Toe with Q-Learning')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train Q-Learning agents')
    train_parser.add_argument('--episodes', type=int, default=50000,
                            help='Number of training episodes (default: 50000)')
    train_parser.add_argument('--alpha', type=float, default=0.1,
                            help='Learning rate (default: 0.1)')
    train_parser.add_argument('--gamma', type=float, default=0.9,
                            help='Discount factor (default: 0.9)')
    train_parser.add_argument('--epsilon', type=float, default=0.1,
                            help='Exploration rate (default: 0.1)')
    train_parser.add_argument('--save-interval', type=int, default=10000,
                            help='Save interval in episodes (default: 10000)')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against trained AI')
    play_parser.add_argument('--agent', type=str, default='agent1_final.pkl',
                           help='Path to agent file (default: agent1_final.pkl)')
    play_parser.add_argument('--ai-first', action='store_true',
                           help='AI plays first')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch AI vs AI')
    watch_parser.add_argument('--agent1', type=str, default='agent1_final.pkl',
                            help='Path to first agent file')
    watch_parser.add_argument('--agent2', type=str, default='agent2_final.pkl',
                            help='Path to second agent file')
    watch_parser.add_argument('--games', type=int, default=10,
                            help='Number of games to play (default: 10)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_agents(
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            save_interval=args.save_interval,
            verbose=True
        )
    elif args.command == 'play':
        play_human_vs_ai(
            agent_file=args.agent,
            human_first=not args.ai_first
        )
    elif args.command == 'watch':
        play_ai_vs_ai(
            agent1_file=args.agent1,
            agent2_file=args.agent2,
            num_games=args.games
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
