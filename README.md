# Tic-Tac-Toe RL Q-Learning (5x5 Board)

A reinforcement learning implementation of 5x5 Tic-Tac-Toe using Q-Learning algorithm. Train AI agents to play the game through self-play and compete against them!

## Features

- **5x5 Board**: Extended Tic-Tac-Toe with 5-in-a-row win condition
- **Q-Learning Algorithm**: Reinforcement learning using tabular Q-learning
- **Self-Play Training**: Agents learn by playing against each other
- **Interactive Play**: Play against trained AI agents
- **AI vs AI Mode**: Watch trained agents compete
- **Model Persistence**: Save and load trained agents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ABDUL200408/Tic_tac_toe_RL_Q_Learning.git
cd Tic_tac_toe_RL_Q_Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Agents

Train two agents through self-play:

```bash
python main.py train
```

With custom parameters:
```bash
python main.py train --episodes 100000 --alpha 0.1 --gamma 0.9 --epsilon 0.1
```

Parameters:
- `--episodes`: Number of training episodes (default: 50000)
- `--alpha`: Learning rate (default: 0.1)
- `--gamma`: Discount factor (default: 0.9)
- `--epsilon`: Exploration rate (default: 0.1)
- `--save-interval`: Save checkpoint every N episodes (default: 10000)

### Playing Against AI

Play against a trained agent:

```bash
python main.py play
```

Options:
- `--agent`: Path to agent file (default: agent1_final.pkl)
- `--ai-first`: AI plays first (default: human plays first)

Example:
```bash
python main.py play --agent agent1_final.pkl --ai-first
```

### Watch AI vs AI

Watch two trained agents play against each other:

```bash
python main.py watch
```

Options:
- `--agent1`: Path to first agent (default: agent1_final.pkl)
- `--agent2`: Path to second agent (default: agent2_final.pkl)
- `--games`: Number of games to play (default: 10)

Example:
```bash
python main.py watch --games 20
```

## How It Works

### Game Environment

The 5x5 Tic-Tac-Toe environment (`tictactoe_env.py`):
- Players alternate placing marks (X and O) on a 5x5 grid
- First player to get 5 marks in a row (horizontal, vertical, or diagonal) wins
- Game ends in a draw if the board is full with no winner

### Q-Learning Agent

The Q-Learning agent (`q_learning_agent.py`) learns by:
1. **Exploration vs Exploitation**: Using epsilon-greedy policy to balance trying new moves vs using learned knowledge
2. **Q-Value Updates**: Learning which state-action pairs lead to wins
3. **Temporal Credit Assignment**: Propagating rewards backwards through the game

The Q-learning update rule:
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

Where:
- `α` (alpha): Learning rate - how much new information overrides old
- `γ` (gamma): Discount factor - importance of future rewards
- `ε` (epsilon): Exploration rate - probability of random action

### Training Process

During training (`train.py`):
1. Two agents play against each other
2. Each agent stores its state-action pairs during the game
3. After the game ends, both agents update their Q-tables based on the outcome
4. Winners get positive reward (+1), losers get negative (-1), draws get 0
5. Q-values are propagated backwards through the game trajectory

## File Structure

```
.
├── main.py                 # Main entry point
├── tictactoe_env.py        # Game environment
├── q_learning_agent.py     # Q-Learning agent implementation
├── train.py                # Training module
├── play.py                 # Interactive play module
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Examples

### Quick Start

1. Train agents (this may take a few minutes):
```bash
python main.py train --episodes 50000
```

2. Play against the trained AI:
```bash
python main.py play
```

3. Enter moves as row and column numbers (0-4):
```
Enter your move (row col): 2 2
```

### Training Output

```
Starting training...
Episodes: 50000, Alpha: 0.1, Gamma: 0.9, Epsilon: 0.1

Episode 10000/50000
  Agent 1 (X) wins: 4523 (45.2%)
  Agent 2 (O) wins: 4892 (48.9%)
  Draws: 585 (5.9%)
  Agent 1 Q-table size: 45234
  Agent 2 Q-table size: 46891
```

## Technical Details

### State Representation

- Board state is represented as a numpy array of shape (5, 5)
- Values: 1 (Player X), -1 (Player O), 0 (empty)
- States are normalized to current player's perspective for Q-table

### Action Space

- Actions are (row, col) tuples
- Valid actions are empty cells on the board
- 25 possible positions initially, decreasing as game progresses

### Reward Structure

- Win: +1
- Loss: -1
- Draw: 0
- Invalid move: -10 (game ends)

## Performance

Training 50,000 episodes typically results in:
- Q-table size: ~50,000-100,000 states per agent
- Win rate: ~45-50% for each agent (balanced)
- Draw rate: ~5-10%
- Training time: 5-10 minutes on modern CPU

## Future Improvements

- Implement neural network-based Q-learning (DQN)
- Add different board sizes
- Implement other RL algorithms (SARSA, Actor-Critic)
- Add GUI interface
- Optimize state representation for faster lookup

## License

MIT License

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
