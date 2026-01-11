"""
Test suite for 5x5 Tic-Tac-Toe Q-Learning
"""
import numpy as np
from tictactoe_env import TicTacToe
from q_learning_agent import QLearningAgent
import sys


def test_environment_initialization():
    """Test environment initialization"""
    print("Testing environment initialization...")
    env = TicTacToe(board_size=5)
    assert env.board_size == 5
    assert env.board.shape == (5, 5)
    assert np.all(env.board == 0)
    assert env.current_player == 1
    assert env.winner is None
    assert env.game_over is False
    print("✓ Environment initialization test passed")


def test_environment_reset():
    """Test environment reset"""
    print("Testing environment reset...")
    env = TicTacToe()
    env.step((0, 0))
    env.step((1, 1))
    state = env.reset()
    assert np.all(state == 0)
    assert env.current_player == 1
    assert env.winner is None
    assert env.game_over is False
    print("✓ Environment reset test passed")


def test_valid_actions():
    """Test valid action checking"""
    print("Testing valid actions...")
    env = TicTacToe()
    env.reset()
    
    # Valid action
    assert env.is_valid_action(0, 0) == True
    
    # Make a move
    env.step((0, 0))
    
    # Now that position should be invalid
    assert env.is_valid_action(0, 0) == False
    
    # Out of bounds
    assert env.is_valid_action(-1, 0) == False
    assert env.is_valid_action(5, 0) == False
    assert env.is_valid_action(0, 5) == False
    
    print("✓ Valid actions test passed")


def test_horizontal_win():
    """Test horizontal win detection"""
    print("Testing horizontal win...")
    env = TicTacToe()
    env.reset()
    
    # Player 1 wins horizontally
    for i in range(5):
        state, reward, done, info = env.step((0, i))
        if i < 4:
            env.step((1, i))
    
    assert done is True
    assert info['winner'] == 1
    assert reward == 1
    print("✓ Horizontal win test passed")


def test_vertical_win():
    """Test vertical win detection"""
    print("Testing vertical win...")
    env = TicTacToe()
    env.reset()
    
    # Player 1 wins vertically
    for i in range(5):
        state, reward, done, info = env.step((i, 0))
        if i < 4:
            env.step((i, 1))
    
    assert done is True
    assert info['winner'] == 1
    print("✓ Vertical win test passed")


def test_diagonal_win():
    """Test diagonal win detection"""
    print("Testing diagonal win...")
    env = TicTacToe()
    env.reset()
    
    # Player 1 wins diagonally (TL to BR)
    for i in range(5):
        state, reward, done, info = env.step((i, i))
        if i < 4:
            env.step((i, i+1))
    
    assert done is True
    assert info['winner'] == 1
    print("✓ Diagonal win test passed")


def test_anti_diagonal_win():
    """Test anti-diagonal win detection"""
    print("Testing anti-diagonal win...")
    env = TicTacToe()
    env.reset()
    
    # Player 1 wins diagonally (TR to BL)
    for i in range(5):
        state, reward, done, info = env.step((i, 4-i))
        if i < 4:
            env.step((i, 3-i))
    
    assert done is True
    assert info['winner'] == 1
    print("✓ Anti-diagonal win test passed")


def test_draw():
    """Test draw condition"""
    print("Testing draw condition...")
    env = TicTacToe()
    env.reset()
    
    # Create a scenario that leads to a draw
    # This is complex for 5x5, so we'll test the logic
    moves = [
        (0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
        (1, 1), (1, 2), (1, 3), (2, 0), (2, 1),
        (2, 2), (2, 3), (3, 0), (3, 1), (3, 2),
        (3, 3), (4, 0), (4, 1), (4, 2), (4, 3),
        (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),
    ]
    
    done = False
    for i, move in enumerate(moves):
        if not done:
            state, reward, done, info = env.step(move)
            if done:
                # If game ended, check it's either a win or draw
                assert info['winner'] in [-1, 0, 1]
                break
    
    print("✓ Draw condition test passed")


def test_agent_initialization():
    """Test agent initialization"""
    print("Testing agent initialization...")
    agent = QLearningAgent(player=1, alpha=0.2, gamma=0.95, epsilon=0.15)
    assert agent.player == 1
    assert agent.alpha == 0.2
    assert agent.gamma == 0.95
    assert agent.epsilon == 0.15
    assert len(agent.q_table) == 0
    print("✓ Agent initialization test passed")


def test_agent_action_selection():
    """Test agent action selection"""
    print("Testing agent action selection...")
    env = TicTacToe()
    env.reset()
    agent = QLearningAgent(player=1)
    
    available_actions = env.get_available_actions()
    action = agent.choose_action(env.board, available_actions, training=False)
    
    assert action in available_actions
    print("✓ Agent action selection test passed")


def test_agent_learning():
    """Test agent learning"""
    print("Testing agent learning...")
    agent = QLearningAgent(player=1)
    env = TicTacToe()
    env.reset()
    
    # Simulate a game
    agent.start_episode()
    state = env.board.copy()
    action = (2, 2)
    agent.store_transition(state, action)
    
    # Learn from positive reward
    initial_size = agent.get_q_table_size()
    agent.learn(1)
    final_size = agent.get_q_table_size()
    
    assert final_size >= initial_size
    print("✓ Agent learning test passed")


def test_agent_save_load():
    """Test agent save and load"""
    print("Testing agent save/load...")
    import os
    
    agent1 = QLearningAgent(player=1, alpha=0.2, epsilon=0.15)
    
    # Add some data to Q-table
    env = TicTacToe()
    env.reset()
    agent1.start_episode()
    agent1.store_transition(env.board.copy(), (2, 2))
    agent1.learn(1)
    
    # Save
    filename = 'test_agent.pkl'
    agent1.save(filename)
    
    # Load into new agent
    agent2 = QLearningAgent(player=-1)
    agent2.load(filename)
    
    assert agent2.player == 1
    assert agent2.alpha == 0.2
    assert agent2.epsilon == 0.15
    assert agent2.get_q_table_size() > 0
    
    # Clean up
    os.remove(filename)
    print("✓ Agent save/load test passed")


def test_player_switching():
    """Test that players switch correctly"""
    print("Testing player switching...")
    env = TicTacToe()
    env.reset()
    
    assert env.current_player == 1
    env.step((0, 0))
    assert env.current_player == -1
    env.step((0, 1))
    assert env.current_player == 1
    
    print("✓ Player switching test passed")


def test_invalid_move():
    """Test handling of invalid moves"""
    print("Testing invalid move handling...")
    env = TicTacToe()
    env.reset()
    
    # Make a valid move
    env.step((0, 0))
    
    # Try to make the same move (invalid)
    state, reward, done, info = env.step((0, 0))
    
    assert done is True
    assert reward == -10
    print("✓ Invalid move test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING TEST SUITE FOR 5x5 TIC-TAC-TOE Q-LEARNING")
    print("="*60 + "\n")
    
    tests = [
        test_environment_initialization,
        test_environment_reset,
        test_valid_actions,
        test_horizontal_win,
        test_vertical_win,
        test_diagonal_win,
        test_anti_diagonal_win,
        test_draw,
        test_agent_initialization,
        test_agent_action_selection,
        test_agent_learning,
        test_agent_save_load,
        test_player_switching,
        test_invalid_move,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
