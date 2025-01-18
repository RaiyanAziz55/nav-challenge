import sys
from dqn_agent import DQNAgent




def get_ai_recommendation(env, agent):
    """
    Generate AI recommendation for the given Tetris environment.
    Args:
        env (Tetris): The Tetris game environment.
        agent (DQNAgent): The AI agent.
    Returns:
        tuple: Recommended action (position, rotation).
    """
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    return best_action