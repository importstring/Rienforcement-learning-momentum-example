import gymnasium as gym  # Use gymnasium instead of gym
import numpy as np

# Create the MountainCar environment
env = gym.make("MountainCar-v0", render_mode="human")

# Hyperparameters
learning_rate = 1.1
discount_factor = 0.1
epsilon = 0.01  # Exploration rate

# Discretization settings (similar to bucketing)
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Q-table initialized to random values between -2 and 0
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Discretize the continuous state into a bucket index
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

# Reset environment and get initial discrete state
discrete_state = get_discrete_state(env.reset()[0])  # reset() returns a tuple, access the first element

done = False

while not done:
    try:
        print(q_table)
    except Exception:
        pass  # Printing the Q-table may cause an error in a notebook environment
    # Exploration vs. exploitation
    if np.random.random() > epsilon:
        # Exploit: choose action with the highest Q-value
        action = np.argmax(q_table[discrete_state])
    else:
        # Explore: choose a random action
        action = np.random.randint(0, env.action_space.n)
    
    # Step the environment
    new_state, reward, done, truncated, info = env.step(action)  # New API provides 'truncated'
    new_discrete_state = get_discrete_state(new_state)
    goal_position = env.unwrapped.goal_position
    if not done:
        # Update Q-table using Q-learning algorithm
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action,)]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[discrete_state + (action,)] = new_q
    elif new_state[0] >= env.goal_position:
        # Reached the goal: reward is 0, set Q-value to 0
        q_table[discrete_state + (action,)] = 0
    
    # Move to the next state
    discrete_state = new_discrete_state
    
    # Render the environment
    env.render()

env.close()