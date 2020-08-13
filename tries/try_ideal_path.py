from warehouse import Warehouse
import numpy as np
import pandas as pd

ideal_steps = [0, 0, 0, 3, 1, 2, 2, 2]

l_rate = 0.75  # learning rate
d_rate = 0.99  # discount rate
e_rate = 1  # exploration rate
max_e_rate = 1
min_e_rate = 0.001
e_d_rate = 0.0005  # exploration decay route

env = Warehouse()

q_table = np.zeros((env.n_states, env.n_actions))

for step in ideal_steps:
    old_state_idx = env.states.index(env.state)
    new_state, reward, done = env.step(step)
    q_table[old_state_idx, step] = q_table[old_state_idx, step] * (1 - l_rate) \
        + l_rate * (reward + d_rate * np.max(q_table[env.states.index(new_state), :]))

q_table_pd = pd.DataFrame(q_table, index=env.states, columns=env.action_str).round(5)
print(q_table_pd)
