from warehouse import Warehouse, train
from warehouse_parallel import train_parallel
from time import time
import matplotlib.pyplot as plt
import pandas as pd

env = Warehouse(8, 5, 6, False)
env.render()
n_episodes_max = 20000

n_proc = 4
update_interval = 100
cache_size = 10

s = time()
actions_seq, rewards_seq = train(env, n_episodes=n_episodes_max, n_steps=100, l_rate=1., d_rate=1., max_e_rate=1,
                                 min_e_rate=0.001, e_d_rate=0.005, r_threshold=117)
e_seq = time() - s

actions_par, rewards_par, e_par = train_parallel(env, n_proc=n_proc, update_interval=update_interval,
                                                 n_episodes=n_episodes_max, cache_size=cache_size, n_steps=100,
                                                 l_rate=1, d_rate=1., max_e_rate=1, min_e_rate=0.001,
                                                 e_d_rate=0.005, r_threshold=117)

print("Sequential done in {}s, parallel done in {}s.".format(round(e_seq, 3), round(e_par, 3)))

plt.plot(range(len(rewards_seq)), pd.DataFrame(rewards_seq).rolling(5).mean().to_numpy(), label="Sequential")
plt.plot(range(0, len(rewards_par)*update_interval, update_interval), rewards_par, label="Parallel")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.show()
