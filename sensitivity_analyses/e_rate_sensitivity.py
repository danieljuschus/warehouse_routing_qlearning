from warehouse import Warehouse, train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

e_decays = [0.001, 0.01, 1]

env = Warehouse(2, 2, 2, True)  # create warehouse
# n_shelve_units, unit_width, n_pick_pts
env.render()
n_episodes = 1000
qs = []

for e_decay in e_decays:
    # plt.plot(range(n_episodes), [0.001 + (1 - 0.001) * np.exp(-e_decay * e) for e in range(n_episodes)],
    #          label=str(e_decay))
    r, q = train(env, n_episodes=n_episodes, n_steps=100, l_rate=0.5, d_rate=0.99, max_e_rate=1,
                 min_e_rate=0.001, e_d_rate=e_decay)
    plt.plot(range(n_episodes), pd.DataFrame(r).rolling(100).mean().to_numpy(), label=str(e_decay))

plt.grid()
plt.legend(title="Exploration decay rate:")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("../basic_e_sens.pdf")
plt.show()
