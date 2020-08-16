from warehouse import Warehouse, train
import pandas as pd
import matplotlib.pyplot as plt

l_rates = [0.1, 1.]

env = Warehouse(2, 2, 2, True)  # create warehouse
# n_shelve_units, unit_width, n_pick_pts
env.render()
n_episodes = 7000
qs = []

for l_rate in l_rates:
    _, r = train(env, n_episodes=n_episodes, n_steps=100, l_rate=l_rate, d_rate=0.99, max_e_rate=1,
                 min_e_rate=0.001, e_d_rate=0.0005)
    plt.plot(range(n_episodes), pd.DataFrame(r).rolling(100).mean().to_numpy(), label=str(l_rate))

plt.grid()
plt.legend(title="Learning rate:")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("../../basic_lres_sens.pdf")
plt.show()
