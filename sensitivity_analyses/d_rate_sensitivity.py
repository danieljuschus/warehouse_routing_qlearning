from warehouse import Warehouse, train
import pandas as pd
import matplotlib.pyplot as plt

d_rates = [0., 0.00001, 0.99, 1.]

env = Warehouse(2, 2, 2, True)  # create warehouse
# n_shelve_units, unit_width, n_pick_pts
env.render()
n_episodes = 7000
qs = []

for d_rate in d_rates:
    r, q = train(env, n_episodes=n_episodes, n_steps=100, l_rate=0.5, d_rate=d_rate, max_e_rate=1,
                 min_e_rate=0.001, e_d_rate=0.0005)
    qs.append(pd.DataFrame(q, index=[str(state) for state in env.states], columns=env.action_str).round(3))
    plt.plot(range(n_episodes), pd.DataFrame(r).rolling(100).mean().to_numpy(), label=str(d_rate))

plt.grid()
plt.legend(title="Discount rate:")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("../basic_d_sens.pdf")
plt.show()
