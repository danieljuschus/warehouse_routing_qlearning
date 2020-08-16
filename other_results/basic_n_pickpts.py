import time
from warehouse import Warehouse, train
import matplotlib.pyplot as plt

times = []
ns = []
nps = range(1, 6)

for n_pick_pts in nps:
    env = Warehouse(4, 4, n_pick_pts)
    s = time.time()
    train(env, n_episodes=1000, n_steps=100, l_rate=1., d_rate=1., max_e_rate=1, min_e_rate=0.001, e_d_rate=0.1)
    times.append(time.time()-s)
    ns.append(env.n_states)

fig, ax1 = plt.subplots()

ax1.plot(nps, times, color="red")
ax1.set_xlabel("Number of pick points")
ax1.set_xticks(nps)
ax1.set_ylabel("Execution time in seconds", color="red")
ax1.tick_params(axis='y', labelcolor="red")
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(nps, ns, color="blue")
ax2.set_ylabel("Number of states", color="blue")
ax2.tick_params(axis='y', labelcolor="blue")

plt.savefig("../../pickpt_vs_exectime.pdf")
plt.show()
