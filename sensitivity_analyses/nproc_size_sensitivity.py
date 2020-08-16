from warehouse import Warehouse
from warehouse_parallel import train_parallel
import matplotlib.pyplot as plt

env = Warehouse(8, 5, 5, False)
env.render()
cache_size = 10
n_episodes = 20000
update_interval = 75

exec_times = []

plt.figure(1)
nprocs = range(1, 7)
for n_proc in nprocs:
    _, rewards, exec_time = train_parallel(env, n_proc=n_proc, update_interval=update_interval, n_episodes=n_episodes,
                                           cache_size=cache_size, n_steps=100, l_rate=0.9, d_rate=0.99, max_e_rate=1,
                                           min_e_rate=0.001, e_d_rate=0.005, r_threshold=110)
    exec_times.append(exec_time)
    plt.plot(range(5, len(rewards)*update_interval, update_interval), rewards, label=str(n_proc))
    print(max(rewards))

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend(title="Number of workers:")
plt.grid()
plt.savefig("../../nproc_rewards.pdf")
plt.show()

plt.figure(2)
plt.plot(nprocs, exec_times)
plt.xlabel("Number of workers")
plt.ylabel("Execution time in s")
plt.grid()
plt.savefig("../../nproc_exec_time.pdf")
plt.show()
