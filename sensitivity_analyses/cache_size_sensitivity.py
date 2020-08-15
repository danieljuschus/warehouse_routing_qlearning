from warehouse import Warehouse
from warehouse_parallel import train_parallel
import matplotlib.pyplot as plt

env = Warehouse(8, 5, 5)
env.render()
n_proc = 4
n_episodes = 7000
update_interval = 1000

exec_times = []

plt.figure(1)
cache_sizes = range(5, 100, 20)
for cache_size in cache_sizes:
    _, rewards, exec_time = train_parallel(env, n_proc=n_proc, update_interval=update_interval, n_episodes=n_episodes,
                                           cache_size=cache_size, n_steps=100, l_rate=0.7, d_rate=0.99, max_e_rate=1,
                                           min_e_rate=0.001, e_d_rate=0.2)
    exec_times.append(exec_time)
    plt.plot(range(5, len(rewards)*update_interval, update_interval), rewards, label=str(cache_size))

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend(title="Cache size:")
plt.grid()
plt.show()

plt.figure(2)
plt.plot(cache_sizes, exec_times)
plt.xlabel("Cache size")
plt.ylabel("Execution time in s")
plt.grid()
plt.show()
