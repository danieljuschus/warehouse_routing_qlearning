from warehouse import Warehouse
import time

env = Warehouse()
print(env.state)

steps_winning = [0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3]
steps_missing_one = [0, 0, 0, 3, 3, 2, 2, 2, 1, 1]*10000

start_time = time.time()

for step in steps_winning:
    env.step(step)

env.reset()
print(env.state)
for step in steps_missing_one:
    env.step(step)

print("Done in {} seconds.".format(round(time.time()-start_time, 6)))