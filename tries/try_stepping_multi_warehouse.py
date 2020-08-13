from warehouse_multiagent import Warehouse

env = Warehouse()

print(env.state)

env.render()

new_state, reward, done = env.step((1, 3))

print(env.state, done)

env.render()


new_state, reward, done = env.step((3, 1))

print(env.state, done)

env.render()
