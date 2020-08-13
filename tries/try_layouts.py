from warehouse import Warehouse
import numpy as np

for param in [(2, 1), (4, 1), (2, 4), (6, 2)]:
    env = Warehouse(*param)
    env.render()
    env.render_possible_actions()
