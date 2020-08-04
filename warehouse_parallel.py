import time
import random
import ray
import numpy as np
from more_itertools import split_into, powerset
from warehouse import Warehouse, q_table_to_action_list


@ray.remote
class Main:
    """ Main process. Contains global q-table. Receives local q-tables and sends max q values for a given state to
    worker process."""
    def __init__(self, env):
        self.env = env
        self.n_episodes = n_episodes
        self.q_table = np.zeros((self.env.n_states, self.env.n_actions))
        self.cache = []

    def receive_q_loc(self, qt_loc, states):
        """Receive local q table from worker and update global qt."""
        self.q_table[self.env.states.index(states[0]):self.env.states.index(states[-1])+1] = qt_loc

    def send_max_q(self, state):
        """Get max q value of state from global qt (send to worker)."""
        return np.max(self.q_table[self.env.states.index(state), :])

    def send_q_table(self):
        """ Send final q-table."""
        return self.q_table


@ray.remote
class Worker:
    """ Worker process. Performs actions in its own part of the state space and communicates with Main periodically."""
    def __init__(self, states, possible_actions, corridors, grid_size, start_global, pick_pts_global, main, id):
        self.states = states
        self.grid_size = grid_size
        self.possible_actions = possible_actions
        self.corridors = corridors
        self.pick_pts_global = pick_pts_global  # use global variable instead?
        self.qt_loc = np.zeros((len(self.states), 4))
        self.start_global = start_global  # use global variable instead?
        self.start = self.corridors[0]
        self.position = self.start
        self.state = (self.position, ())
        self.main = main
        self.id = id

    def reset(self):
        """Reset local state"""
        self.state = random.choice(self.states)  # pick random state to start the episode

    def step(self, action):
        """Take step, update qt loc."""
        done = False
        # Determine new position
        if action in self.possible_actions[np.where(self.corridors == self.state[0])[0][0]]:  # if action is valid for
            # current field
            reward = -1  # general reward for motion
            if action == 0:  # up
                new_position = self.state[0] - self.grid_size[1]
            if action == 1:  # right
                new_position = self.state[0] + 1
            if action == 2:  # down
                new_position = self.state[0] + self.grid_size[1]
            if action == 3:  # left
                new_position = self.state[0] - 1
        else:  # action was invalid, so no movement
            reward = -2
            new_position = self.state[0]
        self.position = new_position

        # Check if new position is above or below unvisited point
        if new_position + self.grid_size[1] in np.setdiff1d(self.pick_pts_global, self.state[1]):  # if above
            new_pick_state = tuple(sorted(self.state[1] + tuple([new_position + self.grid_size[1]])))
            reward = 10
        elif new_position - self.grid_size[1] in np.setdiff1d(self.pick_pts_global, self.state[1]):  # if below
            new_pick_state = tuple(sorted(self.state[1] + tuple([new_position - self.grid_size[1]])))
            reward = 10
        elif new_position == self.start_global and not np.setdiff1d(self.pick_pts_global, self.state[1]).any():  # if
            # done
            new_pick_state = self.state[1]
            reward = 100
            done = True
        else:  # don't change pick state
            new_pick_state = self.state[1]

        # Update state
        old_state_idx = self.states.index(self.state)
        new_state = (new_position, new_pick_state)
        self.state = new_state

        # Check if new position is local
        if new_position in [state[0] for state in self.states]:  # determine max next from local q table
            max_next = np.max(self.qt_loc[self.states.index(new_state), :])
        else:  # finish episode and get max next from global q table from main
            done = True
            max_next = self.main.send_max_q.remote(new_state)  # no cache for now
            max_next = ray.get(max_next)

        # Update local q-table
        self.qt_loc[old_state_idx, action] = self.qt_loc[old_state_idx, action] * (1 - l_rate) \
            + l_rate * (reward + d_rate * max_next)

        return done, reward

    def train(self):
        """Run local training process"""
        e_rate = 1
        update_interval = 50
        rewards = []
        # For each episode
        for episode in range(n_episodes):
            self.reset()
            rewards_current = 0

            # For each step
            for step in range(n_steps):
                # Pick action
                if random.uniform(0, 1) > e_rate:  # exploit
                    action = np.argmax(self.qt_loc[self.states.index(self.state), :])  # pick best action from
                    # current state
                else:  # explore
                    action = random.choice(env.actions)  # choose random action

                # Take action
                done, reward = self.step(action=action)
                rewards_current += reward

                # Break loop if done
                if done:
                    break

            # Add episode reward to list
            rewards.append(rewards_current)

            # Update exploration rate
            e_rate = min_e_rate + (max_e_rate - min_e_rate) * np.exp(-e_d_rate * episode)

            # Update main q-table periodically
            if not episode % update_interval:
                self.main.receive_q_loc.remote(qt_loc=self.qt_loc, states=self.states)


env = Warehouse(8, 5, 5)
n_proc = 6
n_episodes = 5000

n_steps = 100
l_rate = 0.5  # learning rate
d_rate = 0.99  # discount rate
max_e_rate = 1
min_e_rate = 0.001
e_d_rate = 0.0005  # exploration decay rate

corridors_split = np.array_split(env.corridors, n_proc)
possible_actions_split = list(split_into(env.possible_actions, [len(c) for c in corridors_split]))
states_split = list(split_into(env.states, [len(c)*len(list(powerset(env.pick_pts))) for c in corridors_split]))

ray.init()
main = Main.remote(env=env)
workers = [Worker.remote(states=states_slice, possible_actions=actions_slice, corridors=corridors_slice, id=i,
                         grid_size=env.grid_size, start_global=env.start, pick_pts_global=env.pick_pts, main=main)
           for i, (states_slice, actions_slice, corridors_slice)
           in enumerate(zip(states_split, possible_actions_split, corridors_split))]
s = time.time()
ray.get([worker.train.remote() for worker in workers])
q_table_final = ray.get(main.send_q_table.remote())
e = time.time() - s
print("Finished in {} seconds. Determining optimal actions from q-table...".format(e))
ray.shutdown()
actions_final = q_table_to_action_list(q_table_final, env)
