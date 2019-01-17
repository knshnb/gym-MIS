import gym
import numpy as np

# sample graph
# optimal solution is {0, 2, 3}
SAMPLE_GRAPH = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
], dtype=np.bool)

"""
graph: original graph
A: current subgraph
to_vertex: index of A -> index of graph
ans: current answer for MIS
reward: current number of vertices in answer
"""
class MISEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        # not implemented!
        # self.action_space = N
        # self.observation_space = []
        # self.reward_range = []
        pass

    def set_graph(self, graph=SAMPLE_GRAPH):
        self.graph = graph
        self.reset()

    def reset(self):
        self.A = self.graph
        self.to_vertex = np.arange(self.A.shape[0])
        self.ans = []
        self.reward = 0  # number of vertices already counted in the solution
        return self.A

    def step(self, action):  # action: index of a vertex
        self.ans.append(self.to_vertex[action])
        # delete neighbors
        remain = self.A[action] == 0
        # delete itself
        remain[action] = False
        self.to_vertex = self.to_vertex[remain]
        self.A = self.A[remain][:, remain]
        self.reward += 1
        assert self.A.shape[0] == 0 or self.A.shape[0] == self.A.shape[1]
        return self.A, self.reward, self.A.shape[0] == 0, {'ans': self.ans}

    def render(self, mode='human', close=False):
        print(self.A)

    def seed(self, seed=None):
        pass