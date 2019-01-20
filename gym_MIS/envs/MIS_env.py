import gym
import numpy as np
import scipy.sparse as sp

# sample graph
# optimal solution is {0, 2, 3}
SAMPLE_GRAPH = sp.coo_matrix([
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
        self.to_vertex = np.arange(self.A.shape[0], dtype=np.int)
        self.ans = []
        self.reward = 0  # number of vertices already counted in the solution
        return self.A

    def step(self, action):  # action: index of a vertex
        self.ans.append(self.to_vertex[action])
        csr = self.A.tocsr()
        # delete neighbors
        mask = csr.getrow(action).toarray()[0] == 0
        # delete itself
        mask[action] = False
        self.to_vertex = self.to_vertex[mask]
        csr = csr[mask][:, mask]
        self.A = csr.tocoo()
        self.reward += 1
        assert self.A.shape[0] == 0 or self.A.shape[0] == self.A.shape[1]
        return self.A, self.reward, self.A.shape[0] == 0, {'ans': self.ans}

    def render(self, mode='human', close=False):
        print(self.A)

    def seed(self, seed=None):
        pass