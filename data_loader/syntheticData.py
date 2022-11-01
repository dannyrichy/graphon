"""
Helps generate synthetic dataset for validating different task
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import DEVICE


class SynthGraphons:
    """
    Class to generate synthetic data_loader. Graphon probability functions for each graphons

    - graphon_1 : f(u,v) -> u*v
    - graphon_2 : f(u,v) -> exp{-(u^0.7 + v^0.7))}
    - graphon_3 : f(u,v) -> (1/4) * [u^2 + v^2 + u^(1/2) + v^(1/2)]
    - graphon_4 : f(u,v) -> 0.5 * (u + v)
    - graphon_5 : f(u,v) -> 1 / (1 + exp(-10 * (u^2 + v^2)))
    - graphon_6 : f(u,v) -> w(u,v) = |u - v|
    - graphon_7 : f(u,v) -> 1 / (1 + exp(-(max(u,v)^2 + min(u,v)^4)))
    - graphon_8 : f(u,v) -> exp(-max(u, v)^(3/4))
    - graphon_9 : f(u,v) -> exp(-0.5 * (min(u, v) + u^0.5 + v^0.5))
    - graphon_10: f(u,v) -> log(1 + 0.5 * max(u, v))
    """

    graphs = list()
    labels = list()

    def __init__(self, num_nodes, num_graphs, graphons_keys):
        """
        :param num_graphs:
        :type num_graphs:
        :param graphons_keys:
        :type graphons_keys:
        """
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.graphons_keys = [int(item) for item in graphons_keys]  # 0 to 9

    @staticmethod
    def graphon_1(u, v):
        """
        u*v

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        graphon = u * v
        return graphon

    @staticmethod
    def graphon_2(u, v):
        """
        exp{-(u^0.7 + v^0.7))}

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return torch.exp(-(torch.pow(u, 0.7) + torch.pow(v, 0.7)))

    @staticmethod
    def graphon_3(u, v):
        """
        (1/4) * [u^2 + v^2 + u^(1/2) + v^(1/2)]

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return 0.25 * (torch.pow(u, 2) + torch.pow(v, 2) + torch.pow(u, 0.5) + torch.pow(u, 0.5))

    @staticmethod
    def graphon_4(u, v):
        """
        0.5 * (u + v)

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return 0.5 * (u + v)

    @staticmethod
    def graphon_5(u, v):
        """
        1 / (1 + exp(-10 * (u^2 + v^2)))

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return 1 / (1 + torch.exp(-10 * (torch.pow(u, 2) + torch.pow(v, 2))))

    @staticmethod
    def graphon_6(u, v):
        """
        w(u,v) = |u - v|

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return torch.abs(u - v)

    @staticmethod
    def graphon_7(u, v):
        """
        1 / (1 + exp(-(max(u,v)^2 + min(u,v)^4)))

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return 1 / (1 + torch.exp(-(torch.pow(torch.max(u, v), 2) + torch.pow(torch.min(u, v), 4))))

    @staticmethod
    def graphon_8(u, v):
        """
        exp(-max(u, v)^(3/4))

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return torch.exp(-torch.pow(torch.max(u, v), 0.75))

    @staticmethod
    def graphon_9(u, v):
        """
        exp(-0.5 * (min(u, v) + u^0.5 + v^0.5))

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return torch.exp(-0.5 * (torch.min(u, v) + torch.pow(u, 0.5) + torch.pow(v, 0.5)))

    @staticmethod
    def graphon_10(u, v):
        """
        log(1 + 0.5 * max(u, v))

        :param u: Node value as tensors
        :type u: torch.Tensor

        :param v: Node value as tensors
        :type v: torch.Tensor

        :return: Probability matrix
        :rtype: torch.Tensor
        """
        return torch.log(1 + 0.5 * torch.max(u, v))

    def _generate_graphs(self, graphon_key, n):
        """

        :param graphon_key:
        :type graphon_key:

        :param n:
        :type n:

        :return:
        :rtype:
        """
        graph_gen = []

        for nn in n:
            x = torch.distributions.uniform.Uniform(0, 1).sample([nn]).to(device=DEVICE)
            p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
            u = p + x.reshape(1, -1)
            v = p + x.reshape(-1, 1)
            graph_prob = eval('self.graphon_' + str(graphon_key) + '(u, v)')

            graph = torch.distributions.binomial.Binomial(1, graph_prob).sample()
            graph = torch.triu(graph, diagonal=1)
            graph = graph + graph.t()
            graph_gen.append(graph)

        return graph_gen

    def data_simulation(self, start=100, stop=1000):
        """
        Simulate data_loader for the data_loader model

        :param start: start of the range of the number of nodes
        :type start: int

        :param stop: end of the range of the number of nodes
        :type stop: int


        :return: list of graphs and list of labels
        :rtype: list, list
        """
        for graphon in tqdm(self.graphons_keys):
            p = torch.randperm(stop)
            if self.num_nodes == 'None':
                n = p[p > start][:self.num_graphs]
            else:
                n = [self.num_nodes] * self.num_graphs
            g = self._generate_graphs(graphon, n)
            self.graphs = self.graphs + g

        for i in range(len(self.graphons_keys)):
            _label = int(i) * np.ones(self.num_graphs)
            self.labels = self.labels + _label.astype(int).tolist()

        return self.graphs, self.labels

    def save_graphs(self, path):
        """
        Save graphs and labels to a file
        """
        print('Storing graphs at ', path)
        with open(path, 'wb') as f:
            pickle.dump((self.graphs, self.labels), f)

    @staticmethod
    def load_graphs(path):
        """
        Load graphs and labels from a file

        :return: list of graphs and list of labels
        :rtype: list, list
        """
        print('Loading graphs from ', path)
        with open(path, 'rb') as f:
            graphs, labels = pickle.load(f)
        return graphs, labels

    @staticmethod
    def plot_graphons(key):
        """
        Plot the data_loader functions

        :param key: the key of the data_loader function
        :type key: int

        :return: Noothing
        :rtype: None
        """
        x = torch.linspace(0, 1, 1000)
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        y = eval('self.graphon_' + str(key) + '(u, v)')
        func_name = eval('self.graphon_' + str(key) + '.__doc__').split('\n')[1].lstrip()
        plt.title('Graphon ' + str(key) + f' --> ({func_name})')
        plt.imshow(y, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
