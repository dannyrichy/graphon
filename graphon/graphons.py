import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SynthGraphons():
    
    def __init__(self, num_graphs, graphons_keys, num_nodes, save_graphons_loc): # both graphons_keys and num_nodes are lists
                                                         
        self.num_graphs = num_graphs
        self.graphons_keys = [int(item) for item in graphons_keys] # 0 to 9
        self.name = ''
        self.num_nodes = num_nodes
        self.save_graphons_loc = save_graphons_loc
        
        # graphons for simulated data
    def graphon_1(self, x):
        self.name = 'w(u,v) = u * v'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = u * v
        return graphon


    def graphon_2(self, x):
        self.name = 'w(u,v) = exp{-(u^0.7 + v^0.7))}'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = torch.exp(-(torch.pow(u, 0.7) + torch.pow(v, 0.7)))
        return graphon


    def graphon_3(self, x):
        self.name = 'w(u,v) = (1/4) * [u^2 + v^2 + u^(1/2) + v^(1/2)]'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = 0.25 * (torch.pow(u, 2) + torch.pow(v, 2) + torch.pow(u, 0.5) + torch.pow(u, 0.5))
        return graphon


    def graphon_4(self, x):
        self.name = 'w(u,v) = 0.5 * (u + v)'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = 0.5 * (u + v)
        return graphon


    def graphon_5(self, x):
        self.name = 'w(u,v) = 1 / (1 + exp(-10 * (u^2 + v^2)))'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = 1 / (1 + torch.exp(-10 * (torch.pow(u, 2) + torch.pow(v, 2))))
        return graphon


    def graphon_6(self, x):
        self.name = 'w(u,v) = |u - v|'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = torch.abs(u - v)
        return graphon


    def graphon_7(self, x):
        self.name =  'w(u,v) = 1 / (1 + exp(-(max(u,v)^2 + min(u,v)^4)))'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = 1 / (1 + torch.exp(-(torch.pow(torch.max(u, v), 2) + torch.pow(torch.min(u, v), 4))))
        return graphon


    def graphon_8(self, x):
        self.name = 'w(u,v) = exp(-max(u, v)^(3/4))'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = torch.exp(-torch.pow(torch.max(u, v), 0.75))
        return graphon


    def graphon_9(self, x):
        self.name = 'w(u,v) = exp(-0.5 * (min(u, v) + u^0.5 + v^0.5))'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = torch.exp(-0.5 * (torch.min(u, v) + torch.pow(u, 0.5) + torch.pow(v, 0.5)))
        return graphon


    def graphon_10(self, x):
        self.name = 'w(u,v) = log(1 + 0.5 * max(u, v))'
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=DEVICE)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        graphon = torch.log(1 + 0.5 * torch.max(u, v))
        return graphon
     

    def _generate_graphs(self, graphon_key, n):
        graph_gen = []

        for nn in n:
            x = torch.distributions.uniform.Uniform(0, 1).sample([nn]).to(device=DEVICE)
            graph_prob = eval('self.graphon_' + str(graphon_key+1) + '(x)')

            graph = torch.distributions.binomial.Binomial(1, graph_prob).sample()
            graph = torch.triu(graph, diagonal=1)
            graph = graph + graph.t()
            graph_gen.append(graph)

        return graph_gen


    def data_simulation(self, start=100, stop=1000, save=False):
        """
        Simulate data for the graphon model

        :param start: start of the range of the number of nodes
        :type start: int

        :param stop: end of the range of the number of nodes
        :type stop: int

        :return: list of graphs and list of labels
        :rtype: list, list
        """
        graphs = []
        labels = []
        for graphon in tqdm(self.graphons_keys):
            p = torch.randperm(stop)
            if self.num_nodes == 'None':
                n = p[p > start][:self.num_graphs]
            else:
                n = [self.num_nodes] * self.num_graphs
            #print('nodes ', n)
            g = self._generate_graphs(graphon, n)
            graphs = graphs + g

        for i in range(len(self.graphons_keys)):
            l = i * np.ones(self.num_graphs)
            labels = labels + l.tolist()
        #print('graphs generated', len(graphs))
        #print('true labels ', labels)
        if save:
            self.save_graphs(graphs, labels)
        
        return graphs, labels

    def save_graphs(self, graphs, labels):
        """
        Save graphs and labels to a file

        :param graphs: list of graphs
        :type graphs: list

        :param labels: list of labels
        :type labels: list
        """
        with open(self.save_graphons_loc, 'wb') as f:
            pickle.dump((graphs, labels), f)
    
    def load_graphs(self):
        """
        Load graphs and labels from a file

        :return: list of graphs and list of labels
        :rtype: list, list
        """
        with open(self.save_graphons_loc, 'rb') as f:
            graphs, labels = pickle.load(f)
        return graphs, labels

    def plot_graphons(self, key):
        """
        Plot the graphon functions

        :param key: the key of the graphon function
        :type key: int
        """
        x = torch.linspace(0, 1, 1000)
        y = eval('self.graphon_' + str(key) + '(x)')
        plt.title('Graphon ' + str(key) + f' --> ({self.name})')
        plt.imshow(y, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
