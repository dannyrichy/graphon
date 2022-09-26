import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class graphons_graphs():
    
    def __init__(self, num_graphs, graphons_keys, num_nodes):#both graphons_keys and num_nodes 
                                                             #are lists
        self.num_graphs = num_graphs
        self.graphons_keys = graphons_keys #0 to 9
        self.num_nodes = num_nodes #[n1,n2,...]
        

        
    # graphons for simulated data
    def graphon_functions(self, x):
        graphons = []
        p = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64).to(device=device)
        u = p + x.reshape(1, -1)
        v = p + x.reshape(-1, 1)
        'w(u,v) = u * v'
        graphons.append(u * v) #graphon_0
        'w(u,v) = exp{-(u^0.7 + v^0.7))}'
        graphons.append(torch.exp(-(torch.pow(u, 0.7) + torch.pow(v, 0.7))))#graphon_1
        'w(u,v) = (1/4) * [u^2 + v^2 + u^(1/2) + v^(1/2)]'
        graphons.append(0.25 * (torch.pow(u, 2) + torch.pow(v, 2) + torch.pow(u, 0.5) + torch.pow(u, 0.5)))#graphon_2
        'w(u,v) = 0.5 * (u + v)'
        graphons.append(0.5 * (u + v))#graphon_3
        'w(u,v) = 1 / (1 + exp(-10 * (u^2 + v^2)))'
        graphons.append(1 / (1 + torch.exp(-10 * (torch.pow(u, 2) + torch.pow(v, 2)))))#graphon_4
        'w(u,v) = |u - v|'
        graphons.append(torch.abs(u - v))#graphon_5
        'w(u,v) = 1 / (1 + exp(-(max(u,v)^2 + min(u,v)^4)))'
        graphons.append(1 / (1 + torch.exp(-(torch.pow(torch.max(u, v), 2) + torch.pow(torch.min(u, v), 4)))))#graphon_6
        'w(u,v) = exp(-max(u, v)^(3/4))'
        graphons.append(torch.exp(-torch.pow(torch.max(u, v), 0.75)))#graphon_7
        'w(u,v) = exp(-0.5 * (min(u, v) + u^0.5 + v^0.5))'
        graphons.append(torch.exp(-0.5 * (torch.min(u, v) + torch.pow(u, 0.5) + torch.pow(v, 0.5))))#graphon_8
        'w(u,v) = log(1 + 0.5 * max(u, v))'
        graphons.append(torch.log(1 + 0.5 * torch.max(u, v)))#graphon_9
        
        return graphons[self.graphons_keys]
     

    def data_simulation(self, start=100, stop=1000):
        '''
        Generates number_of_graphs random graphs for each graphon, with a random number of nodes between start and stop.
        :param graphons: list - keys of the graphons we want to use.
        :param number_of_graphs: int - number of graphs for each graphon.
        :param start: int - minimum number of nodes in the graphs.
        :param stop: int - maximum number of nodes in the graphs.
        :return graph: list - adjacency matrix for each generated graph.
        :return labels: int - number representing the graphon which generated each graph.
        '''
        
        graphs = []
        labels = []
        graphons = []
        
        for i, j in zip(self.num_nodes, self.graphons_keys):
            x = torch.distributions.uniform.Uniform(0, 1).sample([i]).to(device=device)
            graphons.append(self.graphon_functions(x)[j])
        
        for graphon in graphons:
            p = torch.randperm(stop)
            n = p[p > start][:self.num_graphs]
            print('nodes ', n)
            graphs = graphs + graphon

        for i in range(len(graphons)):
            l = i * np.ones(self.num_graphs)
            labels = labels + l.tolist()
        print('graphs generated', len(graphs))
        print('true labels ', labels)
        return graphs, labels