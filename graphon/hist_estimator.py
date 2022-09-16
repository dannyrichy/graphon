import torch

from graphon.config import DEVICE


def hist_approximate(graph, n0=30):
    """
    Approximating graphon using histogram estimate
    Reference: https://github.com/mahalakshmi-sabanayagam/Clustering-Testing-Networks/blob/96989cbade5eb14d2426de7e1b6d277e55b76766/DSC_SSDP.py

    :param graph: Graph for which graphon has to be estimated
    :type graph: torch.Tensor

    :param n0: Size of the binned matrix
    :type n0: int

    :return: Approximate list of graphs
    :rtype: list
    """
    nn = graph.shape[0]
    h = int(nn / n0)

    deg = torch.sum(graph, axis=1)
    id_sort = torch.argsort(-deg)

    graph_sorted = graph[id_sort]
    graph_sorted = graph_sorted[:, id_sort]

    # histogram approximation
    graph_apprx = torch.zeros((n0, n0), dtype=torch.float64).to(device=DEVICE)
    for i in range(n0):
        for j in range(i + 1):
            graph_apprx[i][j] = torch.sum(graph_sorted[i * h:i * h + h, j * h:j * h + h]) / (h * h)
            graph_apprx[j][i] = graph_apprx[i][j]

    return graph_apprx