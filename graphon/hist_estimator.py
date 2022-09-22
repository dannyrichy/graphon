import logging

import torch

import graphon
import sys
import os
sys.path.append(os.path.abspath('../'))
from config import DEVICE

logger = logging.getLogger(__name__)


def hist_approximate(graphs, n0=30):
    """
    Approximating graphon using histogram estimate
    Reference: https://github.com/mahalakshmi-sabanayagam/Clustering-Testing-Networks/blob/96989cbade5eb14d2426de7e1b6d277e55b76766/DSC_SSDP.py

    :param graphs: List of graphs for which graphon has to be estimated
    :type graph: List of torch.Tensor

    :param n0: Size of the binned matrix
    :type n0: int

    :return: Approximate list of graphs
    :rtype: list
    """
    graphs_approx = []
    for graph in graphs:
        nn = graph.shape[0]
        h = int(nn / n0)
        logger.info(f"Shape of graph: {nn}")
        logger.debug(f"h: {h}")

        deg = torch.sum(graph, axis=1)
        logger.debug(f"Degree of each node: \n {deg}")
        id_sort = torch.argsort(-deg)

        logger.info("Sorting graph")
        graph_sorted = graph[id_sort]
        graph_sorted = graph_sorted[:, id_sort]
        logger.debug(f"Sorted graph:\n{graph_sorted}")

        # histogram approximation
        logger.info("Approximating the graphon")
        graph_apprx = torch.zeros((n0, n0), dtype=torch.float64).to(device=DEVICE)
        for i in range(n0):
            for j in range(i + 1):
                graph_apprx[i][j] = torch.sum(graph_sorted[i * h:i * h + h, j * h:j * h + h]) / (h * h)
                graph_apprx[j][i] = graph_apprx[i][j]

        logger.info("Approximation done!")
        graphs_approx.append(graph_apprx)
    return graphs_approx
