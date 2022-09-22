import logging
import os
import pickle
from functools import reduce
from pathlib import Path

from karateclub import Graph2Vec
from tqdm import tqdm

from config import DOWNLOAD_DATA
from utils import *

from graphon.hist_estimator import *

logging.basicConfig(level=logging.INFO)


def combine_datasets(li_dataset):
    """
    Combining datasets

    :param li_dataset: list of datasets
    :type li_dataset: list

    :return: Reduced datasets with labels
    :rtype: tuple
    """
    result = reduce(lambda x, y: x + y, li_dataset)
    labels = np.array([i for i, dataset in enumerate(li_dataset) for _ in range(len(dataset))])
    return result, labels


def main():
    """
    Main function

    :return:
    :rtype:
    """
    if DOWNLOAD_DATA:
        download_datasets()

    # loading graphs
    fb = load_graph(min_num_nodes=100, name='facebook_ct1')
    # github = load_graph(min_num_nodes=950, name='github_stargazers')
    # reddit = load_graph(min_num_nodes=3200, name='REDDIT-BINARY')
    # deezer = load_graph(min_num_nodes=200, name='deezer_ego_nets')

    # fb_github_reddit, gt_fb_github_reddit = combine_datasets([fb, github, reddit])
    # # gt_fb_github_reddit

    # fb_github_deezer, gt_fb_github_deezer = combine_datasets([fb, github, deezer])
    # # gt_fb_github_deezer

    # fb_reddit_deezer, gt_fb_reddit_deezer = combine_datasets([fb, reddit, deezer])
    # # gt_fb_reddit_deezer

    # github_reddit_deezer, gt_github_reddit_deezer = combine_datasets([github, reddit, deezer])
    # gt_github_reddit_deezer

    # fb_github_reddit_deezer, gt_fb_github_reddit_deezer = combine_datasets([fb, github, reddit, deezer])
    # gt_fb_github_reddit_deezer
    approxs = hist_approximate(fb, n0=10)
    plt.imshow(approxs[0], cmap='hot')
    plt.show()


def graph2vec(graphs, emb_dir, savename=None):
    '''
    Creates a graph2vec embedding for the graphs in graphs.
    :param graphs: list of graphs to embed.
    :param emb_dir: name of the directory where the embedding will be stored.
    '''
    graph2vec = Graph2Vec()
    graph2vec.fit(graphs)
    embeddings = graph2vec.get_embedding()
    with open(f'{emb_dir}/{savename}.pkl', 'wb') as f:
        pickle.dump(embeddings, f)


def embed_all_graph2vec(emb_dir, datasets=None):
    '''
    Creates a graph2vec embedding for all the datasets in datasets.
    :param emb_dir: name of the directory where all the embeddings will be stored
    :param datasets: list containing the datasets to embed (if None it does the default ones)
    '''
    if datasets is None:
        datasets = ['facebook_ct1', 'deezer_ego_nets', 'github_stargazers', 'REDDIT-BINARY']
    Path(emb_dir).mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        graphs = load_graph(min_num_nodes=10, name=ds)
        for idx, graph in tqdm(enumerate(graphs)):
            graphs[idx] = nx.from_numpy_array(graph.numpy())
        savename = f'{ds}'
        graph2vec(graphs=graphs, emb_dir=emb_dir, savename=savename)


def load_embeddings(embedding_dir):
    '''
    Loads all the embeddings in the embedding_dir directory
    :param embedding_dir: directory where the embeddings are stored
    :return: a list containing all the embeddings
    '''
    embeddings = []
    for file in os.listdir(embedding_dir):
        if file.endswith('.pkl'):
            with open(f'{embedding_dir}/{file}', 'rb') as f:
                embeddings.append(pickle.load(f))
    return embeddings


# %%

if __name__ == '__main__':
    main()
    # logging.info("Embedding all graph using graph2vec")
    # embed_all_graph2vec(emb_dir='graph2vec_embeddings_0')
