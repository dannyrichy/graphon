from utils import *

from node2vec import Node2Vec
import networkx as nx
from pathlib import Path


def combine_datasets(dataset_names=[]):
    result = []
    size = 0
    for dataset in dataset_names:
        size += len(dataset)
        result += dataset

    gt_result = []

    label = 0
    for dataset in dataset_names:
        gt_result += [label] * len(dataset)
        label += 1

    return result, np.array(gt_result)


def main():
    DOWNLOAD_DATA = False

    if DOWNLOAD_DATA:
        download_datasets()

    # loading graphs
    fb = load_graph(min_num_nodes=100, name='facebook_ct1')
    github = load_graph(min_num_nodes=950, name='github_stargazers')
    reddit = load_graph(min_num_nodes=3200, name='REDDIT-BINARY')
    deezer = load_graph(min_num_nodes=200, name='deezer_ego_nets')

    fb_github_reddit, gt_fb_github_reddit = combine_datasets([fb, github, reddit])
    # gt_fb_github_reddit

    fb_github_deezer, gt_fb_github_deezer = combine_datasets([fb, github, deezer])
    # gt_fb_github_deezer

    fb_reddit_deezer, gt_fb_reddit_deezer = combine_datasets([fb, reddit, deezer])
    # gt_fb_reddit_deezer

    github_reddit_deezer, gt_github_reddit_deezer = combine_datasets([github, reddit, deezer])
    # gt_github_reddit_deezer

    fb_github_reddit_deezer, gt_fb_github_reddit_deezer = combine_datasets([fb, github, reddit, deezer])
    # gt_fb_github_reddit_deezer


def node2vec(graph, emb_dir=None, savename=None, dimensions=64, walk_length=30, num_walks=200):
    graph = nx.from_numpy_array(graph.numpy())
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)

    # saves the description of the parameters used for the embedding
    with open(f'{emb_dir}/AA_emb_params.txt', 'w') as f:
        f.write(f'dimensions --> {dimensions}\nwalk_length --> {walk_length}\nnum_walks --> {num_walks}')

    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    if savename is not None:
        model.wv.save_word2vec_format(f'{emb_dir}/{savename}')


def embed_all_node2vec(emb_dir, datasets=None, dimensions=64, walk_length=30, num_walks=200):
    '''
    Creates a node2vec embedding for all the datasets in datasets.
    :param emb_dir: name of the directory where all the embeddings will be stored
    :param datasets: list containing the datasets to embed (if None it does the default ones)
    :param dimensions: dimensions for the node2vec embeddings
    :params walk_length: number of nodes in each random walk
    :param num_walks: number of walks per node
    '''
    if datasets is None:
        datasets = ['deezer_ego_nets', 'facebook_ct1', 'github_stargazers', 'REDDIT-BINARY']
    Path(emb_dir).mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        graphs = load_graph(min_num_nodes=10, name=ds)
        for idx, g in enumerate(graphs):
            Path(f'{emb_dir}/{ds}').mkdir(parents=True, exist_ok=True)
            savename = f'{ds}/{ds}_{idx}'
            node2vec(graph=g, emb_dir=emb_dir, savename=savename, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)


if __name__ == '__main__':
    main()
    # embed_all_node2vec(emb_dir='node2vec_embeddings_0')
