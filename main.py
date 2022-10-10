import numpy as np

from config import NUM_GRAPHS_PER_GRAPHONS, DATA, CREATE_EMBEDDINGS, SAVE_GRAPHONS
from data_loader.syntheticData import SynthGraphons
from embedding import create_g2v_embeddings, load_embeddings, histogram_embeddings
from utils import classification, clustering


def main():
    # synthetic data_loader
    if SAVE_GRAPHONS:
        syn_graphons = SynthGraphons(NUM_GRAPHS_PER_GRAPHONS, DATA['SYNTHETIC_DATA'])
        graphs, labels = syn_graphons.data_simulation(start=100, stop=1000, save=SAVE_GRAPHONS)
    else:
        graphs, labels = SynthGraphons.load_graphs()

    # creating embeddings
    if CREATE_EMBEDDINGS:
        print('Creating graph2vec embeddings')
        create_g2v_embeddings(graph_list=graphs, true_labels=labels, dir_name="G2V_LOC")

    # classification of graph2vec embeddings
    embeddings, true_labels = load_embeddings(dir_name="G2V_LOC")
    embeddings = np.squeeze(embeddings)
    print('Number of labels: ', len(true_labels))
    classification(embeddings, true_labels)
    clustering(embeddings, labels, k=2, GRAPH2VEC=True)

    # classification of data_loader embeddings
    hist_embeddings = histogram_embeddings(graphs, n0=30)
    embeddings = []
    for i in range(len(hist_embeddings)):
        flattened_emb = hist_embeddings[i].numpy().flatten()
        embeddings.append(flattened_emb)
    classification(embeddings, labels)
    clustering(hist_embeddings, labels, k=2)

# if DOWNLOAD_DATA:
#     download_datasets()
#
#
# # loading graphs
# print(DATASETS)
# fb = load_graph(min_num_nodes=100, name=DATASETS[0])
# github = load_graph(min_num_nodes=950, name=DATASETS[1])
# reddit = load_graph(min_num_nodes=3200, name=DATASETS[2])
# deezer = load_graph(min_num_nodes=200, name=DATASETS[3])
#
# fb_github_reddit, gt_fb_github_reddit = combine_datasets([fb, github, reddit])
# fb_github_deezer, gt_fb_github_deezer = combine_datasets([fb, github, deezer])
# fb_reddit_deezer, gt_fb_reddit_deezer = combine_datasets([fb, reddit, deezer])
# github_reddit_deezer, gt_github_reddit_deezer = combine_datasets([github, reddit, deezer])
# fb_github_reddit_deezer, gt_fb_github_reddit_deezer = combine_datasets([fb, github, reddit, deezer])
