from pathlib import Path
from embedding import create_g2v_embeddings, load_embeddings, histogram_embeddings
from data_loader.syntheticData import SynthGraphons
from utils import classification, clustering, update_config, download_datasets, load_graph, combine_datasets
import numpy as np
import wandb
import yaml
import time
import os


def clustering_classification(
    NUM_GRAPHS_PER_GRAPHON=100,
    NUM_NODES='None',
    N0=30,
    SAVE_GRAPHONS=False,
    CREATE_EMBEDDINGS=False,
    G2V_EMBEDDING_DIR=None,
    DATA=None,
    SWEEP=False,
    DOWNLOAD_DATA=False,
    SYNTH_DATA=True
    
):
    if SYNTH_DATA:
        NUM_GRAPHONS = len(DATA[1])
        k = NUM_GRAPHONS
        parent_dir = Path('graphons_dir')
        parent_dir.mkdir(exist_ok=True, parents=True)
        GRAPHONS_DIR = parent_dir.joinpath(f'{NUM_GRAPHONS}_graphons_{NUM_GRAPHS_PER_GRAPHON}_graphs.pkl')

        # synthetic data_loader
        if SAVE_GRAPHONS:
            syn_graphons = SynthGraphons(NUM_NODES, NUM_GRAPHS_PER_GRAPHON, DATA[1])
            graphs, true_labels = syn_graphons.data_simulation(start=100, stop=1000)
            syn_graphons.save_graphs(GRAPHONS_DIR)
        else:
            graphs, true_labels = SynthGraphons.load_graphs(path=GRAPHONS_DIR)
    else:
        if DOWNLOAD_DATA:
            download_datasets()
        # loading graphs
        fb = load_graph(min_num_nodes=100, name=DATA[0][0])
        github = load_graph(min_num_nodes=100, name=DATA[0][2])
        reddit = load_graph(min_num_nodes=100, name=DATA[0][3])
        deezer = load_graph(min_num_nodes=100, name=DATA[0][1])
        # graphs, true_labels = combine_datasets([fb, github, reddit])
        # graphs, true_labels = combine_datasets([fb, github, deezer])
        # graphs, true_labels = combine_datasets([fb, reddit, deezer])
        # graphs, true_labels = combine_datasets([github, reddit, deezer])
        graphs, true_labels = combine_datasets([fb, github, reddit, deezer])

        k = len(np.unique(true_labels))
    
    # start recording time for embedding creation
    
    start_t_g2v = time.time()

    # creating graph2vec embeddings of the graphs from graphons and storing them
    if CREATE_EMBEDDINGS:
        print('\nCreating graph2vec embeddings')
        create_g2v_embeddings(graph_list=graphs, true_labels=true_labels, dir_name=G2V_EMBEDDING_DIR)
    time_g2v = time.time() - start_t_g2v
    print(f'Graph2vec embeddings created in {time_g2v} seconds')

    
    # classification of graph2vec embeddings
    embeddings, true_labels = load_embeddings(dir_name=G2V_EMBEDDING_DIR)
    embeddings = np.squeeze(embeddings)
    print('Number of labels: ', len(true_labels))

    print('performing clustering on histogram approximation')
    clustering_rand_score, clustering_error = clustering(embeddings, true_labels, k=k, GRAPH2VEC=True)

    print('\nPerforming classification on histogram approximation')
    classification_train_acc, classification_test_acc = classification(embeddings, true_labels, GRAPH2VEC=True)

    if SWEEP:
        wandb.log({'g2v_class_train_accuracy': classification_train_acc, 
                    'g2v_class_test_accuracy': classification_test_acc,
                    'g2v_clustering_rand_score': clustering_rand_score,
                    'g2v_clustering_error': clustering_error,
                    'time_g2v': time_g2v})


    start_t_graphons = time.time()
    # classification of graphon embeddings
    print('creating histogram approximation of graphs')
    hist_embeddings = histogram_embeddings(graphs, n0=N0) 
    embeddings = []
    for i in range(len(hist_embeddings)):
        flattened_emb = hist_embeddings[i].numpy().flatten()
        embeddings.append(flattened_emb)
    time_graphons = time.time() - start_t_graphons
    print(f'Graphon embeddings created in {time_graphons} seconds')

    print('\nPerforming classification on histogram approximation')
    classification_train_acc, classification_test_acc = classification(embeddings, true_labels)

    print('performing clustering on histogram approximation')
    clustering_rand_score, clustering_error = clustering(embeddings, true_labels, k=k, GRAPH2VEC=False)

    if SWEEP:
        wandb.log({
            'graphons_class_train_accuracy': classification_train_acc, 
                    'graphons_class_test_accuracy': classification_test_acc,
                    'graphons_clustering_rand_score': clustering_rand_score,
                    'graphons_clustering_error': clustering_error,
                    'time_graphons': time_graphons})



def sweep(config=None):
    with wandb.init(config=config):
            clustering_classification(**wandb.config)


if __name__ == '__main__':
    # loads the config file
    with open("config.yaml", 'r') as stream:
        config_def = yaml.load(stream, Loader=yaml.FullLoader)

    with open('sweep_config.yaml', 'r') as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
    final_config = update_config(sweep_configuration, config_def)

    # if we are sweeping, we update the config with the default values and start the sweep
    # else we run the code using the config_def values
    if config_def['SWEEP']:
        wandb.login()
        sweep_id = wandb.sweep(sweep_configuration, project="graphon", entity='seb-graphon')
        wandb.agent(sweep_id, sweep)
    else:
        clustering_classification(**final_config)



'''
if DOWNLOAD_DATA:
    download_datasets()
# loading graphs
print(DATASETS)
fb = load_graph(min_num_nodes=100, name=DATASETS[0])
github = load_graph(min_num_nodes=950, name=DATASETS[1])
reddit = load_graph(min_num_nodes=3200, name=DATASETS[2])
deezer = load_graph(min_num_nodes=200, name=DATASETS[3])
fb_github_reddit, gt_fb_github_reddit = combine_datasets([fb, github, reddit])
fb_github_deezer, gt_fb_github_deezer = combine_datasets([fb, github, deezer])
fb_reddit_deezer, gt_fb_reddit_deezer = combine_datasets([fb, reddit, deezer])
github_reddit_deezer, gt_github_reddit_deezer = combine_datasets([github, reddit, deezer])
fb_github_reddit_deezer, gt_fb_github_reddit_deezer = combine_datasets([fb, github, reddit, deezer])
'''