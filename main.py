from pathlib import Path
from embedding import create_g2v_embeddings, load_embeddings, histogram_embeddings
from data_loader.syntheticData import SynthGraphons
from utils import classification, clustering
import numpy as np
import wandb
import yaml
import os

'''
Have to make the code in the form of a function that takes in all the needed parameters
Then when we want to sweep we pass the sweep parameters, else, the normal ones.
'''

def update_config(sweep_config, config_def):
    """
    Updated the sweep config with the default config, adding the default values 

    :param sweep_config: the sweep config to be updated
    :type sweep_config: dict

    :param config_def: the default config
    :type config_def: dict

    :return: the updated sweep config
    :rtype: dict
    """
    real_data_params = ['DOWNLOAD_DATA']
    synth_data_params = ['NUM_GRAPHS_PER_GRAPHON', 'NUM_NODES', 'N0']

    if config_def['SYNTH_DATA']:
        to_remove = real_data_params
    else:
        to_remove = synth_data_params
    
    for param in to_remove:
        if param in sweep_config['parameters']:
            del sweep_config['parameters'][param]
        del config_def[param]
    del config_def['SYNTH_DATA']

    if config_def['SWEEP']:
        for key, value in config_def.items():
            if key not in sweep_config['parameters'].keys(): 
                sweep_config['parameters'][key] = {'value': value}
        return sweep_config
    return config_def

def clustering_classification_synth(
    NUM_GRAPHS_PER_GRAPHON=100,
    NUM_NODES=None,
    N0=30,
    SAVE_GRAPHONS=False,
    CREATE_EMBEDDINGS=False,
    G2V_EMBEDDING_DIR=None,
    DATA=None,
    SWEEP=False,
    
):
    NUM_GRAPHONS = len(DATA[1])
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

    # creating graph2vec embeddings of the graphs from graphons and storing them
    if CREATE_EMBEDDINGS:
        print('Creating graph2vec embeddings')
        create_g2v_embeddings(graph_list=graphs, true_labels=true_labels, dir_name=G2V_EMBEDDING_DIR)

    # classification of graph2vec embeddings
    embeddings, true_labels = load_embeddings(dir_name=G2V_EMBEDDING_DIR)
    embeddings = np.squeeze(embeddings)
    print('Number of labels: ', len(true_labels))
    
    print('\nPerforming classification on histogram approximation')
    classification_train_acc, classification_test_acc = classification(embeddings, true_labels, GRAPH2VEC=True)

    # print('performing clustering on histogram approximation')
    # clustering_rand_score, clustering_error = clustering(embeddings, true_labels, k=NUM_GRAPHONS, GRAPH2VEC=True)

    if SWEEP:
        wandb.log({
            'g2v_class_train_accuracy': classification_train_acc, 
                    'g2v_class_test_accuracy': classification_test_acc,})
                    # 'g2v_clustering_rand_score': clustering_rand_score,
                    # 'g2v_clustering_error': clustering_error})



    # classification of graphon embeddings
    print('creating histogram approximation of graphs')
    hist_embeddings = histogram_embeddings(graphs, n0=N0) 
    embeddings = []
    for i in range(len(hist_embeddings)):
        flattened_emb = hist_embeddings[i].numpy().flatten()
        embeddings.append(flattened_emb)

    print('\nPerforming classification on histogram approximation')
    classification_train_acc, classification_test_acc = classification(embeddings, true_labels)

    # print('performing clustering on histogram approximation')
    # clustering_rand_score, clustering_error = clustering(embeddings, true_labels, k = NUM_GRAPHONS, GRAPH2VEC=False)

    if SWEEP:
        wandb.log({
            'graphons_class_train_accuracy': classification_train_acc, 
                    'graphons_class_test_accuracy': classification_test_acc,})
                    # 'graphons_clustering_rand_score': clustering_rand_score,
                    # 'graphons_clustering_error': clustering_error})

    

def sweep(config=None):
    with wandb.init(config=config):
            clustering_classification_synth(**wandb.config)


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
        clustering_classification_synth(**final_config)


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