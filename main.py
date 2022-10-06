# from config import NUM_GRAPHONS, NUM_GRAPHS_PER_GRAPHONS, DATA, CREATE_EMBEDDINGS, EMBEDDING_DIR, SAVE_GRAPHONS, SAVE_GRAPHONS_LOC
import os
from graph2vec.utils import embed_all_graph2vec, load_graphons_embeddings
from graphon.hist_estimator import hist_approximate
from sklearn.ensemble import RandomForestClassifier
from graphon.graphons import SynthGraphons
from utils import classification, clustering
import numpy as np
import wandb
import yaml

'''
Have to make the code in the form of a function that takes in all the needed parameters
Then when we want to sweep we pass the sweep parameters, else, the normal ones.
'''

def update_sweep_config(sweep_config, config_def):
    """
    Updated the sweep config with the default config, adding the default values 

    :param sweep_config: the sweep config to be updated
    :type sweep_config: dict

    :param config_def: the default config
    :type config_def: dict

    :return: the updated sweep config
    :rtype: dict
    """
    for key, value in config_def.items():
        if key not in sweep_config['parameters'].keys(): 
            sweep_config['parameters'][key] = {'value': value}
    return sweep_config


def clustering_classification(
    NUM_GRAPHS_PER_GRAPHON=100,
    NUM_NODES=None,
    N0=30,
    SAVE_GRAPHONS=False,
    CREATE_EMBEDDINGS=False,
    G2V_EMBEDDING_DIR=None,
    DATA=None,
    DOWNLOAD_DATA=False,
    SWEEP=False,
    
):

    NUM_GRAPHONS = len(DATA[1])
    GRAPHONS_DIR = f'./graphons_dir/{NUM_GRAPHONS}_graphons_{NUM_GRAPHS_PER_GRAPHON}_graphs.pkl'

    syn_graphons = SynthGraphons(NUM_GRAPHS_PER_GRAPHON, 
                                DATA[1], 
                                num_nodes=NUM_NODES, 
                                save_graphons_loc=GRAPHONS_DIR
                                )
    # generate graphs from graphons
    if SAVE_GRAPHONS:
        print('storing graphs at ', GRAPHONS_DIR)
        graphs, labels = syn_graphons.data_simulation(start=100, stop=1000, save=SAVE_GRAPHONS)
    else:
        print('loading graphs from ', GRAPHONS_DIR)
        graphs, labels = syn_graphons.load_graphs() 

    # creating graph2vec embeddings of the graphs from graphons and storing them
    if CREATE_EMBEDDINGS:
        print('creating graph2vec embeddings')
        tmp = np.split(np.array(graphs), NUM_GRAPHONS)
        embed_all_graph2vec(emb_dir=G2V_EMBEDDING_DIR, 
                            graph_list=[list(tmp[i]) for i in range(NUM_GRAPHONS)], data = DATA[1])

    # loading graph2vec embeddings of the graphone
    embeddings, true_labels = load_graphons_embeddings(embedding_dir=G2V_EMBEDDING_DIR, names=DATA[1])
    embeddings = np.squeeze(embeddings)

    print('\nPerforming classification on histogram approximation')
    classification_train_acc, classification_test_acc = classification(embeddings, true_labels, GRAPH2VEC=True)

    # print('performing clustering on histogram approximation')
    # clustering_rand_score, clustering_error = clustering(embeddings, labels, k=NUM_GRAPHONS, GRAPH2VEC=True)

    if SWEEP:
        wandb.log({'g2v_class_train_accuracy': classification_train_acc, 
                    'g2v_class_test_accuracy': classification_test_acc,})
                    # 'g2v_clustering_rand_score': clustering_rand_score,
                    # 'g2v_clustering_error': clustering_error})



    # classification of graphon embeddings
    print('creating histogram approximation of graphs')
    approxs = hist_approximate(graphs, n0=N0) 
    embeddings = []
    for i in range(len(approxs)):
        flattened_emb = approxs[i].numpy().flatten()
        embeddings.append(flattened_emb)

    print('\nPerforming classification on histogram approximation')
    classification_train_acc, classification_test_acc = classification(embeddings, labels)

    # print('performing clustering on histogram approximation')
    # clustering_rand_score, clustering_error = clustering(embeddings, labels, k = NUM_GRAPHONS, GRAPH2VEC=False)

    if SWEEP:
        wandb.log({'graphons_class_train_accuracy': classification_train_acc, 
                    'graphons_class_test_accuracy': classification_test_acc,})
                    # 'graphons_clustering_rand_score': clustering_rand_score,
                    # 'graphons_clustering_error': clustering_error})
    



def sweep(config=None):
    with wandb.init(config=config):
            clustering_classification(**wandb.config)


if __name__ == '__main__':
    # loads the config file
    with open("config.yaml", 'r') as stream:
        config_def = yaml.load(stream, Loader=yaml.FullLoader)

    with open('sweep_config.yaml', 'r') as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists('graphons_dir'):
        os.makedirs('graphons_dir')
    # if we are sweeping, we update the config with the default values and start the sweep
    # else we run the code using the config_def values
    if config_def['SWEEP']:
        wandb.login()
        sweep_configuration = update_sweep_config(sweep_configuration, config_def)
        sweep_id = wandb.sweep(sweep_configuration, project="graphon", entity='seb-graphon')
        wandb.agent(sweep_id, sweep)
    else:
        clustering_classification(**config_def)


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

























