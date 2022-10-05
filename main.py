# from config import NUM_GRAPHONS, NUM_GRAPHS_PER_GRAPHONS, DATA, CREATE_EMBEDDINGS, EMBEDDING_DIR, SAVE_GRAPHONS, SAVE_GRAPHONS_LOC
from graph2vec.utils import embed_all_graph2vec, load_embeddings
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
    NUM_GRAPHONS=1000,
    NUM_GRAPHS_PER_GRAPHON=100,
    NUM_NODES=None,
    SAVE_GRAPHONS=False,
    CREATE_EMBEDDINGS=False,
    EMBEDDING_DIR=None,
    DATA='',
    DOWNLOAD_DATA=False,
    SWEEP=False,
):
  
    SAVE_GRAPHONS_LOC = f'./graphons_dir/{NUM_GRAPHONS}_graphons_{NUM_GRAPHS_PER_GRAPHON}_graphs.pkl'
    syn_graphons = SynthGraphons(NUM_GRAPHS_PER_GRAPHON, DATA['SYNTHETIC_DATA'], num_nodes=NUM_NODES, save_graphons_loc = SAVE_GRAPHONS_LOC)
    if SAVE_GRAPHONS:
        print('storing graphs at ', SAVE_GRAPHONS_LOC)
        graphs, labels = syn_graphons.data_simulation(start=100, stop=1000, save=SAVE_GRAPHONS)
    else:
        print('loading graphs from ', SAVE_GRAPHONS_LOC)
        graphs, labels = syn_graphons.load_graphs() 

    # creating embeddings
    if CREATE_EMBEDDINGS:
        print('creating graph2vec embeddings')
        tmp = np.split(np.array(graphs), NUM_GRAPHONS)
        embed_all_graph2vec(emb_dir=EMBEDDING_DIR, graph_list=[list(tmp[i]) for i in range(NUM_GRAPHONS)], 
                                                                data = DATA['SYNTHETIC_DATA'])

    # classification of graph2vec embeddings
    embeddings, true_labels = load_embeddings(names=DATA['SYNTHETIC_DATA'])
    embeddings = np.squeeze(embeddings)
    print('number of labels: ',len(true_labels))
    classification(embeddings, true_labels)
    # clustering(embeddings, labels, k = 2, GRAPH2VEC=True)


    # classification of graphon embeddings
    approxs = hist_approximate(graphs, n0=30) 
    embeddings = []
    for i in range(len(approxs)):
        flattened_emb = approxs[i].numpy().flatten()
        embeddings.append(flattened_emb)
    train_acc, test_acc = classification(embeddings, labels)
    if SWEEP:
        wandb.log({'train_accuracy': train_acc, 
                    'test_accuracy': test_acc})
    # clustering(approxs, labels, k = 2)

def sweep(config=None):
    with wandb.init(config=config):
            config = wandb.config
            clustering_classification(**config)



if __name__ == '__main__':
    # loads the config file
    with open("config.yaml", 'r') as stream:
        config_def = yaml.load(stream, Loader=yaml.FullLoader)

    # if we are sweeping, we update the config with the default values and start the sweep
    # else we run the code using the config_def values
    if config_def['SWEEP']:
        with open('sweep_config.yaml', 'r') as f:
            sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
        sweep_configuration = update_sweep_config(sweep_configuration, config_def)
        sweep_id = wandb.sweep(sweep_configuration, project="graphon")
        wandb.agent(sweep_id, sweep, count = 5)
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

























