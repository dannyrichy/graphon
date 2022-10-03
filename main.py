from config import NUM_GRAPHONS, NUM_GRAPHS_PER_GRAPHONS, DATA, CREATE_EMBEDDINGS, EMBEDDING_DIR, SAVE_GRAPHONS, SAVE_GRAPHONS_LOC
from graph2vec.utils import embed_all_graph2vec, load_embeddings
from graphon.hist_estimator import hist_approximate
from sklearn.ensemble import RandomForestClassifier
from graphon.graphons import SynthGraphons
from utils import classification, clustering
import numpy as np
import wandb




# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='graphon')



def main(config=None):
    with wandb.init(config=config):
        config = wandb.config    
        SAVE_GRAPHONS_LOC = f'./graphons_dir/{wandb.NUM_GRAPHONS}_graphons_{wandb.NUM_GRAPHS_PER_GRAPHONS}_graphs.pkl'
        # synthetic data
        syn_graphons = SynthGraphons(NUM_GRAPHS_PER_GRAPHONS, DATA['SYNTHETIC_DATA'], num_nodes=config.NUM_NODES, save_graphons_loc = SAVE_GRAPHONS_LOC)
        if SAVE_GRAPHONS:
            print('storing graphs at ', SAVE_GRAPHONS_LOC)
            graphs, labels = syn_graphons.data_simulation(start=100, stop=1000, save=SAVE_GRAPHONS, save_dir=SAVE_GRAPHONS_LOC)
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
        classification(embeddings, labels)
        # clustering(approxs, labels, k = 2)
        


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

























