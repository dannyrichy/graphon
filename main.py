import logging
import os
from tqdm import tqdm
from config import *
from utils import *
from graphon.hist_estimator import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from graph2vec_utils import *
from graphon.graphons import graphons_graphs

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

def classification(embeddings = [], gt = []):
    permutation = np.random.permutation(len(embeddings)) # random shuffling
    X = np.take(embeddings, permutation, axis = 0)
    y = np.take(gt, permutation, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy_classification = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy for classification on embeddings: ",accuracy_classification)


def clustering(self):
    return None




if __name__ == '__main__':

    'synthetic data'
    syn_graphons = graphons_graphs(NUM_GRAPHS_PER_GRAPHONS, DATA['SYNTHETIC_DATA'])
    graphs, labels = syn_graphons.data_simulation(start=100, stop=1000)
    

    'creating embeddings'
    if CREATE_EMBEDDINGS:
        graphs = np.split(np.array(graphs), NUM_GRAPHONS)
        embed_all_graph2vec(emb_dir=EMBEDDING_DIR, graph_list=[list(graphs[i]) for i in range(NUM_GRAPHONS)], 
                                                                data = DATA['SYNTHETIC_DATA'])

    'classification of graph2vec embeddings'
    embeddings, gt = load_embeddings(names=DATA['SYNTHETIC_DATA'])
    embeddings = np.squeeze(embeddings)
    logger.info('number of datasamples (embeddings): ',len(embeddings))
    print('number of labels: ',len(gt))
    classification(embeddings, gt)


    'classification of graphon embeddings'
    approxs = hist_approximate(graphs, n0 = 30) #these are the graphon embeddings
    embeddings = []
    for i in range(len(approxs)):
        flattened_emb = approxs[i].numpy().flatten()
        embeddings.append(flattened_emb) #using only the eigen vector corresponding to the largest eigen value
    classification(embeddings, labels)

    

    






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

























