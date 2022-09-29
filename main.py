import logging
from config import NUM_GRAPHONS, NUM_GRAPHS_PER_GRAPHONS, DATA, CREATE_EMBEDDINGS, EMBEDDING_DIR
from graphon.hist_estimator import hist_approximate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from graph2vec.utils import embed_all_graph2vec, load_embeddings
from clustering.spectral_clustering import graphon_clustering
from clustering.graph2vec_clustering import graph2vec_clustering
from graphon.graphons import SynthGraphons
import numpy as np

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

def classification(embeddings, true_labels):
    """
    Classification of graph embeddings using Random Forests
    
    :param embeddings: List of embeddings
    :type embeddings: list

    :param gt: Ground truth labels
    :type gt: list
    """
    permutation = np.random.permutation(len(embeddings)) # random shuffling
    X = np.take(embeddings, permutation, axis=0)
    y = np.take(true_labels, permutation, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy_classification = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy for classification on embeddings: ",accuracy_classification)


def clustering(graphs, true_labels, k = 2, GRAPH2VEC=False):
    if GRAPH2VEC:
        adjusted_rand_score, error = graph2vec_clustering(li_emb = graphs, true_labels=true_labels, k=2)  
    else: 
        graphon_clustering(graphs, true_labels, num_clusters=k)
    print(f'Adjusted Rand Score: {adjusted_rand_score} and Error: {error} for {k} clusters.') 

if __name__ == '__main__':
    # synthetic data
    syn_graphons = SynthGraphons(NUM_GRAPHS_PER_GRAPHONS, DATA['SYNTHETIC_DATA'])
    graphs, labels = syn_graphons.data_simulation(start=100, stop=1000)
    

    # creating embeddings
    if CREATE_EMBEDDINGS:
        graphs = np.split(np.array(graphs), NUM_GRAPHONS)
        embed_all_graph2vec(emb_dir=EMBEDDING_DIR, graph_list=[list(graphs[i]) for i in range(NUM_GRAPHONS)], 
                                                                data = DATA['SYNTHETIC_DATA'])

    # classification of graph2vec embeddings
    embeddings, true_labels = load_embeddings(names=DATA['SYNTHETIC_DATA'])
    embeddings = np.squeeze(embeddings)
    # logger.info('number of datasamples (embeddings): ',len(embeddings))
    print('number of labels: ',len(true_labels))
    classification(embeddings, true_labels)
    clustering(embeddings, labels, k = 2, GRAPH2VEC=True)


    # classification of graphon embeddings
    # Graphon embeddings
    approxs = hist_approximate(graphs, n0 = 30) 
    embeddings = []
    for i in range(len(approxs)):
        flattened_emb = approxs[i].numpy().flatten()
        embeddings.append(flattened_emb)
    classification(embeddings, labels)
    clustering(approxs, labels, k = 2)
    

    






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

























