from errno import EMEDIUMTYPE
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

logging.basicConfig(level=logging.INFO)


class classifier_clustering():
    def __init__(self, classification, graphs, approx_nodes):
        self.classification = classification
        self.graphs = graphs
        self.approx_nodes

    def classify(self, gt = None, graph2vec_embeddings = True):

        if graph2vec_embeddings:
            ##### classification using the embeddings from graph2vec
            embeddings, gt = load_embeddings(names=DATA['SYNTHETIC_DATA'])
            embeddings = np.squeeze(embeddings)
            print('number of datasamples (embeddings): ',len(embeddings))
            print('number of labels: ',len(gt))

        else:
            if gt is None:
                print('You need to pass the ground truth labels')
            approxs = hist_approximate(self.graphs, self.approx_nodes) #these are the graphon embeddings
            ##### classification using the embeddings from the graphon - since the graphon approximation is a matrix, we compute the 
            ##### eigen vector corresponding to the largest eigen value, and use this vector as a further embedding of a graphon
            embeddings = []
            for i in range(len(approxs)):
                eigen_val, eigen_vec = compute_spectrum_graph_laplacian(approxs[i])
                embeddings.append(eigen_vec[0]) #using only the eigen vector corresponding to the largest eigen value
        
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

    syn_graphons = graphons_graphs(NUM_GRAPHS_PER_GRAPHONS, DATA['SYNTHETIC_DATA'])
    graphs, lables = syn_graphons.data_simulation(start=100, stop=1000)
    graphs = np.split(np.array(graphs), NUM_GRAPHONS)
    if CREATE_EMBEDDINGS:
        embed_all_graph2vec(emb_dir=EMBEDDING_DIR, graph_list=[list(graphs[i]) for i in range(NUM_GRAPHONS)], 
                                                                data = DATA['SYNTHETIC_DATA'])

    a = 1




























