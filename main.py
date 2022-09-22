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

logging.basicConfig(level=logging.INFO)



def main():
    """
    Main function

    :return:
    :rtype:
    """
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

    #generate embeddings if the there are none
    if len(os.listdir(EMBEDDING_DIR)) == 0:
        embed_all_graph2vec(emb_dir=EMBEDDING_DIR, graph_list=[fb, github, reddit, deezer])


    approxs = hist_approximate(fb_github_reddit, n0=30) #these are the graphon embeddings

    ##### classification using the embeddings from the graphon - since the graphon approximation is a matrix, we compute the 
    ##### eigen vector corresponding to the largest eigen value, and use this vector as a further embedding of a graphon
    graphon_embeddings = []
    for i in range(len(approxs)):
        eigen_val, eigen_vec = compute_spectrum_graph_laplacian(approxs[i])
        graphon_embeddings.append(eigen_vec[0]) #using only the eigen vector corresponding to the largest eigen value
    
    permutation = np.random.permutation(len(graphon_embeddings)) # random shuffling
    X = np.take(graphon_embeddings, permutation, axis = 0)
    y = gt_fb_github_reddit[permutation]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy_graphon_classification = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy for classification on graphon embeddings: ",accuracy_graphon_classification)



    ##### classification using the embeddings from graph2vec
    embeddings, gt = load_embeddings(names=DATASETS[:2])
    embeddings = np.squeeze(embeddings)
    print('number of datasamples (embeddings): ',len(embeddings))
    print('number of labels: ',len(gt))
    permutation = np.random.permutation(len(embeddings))
    X = np.take(embeddings, permutation, axis = 0)
    y = np.take(gt, permutation, axis = 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy_grap2vec_classification = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy for graph2vec embeddings: ",accuracy_grap2vec_classification)



if __name__ == '__main__':
    main()

