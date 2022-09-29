from config import *
from utils import *
from karateclub import Graph2Vec
import pickle
from pathlib import Path
from tqdm import tqdm


def graph2vec(graphs, emb_dir, savename=None):
    '''
    Creates a graph2vec embedding for the graphs in graphs.
    :param graphs: list of graphs to embed.
    :param emb_dir: name of the directory where the embedding will be stored.
    '''
    graph2vec = Graph2Vec()
    graph2vec.fit(graphs)
    embeddings = graph2vec.get_embedding()
    with open(f'{emb_dir}/{savename}.pkl', 'wb') as f:
        pickle.dump(embeddings, f)


def embed_all_graph2vec(emb_dir = EMBEDDING_DIR, graph_list = [], data = []):
    '''
    Creates a graph2vec embedding for the graphs in the graphlist.
    :param emb_dir: name of the directory where all the embeddings will be stored
    :param datasets: list containing the datasets to embed (if None it does the default ones)
    '''
    
    if graph_list is None:
        print('you have not loaded the graphs yet')
    Path(emb_dir).mkdir(parents=True, exist_ok=True)
    for name_id, graphs in enumerate(graph_list):
        for idx, graph in tqdm(enumerate(graphs)):
            graphs[idx] = nx.from_numpy_array(graph.numpy())
        savename = f'{data[name_id]}'
        graph2vec(graphs=graphs, emb_dir=emb_dir, savename=savename)


def load_embeddings(embedding_dir = EMBEDDING_DIR, names = []):
    '''
    Loads the embeddings by name
    :param embedding_dir: directory where the embeddings are stored
    :param name: which embedding/embeddings to load
    :return: a list containing all the embeddings and the corresponding labels
    '''
    embeddings = []
    labels = []
    gt = []
    for id, name in enumerate(names):
        with open(f'{embedding_dir}{name}.pkl', 'rb') as f:
            tmp = pickle.load(f)
            gt += [id]*len(tmp)
            embeddings.append(tmp)
    return np.reshape( embeddings, (len(embeddings)*tmp.shape[0], -1) ), gt
