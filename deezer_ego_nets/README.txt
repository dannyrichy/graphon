README for dataset Deezer Ego Nets


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

There are OPTIONAL files if the respective information is available:

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description of the dataset === 

Deezer Ego Nets

The ego-nets of Eastern European users collected from the music streaming service Deezer in February 2020. Nodes are users and edges are mutual follower relationships. The related task is the prediction of gender for the ego node in the graph.

Properties

- Number of graphs: 9,629
- Directed: No.
- Node features: No.
- Edge features: No.
- Graph labels: Yes. Binary-labeled.
- Temporal: No.

|  Stat.   |  Min  |  Max  |
|   ---    |  ---  |  ---  |
| Nodes    |   11  |  363  | 
| Density  | 0.015 | 0.909 | 
| Diameter |   2   |  2    |  


=== Source ===

If you find this dataset useful in your research, please consider citing the following paper:

>@misc{karateclub2020,
       title={An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs},
       author={Benedek Rozemberczki and Oliver Kiss and Rik Sarkar},
       year={2020},
       eprint={2003.04819},
       archivePrefix={arXiv},
       primaryClass={cs.LG}
}

And take a look at the project itself:

https://github.com/benedekrozemberczki/karateclub
