README for dataset OHSU


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

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

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


=== Node Label Conversion === 


Node labels were converted to integer values using this map:

Component 0:
	0	n_0
	1	n_157
	2	n_159
	3	n_70
	4	n_109
	5	n_120
	6	n_59
	7	n_74
	8	n_122
	9	n_5
	10	n_164
	11	n_21
	12	n_80
	13	n_83
	14	n_102
	15	n_65
	16	n_13
	17	n_39
	18	n_69
	19	n_149
	20	n_10
	21	n_184
	22	n_76
	23	n_170
	24	n_43
	25	n_77
	26	n_124
	27	n_23
	28	n_31
	29	n_110
	30	n_136
	31	n_1
	32	n_35
	33	n_118
	34	n_167
	35	n_171
	36	n_58
	37	n_121
	38	n_177
	39	n_3
	40	n_84
	41	n_49
	42	n_89
	43	n_47
	44	n_131
	45	n_183
	46	n_161
	47	n_87
	48	n_135
	49	n_148
	50	n_94
	51	n_151
	52	n_168
	53	n_17
	54	n_37
	55	n_130
	56	n_85
	57	n_128
	58	n_173
	59	n_51
	60	n_72
	61	n_95
	62	n_96
	63	n_143
	64	n_185
	65	n_98
	66	n_15
	67	n_101
	68	n_158
	69	n_55
	70	n_46
	71	n_111
	72	n_133
	73	n_137
	74	n_188
	75	n_123
	76	n_12
	77	n_90
	78	n_71
	79	n_113
	80	n_129
	81	n_82
	82	n_100
	83	n_68
	84	n_117
	85	n_178
	86	n_187
	87	n_18
	88	n_91
	89	n_103
	90	n_106
	91	n_160
	92	n_7
	93	n_30
	94	n_53
	95	n_63
	96	n_73
	97	n_97
	98	n_107
	99	n_144
	100	n_152
	101	n_163
	102	n_186
	103	n_19
	104	n_146
	105	n_14
	106	n_56
	107	n_81
	108	n_41
	109	n_22
	110	n_86
	111	n_162
	112	n_33
	113	n_153
	114	n_165
	115	n_115
	116	n_145
	117	n_36
	118	n_8
	119	n_9
	120	n_29
	121	n_57
	122	n_132
	123	n_180
	124	n_16
	125	n_24
	126	n_125
	127	n_182
	128	n_147
	129	n_142
	130	n_45
	131	n_114
	132	n_66
	133	n_179
	134	n_119
	135	n_75
	136	n_126
	137	n_140
	138	n_155
	139	n_108
	140	n_174
	141	n_4
	142	n_105
	143	n_27
	144	n_78
	145	n_44
	146	n_38
	147	n_60
	148	n_150
	149	n_175
	150	n_62
	151	n_127
	152	n_92
	153	n_189
	154	n_42
	155	n_88
	156	n_172
	157	n_67
	158	n_154
	159	n_139
	160	n_156
	161	n_141
	162	n_99
	163	n_181
	164	n_6
	165	n_112
	166	n_2
	167	n_32
	168	n_169
	169	n_26
	170	n_79
	171	n_134
	172	n_166
	173	n_48
	174	n_54
	175	n_176
	176	n_64
	177	n_20
	178	n_40
	179	n_61
	180	n_104
	181	n_52
	182	n_25
	183	n_28
	184	n_11
	185	n_116
	186	n_93
	187	n_138
	188	n_34
	189	n_50


=== References ===
https://github.com/shiruipan/graph_datasets/tree/master/Graph_Repository

=== Previous Use of the Dataset ===
Shirui Pan, Jia Wu, Xingquan Zhu, Guodong Long, and Chengqi Zhang. " Task Sensitive Feature Exploration and Learning for Multi-Task Graph Classification."  IEEE Trans. Cybernetics (TCYB) 47(3): 744-758 (2017)
