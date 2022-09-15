README for dataset Peking_1


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
	0	n_1
	1	n_133
	2	n_187
	3	n_172
	4	n_91
	5	n_73
	6	n_103
	7	n_114
	8	n_126
	9	n_147
	10	n_27
	11	n_41
	12	n_45
	13	n_54
	14	n_81
	15	n_145
	16	n_4
	17	n_53
	18	n_60
	19	n_105
	20	n_108
	21	n_115
	22	n_16
	23	n_75
	24	n_162
	25	n_29
	26	n_175
	27	n_111
	28	n_119
	29	n_155
	30	n_94
	31	n_140
	32	n_50
	33	n_168
	34	n_156
	35	n_134
	36	n_160
	37	n_182
	38	n_128
	39	n_125
	40	n_138
	41	n_9
	42	n_37
	43	n_57
	44	n_151
	45	n_8
	46	n_84
	47	n_123
	48	n_17
	49	n_58
	50	n_144
	51	n_183
	52	n_56
	53	n_14
	54	n_77
	55	n_117
	56	n_82
	57	n_90
	58	n_71
	59	n_121
	60	n_65
	61	n_97
	62	n_11
	63	n_30
	64	n_43
	65	n_102
	66	n_113
	67	n_186
	68	n_152
	69	n_137
	70	n_101
	71	n_18
	72	n_42
	73	n_59
	74	n_100
	75	n_146
	76	n_38
	77	n_127
	78	n_167
	79	n_12
	80	n_46
	81	n_47
	82	n_129
	83	n_188
	84	n_52
	85	n_63
	86	n_35
	87	n_157
	88	n_163
	89	n_7
	90	n_158
	91	n_20
	92	n_88
	93	n_70
	94	n_109
	95	n_2
	96	n_184
	97	n_130
	98	n_64
	99	n_122
	100	n_10
	101	n_178
	102	n_3
	103	n_142
	104	n_106
	105	n_22
	106	n_66
	107	n_169
	108	n_131
	109	n_0
	110	n_21
	111	n_74
	112	n_80
	113	n_159
	114	n_120
	115	n_171
	116	n_13
	117	n_118
	118	n_141
	119	n_44
	120	n_55
	121	n_62
	122	n_72
	123	n_96
	124	n_99
	125	n_143
	126	n_154
	127	n_51
	128	n_95
	129	n_148
	130	n_185
	131	n_67
	132	n_36
	133	n_165
	134	n_83
	135	n_176
	136	n_24
	137	n_85
	138	n_87
	139	n_76
	140	n_132
	141	n_180
	142	n_69
	143	n_110
	144	n_170
	145	n_31
	146	n_19
	147	n_68
	148	n_78
	149	n_124
	150	n_166
	151	n_177
	152	n_89
	153	n_139
	154	n_189
	155	n_5
	156	n_39
	157	n_104
	158	n_40
	159	n_32
	160	n_107
	161	n_33
	162	n_26
	163	n_48
	164	n_98
	165	n_150
	166	n_6
	167	n_149
	168	n_112
	169	n_135
	170	n_173
	171	n_153
	172	n_174
	173	n_179
	174	n_79
	175	n_15
	176	n_181
	177	n_86
	178	n_25
	179	n_93
	180	n_23
	181	n_136
	182	n_161
	183	n_49
	184	n_92
	185	n_61
	186	n_28
	187	n_116
	188	n_34
	189	n_164


=== References ===
https://github.com/shiruipan/graph_datasets/tree/master/Graph_Repository

=== Previous Use of the Dataset ===
Shirui Pan, Jia Wu, Xingquan Zhu, Guodong Long, and Chengqi Zhang. " Task Sensitive Feature Exploration and Learning for Multi-Task Graph Classification."  IEEE Trans. Cybernetics (TCYB) 47(3): 744-758 (2017)
