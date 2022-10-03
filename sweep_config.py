# Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_1_classification',
    'metric': {'goal': 'maximize', 'name': 'test_accuracy'},
    'parameters': 
    {
        'NUM_GRAPHONS': {'values': [4]},
        'NUM_GRAPHS_PER_GRAPHON': {'values': [50, 100]},
        'NUM_NODES': {'values': [None, 100, 300]}
     }
}