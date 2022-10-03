

# Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_1_classification',
    'metric': {'goal': 'maximize', 'name': 'test_accuracy'},
    'parameters': 
    {
        'NUM_GRAPHS_PER_GRAPHON': {'values': [50, 100, 200]},
        'NUM_NODES': {'values': [None, 50, 100]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}