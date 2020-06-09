import copy

def load_config(dataset):
    if dataset == "BBBP":
        config = bbbp_cls
    elif dataset == "BACE":
        config = bace_cls
    elif dataset == "TOX21":
        config = tox21_cls
    return config


#NB: "base" != "bace"
base_config = {}
base_config['data_dir'] = '../data'
base_config['saved_models_dir'] = '../saved_models'
base_config['results_dir'] = '../results'
base_config['fig_dir'] = '../figs'

bbbp_cls = copy.deepcopy(base_config)
bbbp_cls['data_fn'] = 'BBBP.csv'
bbbp_cls['d'] = 75
bbbp_cls['init_stddev'] = 0.1
bbbp_cls['L1'] = 128
bbbp_cls['L2'] = 256
bbbp_cls['L3'] = 512
bbbp_cls['N'] = None
bbbp_cls['batch_size'] = 1
bbbp_cls['num_epochs'] = 100
bbbp_cls['num_classes'] = 2

bace_cls = copy.deepcopy(base_config)
bace_cls['data_fn'] = 'bace_processed.csv'
bace_cls['d'] = 75
bace_cls['init_stddev'] = 0.1
bace_cls['L1'] = 128
bace_cls['L2'] = 256
bace_cls['L3'] = 512
bace_cls['N'] = None
bace_cls['batch_size'] = 1
bace_cls['num_epochs'] = 100
bace_cls['num_classes'] = 2

tox21_cls = copy.deepcopy(base_config)
tox21_cls['data_fn'] = 'tox21_NR-ER_processed.csv'
tox21_cls['d'] = 75
tox21_cls['init_stddev'] = 0.1
tox21_cls['L1'] = 128
tox21_cls['L2'] = 256
tox21_cls['L3'] = 512
tox21_cls['N'] = None
tox21_cls['batch_size'] = 1
tox21_cls['num_epochs'] = 100
tox21_cls['num_classes'] = 2
