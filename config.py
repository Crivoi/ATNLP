overall_best = {
    'HIDDEN_SIZE': 200,
    'RNN_TYPE': 'LSTM',
    'N_LAYERS': 2,
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': False,  # True or False
}

experiment_1_best = {
    'HIDDEN_SIZE': 50,  # 25, 50, 100, 200, or 400
    'RNN_TYPE': 'GRU',  # RNN, GRU or LSTM
    'N_LAYERS': 1,  # 1 or 2
    'DROPOUT': 0.5,  # 0, 0.1 or 0.5
    'ATTENTION': True,  # True or False
}

experiment_2_best = {
    'HIDDEN_SIZE': 200,
    'RNN_TYPE': 'LSTM',
    'N_LAYERS': 2,
    'DROPOUT': 0,
    'ATTENTION': False,
}

if __name__ == '__main__':
    pass
