import torch
import wandb
import pipeline
import numpy as np
import scan_dataset
import rnn_models as models

from config import overall_best
from matplotlib import pyplot as plt
from config import experiment_2_best as experiment_best

input_lang = scan_dataset.Lang()
output_lang = scan_dataset.Lang()

train_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True
)

test_dataset = scan_dataset.ScanDataset(
    split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False
)

MAX_LENGTH = max(train_dataset.input_lang.max_length, train_dataset.output_lang.max_length)

dump = False
n_iter = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

wandb.login()

wandb.init(project="experiment-1", entity="atnlp")

results = []
# Train 5 times and average the results
for _ in range(5):
    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device,
                                overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'],
                                overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT'],
                                overall_best['ATTENTION']).to(device)
    if dump:
        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device)
    results.append(
        pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False, device=device))

avg_accuracy = sum(results) / len(results)
print('Average accuracy for overall best: {}'.format(avg_accuracy))
wandb.run.summary["Average accuracy for overall best"] = avg_accuracy

results = []
# Train 5 times and average the results
for _ in range(5):
    encoder = models.EncoderRNN(train_dataset.input_lang.n_words, experiment_best['HIDDEN_SIZE'], device,
                                experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'],
                                experiment_best['DROPOUT']).to(device)
    decoder = models.DecoderRNN(train_dataset.output_lang.n_words, experiment_best['HIDDEN_SIZE'],
                                experiment_best['N_LAYERS'], experiment_best['RNN_TYPE'], experiment_best['DROPOUT'],
                                experiment_best['ATTENTION']).to(device)

    encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                      device=device)
    results.append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=MAX_LENGTH, verbose=False))

avg_accuracy = sum(results) / len(results)
print('Average accuracy for experiment best: {}'.format(avg_accuracy))
wandb.run.summary["Average accuracy for experiment best"] = avg_accuracy

splits = ['p1', 'p2', 'p4', 'p8', 'p16', 'p32', 'p64']

results = {}

for split in splits:
    results[split] = []
    for _ in range(5):
        input_lang = scan_dataset.Lang()
        output_lang = scan_dataset.Lang()

        train_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
            split_variation=split,
            input_lang=input_lang,
            output_lang=output_lang,
            train=True
        )

        test_dataset = scan_dataset.ScanDataset(
            split=scan_dataset.ScanSplit.SIMPLE_SPLIT,
            split_variation=split,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False
        )

        encoder = models.EncoderRNN(train_dataset.input_lang.n_words, overall_best['HIDDEN_SIZE'], device,
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], overall_best['DROPOUT']).to(
            device)
        decoder = models.DecoderRNN(train_dataset.output_lang.n_words, overall_best['HIDDEN_SIZE'],
                                    overall_best['N_LAYERS'], overall_best['RNN_TYPE'], experiment_best['DROPOUT'],
                                    experiment_best['ATTENTION']).to(device)

        encoder, decoder = pipeline.train(train_dataset, encoder, decoder, n_iter, print_every=100, learning_rate=0.001,
                                          device=device)
        results[split].append(pipeline.evaluate(test_dataset, encoder, decoder, max_length=100, verbose=False))

# Average results
mean_results = {}
for split, result in results.items():
    mean_results[split] = sum(result) / len(result)

# Find standard deviation
std_results = {}
for split, result in results.items():
    std_results[split] = np.std(result)

# Plot bar chart
plt.bar(list(results.keys()), list(mean_results.values()), align='center', yerr=list(std_results.values()), capsize=5)
plt.xlabel('Percent of commands used for training')
plt.ylabel('Accuracy on new commands (%)')

wandb.log({"Percent commands": plt})
plt.show()

# Print results
for split, result in results.items():
    print('Split: {}, Accuracy: {}'.format(split, sum(result) / len(result)))
