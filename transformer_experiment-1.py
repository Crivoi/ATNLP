import os
import gc
import pickle
from collections import defaultdict

import numpy as np
import torch

from helper import bar_plot
from scan_dataset import ScanSplit
from transformer_dataset import TransformerDataset, TransformerLang
from transformer_models import TransformerModel
from transformer_pipeline import TransformerTrainer

os.environ["WANDB_DISABLED"] = "true"

num_iters = 100

input_lang = TransformerLang()
output_lang = TransformerLang()

train_dataset = TransformerDataset(
    split=ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True
)

test_dataset = TransformerDataset(
    split=ScanSplit.SIMPLE_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False
)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=input_lang.PAD_token)

model = TransformerModel()


def experiment_generalization(splits, title='Percent_of_commands_used_for_training'):
    results = defaultdict(list)

    for split in splits:
        train_dataset = TransformerDataset(
            split=ScanSplit.SIMPLE_SPLIT,
            split_variation=split,
            input_lang=input_lang,
            output_lang=output_lang,
            train=True
        )

        test_dataset = TransformerDataset(
            split=ScanSplit.SIMPLE_SPLIT,
            split_variation=split,
            input_lang=input_lang,
            output_lang=output_lang,
            train=False
        )
        trainer = TransformerTrainer(model, train_dataset, test_dataset)
        results[split].append(trainer.evaluate())

    print(f'Length: {results}')
    pickle.dump(results, open(f'./transformer_runs/results_{title}.sav', 'wb'))

    # Average results
    mean_results = {}
    for split, result in results.items():
        mean_results[split] = sum(result) / len(result)

    # Find standard deviation
    std_results = {}
    for split, result in results.items():
        std_results[split] = np.std(result)

    # Plot bar chart
    bar_plot(splits, mean_results, std_results, title=title.replace('_', ' '))


def test_percent_commands(oracle=False):
    splits = ['p1', 'p2', 'p4', 'p8', 'p16', 'p32', 'p64']
    experiment_generalization(splits)


def report_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    report_gpu()

    model_path = f'model_{num_iters}_exp_1.pth'
    save = num_iters >= 100

    def _eval(trainer):
        accuracy = trainer.evaluate()
        print(f'Test accuracy: {accuracy}')
        test_percent_commands()

    if save:
        try:
            model.load_state_dict(torch.load(f'{TransformerTrainer.save_dir}/{model_path}'))
            trainer = TransformerTrainer(model, train_dataset, test_dataset)
            _eval(trainer)
        except FileNotFoundError:
            trainer = TransformerTrainer(model, train_dataset, test_dataset)
            trainer.train(num_iters, save=save, path=model_path)
            _eval(trainer)


if __name__ == '__main__':
    main()
