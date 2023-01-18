from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ATNLP import helper
from ATNLP.transformer_dataset import TransformerDataset
from ATNLP.transformer_models import TransformerModel


class TransformerTrainer:
    batch_size = 64
    save_dir = './transformer_models'

    def __init__(self,
                 model: TransformerModel,
                 train_dataset: TransformerDataset,
                 test_dataset: TransformerDataset):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.input_lang.PAD_token)

    def train_iteration(self, input_tensor, target_tensor):
        """A single training iteration."""
        # Reset the gradients and loss
        self.model.train()
        self.model.zero_grad()

        output = self.model(input_ids=input_tensor, labels=target_tensor)
        # loss = self.loss_fn(output, target_tensor)
        loss = output.loss

        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, num_iters, save=False, plot=True):
        self.model.train()

        # print_loss_total = 0
        plot_loss_total = 0
        plot_losses = []

        for iteration in tqdm(range(num_iters), desc="Training"):
            batch = np.random.choice(len(self.train_dataset), self.batch_size)
            X, y = self.train_dataset[batch]
            input_tensor, target_tensor = self.train_dataset.convert_to_tensor(X, y)

            loss = self.train_iteration(input_tensor, target_tensor)

            # print_loss_total += loss
            plot_loss_total += loss

            # if iteration % 1000 == 0:
            #     print_loss_avg = print_loss_total / print_every
            #     print_loss_total = 0
            #     print('%d (%d%%): %.4f' % (iteration, iteration / n_iters * 100, print_loss_avg))
            #
            # if iteration % 100 == 0:
            plot_loss_avg = plot_loss_total / 100
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if plot:
            helper.show_plot(plot_losses)

        if save:
            torch.save(self.model.state_dict(),
                       f'{self.save_dir}/model_{num_iters}_{datetime.now().strftime("%d%mT%H%M")}.pth')
        return self.model

    def evaluate(self):
        self.model.eval()

        n_correct = []
        i = 0
        with torch.no_grad():
            for X, y in tqdm(self.test_dataset, desc="Evaluating"):
                input_tensor, target_tensor = self.test_dataset.convert_to_tensor(X, y)
                prediction = self.model.generate(input_tensor, oracle_length=target_tensor.size(1))
                pred = prediction.squeeze().cpu().numpy()
                ground_truth = target_tensor.numpy().squeeze()
                if i < 10:
                    print('\n')
                    print(pred)
                    print(ground_truth)
                    print(pred.shape, ground_truth.shape)
                    i += 1
                else:
                    break

                n_correct.append(np.all(pred == ground_truth))
        accuracy = np.mean(n_correct)

        self.model.train()
        return accuracy

    def oracle_eval(self):
        pass
