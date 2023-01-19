import helper
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from transformer_dataset import TransformerDataset
from transformer_models import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # self.model.to(device)
        # print(f'Device: {device}')

    def train_iteration(self, input_tensor, target_tensor):
        """A single training iteration."""
        # Reset the gradients and loss
        self.model.train()
        self.model.zero_grad()

        output = self.model(input_ids=input_tensor, labels=target_tensor)
        loss = output.loss

        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, num_iters, path, save=False, plot=True):
        self.model.train()

        # print_loss_total = 0
        plot_loss_total = 0
        plot_losses = []

        for _ in tqdm(range(num_iters), desc="Training", position=0, leave=True):
            batch = np.random.choice(len(self.train_dataset), self.batch_size)
            X, y = self.train_dataset[batch]
            input_tensor, target_tensor = self.train_dataset.convert_to_tensor(X, y)
            loss = self.train_iteration(input_tensor, target_tensor)

            plot_loss_total += loss
            plot_loss_avg = plot_loss_total / 100
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if save:
            torch.save(self.model.state_dict(), f'{self.save_dir}/{path}')

        if plot:
            helper.show_plot(plot_losses)

        return self.model

    def evaluate(self):
        self.model.eval()

        n_correct = []
        # i = 0
        with torch.no_grad():
            for X, y in tqdm(self.test_dataset, desc="Evaluating", position=0, leave=True):
                input_tensor, target_tensor = self.test_dataset.convert_to_tensor(X, y)
                prediction = self.model.generate(input_tensor, oracle_length=target_tensor.size(1))
                pred = prediction.squeeze().cpu().numpy()
                ground_truth = target_tensor.cpu().numpy().squeeze()
                # i += 1
                # if i >= 10:
                #     break

                n_correct.append(np.all(pred == ground_truth))
        accuracy = np.mean(n_correct)

        self.model.train()
        return accuracy

    def oracle_eval(self):
        pass
