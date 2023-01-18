import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

from ATNLP.transformer_dataset import TransformerDataset, TransformerLang
from ATNLP.scan_dataset import ScanSplit
from ATNLP.transformer_models import TransformerModel
from ATNLP.transformer_pipeline import TransformerTrainer

os.environ["WANDB_DISABLED"] = "true"

num_epochs = 3
batch_size = 32

input_lang = TransformerLang()
output_lang = TransformerLang()

train_dataset = TransformerDataset(
    split=ScanSplit.LENGTH_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=True
)

test_dataset = TransformerDataset(
    split=ScanSplit.LENGTH_SPLIT,
    input_lang=input_lang,
    output_lang=output_lang,
    train=False
)


def collate(batch):
    def _flatten(x):
        return list(sum(x, []))

    X, y = zip(*batch)

    X = _flatten(X)
    y = _flatten(y)

    input_tensor, target_tensor = train_dataset.convert_to_tensor(X, y)

    return input_tensor, target_tensor
    # src_batch = pad_sequence(src_batch, padding_value=train_dataset.input_lang.PAD_token, batch_first=True)
    # tgt_batch = pad_sequence(tgt_batch, padding_value=train_dataset.output_lang.PAD_token, batch_first=True)
    # return src_batch, tgt_batch


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=input_lang.PAD_token)

model = TransformerModel()


def main():
    # optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    # model.train()
    # for epoch in range(num_epochs):
    #     for input_tensor, target_tensor in tqdm(train_loader):
    #         optim.zero_grad()
    #         # input_tensor, target_tensor = train_dataset.convert_to_tensor(X, y)
    #         # input_ids = batch['input_ids']
    #         # attention_mask = batch['attention_mask']
    #         # labels = batch['labels']
    #         outputs = model(input_ids=input_tensor, labels=target_tensor)
    #         loss = outputs[0]
    #         loss.backward()
    #         optim.step()

    trainer = TransformerTrainer(model, train_dataset, test_dataset)
    save = num_epochs >= 10
    trainer.train(num_epochs, save=save)
    # print(trainer.evaluate())


if __name__ == '__main__':
    main()
