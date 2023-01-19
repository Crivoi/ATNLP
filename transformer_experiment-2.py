import os
import gc
import torch
from torch.utils.data import DataLoader

from scan_dataset import ScanSplit
from transformer_dataset import TransformerDataset, TransformerLang
from transformer_models import TransformerModel
from transformer_pipeline import TransformerTrainer

os.environ["WANDB_DISABLED"] = "true"

num_epochs = 10

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


# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=input_lang.PAD_token)

model = TransformerModel()


def report_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    report_gpu()
    trainer = TransformerTrainer(model, train_dataset, test_dataset)
    save = num_epochs >= 10
    trainer.train(num_epochs, save=save)
    # print(trainer.evaluate())


if __name__ == '__main__':
    main()
