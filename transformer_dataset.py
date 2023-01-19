from operator import itemgetter

import torch
from torch.nn.utils.rnn import pad_sequence
from scan_dataset import ScanDataset, Lang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerLang(Lang):
    def __init__(self):
        super(TransformerLang, self).__init__()
        self.index2word[self.PAD_token] = "<PAD>"
        self.word2index = dict((v, k) for k, v in self.index2word.items())
        self.n_words = len(self.index2word)

    def tensor_from_sentence(self, sentence: str):
        """Convert sentence to torch tensor"""
        indexes = self.indexes_from_sentence(sentence)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long)


class TransformerDataset(ScanDataset):
    input_lang: TransformerLang
    output_lang: TransformerLang

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return [self.X[idx]], [self.y[idx]]
        elif len(idx) == 1:
            return [self.X[idx[0]]], [self.y[idx[0]]]

        def _getter(x: list):
            return list(itemgetter(*idx)(x))

        X = _getter(self.X)
        y = _getter(self.y)

        return X, y

    def convert_to_tensor(self, X, y):
        for i in range(len(X)):
            X[i] = self.input_lang.tensor_from_sentence(X[i]).to(device)
            y[i] = self.output_lang.tensor_from_sentence(y[i]).to(device)

        input_tensor, target_tensor = self.collate(X, y)
        return input_tensor, target_tensor

    def convert_to_string(self, X, y):
        input_string = self.input_lang.sentence_from_indexes(X)
        target_string = self.output_lang.sentence_from_indexes(y)
        return input_string, target_string

    def collate(self, src_batch, tgt_batch):
        src_batch = pad_sequence(src_batch, padding_value=self.input_lang.PAD_token, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.output_lang.PAD_token, batch_first=True)
        return src_batch, tgt_batch
