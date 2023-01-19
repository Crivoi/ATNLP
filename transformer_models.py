import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5ForConditionalGeneration, T5Config

from scan_dataset import PAD_token, SOS_token, EOS_token


class TransformerModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.transformer = T5ForConditionalGeneration.from_pretrained('t5-base')

    def forward(self, input_ids, labels):
        return self.transformer(input_ids=input_ids, labels=labels)

    def to(self, device):
        self.transformer.to(device)

    def generate(self, input_tensor, oracle_length):
        max_length = 50
        min_length = 1

        if oracle_length is not None:
            max_length = oracle_length
            min_length = oracle_length

        prediction = self.transformer.generate(
            input_ids=input_tensor,
            pad_token_id=PAD_token,
            bos_token_id=SOS_token,
            eos_token_id=EOS_token,
            forced_eos_token_id=EOS_token,
            num_beams=1,
            max_length=max_length + 1,
            min_length=min_length
        )

        return prediction[:, 1:]
