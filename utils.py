import os
import json
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

from transformers.tokenization_utils import trim_batch
from rouge import Rouge
rouge = Rouge()

def calculate_rouge(preds, gts):
    assert len(preds) == len(gts)
    result = rouge.get_scores(preds, gts, avg=True)
    return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smooth, tgt_vocab_size, ignore_index=-100):
        assert 0. < label_smooth <= 1.
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smooth / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0).unsqueeze(0))

        self.confidence = 1.0 - label_smooth
        self.lossfct = torch.nn.KLDivLoss(reduction='none')

    def forward(self, pred, target):
        """
        Args:
            pred: [bsz, seq_len, vocab_size]
            target: [bsz, seq_len]

        Returns:
        """
        model_prob = self.one_hot.repeat(target.size(0), target.size(1), 1)  # [bsz, seq_len, vocab_size]
        model_prob.scatter_(2, target.unsqueeze(2), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(2), 0)
        pred_prob = F.log_softmax(pred, dim=2)

        #return F.kl_div(pred_prob, model_prob, reduction='mean')
        loss = self.lossfct(pred_prob, model_prob)
        loss = torch.sum(loss, dim=2).masked_fill_((target == self.ignore_index), 0)
        avg_loss = torch.sum(loss) / torch.sum((target != self.ignore_index).to(torch.float))
        return avg_loss


def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt", max_examples=None):
    examples = []
    if data_path[-3:] == '.gz':
        print('Data file is gzipped')
        f = gzip.open(data_path, "rt")
    else:
        print('Data file is plain text')
        print(data_path)
        f = open(data_path, "r", encoding='utf-8')

    for i, text in enumerate(f.readlines()):
        tokenized = tokenizer.batch_encode_plus( [text + ' </s>'], max_length=max_length, 
            pad_to_max_length=pad_to_max_length, return_tensors=return_tensors )

        if max_examples and i >= max_examples:
            break
        examples.append(tokenized)

    f.close()
    return examples


class SummarizationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir="./cnn-dailymail/cnn_dm/",
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        tokenized=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenized = tokenized
        if not tokenized:
            self.source = encode_file(tokenizer, os.path.join(data_dir, type_path + ".source"), max_source_length)
            self.target = encode_file(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)
        else:
            self.dataset = torch.load(os.path.join(data_dir, type_path))

    def __len__(self):
        if self.tokenized:
            return len(self.dataset)
        else:
            return len(self.source)

    def __getitem__(self, index):
        if self.tokenized:
            dp = self.dataset[index]
            source_ids, src_mask, target_ids = dp[0], dp[1], dp[2]
            source_ids = source_ids[:self.max_source_length]
            src_mask = src_mask[:self.max_source_length]
            target_ids = target_ids[:self.max_target_length]
        else:
            source_ids = self.source[index]["input_ids"].squeeze()
            target_ids = self.target[index]["input_ids"].squeeze()
            src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id, trim_y=True):
        if trim_y:
            y = trim_batch(batch["target_ids"], pad_token_id)
        else:
            y = batch["target_ids"]
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}

