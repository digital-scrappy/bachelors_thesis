import os
import torch
import torch.autograd as autograd
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import sentencepiece as spm
import numpy as np

class SrcTrgLblData(Dataset):
    def __init__(self, src_filename, trg_filename, labels_filename, bpe_model_path):
        self.src_sents = open(src_filename).readlines()
        self.trg_sents = open(trg_filename).readlines()
        self.labels = open(labels_filename).readlines()

        self.total_sents = len(self.labels)
        self.bpe_model = spm.SentencePieceProcessor(model_file= bpe_model_path)

    def __getitem__(self, idx):
        """Returns three Tensors: src, trg and label."""
        src_sent_txt = self.src_sents[idx]
        trg_sent_txt = self.trg_sents[idx]
        label_txt = self.labels[idx]

        src_idx = self.bpe_model.encode(src_sent_txt, out_type=int)
        trg_idx = self.bpe_model.encode(trg_sent_txt, out_type=int)
        label = label_txt
        return torch.LongTensor(src_idx), torch.LongTensor(trg_idx), torch.FloatTensor([float(label)])

    def __len__(self):
        return self.total_sents

class DataLD(object):
    def __init__(self, src_path, trg_path, labels_path, bpe_model_path, bpe_vocab_size= 88500):

        self.data_dict_size = bpe_vocab_size
        self.dataset = SrcTrgLblData(src_path, trg_path, labels_path, bpe_model_path)

    def collate_fn(self, data):
        """ Creates mini-batch tensors from the list of tuples (src_trg_combo, label).
            We should build a custom collate_fn rather than using default collate_fn,
            because merging sequences (including padding) is not supported in default.
            Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

            :params data: list of tuples (src_trg_combo, label).
            :returns: a torch tensor for the src_trg_combo, a torch tensor for the lables
                      and a list of length (batch_size).
        """
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src, trg, labels = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_sents, src_lengths = merge(src)
        trg_sents, trg_lengths = merge(trg)
        labels, label_lengths = merge(labels)
        return src_sents, src_lengths, trg_sents, trg_lengths, labels, label_lengths

    def get_dict_size(self):
        """ returns the size of the data dictionary """
        return self.data_dict_size

    def get_loader(self, shuf=True, batch_size=100):
        """Returns data loader for custom dataset.

            :params batch_size: the batch size
            :returns: a data loader for the dataset.
        """
        # data loader for custome dataset
        # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
        # please see collate_fn for details
        data_loader = DataLoader(dataset=self.dataset,
                                 batch_size=batch_size,
                                 shuffle=shuf,
                                 collate_fn=self.collate_fn)

        return data_loader
