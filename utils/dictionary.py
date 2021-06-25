# -*- coding: utf-8 -*-
import os
import torch
import codecs
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class Dictionary(object):
    def __init__(self, filename):
        """ Initialise the dictionary """
        self.tok2idx = {'<sos>':0, '<break>':1, '<eos>':2, '<unk>':3}
        self.idx2tok = ['<sos>', '<break>', '<eos>', '<unk>']

        self.filename = filename
        
    def load_dictionary(self):
        """ add a new token to the dictionary
        
        :param word: the word to be added to the dictionary
        """
        with codecs.open(self.filename, 'r', 'utf8') as f:
            for token in f:
                token = token.strip()
                if token not in self.tok2idx:
                    self.idx2tok.append(token)
                    self.tok2idx[token] = len(self.idx2tok) - 1

    def extract_dictionary(self):
        """ read a file and create a dictionary from its words/tokens """
        with codecs.open(self.filename, 'r', 'utf8') as f:
            for sent in f:
                for token in sent.strip().split(): # tokenisation is based on space
                    if token not in self.tok2idx:
                        self.idx2tok.append(token)
                        self.tok2idx[token] = len(self.idx2tok) - 1
                    
    def print_dictionary(self):
        """ print the tokens one per line """
        for token in self.idx2tok:
            print(token)
            
    def get_idx(self, token):
        """ Returns an index from the dictionary for the given token
        
        :param token: a word or a token that we want the index of
        :returns: the index of the token (i.e., one-k encoding).
        """
        if token in self.tok2idx:
            idx = self.tok2idx[token]
        else:
            idx = self.tok2idx['<unk>']
        
        return idx

    def get_token(self, idx):
        """ Returns the token that corresponds to the index
        
        :param idx: an index
        :returns: the token
        """
        if idx < len(self.idx2tok):
            token = self.idx2tok[idx]
        else:
            token = '<unk>'

        # token = one_hot_encoding(token)
        return token

    def __len__(self):
        return len(self.idx2tok)

class Labels(Dictionary): 
    def __init__(self, filename):
        """ Initialise the dictionary """
        self.tok2idx = {}
        self.idx2tok = []
        self.filename = filename
        
    def load_labels(self):
        super.load_dictionary()
                    
    def print_labels(self):
        super.print_dictionary()
            
    def get_idx(self, label):
        """ Returns the index of the given label
        
        :param label: a label that we want the index of
        :returns: the index of the label (i.e., one-k encoding).
        """
        idx = self.tok2idx[label]
        return idx

    def get_label(self, idx):
        """ Returns the label that corresponds to the index
        
        :param idx: an index
        :returns: the label
        """
        label = self.idx2tok[idx]
        return label
