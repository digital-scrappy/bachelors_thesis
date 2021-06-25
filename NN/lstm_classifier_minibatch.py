import torch
import torch.autograd as autograd
import torch.nn as nn
import torchnlp.nn as nlp
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class SoftDotAttention(nn.Module):
    """ Soft Dot Attention.
        Ref: http://www.aclweb.org/anthology/D15-1166
        Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """ Initialize layer.

        :param dim: Dimmension
        """
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, y, h):
        """Propogates input through the network.

        :param y: batch of sentences, T x batch x dim
        :param h: the hiddent states, batch x dim
        """
        y = y.transpose(1, 0) # batch x T x dim

        t = self.linear_in(h)
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(y, target).squeeze(2)  # batch x T
        attn = F.softmax(attn, dim=1)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x T

        weighted_y = torch.bmm(attn3, y).squeeze(1)  # batch x dim

        h_tilde = torch.cat((weighted_y, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

class RTEAttention(nn.Module):
    """ Word by Word attention.
        Ref: https://arxiv.org/pdf/1509.06664.pdf
    """

    def __init__(self, dim, deviceid=-1):
        """ Initialize network.

        :param dim: Dimmension
        :param deviceid: the device ID to run the training/testing on, deviceid = -1 declares using CPU
        """
        super(RTEAttention, self).__init__()

        self.dim = dim
        self.deviceid = deviceid

        # Attention Parameters
        if self.deviceid == -1: # NO GPU
            self.W_y = nn.Parameter(torch.randn(self.dim, self.dim))
            self.W_h = nn.Parameter(torch.randn(self.dim, self.dim))
            self.W_r = nn.Parameter(torch.randn(self.dim, self.dim))
            self.W_alpha = nn.Parameter(torch.randn(self.dim, 1))

            # Final combination Parameters
            self.W_x = nn.Parameter(torch.randn(self.dim, self.dim))  # dim x dim
            self.W_p = nn.Parameter(torch.randn(self.dim, self.dim))  # dim x dim
        else: # WITH GPU
            self.W_y = nn.Parameter(torch.randn(self.dim, self.dim).cuda())
            self.W_h = nn.Parameter(torch.randn(self.dim, self.dim).cuda())
            self.W_r = nn.Parameter(torch.randn(self.dim, self.dim).cuda())
            self.W_alpha = nn.Parameter(torch.randn(self.dim, 1).cuda())

            # Final combination Parameters
            self.W_x = nn.Parameter(torch.randn(self.dim, self.dim))  # dim x dim
            self.W_p = nn.Parameter(torch.randn(self.dim, self.dim))  # dim x dim

        self.register_parameter('W_y', self.W_y)
        self.register_parameter('W_h', self.W_h)
        self.register_parameter('W_r', self.W_r)
        self.register_parameter('W_alpha', self.W_alpha)
        self.register_parameter('W_x', self.W_x)
        self.register_parameter('W_p', self.W_p)

    def forward(self, y, mask_y, h):
        """ Computes the attention weights over y using h
            Returns an attention weighted representation of y, and the alphas

        :param y: The input of sentences, T x batch x dim
        :param mask_y: Mask for the input, T x batch
        :param h: Hidden states, batch x dim
        :returns: r, batch x dim
                  alpha, batch x T
        """
        y = y.transpose(1, 0)  # batch x T x dim

        mask_y = mask_y.transpose(1, 0)  # batch x T
        Wy = torch.bmm(y, self.W_y.unsqueeze(0).expand(y.size(0), *self.W_y.size()))  # batch x T x dim
        Wh = torch.mm(h, self.W_h)  # batch x dim

        M = torch.tanh(Wy + Wh.unsqueeze(1).expand(Wh.size(0), y.size(1), Wh.size(1)))  # batch x T x dim
        alpha = torch.bmm(M, self.W_alpha.unsqueeze(0).expand(y.size(0), *self.W_alpha.size())).squeeze(-1)  # batch x T

        alpha = alpha + (-1000.0 * (1. - mask_y))  # To ensure probability mass doesn't fall on non tokens
        alpha = F.softmax(alpha, dim=1)
        r = torch.bmm(alpha.unsqueeze(1), y).squeeze(1)  # batch x dim

        h_star = self.combine_last(r, h)

        return h_star, alpha

    def combine_last(self, r, hidden):
        """ Combining two matrixes

        :param r: r, batch x dim
        :param hidden: hidden states, batch x dim
        :returns: the tanh transformation of the combined matrixes
        """
        W_p_r = torch.mm(r, self.W_p)
        W_x_h = torch.mm(hidden, self.W_x)
        h_star = torch.tanh(W_p_r + W_x_h)

        return h_star


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def where(self, cond, x_1, x_2):
        cond = cond.float()
        return (cond * x_1) + ((1-cond) * x_2)

#    def forward(self, similarity, label):
#        ''' Computes the contrastive loss based on a similarity measure
#
#        :param similarity: the similarity score
#        :param label: the label towards which it should compare (the similarity)
#        :returns: the contrastive loss
#        '''
#        loss_contrastive = torch.mean((1.0-label) * self.where(similarity < self.margin, torch.pow(similarity, 2), 0) +
#                                      (label) * 0.25 * torch.pow((1.0 - similarity), 2))
#        return loss_contrastive

    def forward(self, similarity, label):
        ''' Computes the contrastive loss based on a similarity measure

        :param similarity: the similarity score
        :param label: the label towards which it should compare (the similarity)
        :returns: the contrastive loss
        '''
        loss_contrastive = torch.mean((1.0-label) * torch.pow(similarity, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))
        return loss_contrastive



class SiameseSimilarity(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dict_size, batch_size, metric='eucledian', attention_type='dot',dropout = 0.2, deviceid=-1):
        ''' Init method for the NN - bidirectional LSTM

        :param embedding_dim: The embedding dimension
        :param hidden_dim: the dimmention for the hidden states
        :param dict_size: the dictionary size
        :param batch_size: the batch size
        :param metric: the metric used for the similarity; default is Eucledian (options are Eucledian, Manhattan, Cosine similarity)
        :param device: the device to run the training/test; default is -1 = CPU
        '''
        super(SiameseSimilarity, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.deviceid = deviceid
        self.metric = metric
        self.dropout = dropout

        self.dropout = nn.Dropout(dropout) # Add a variable

        self.dtype = torch.FloatTensor
        if self.deviceid > -1:
            self.dtype = torch.cuda.FloatTensor

        self.attention_type = attention_type
        if self.attention_type == 'dot':
            self.attention_left = SoftDotAttention(hidden_dim)
            self.attention_right = SoftDotAttention(hidden_dim)
        elif self.attention_type == 'rte':
            self.attention_left = RTEAttention(hidden_dim, deviceid)
            self.attention_right = RTEAttention(hidden_dim, deviceid)
        elif self.attention_type == 'nlpAttDot':
            self.attention_left = nlp.Attention(hidden_dim)
            self.attention_right = nlp.Attention(hidden_dim)
            # print(hidden_dim)
        else:
            self.attention_left = None
            self.attention_right = None

        self.word_embeddings = nn.Embedding(dict_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        self.fc1 = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim ),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, embedding_dim))
        self.fc2 = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, embedding_dim))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        ''' initializes the hidden states

        :returns: first hidden and first cell
        '''
        # the first is the hidden h            sim = F.pairwise_distance(h1, h2)
        # the second is the cell  c
        if self.deviceid > -1:
            h0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2).cuda())
            c0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2))
            c0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2))
        return (h0, c0)



    def forward_once(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.dropout(embeds)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        return lstm_out, self.hidden


    def forward(self, sentence_left, sentence_right):
        ''' Forward method for the Siamese architecture

        :param sentence_left: the first sentence to compare
        :param sentence_right: the second sentence to compare
        :returns: output of the network for the first and the second sentence, as well as the similarity
        '''

        lstm_out_left, hidden_left = self.forward_once(sentence_left)
        lstm_out_right, hidden_right = self.forward_once(sentence_right)

        mask_left = torch.ne(sentence_left, 0).type(self.dtype)
        mask_right = torch.ne(sentence_right, 0).type(self.dtype)

        h_left = hidden_left[0] #view(self.batch_size, -1)
        h_right = hidden_right[0] #view(self.batch_size, -1)

        if self.attention_type == 'rte':
            h_left_star, alpha_left_vec = self.attention_left.forward(lstm_out_left, mask_left, lstm_out_right[-1])
            h_right_star, alpha_right_vec = self.attention_right.forward(lstm_out_right, mask_right, lstm_out_left[-1])
        elif self.attention_type == 'dot':
            h_left_star, alpha_left_vec = self.attention_left.forward(lstm_out_left, lstm_out_right[-1])
            h_right_star, alpha_right_vec = self.attention_right.forward(lstm_out_right, lstm_out_left[-1])
        elif self.attention_type == 'nlpAttDot':
            output_left, weights_left = self.attention_left(lstm_out_left[-1].unsqueeze(0).transpose(0, 1), lstm_out_right.transpose(0, 1))
            output_right, weights_right = self.attention_right(lstm_out_right[-1].unsqueeze(0).transpose(0, 1), lstm_out_left.transpose(0, 1))

            h_left_star = torch.bmm(weights_right, lstm_out_left.transpose(0, 1)).squeeze(1)
            h_right_star = torch.bmm(weights_left, lstm_out_right.transpose(0, 1)).squeeze(1)
        else:
            h_left_star = h_left.view(h_left.shape[1], -1)
            h_right_star = h_right.view(h_right.shape[1], -1)

        if self.metric == 'manhattan':
            sim = torch.exp(-torch.norm((h_left_star - h_right_star), 1))
        elif self.metric == 'eucledian':
            sim = F.pairwise_distance(h_left_star, h_right_star)
        elif self.metric == 'cosine':
            sim = F.cosine_similarity(h_left_star, h_right_star)

        return lstm_out_left[-1], lstm_out_right[-1], sim
