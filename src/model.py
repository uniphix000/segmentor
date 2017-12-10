#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from data_utils import predict_extraction
from data_utils import load_embedding

class Model(nn.Module):
    def __init__(self, flag, embed_size, hidden_size, label_size, word2idx, dropout, embed_path):
        super(Model, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.V = word2idx.get_word_size()
        self.embedding = EmbeddingLayer(flag, embed_path, word2idx)
        self.Linear = nn.Linear(2 * hidden_size, label_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, dropout=dropout, bidirectional=True, batch_first=True)
        self.Nloss = nn.NLLLoss()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, y, sentences_lens):
        '''

        :param input:
        :return: loss, output
        '''

        embed = self.embedding.forward(input)
        embed = self.dropout(embed)
        output, _ = self.lstm(embed)  #(batch_size, max_length, hidden_size)
        linear_output = self.Linear(output)  #(batch_size, max_length, label_size)
        y_predict = predict_extraction(linear_output, sentences_lens)
        if (self.training):
            y_score = F.log_softmax(y_predict)
            loss = self.Nloss(y_score, y)
            return loss
        else:
            _ ,max_id= torch.max(y_predict, 1)
            minus = list(max_id.data - y.data)
            correct_count = minus.count(0)
            return correct_count, len(y), max_id.data, y.data


class EmbeddingLayer(nn.Module):

    def __init__(self, flag , embed_path, word_2_idx):
        super(EmbeddingLayer, self).__init__()
        self.embed_num, self.embed_size, self.embed_words, self.embed_vecs = load_embedding(embed_path)
        self.V = word_2_idx.get_word_size()
        self.d = self.embed_size
        self.pad = word_2_idx.get_word2idx()['pad']
        self.embedding = nn.Embedding(self.V, self.d, padding_idx=self.pad)
        #self.embedding.weight.data.uniform(-0.25, 0.25)
        weight = self.embedding.weight
        if (flag == True):
            weight.data[:self.embed_num].copy_(torch.FloatTensor(self.embed_vecs))

    def forward(self, input):
        return self.embedding(input)







