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
    def __init__(self, uni_flag, bi_flag, embed_size_uni, embed_size_bi, hidden_size, label_size, word2idx_uni, word2idx_bi, dropout, uni_embed_path, bi_embed_path):
        super(Model, self).__init__()
        self.embed_size_uni = embed_size_uni
        self.embed_size_bi = embed_size_bi if bi_flag == True else 0
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.uni_flag = uni_flag
        self.bi_flag = bi_flag
        print ('flag:',(uni_flag, bi_flag))
        self.embedding_uni = EmbeddingLayer(self.uni_flag, uni_embed_path, word2idx_uni)
        self.embedding_bi = EmbeddingLayer(self.bi_flag, bi_embed_path, word2idx_bi)
        self.Linear = nn.Linear(2 * hidden_size, label_size)
        self.lstm = nn.LSTM(self.embed_size_uni + 2 * self.embed_size_bi, hidden_size, dropout=dropout, bidirectional=True, batch_first=True)
        self.Nloss = nn.NLLLoss()
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, y, sentences_lens):
        '''

        :param input:
        :return: loss, output
        '''
        if (self.bi_flag == True):
            embed_bi_left = input[1][:,:-1]
            embed_bi_right = input[1][:,1:]
            embed_uni = self.embedding_uni.forward(input[0])
            embed_bi_left = self.embedding_bi.forward(embed_bi_left)
            embed_bi_right = self.embedding_bi.forward(embed_bi_right)
            embed = torch.cat((embed_uni, embed_bi_left, embed_bi_right), 2)
        else:
            embed = self.embedding_uni.forward(input[0])

        embed = self.dropout(embed)  #(b_s, m_l, e_s_u + 2*e_s_b)
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







