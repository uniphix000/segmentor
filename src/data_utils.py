#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import codecs
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
oov = 0
label2idx = {'B':0, 'I':1, 'E':2, 'S':3}
idx2label = {idx:label for label, idx in label2idx.items()}


def read_data(train_path, valid_path, test_path):
    '''
    数据格式: [[],]
    :param train_path:
    :param eval_path:
    :param test_path:
    :return:
    '''
    train_x, train_y = read_corpus(train_path)
    valid_x, valid_y = read_corpus(valid_path)
    test_x, test_y  = read_corpus(test_path)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def read_corpus(path):
    words = []
    labels = []
    with codecs.open(path, 'r', encoding = 'utf-8') as f_train:
        f_train_lines = f_train.read().strip().split('\n')
        for line in f_train_lines:
            sentences, sentences_labels = line.split('\t')
            words.append(sentences.split())
            labels.append(sentences_labels.split())
    return words, labels


def load_embedding(embed_path):
    '''

    :param embed_path:
    :return:
    '''
    embed_words, embed_vecs = [], []
    embed_num , embed_size = 0, 0
    with codecs.open(embed_path, 'r', encoding='utf-8') as fp_embed:
        embed_lines = fp_embed.read().strip().split('\n')

        for idx, line in enumerate(embed_lines):
            if (idx == 0):
                embed_num, embed_size = line.split()
                embed_num, embed_size = int(embed_num), int(embed_size)
            else:
                line = line.split()
                embed_words.append(line[0])
                embed_vecs.append([float(item) for item in line[1:]])
    #print (embed_num, embed_size, embed_words, embed_vecs)
    return embed_num, embed_size, embed_words, embed_vecs


class Lang:
    def __init__(self):
        self.word2idx = {'oov':0, 'pad':1}
        self.idx2word = {}
        self.word_size = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.word_size +=1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def get_idx2word(self):
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}
        return self.idx2word

    def get_word2idx(self):
        return self.word2idx

    def get_word_size(self):
        return self.word_size


def word_to_idx(train_x, word2idx):
    '''

    :param train_x:
    :param word2idx:
    :return:
    '''
    train_x_idx =[[word2idx.get(word, oov) for word in sentence] for sentence in train_x]

    # train_x_idx = [[word2idx[word] for word in sentence] for sentence in train_x]

    return train_x_idx


def generate_dict(flag, word_2_idx, embed_path, train_x):
    if (flag == True):
        embed_num, embed_size, embed_words, embed_vecs = load_embedding(embed_path)
        for word in embed_words:
            word_2_idx.add_word(word)
    train_word = flatten(train_x)
    #print(train_word)
    for word in train_word:
        #print(word)
        word_2_idx.add_word(word)

    return word_2_idx


def label_to_idx(train_y, label2idx):
    '''

    :param train_y:
    :param label2idx:
    :return:
    '''
    train_y_idx = [[label2idx[label] for label in sentence] for sentence in train_y]

    return train_y_idx


def flatten(lst):
    return list(itertools.chain.from_iterable(lst))


def create_batches(train_x_idx, train_y_idx, order, batch_start, batch_end, pad_idx):
    '''

    :param train_x_idx:
    :param train_y_idx:
    :param order:
    :param batch_start:
    :param batch_end:
    :return:
    '''
    #print (order[batch_start:batch_end])
    batch_x = [train_x_idx[ids] for ids in order[batch_start:batch_end]]
    batch_y = [train_y_idx[ids] for ids in order[batch_start:batch_end]]

    max_length = max([len(sentence) for sentence in batch_x])
    batch_x_padded = [[sentence + [pad_idx] * (max_length - len(sentence))] \
                      for sentence in batch_x]

    sentences_lens = [len(sentence) for sentence in batch_y]
    flatten_y = flatten(batch_y)
    x_return = Variable(torch.LongTensor(batch_x_padded)).cuda() if use_cuda else Variable(torch.LongTensor(batch_x_padded))
    y_return = Variable(torch.LongTensor(flatten_y)).cuda() if use_cuda else Variable(torch.LongTensor(flatten_y))
    return x_return, y_return, sentences_lens


def predict_extraction(linear_output, sentences_lens):
    '''

    :param linear_output: (batch_size, max_length, label_size)
    :param sentences_lens:
    :return:  (labels_num, label_size)
    '''
    predict = []
    batch_size = len(linear_output)
    for i in range(batch_size):
        predict.append(linear_output[i][:sentences_lens[i]])
    y_predict = torch.cat(predict, 0)

    return y_predict








