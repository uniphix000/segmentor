#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import argparse
from data_utils import *
from utils import *
import random
from model import Model
import torch.optim as optim
import logging
import codecs
import os
import json
import sys
import collections
from bies_to_sentence import bies_to_segmentation


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')


def train(model, batch_x, batch_y, sentences_lens, optimizer):
    '''

    :param model:
    :param batch_x:
    :param batch_y:
    :param sentences_lens:
    :param optimizer:
    :return:
    '''


def eval_model(model, x_idx, y_idx, batch_size, pad_idx):
    '''

    :param x_idx: ([],[])
    :param y_idx:
    :param batch_size:
    :param pad_idx:
    :return:
    '''
    model.eval()
    x_size = len(x_idx[0])
    order = range(x_size)
    random.shuffle(list(order))
    correct_count , all_count = 0, 0
    predict_all = []
    gold_all = []
    for batch_start in range(0, x_size, batch_size):
        batch_end = batch_start + batch_size if batch_start \
                    + batch_size < x_size else x_size
        batch_size = batch_end - batch_start
        # print("valid_x_idx:{0}".format(valid_x_idx))

        batch_x, batch_y, sentences_lens  = create_batches(x_idx, y_idx, order, batch_start, batch_end, pad_idx)
        #print ("batch_x = {0}, batch_y = {1}".format(batch_x, batch_y))
        batch_x = (batch_x_uni, batch_x_bi) = (batch_x[0].view(batch_size, -1), batch_x[1].view(batch_size, -1))
        #train(model, batch_x, batch_y, sentences_lens, optimizer)
        correct_batch, all_count_batch, predict, y = model.forward(batch_x, batch_y, sentences_lens)
        correct_count += correct_batch
        all_count += all_count_batch
        predict_all.append(predict)
        gold_all.append(y)
        acc = correct_count*1.0 / all_count
    _, _, F = evaluate(predict_all, gold_all)

    return  acc, F, predict_all


def main():
    cmd0 = argparse.ArgumentParser('data from local or hpc')
    cmd0.add_argument("--env", help='', type=int, default=0)
    args0 = cmd0.parse_args([sys.argv[2]])

    cmd = argparse.ArgumentParser("lstm for segmentor")
    cmd.add_argument("--seed", help = "path of test data", type=int, default=1234)
    cmd.add_argument("--env", help = "environment", type=int, default=0)

    if (args0.env == 0):
        cmd.add_argument("--train_path", help = "path of train data", default='../data/train.txt')
        cmd.add_argument("--devel_path", help = "path of devel data", default='../data/valid.txt')
        cmd.add_argument("--test_path", help = "path of test data", default='../data/test.txt')
        cmd.add_argument("--unigram_embed_path", help = "path of test data", default='../data/unigram_100.embed')
        cmd.add_argument("--bigram_embed_path", help = "path of test data", default='../data/bigram_100_test.embed')
    elif (args0.env == 1):
        cmd.add_argument("--train_path", help = "path of train data", default='../data/data/train.txt')
        cmd.add_argument("--devel_path", help = "path of devel data", default='../data/data/valid.txt')
        cmd.add_argument("--test_path", help = "path of test data", default='../data/data/test.txt')
        cmd.add_argument("--unigram_embed_path", help = "path of test data", default='../data/data/unigram_100.embed')
        cmd.add_argument("--bigram_embed_path", help = "path of test data", default='../data/data/bigram_100.embed')

    cmd.add_argument("--model_save_path", help = "path of test data", default='../model/')
    cmd.add_argument("--use_pretrain_unigram_embedding", type = bool, help = "path of test data", default=True)
    cmd.add_argument("--use_pretrain_bigram_embedding", type = bool, help = "path of test data", default=True)
    cmd.add_argument("--batch_size", help = "path of test data", type=int, default=16)
    cmd.add_argument("--max_epoch", help = "path of test data", type=int, default=20)
    cmd.add_argument("--optimizer", help = "path of test data", default='Adam')
    cmd.add_argument("--lr", help = "path of test data", default=0.001)
    cmd.add_argument("--lr_decay", help = "path of test data", type=float, default=1.0)
    cmd.add_argument("--embed_size_uni", help = "path of test data", default=100)
    cmd.add_argument("--embed_size_bi", help = "path of test data", default=100)
    cmd.add_argument("--hidden_size", help = "path of test data", type=int, default=64)
    cmd.add_argument("--dropout", help = "path of test data", type=float, default=0.4)
    cmd.add_argument("--clip_grad", help = "path of test data", default=5)

    args = cmd.parse_args(sys.argv[3:])



    print ('Training Parameters As Following:',args)


    with open ('../record/record', 'a+') as f:
        f.write('-----------------------------\n')
        f.write(str(args))
        f.write('\n')
        f.close()
    logging.info('unigram: {0}, bigram: {1}'.format(args.use_pretrain_unigram_embedding, args.use_pretrain_bigram_embedding))

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logging.info('Data Loading...')
    train_x, train_y, valid_x, valid_y, test_x, test_y = read_data(args.train_path, args.devel_path, args.test_path) #train_x [([],[]),]
    logging.info('Data Loading Complete.')
    logging.info('training_data_size = {0}, valid_data_size = {1}, test_data_size = {2}'.format(len(train_x), len(valid_x), len(test_x)))

    #print (train_x, train_y)
    uni_words, bi_words = [pair[0] for pair in train_x], [pair[1] for pair in train_x]
    word_2_idx_uni = Lang()
    word_2_idx_uni = generate_dict(args.use_pretrain_unigram_embedding, word_2_idx_uni, args.unigram_embed_path, uni_words)
    word_2_idx_bi = Lang()
    word_2_idx_bi = generate_dict(args.use_pretrain_bigram_embedding, word_2_idx_bi, args.bigram_embed_path, bi_words)

    # 保存词典和参数设置信息
    with codecs.open(os.path.join(args.model_save_path, 'uni_dict'), 'w', encoding='utf-8') as fs:
        res_uni_dict = []
        for word, idx in word_2_idx_uni.word2idx.items():
            temp = word+'\t'+str(idx)
            res_uni_dict.append(temp)
        res_uni_dict = '\n'.join(res_uni_dict)
        fs.write(res_uni_dict)
        fs.close()

    with codecs.open(os.path.join(args.model_save_path, 'bi_dict'), 'w', encoding='utf-8') as fs:
        res_bi_dict = []
        for word, idx in word_2_idx_bi.word2idx.items():
            temp = word+'\t'+str(idx)
            res_bi_dict.append(temp)
        res_bi_dict = '\n'.join(res_bi_dict)
        fs.write(res_bi_dict)
        fs.close()

    json.dump(vars(args), codecs.open(os.path.join(args.model_save_path, 'config.json'), 'w', encoding='utf-8'))

    train_x_idx = word_to_idx(train_x, word_2_idx_uni.word2idx, word_2_idx_bi.word2idx)  #([],[])
    train_y_idx = label_to_idx(train_y, label2idx)
    valid_x_idx = word_to_idx(valid_x, word_2_idx_uni.word2idx, word_2_idx_bi.word2idx)
    valid_y_idx = label_to_idx(valid_y, label2idx)
    test_x_idx = word_to_idx(test_x, word_2_idx_uni.word2idx, word_2_idx_bi.word2idx)
    test_y_idx = label_to_idx(test_y,label2idx)
    logging.info('Word2idx Dictionary Generated! Dict Size = {0}, {1}'.format(word_2_idx_uni.get_word_size(),word_2_idx_bi.get_word_size()))
    train_x_size = len(train_x_idx[0]) #句子数
    valid_x_size = len(valid_x_idx[0])
    order = range(train_x_size)
    pad_idx = word_2_idx_uni.word2idx['pad']
    model = Model(args.use_pretrain_unigram_embedding, args.use_pretrain_bigram_embedding, args.embed_size_uni, args.embed_size_bi, \
                  args.hidden_size, len(label2idx), word_2_idx_uni.get_word_size(), word_2_idx_bi.get_word_size(),
                  args.dropout, args.unigram_embed_path, args.bigram_embed_path)
    model = model.cuda() if use_cuda else model
    if (args.optimizer == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif (args.optimizer == 'Adam'):
        optimizer = optim.Adam(model.parameters(), lr=args.lr)


    best_valid_F, best_test_F, best_valid_acc, best_test_acc = 0, 0, 0, 0
    for t in range(args.max_epoch):
        random.shuffle(list(order))
        for batch_start in range(0, train_x_size, args.batch_size):
            batch_end = batch_start + args.batch_size if batch_start \
                        + args.batch_size < train_x_size else train_x_size
            batch_size = batch_end - batch_start
            batch_x, batch_y, sentences_lens  = create_batches(train_x_idx, train_y_idx, order, batch_start, batch_end, pad_idx)
            #到这里batch_x:([],[])
            #batch_x_uni, batch_x_bi = [pair[0] for pair in batch_x], [pair[1] for pair in batch_x]
            batch_x = (batch_x_uni, batch_x_bi) = (batch_x[0].view(batch_size, -1), batch_x[1].view(batch_size, -1))
            #train(model, batch_x, batch_y, sentences_lens, optimizer)
            model.train()
            optimizer.zero_grad()
            loss = model.forward(batch_x, batch_y, sentences_lens)
            #print ("loss = {0}".format(loss))
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            optimizer.step()

        #lr_decay
        if (args.lr_decay > 0):
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        #valid
        valid_acc, valid_F, _ = eval_model(model, valid_x_idx, valid_y_idx, args.batch_size, pad_idx)
        logging.info('--------------------Round {0}---------------------'.format(t))
        if (valid_F > best_valid_F):
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'model.pkl'))  # 存储训练参数
            test_acc, test_F, _ =  eval_model(model, test_x_idx, test_y_idx, args.batch_size, pad_idx)
            logging.info('New Record!!! In Round {0}: test acc: {1}% test F: {2}%'.format(t, test_acc*100, test_F*100))
            best_test_F = max(best_test_F, test_F)
            best_test_acc = max(best_test_acc, test_acc)
        best_valid_F = max(best_valid_F, valid_F)
        best_valid_acc = max(best_valid_acc, valid_acc)
        logging.info('Round {0} ended: valid acc: {1}% F: {2}%'.format(t, valid_acc*100, valid_F*100))
    logging.info('Training Complete!: best test acc:{0}%, F: {1}%; best valid acc: {2}%, F: {3}%'.format(best_test_acc*100,\
                                                            best_test_F*100, best_valid_acc*100, best_valid_F*100))

    with open ('../record/record', 'a+') as f:
        f.write('Training Complete!: best test acc:{0}%, F: {1}%; best valid acc: {2}%, F: {3}%'.format(best_test_acc*100,\
                                                            best_test_F*100, best_valid_acc*100, best_valid_F*100))
        f.write('\n')
        f.close()


def test():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--env', help='it is useless', default=0)
    cmd.add_argument('--input_path', help='', default='../to_be_segmented/input')
    cmd.add_argument('--output_path', help='', default='../segment_result/output')
    cmd.add_argument('--bies_path', help='', default='../to_be_segmented/bies')
    cmd.add_argument('--load_model_path', help='', default='../model')
    args = cmd.parse_args(sys.argv[2:])
    print (args)

    #args_config = json.load(codecs.open(os.path.join(args.load_model_path, 'config.json'), 'r'))  # 会返回一个词典
    args_config = dict2nametuple(json.load(codecs.open(os.path.join(args.load_model_path, 'config.json'), 'r', encoding='utf-8')))

    with codecs.open(os.path.join(args.load_model_path, 'uni_dict'), 'r', encoding='utf-8') as funi:
        word_2_idx_uni = {}
        idx_2_word_uni = {}
        lines = funi.read().strip().split('\n')
        for line in lines:
            word, idx = line.split('\t')  # 这里可以装逼 fixme
            word_2_idx_uni[word] = int(idx)
        idx_2_word_uni = {idx: word for idx,word in word_2_idx_uni.items()}

    with codecs.open(os.path.join(args.load_model_path, 'bi_dict'), 'r', encoding='utf-8') as funi:
        word_2_idx_bi = {}
        idx_2_word_bi = {}
        lines = funi.read().strip().split('\n')
        for line in lines:
            word, idx = line.split('\t')  # 这里可以装逼 fixme
            word_2_idx_bi[word] = int(idx)
        idx_2_word_bi = {idx: word for idx,word in word_2_idx_bi.items()}

    test_x, uni_x = read_corpus_for_test(args.input_path)
    x_idx = word_to_idx(test_x, word_2_idx_uni, word_2_idx_bi)  # ([],[])
    y_idx = x_idx[0]  # 伪造方便程序复用


    model = Model(args_config.use_pretrain_unigram_embedding, args_config.use_pretrain_bigram_embedding, args_config.embed_size_uni, args_config.embed_size_bi, \
                  args_config.hidden_size, len(label2idx), len(word_2_idx_uni), len(word_2_idx_bi),
                  args_config.dropout, args_config.unigram_embed_path, args_config.bigram_embed_path)
    model = model.cuda() if use_cuda else model
    model.load_state_dict(torch.load(os.path.join(args.load_model_path, 'model.pkl')))
    _, _, predict_idx = eval_model(model, x_idx, y_idx, args_config.batch_size, word_2_idx_uni['pad'])
    generate_output(uni_x, predict_idx, args.bies_path, args.output_path)


def generate_output(x, predict_idx, bies_path, output_path):
    res_bies = []
    tmp = 0
    predict_idx = [str(predict_idx[0][i]) for i in range(len(predict_idx[0]))]
    for sentence in x:
        string = ''
        string += ' '.join(sentence)
        string += '\t'
        string += ' '.join(predict_idx[tmp:tmp+len(sentence)])
        tmp += len(sentence)
        res_bies.append(string)
    print (res_bies)
    res_bies = '\n'.join(res_bies)
    with codecs.open(bies_path, 'w', encoding='utf-8') as f:
        f.write(res_bies)
        f.close()
    output =[]
    for line in codecs.open(bies_path, 'r', encoding='utf-8'):
        chars, tags = line.strip().split('\t')
        sentence_split = bies_to_segmentation(chars, tags)
        output.append(sentence_split)
    with codecs.open(output_path, 'w', encoding='utf-8') as fo:
        fo.write('\n'.join(output))
        fo.close()


def dict2nametuple(dict):
    return collections.namedtuple('Namespace', dict.keys())(**dict)





if __name__ == "__main__":
    cmd = argparse.ArgumentParser('train or test')
    cmd.add_argument('--function', help = "running model", default='train')
    args = cmd.parse_args([sys.argv[1]])  #
    if (args.function == 'train'):
        main()
    elif (args.function == 'test'):
        test()
    else:
        raise IOError



































