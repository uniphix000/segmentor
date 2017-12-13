#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import codecs
with codecs.open('output_seg.txt', 'r', encoding = 'utf-8') as f_train:
        f_train_lines = f_train.read().strip().split('\n')
        num = len(f_train_lines)
        count = 0
        res_train = ""
        res_valid = ""

        for index, line in enumerate(f_train_lines):
            if index > 0.02*num:
                break
            if(index < 0.02*0.7*num):
                res_train+=line
                res_train+='\n'
            else:
                res_valid += line
                res_valid += '\n'
        fp_train = open('train.txt', 'w')
        fp_train.write(res_train)
        fp_valid = open('valid.txt', 'w')
        fp_valid.write(res_valid)
