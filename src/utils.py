#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')


def evaluate2(predict_all, gold_all):
    '''
    计算分词结果的P,R,F值,对每个(predict,gold)对,将其按照分词结果分成若干形同(i,j)的元组,只有元组完全
    相同才算一次分词正确
    :param predict_all:
    :param gold_all:
    :return:
    '''
    assert len(predict_all) == len(gold_all)
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(gold_all)):
        predict_sets = get_sets(predict_all[i])
        gold_sets = get_sets(gold_all[i])
    fn += len(gold_sets)
    for predict_set in predict_sets:
        if (predict_set) in gold_sets:
            tp += 1
            fn -= 1
        else:
            fp += 1
    P = 0 if tp == 0 else 1.0 * tp / (tp + fp)
    R = 0 if tp == 0 else 1.0 * tp / (tp + fn)
    F = 0 if P * R == 0 else 2.0 * P * R / (P + R)
    return P, R, F


def get_sets(labels):
    '''
    BIES:0123
    :param labels: 一句话的分词结果
    :return:
    '''
    labels = list(labels)
    l = len(labels)
    start, end = 0, 0
    sets = []
    for i in range(l):
        if (i == l-1):
            sets.append((start,i))
            return sets
        if (labels[i] == 0):
            start = end = i
        elif (labels[i] == 1):
            end += 1
        elif (labels[i] == 2):
            end += 1
            sets.append((start, end))
        elif (labels[i] == 3):
            sets.append((i,i))
            start += 1
            end += 1

    return sets


def evaluate(gold, predicted):
  assert len(gold) == len(predicted)
  tp = 0
  fp = 0
  fn = 0
  for i in range(len(gold)):
    gold_intervals = get_intervals(gold[i])
    predicted_intervals = get_intervals(predicted[i])
    seg = set()
    for interval in gold_intervals:
      seg.add(interval)
      fn += 1
    for interval in predicted_intervals:
      if (interval in seg):
        tp += 1
        fn -= 1
      else:
        fp += 1
  P = 0 if tp == 0 else 1.0 * tp / (tp + fp)
  R = 0 if tp == 0 else 1.0 * tp / (tp + fn)
  F = 0 if P * R == 0 else 2.0 * P * R / (P + R)
  return P, R, F


def get_intervals(tag):
  intervals = []
  l = len(tag)
  i = 0
  while (i < l):
    if (tag[i] == 2 or tag[i] == 3):
      intervals.append((i, i))
      i += 1
      continue
    j = i + 1
    while (True):
      if (j == l or tag[j] == 0 or tag[j] == 3):
        intervals.append((i, j - 1))
        i = j
        break
      elif (tag[j] == 2):
        intervals.append((i, j))
        i = j + 1
        break
      else:
        j += 1
  return intervals