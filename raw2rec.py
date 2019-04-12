import os

import click
import numpy as np
import pandas as pd

from sknlp.dataset import SimpleIndexedRecordIO


def str_w2n(wide_str):
    """
    全角转半角
    """
    chars = []
    for uchar in wide_str:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        chars.append(chr(inside_code))
    return ''.join(chars)


def _create_tags(length, suffix):
    if length == 0:
        return []
    if suffix == 'O':
        tags = ['O'] * length
    else:
        tags = ['-'.join(['I', suffix])] * length
        tags[0] = '-'.join(['B', suffix])
    return tags


def msra2rec(file_dir):
    for data_type in ('test', 'train'):
        filename = os.path.join(file_dir, f'{data_type}.txt')
        idx = os.path.join(file_dir, f'{data_type}.idx')
        rec = os.path.join(file_dir, f'{data_type}.rec')

        dataset = SimpleIndexedRecordIO(idx, rec, 'w')
        dataset.open()
        with open(filename) as f:
            for line in f:
                texts = []
                tags = []
                for cell in line.split():
                    text, tag = cell.split('/')
                    tag = tag.upper()
                    texts.append(str_w2n(text).replace('\t', ' '))
                    tags.append('|'.join(_create_tags(len(text), tag)))
                r = '\t'.join([''.join(texts), '|'.join(tags)]).encode('utf-8')
                dataset.write(r)
        dataset.close()


def waimai2rec(file_dir):
    filename = 'waimai_10k.csv'
    df = pd.read_csv(os.path.join(file_dir, filename), dtype=str)
    df.fillna('', inplace=True)
    size = df.shape[0]
    idx = np.random.permutation(size)

    train_size = int(size * 0.7)
    train = df.loc[idx[:train_size]]
    test = df.loc[idx[train_size:]]
    data = {'train': train, 'test': test}
    for data_type in ('train', 'test'):
        idx = os.path.join(file_dir, f'{data_type}.idx')
        rec = os.path.join(file_dir, f'{data_type}.rec')

        dataset = SimpleIndexedRecordIO(idx, rec, 'w')
        dataset.open()
        for i, row in data[data_type].iterrows():
            dataset.write('\t'.join([
                str_w2n(row.review).replace('\t', ' '),
                row.label]).encode('utf-8'))
        dataset.close()


def intent2rec(file_dir):
    for data_type in ('train', 'test'):
        df = pd.read_csv(os.path.join(file_dir, f'{data_type}.csv'),
                         sep='\t', dtype=str)
        df.fillna('', inplace=True)
        idx = os.path.join(file_dir, f'{data_type}.idx')
        rec = os.path.join(file_dir, f'{data_type}.rec')

        dataset = SimpleIndexedRecordIO(idx, rec, 'w')
        dataset.open()
        for i, row in df.iterrows():
            intent = row.intent
            if intent == 'nonsense':
                intent = ''
            dataset.write('\t'.join([
                str_w2n(row.text).replace('\t', ' '), intent]).encode('utf-8'))
        dataset.close()


if __name__ == '__main__':
    intent2rec('datasets/intent')
