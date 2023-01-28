import math
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, AdamW, GPT2LMHeadModel
import random
import datetime

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '

class Dictionary:
    def __init__(self):
        self.idx2term = []
        self.term2idx = {}

    def add_entity(self, term):
        if term not in self.term2idx:
            self.term2idx[term] = len(self.idx2term)
            self.idx2term.append(term)

    def __len__(self):
        return len(self.idx2term)

class Batchify:
    def __init__(self, data, tokenizer, bos, eos, batch_size=128, shuffle=False):
        u, i, r, t, a = [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating']/5)
            a.append(x['aspect'])
            t.append('{} {} {}'.format(bos, x['text'], eos))

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.aspect = torch.tensor(a, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_size = len(data)
        self.index = list(range(self.data_size))
        self.total_step = int(math.ceil(self.data_size / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.data_size)
        self.step += 1
        index = self.index[start:offset]
        user = self.user[index]
        item = self.item[index]
        seq = self.seq[index]
        aspect = self.aspect[index]
        mask = self.mask[index]
        rating = self.rating[index]
        return user, item, seq, mask, aspect, rating
