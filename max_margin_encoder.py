#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
import dynet as dy
import codecs
import random
import glob
import numpy as np
import sys
import argparse
import os.path
import time

BATCH_SIZE = 32
PATIENCE = 30
UNK = 'unk'

class MaxMarginEncoder():
    def __init__(self, embed_size, hidden_size, model_name, train_file=None, val_file=None):
        self.model_name = model_name
        self.model = dy.ParameterCollection()

        self.source_vocab = defaultdict(lambda: len(self.source_vocab))
        self.target_vocab = defaultdict(lambda: len(self.target_vocab))
        self.training_data = self.read_train(train_file)
        if val_file != None:
            self.validation_data = self.read_data(val_file)

        self.source_lookup = self.model.add_lookup_parameters((len(self.source_vocab), embed_size))
        self.target_lookup = self.model.add_lookup_parameters((len(self.target_vocab), embed_size))
        self.source_lstm_forward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)
        self.source_lstm_backward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)
        self.target_lstm_forward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)
        self.target_lstm_backward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)

        if os.path.isfile(self.model_name):
            self.model.populate(self.model_name)
            print("Populated! " + self.model_name)

        print('init done')

    def read_train(self, file_name):
        parallel_data = []
        with codecs.open(file_name, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                source_chars = [self.source_vocab[x] for x in spl[1]]
                target_chars = [self.target_vocab[x] for x in spl[2]]
                parallel_data.append((source_chars, target_chars))
        self.source_vocab[UNK]
        self.target_vocab[UNK]
        return parallel_data

    def read_data(self, file_name):
        parallel_data = []
        with codecs.open(file_name, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                source_chars = [self.char2int(self.source_vocab, x) for x in spl[1]]
                target_chars = [self.char2int(self.target_vocab, x) for x in spl[2]]
                parallel_data.append((source_chars, target_chars))
        return parallel_data

    def char2int(self, vocab, char):
        if char in vocab:
            return vocab[char]
        else:
            return vocab[UNK]

    def encode(self, entries, source=True):
        if source:
            return self.convert(entries, self.source_vocab, self.source_lookup, self.source_lstm_forward, self.source_lstm_backward)
        else:
            return self.convert(entries, self.target_vocab, self.target_lookup, self.target_lstm_forward, self.target_lstm_backward)

    def convert(self, entries, vocab, lookup, fwd, bwd):
        all_reps = []
        batch_size = 512
        for i in range(0, len(entries), batch_size):
            dy.renew_cg()
            cur_size = min(batch_size, len(entries)-i)
            batch = entries[i:i+cur_size]
            temps = [[self.char2int(vocab, x) for x in entry] for entry in batch]
            embs = [[lookup[y] for y in temp] for temp in temps]
            reps = [dy.concatenate([fwd.initial_state().transduce(emb)[-1], bwd.initial_state().transduce(reversed(emb))[-1]]) for emb in embs]
            reps_norm = [dy.cdiv(rep,dy.l2_norm(rep)).value() for rep in reps]
            all_reps += reps_norm
        return np.array(all_reps)

    def get_val_recall(self):
        recall1 = 0.0
        total = 0

        source_reps = []
        target_reps = []

        for i in range(0, len(self.validation_data), BATCH_SIZE):
            dy.renew_cg()
            cur_size = min(BATCH_SIZE, len(self.validation_data)-i)
            batch = self.validation_data[i:i+cur_size]
            embs = [([self.source_lookup[x] for x in s],[self.target_lookup[y] for y in t]) for s, t in batch]
            source_word_reps = [dy.concatenate([self.source_lstm_forward.initial_state().transduce(emb)[-1], self.source_lstm_backward.initial_state().transduce(reversed(emb))[-1]]) for emb,target in embs]
            source_reps_norm = [dy.cdiv(rep, dy.l2_norm(rep)).value() for rep in source_word_reps]
            source_reps += source_reps_norm
            target_word_reps = [dy.concatenate([self.target_lstm_forward.initial_state().transduce(emb)[-1], self.target_lstm_backward.initial_state().transduce(reversed(emb))[-1]]) for source,emb in embs]
            target_reps_norm = [dy.cdiv(rep, dy.l2_norm(rep)).value() for rep in target_word_reps]
            target_reps += target_reps_norm

        for i in range(0, len(self.training_data), BATCH_SIZE):
            dy.renew_cg()
            cur_size = min(BATCH_SIZE, len(self.training_data)-i)
            batch = self.training_data[i:i+cur_size]
            embs = [[self.source_lookup[x] for x in s] for s, t in batch]
            source_word_reps = [dy.concatenate([self.source_lstm_forward.initial_state().transduce(emb)[-1], self.source_lstm_backward.initial_state().transduce(reversed(emb))[-1]]) for emb in embs]
            source_reps_norm = [dy.cdiv(rep, dy.l2_norm(rep)).value() for rep in source_word_reps]
            source_reps += source_reps_norm

        scores = np.array(target_reps).dot(np.array(source_reps).T)

        for entry_idx, entry_scores in enumerate(scores):
            ranks = entry_scores.argsort()[::-1]
            if ranks[0] == entry_idx:
                recall1 += 1
            total += 1

        return recall1/total

    def calculate_loss(self, words):
        dy.renew_cg()
        embs = [([self.source_lookup[x] for x in s],[self.target_lookup[y] for y in t]) for s, t in words]
        source_word_reps = [dy.concatenate([self.source_lstm_forward.initial_state().transduce(emb)[-1], self.source_lstm_backward.initial_state().transduce(reversed(emb))[-1]]) for emb,target in embs]
        source_reps_norm = [dy.cdiv(rep, dy.l2_norm(rep)) for rep in source_word_reps]
        target_word_reps = [dy.concatenate([self.target_lstm_forward.initial_state().transduce(emb)[-1], self.target_lstm_backward.initial_state().transduce(reversed(emb))[-1]]) for source,emb in embs]
        target_reps_norm = [dy.cdiv(rep, dy.l2_norm(rep)) for rep in target_word_reps]

        mtx_src = dy.concatenate_cols(source_reps_norm)
        mtx_trg = dy.concatenate_cols(target_reps_norm)
        similarity_mtx = dy.transpose(mtx_src) * mtx_trg
        loss = dy.hinge_dim(similarity_mtx, list(range(len(words))), d=1)      

        return dy.sum_elems(loss)/(len(words)*len(words))

    def train(self, epochs, trainer):
        if trainer == 'sgd':
            trainer = dy.SimpleSGDTrainer(self.model)
        elif trainer == 'adam':
            trainer = dy.AdamTrainer(self.model)

        best_recall = 0
        last_updated = 0

        for ep in range(epochs):
            ep_loss = 0
            random.shuffle(self.training_data)
            print("Epoch: %d" % ep)
            for i in range(0, len(self.training_data), BATCH_SIZE):
                cur_size = min(BATCH_SIZE, len(self.training_data)-i)
                loss = self.calculate_loss(self.training_data[i:i+cur_size])
                ep_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
            print("Train loss: %f" % (ep_loss/len(self.training_data)))
            
            if ep % 5 == 0:
                recall = self.get_val_recall()
                if recall > best_recall:
                    best_recall = recall
                    last_updated = ep
                    print('Saved: %0.4f' % best_recall)
                    self.model.save(self.model_name)
                elif ep-last_updated == PATIENCE:
                    break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default='300')
    parser.add_argument('--trainer', default='sgd')
    parser.add_argument('--char_embed_size', default='64')
    parser.add_argument('--lstm_size', default='512')
    parser.add_argument('--train_file')
    parser.add_argument('--val_file')
    parser.add_argument('--model_file', default='out')
    args, unknown = parser.parse_known_args()

    sys.stdout=open(args.model_file.split('/')[-1] + '_' + args.trainer + '.log', 'w', 0)
    mm_encoder = MaxMarginEncoder(int(args.char_embed_size), int(args.lstm_size)*2, args.model_file, args.train_file, args.val_file)
    mm_encoder.train(int(args.epochs), args.trainer)
    sys.stdout.close()

