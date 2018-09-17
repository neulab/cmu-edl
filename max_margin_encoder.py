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
import panphon as pp

BATCH_SIZE = 64
ENCODE_BATCH = 512
PATIENCE = 50
EPOCH_CHECK = 5
VEC_SIZE = 22
UNK = 'unk'
SRC = 'src'
TGT = 'tgt'

class MaxMarginEncoder():
    def __init__(self, embed_size, hidden_size, panphon, model_name, load_model=False, train_file=None, val_file=None):
        self.model_name = model_name
        self.model = dy.ParameterCollection()
        self.panphon = panphon
        
        if self.panphon:
            self.ft = pp.FeatureTable()
            self.ws_panphon = self.model.add_parameters((embed_size, VEC_SIZE))
            self.bs_panphon = self.model.add_parameters((embed_size))
        else:
            self.source_vocab = defaultdict(lambda: len(self.source_vocab))
            self.target_vocab = defaultdict(lambda: len(self.target_vocab))
            self.source_lookup = self.model.add_lookup_parameters((len(self.source_vocab), embed_size))
            self.target_lookup = self.model.add_lookup_parameters((len(self.target_vocab), embed_size))

        self.training_data = self.read_train(train_file)
        if val_file:
            self.validation_data = self.read_data(val_file)

        self.source_lstm_forward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)
        self.source_lstm_backward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)
        self.target_lstm_forward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)
        self.target_lstm_backward = dy.LSTMBuilder(1, embed_size, hidden_size/2, self.model)

        # load model only if flag is true. will overwrite existing model if flag is false. set flag to True for fine-tuning or encoding
        if load_model:
            self.model.populate(self.model_name)
            print("Populated! " + self.model_name)

        print('done')

    def read_train(self, file_name):
        parallel_data = []
        with codecs.open(file_name, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                if self.panphon:
                    source = self.get_feature(spl[1])
                    target = self.get_feature(spl[2])
                else:
                    source = [self.source_vocab[char] for char in spl[1]]
                    target = [self.target_vocab[char] for char in spl[2]]
                if len(source) == 0 or len(target) == 0:
                    continue
                parallel_data.append((source, target))
        if not self.panphon:
            self.source_vocab[UNK]
            self.target_vocab[UNK]
        return parallel_data

    def read_data(self, file_name):
        parallel_data = []
        with codecs.open(file_name, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                if self.panphon:
                    source = self.get_feature(spl[1])
                    target = self.get_feature(spl[2])
                else:
                    source = [self.char2int(self.source_vocab, char) for char in spl[1]]
                    target = [self.char2int(self.target_vocab, char) for char in spl[2]]
                if len(source) == 0 or len(target) == 0:
                    continue
                parallel_data.append((source, target))
        return parallel_data

    def char2int(self, vocab, char):
        if char in vocab:
            return vocab[char]
        else:
            return vocab[UNK]

    def get_feature(self, word):
        default_feats = [[0]*VEC_SIZE]*len(word)
        pp_feats = self.ft.word_to_vector_list(word, numeric=True)
        if len(pp_feats) > 0:
            return pp_feats
        return default_feats

    def get_embedding(self, char, word_type):
        # char will be panphon features if panphon (list of 22 integers) and will be character index if not.
        # word_type not needed for panphon since transformation matrix is the same for source and target
        if self.panphon:
            w_panphon = dy.parameter(self.ws_panphon)
            b_panphon = dy.parameter(self.bs_panphon)
            return w_panphon*dy.inputVector(char) + b_panphon
        else:
            if word_type == SRC:
                return self.source_lookup[char]
            elif word_type == TGT:
                return self.target_lookup[char]        

    def encode(self, entries, word_type):
        if word_type == SRC:
            return self.convert(entries, self.source_vocab, word_type, self.source_lstm_forward, self.source_lstm_backward)
        else:
            return self.convert(entries, self.target_vocab, word_type, self.target_lstm_forward, self.target_lstm_backward)

    def convert(self, entries, vocab, word_type, fwd, bwd):
        all_reps = []
        for i in range(0, len(entries), ENCODE_BATCH):
            dy.renew_cg()
            if i % (ENCODE_BATCH*10) == 0:
                print(i)
            cur_size = min(ENCODE_BATCH, len(entries)-i)
            batch = entries[i:i+cur_size]
            if self.panphon:
                temps = [self.get_feature(entry) for entry in batch]
            else:
                temps = [[self.char2int(vocab, char) for char in entry] for entry in batch]
            embs = [[self.get_embedding(y, word_type) for y in temp] for temp in temps]
            all_reps += self.get_normalized_reps(embs, fwd, bwd, encode=True)
        return np.array(all_reps)

    def get_val_recall(self):
        recall1 = 0.0
        total = 0

        source_reps = []
        target_reps = []

        for i in range(0, len(self.validation_data), ENCODE_BATCH):
            dy.renew_cg()
            cur_size = min(ENCODE_BATCH, len(self.validation_data)-i)
            batch = self.validation_data[i:i+cur_size]

            source_embs = [[self.get_embedding(x, SRC) for x in s] for s, t in batch]
            target_embs = [[self.get_embedding(y, TGT) for y in t] for s, t in batch]
            source_reps += self.get_normalized_reps(source_embs, self.source_lstm_forward, self.source_lstm_backward, encode=True)
            target_reps += self.get_normalized_reps(target_embs, self.target_lstm_forward, self.target_lstm_backward, encode=True)

        # get reps for whole training data as validation KB        
        for i in range(0, len(self.training_data), ENCODE_BATCH):
            dy.renew_cg()
            cur_size = min(ENCODE_BATCH, len(self.training_data)-i)
            batch = self.training_data[i:i+cur_size]
            embs = [[self.get_embedding(x, SRC) for x in s] for s, t in batch]
            source_reps += self.get_normalized_reps(embs, self.source_lstm_forward, self.source_lstm_backward, encode=True)            

        scores = np.array(target_reps).dot(np.array(source_reps).T)

        for entry_idx, entry_scores in enumerate(scores):
            ranks = entry_scores.argsort()[::-1]
            if ranks[0] == entry_idx:
                recall1 += 1
            total += 1

        return recall1/total

    def get_normalized_reps(self, embs, forward_lstm, backward_lstm, encode=False):
        word_reps = [dy.concatenate([forward_lstm.initial_state().transduce(emb)[-1], backward_lstm.initial_state().transduce(reversed(emb))[-1]]) for emb in embs]
        if not encode:
            return [dy.cdiv(rep, dy.l2_norm(rep)) for rep in word_reps]
        else:
            return [dy.cdiv(rep, dy.l2_norm(rep)).value() for rep in word_reps]


    def calculate_loss(self, words):
        dy.renew_cg()
        source_embs = [[self.get_embedding(x, SRC) for x in s] for s, t in words]
        target_embs = [[self.get_embedding(y, TGT) for y in t] for s, t in words]

        source_reps_norm = self.get_normalized_reps(source_embs, self.source_lstm_forward, self.source_lstm_backward)
        target_reps_norm = self.get_normalized_reps(target_embs, self.target_lstm_forward, self.target_lstm_backward)

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

            if ep % EPOCH_CHECK == 0:
                recall = self.get_val_recall()
                if recall > best_recall:
                    best_recall = recall
                    last_updated = ep
                    print('Saved: %0.4f' % best_recall)
                    self.model.save(self.model_name)    
            
            if ep-last_updated > PATIENCE:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default='500')
    parser.add_argument('--trainer', default='sgd')
    parser.add_argument('--embed_size', default='64')
    parser.add_argument('--hidden_size', default='1024')
    parser.add_argument('--train_file')
    parser.add_argument('--val_file')
    parser.add_argument('--model_file')
    parser.add_argument('--load_model', default=0)
    parser.add_argument('--panphon', default=0)
    args, unknown = parser.parse_known_args()

    sys.stdout=open(args.model_file.split('/')[-1] + '_' + args.trainer + '.log', 'w', 0)
    mm_encoder = MaxMarginEncoder(int(args.embed_size), int(args.hidden_size), int(args.panphon), args.model_file, int(args.load_model), args.train_file, args.val_file)
    mm_encoder.train(int(args.epochs), args.trainer)
    sys.stdout.close()

