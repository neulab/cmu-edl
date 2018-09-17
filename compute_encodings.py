'''
Encodes input file with given trained encoder (either source or target).
'''
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
import max_margin_encoder as mme
import time

ENCODING_FOLDER = 'encodings'
TARGET_PREFIX = 'target_'
SOURCE_PREFIX = 'source_'
SRC = 'src'
TGT = 'tgt'

t0 = time.time()
print('Start: ' + str(t0))

parser = argparse.ArgumentParser()
parser.add_argument('--model_embed_size', default='64')
parser.add_argument('--model_hidden_size', default='1024')
parser.add_argument('--panphon', default='0')
parser.add_argument('--parallel_file')
parser.add_argument('--lor_file')
parser.add_argument('--mme_trainfile')
parser.add_argument('--source_file')
parser.add_argument('--model_name')
args, unknown = parser.parse_known_args()

mme_model = mme.MaxMarginEncoder(int(args.model_embed_size), int(args.model_hidden_size), int(args.panphon), args.model_name, load_model=True, train_file=args.mme_trainfile)

if args.parallel_file:
    targets = []
    with codecs.open(args.parallel_file, 'r', 'utf8') as f:
        for line in f:
            spl = line.strip().split(' ||| ')
            targets.append(spl[2])

    output_path = os.path.join(ENCODING_FOLDER, args.model_name.split('/')[-1], TARGET_PREFIX + args.parallel_file.split('/')[-1])
    np.save(output_path, mme_model.encode(targets, TGT))

if args.source_file:
    sources = []
    reps = []
    with codecs.open(args.source_file, 'r', 'utf8') as f:
        for line in f:
            spl = line.strip().split(' ||| ')
            sources.append(spl[1].strip())
            
    output_path = os.path.join(ENCODING_FOLDER,  args.model_name.split('/')[-1], SOURCE_PREFIX + args.source_file.split('/')[-1])
    np.save(output_path, mme_model.encode(sources, SRC))

t1 = time.time()
total = t1-t0

print(str(total))
