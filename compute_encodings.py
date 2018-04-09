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

t0 = time.time()
print('Start: ' + str(t0))

parser = argparse.ArgumentParser()
parser.add_argument('--parallel_file')
parser.add_argument('--mme_trainfile')
parser.add_argument('--source_file')
parser.add_argument('--model_name')
args, unknown = parser.parse_known_args()

if not os.path.exists('encodings/' + args.model_name.split('/')[-1]):
    os.makedirs('encodings/' + args.model_name.split('/')[-1])

sources = []
targets = []

mme_model = mme.MaxMarginEncoder(64, 1024, args.model_name, args.mme_trainfile)

if args.parallel_file:
    with codecs.open(args.parallel_file, 'r', 'utf8') as f:
        for line in f:
            spl = line.strip().split(' ||| ')
            sources.append(spl[1])
            targets.append(spl[2])

    # np.save('encodings/source_' + args.parallel_file.split('/')[-1], mme_model.encode(sources, True))
    print('encodings/' + args.model_name.split('/')[-1] + '/target_' + args.parallel_file.split('/')[-1])
    print(len(targets))
    np.save('encodings/' + args.model_name.split('/')[-1] + '/target_' + args.parallel_file.split('/')[-1], mme_model.encode(targets, False))

if args.source_file:
    reps = []
    with codecs.open(args.source_file, 'r', 'utf8') as f:
        for line in f:
            spl = line.strip().split(' ||| ')
            sources.append(spl[1])
    np.save('encodings/' + args.model_name.split('/')[-1] + '/source_' + args.source_file.split('/')[-1], mme_model.encode(sources, True))

t1 = time.time()
total = t1-t0

print(str(total))
