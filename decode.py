# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
import codecs
import numpy as np
import sys
import argparse
import time
from collections import OrderedDict

NIL = 'NIL'
MAX_SCORE_PLACEHOLDER = '1000'
NOPIVOT_RECALL = 'nopivot_recall'
PIVOT_RECALL = 'pivot_recall'
NOPIVOT_TOPK = 'nopivot_topk'
PIVOT_TOPK = 'pivot_topk'

class ExactMatch():
    def __init__(self, kb_file, link_file, test_file, testing):
        self.links = {}
        self.kb = {}
        self.test = {}
        self.testing = testing

        with codecs.open(link_file, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.links[spl[2]] = spl[0]
        with codecs.open(kb_file, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.kb[spl[1]] = spl[0]
        if self.testing:
            with codecs.open(test_file, 'r', 'utf8') as f:
                for line in f:
                    spl = line.strip().split(' ||| ')
                    self.test[int(spl[0])] = spl[2]

    def get_match(self, ent, pivot):
        if self.testing:
            ent = self.test[ent]
        if ent in self.kb:
            return self.kb[ent]
        elif ent in self.links and pivot:
            return self.links[ent]

        return NIL


class MultilingualPivot():
    def __init__(self, database_filename, links_file, test_file):
        self.db = self.read_database(database_filename)
        self.links = self.read_en_links(links_file)
        self.test_data = self.read_test_data(test_file)
        self.exact_model = ExactMatch(database_filename, links_file, test_file, self.testing)

    def read_test_data(self, test_file):
        data = []
        with codecs.open(test_file, 'r', 'utf8') as f:
            line = f.readline()
            if len(line.strip().split(' ||| ')) < 2:
                self.testing = False
            else:
                self.testing = True                
        with codecs.open(test_file, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                data.append(int(spl[0]))        
        return data

    def read_database(self, database_filename):
        db = []
        with codecs.open(database_filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                db.append(int(spl[0]))                
        return db

    def read_en_links(self, filename):
        links = []
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                links.append(int(spl[0]))
        return links  

    def update_recalls(self, recalls, predicted, true):
        if true == predicted[0]:
            recalls[1] += 1
        if true in predicted[:5]:
            recalls[5] += 1
        if true in predicted[:10]:
            recalls[10] += 1
        if true in predicted[:20]:
            recalls[20] += 1
        if true in predicted[:100]:
            recalls[100] += 1

    def write_recalls(self, filename, recalls):
        for i in [1, 5, 10, 20, 100]:
            filename.write("Recall at %d: %0.4f\n" %i,(recalls[i]/len(self.test_data))) 

    def get_ranks(self, cur_scores, topk):
        limit = min([len(cur_scores),200])
        max_idx = np.argpartition(cur_scores, -limit)[-limit:]
        ranked_idxs = max_idx[np.argsort(cur_scores[max_idx])] 
        return ranked_idxs

    def get_predictions(self, ranked_ids, exact):
        pred_len = min(100, len(ranked_ids))
        if exact != NIL:
            return [exact] + ranked_ids[:pred_len-1]
        else:
           return ranked_ids[:pred_len]

    def test_pivot(self, test_reps_file, kb_reps_file, link_reps_file=None, no_pivot=True, pivot=True, print_recall=False, topk=200, outfile=None):
        if outfile:
            outfiles = {}
            if topk > 0:
                outfiles[NOPIVOT_TOPK] = open(outfile + NOPIVOT_TOPK, 'w')
                outfiles[PIVOT_TOPK] = open(outfile + PIVOT_TOPK, 'w')
            if print_recall and self.testing:
                recalls = {}
                recalls[NOPIVOT_RECALL] = defaultdict(lambda: 0.0)
                recalls[PIVOT_RECALL] = defaultdict(lambda: 0.0)
                outfiles[NOPIVOT_RECALL] = open(outfile + NOPIVOT_RECALL, 'w')
                outfiles[PIVOT_RECALL] = open(outfile + PIVOT_RECALL, 'w')

        test_data_reps = np.load(test_reps_file)
        kb_reps = np.load(kb_reps_file)
        if pivot:
            link_reps = np.load(link_reps_file)

        if pivot:
            entity_ids = np.concatenate((self.db, self.links))
        else:
            entity_ids = self.db

        for idx, input_entry in enumerate(self.test_data):
            if no_pivot:
                exact = self.exact_model.get_match(input_entry, False)
                if exact != NIL and outfile and topk > 0:
                    outfiles[NOPIVOT_TOPK].write(str(exact) + ' ||| ' + MAX_SCORE_PLACEHOLDER + '\n')

                scores = test_data_reps[idx].dot(kb_reps.T)
                ranked_idxs = self.get_ranks(scores, topk)
                ranked_ids = entity_ids[ranked_idxs][::-1]
                ranked_scores = scores[ranked_idxs][::-1]
                assert (len(ranked_ids) > 100 or len(scores) < 100)
                if outfile and topk > 0:
                    for idx,score in zip(ranked_ids, ranked_scores):
                        outfiles[NOPIVOT_TOPK].write(str(idx) + ' ||| ' + str(score) + '\n')
                    outfiles[NOPIVOT_TOPK].write('***\n')

                predicted = self.get_predictions(ranked_ids, exact)
                if print_recall and self.testing:
                    self.update_recalls(recalls[NOPIVOT_RECALL], predicted, input_entry)

            if pivot:  
                exact = self.exact_model.get_match(input_entry, True)
                if exact != NIL and outfile and topk > 0:
                    outfiles[PIVOT_TOPK].write(str(exact) + ' ||| ' + MAX_SCORE_PLACEHOLDER + '\n')

                link_scores = test_data_reps[idx].dot(link_reps.T)
                pivot_scores = np.concatenate((scores, link_scores))
                ranked_idxs = self.get_ranks(pivot_scores, topk)
                ranked_ids = entity_ids[ranked_idxs][::-1]
                ranked_scores = pivot_scores[ranked_idxs][::-1]
                assert (len(ranked_ids) > 100 or len(entity_ids) < 100)
                if outfile and topk > 0:
                    for idx,score in zip(ranked_ids, ranked_scores):
                        outfiles[PIVOT_TOPK].write(str(idx) + ' ||| ' + str(score) + '\n')
                    outfiles[PIVOT_TOPK].write('***\n')

                predicted = self.get_predictions(ranked_ids, exact)
                if print_recall and self.testing:
                    self.update_recalls(recalls[PIVOT_RECALL], predicted, input_entry)

        if print_recall and self.testing:
            if no_pivot:
                self.write_recalls(outfile[NOPIVOT_RECALL], recalls[NOPIVOT_RECALL])
            if pivot:
                self.write_recalls(outfile[PIVOT_RECALL], recalls[PIVOT_RECALL])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # non-ipa files so exact match may work. even if encodings are for IPA.
    parser.add_argument('--kb')
    parser.add_argument('--links')
    parser.add_argument('--test') 
    parser.add_argument('--kb_encodings')
    parser.add_argument('--links_encodings')
    parser.add_argument('--test_encodings')
    parser.add_argument('--outfile', default='multilingual-pivot-output')
    parser.add_argument('--nopivot_decode', default='1')
    parser.add_argument('--pivot_decode', default='1')
    parser.add_argument('--print_recall', default='0')
    parser.add_argument('--top_k', default='200')
    args, unknown = parser.parse_known_args()

    pivot_model = MultilingualPivot(args.kb, args.links, args.test)

    pivot_model.test_pivot(args.test_encodings, args.kb_encodings, args.link_encodings, int(args.nopivot_decode), int(args.pivot_decode), int(args.print_recall), int(args.top_k), args.outfile)
    
    pivot_model.test_pivot(args.encodings + 'target_'+args.test.split('/')[-1]+'.npy', args.encodings + 'source_'+args.kb.split('/')[-1]+'.npy', args.encodings + 'target_'+args.links.split('/')[-1]+'.npy', args.outfile)

