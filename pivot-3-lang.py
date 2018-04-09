# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
# import multilabel_classifier as mlc
import codecs
import numpy as np
import sys
import argparse
import time
from collections import OrderedDict

class ExactMatch():
    def __init__(self, kb_file, link_file, test_file):
        self.links = {}
        self.kb = {}
        self.ipa_links = {}
        self.ipa_kb = {}
        self.test = {}
        self.test_ipa = {}

        with codecs.open(link_file, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.links[spl[2]] = spl[0]

        with codecs.open(kb_file, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.kb[spl[1]] = spl[0]

        with codecs.open('data/ipa_' + link_file.split('/')[-1], 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.ipa_links[spl[2]] = spl[0]

        with codecs.open('data/ipa_' + kb_file.split('/')[-1], 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.ipa_kb[spl[1]] = spl[0]

        with codecs.open(test_file, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.test[int(spl[0])] = spl[2]
        
        with codecs.open('data/ipa_' + test_file.split('/')[-1], 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.test_ipa[int(spl[0])] = spl[2]

    def get_match(self, ent_idx, frac):
        up = False
        ent = self.test[ent_idx]
        ipa_ent = self.test_ipa[ent_idx]
        check = False

        if ent in self.kb:# and frac != 0.0:
            check = True
            if self.kb[ent] == str(ent_idx):
                up = True
        elif ent in self.links:# and frac != 0.0:
            check = True
            if self.links[ent] == str(ent_idx):
                up = True        
        elif ipa_ent in self.ipa_kb:# and frac != 1.0:
            check = True
            if self.ipa_kb[ipa_ent] == str(ent_idx):
                up = True
        elif ipa_ent in self.ipa_links:# and frac != 0.0:
            check = True
            if self.ipa_links[ipa_ent] == str(ent_idx):
                up = True

        return check, up


class MultilingualPivot():
    def __init__(self, database_filename, links_file, test_file):
        self.db = self.read_database(database_filename)
        self.links = self.read_en_links(links_file)
        self.test_data = self.get_data(test_file)
        self.exact_model = ExactMatch(database_filename, links_file, test_file)
        print('starting')

    def get_data(self, test_file):
        data = []
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

    def test_pivot(self, pivot, test_reps_file, kb_reps_file, link_reps_file, ipa_test_reps_file=None, ipa_kb_reps_file=None, ipa_link_reps_file=None, outfile=None, g_frac=1.0):
        if outfile:
            recall_file = open(outfile + '_recall', 'w')
        #     score_file = open(outfile + '_score', 'w')

        print('Pivot: %d' % pivot)
        recall1 = defaultdict(lambda: 0.0)
        recall5 = defaultdict(lambda: 0.0)
        recall10 = defaultdict(lambda: 0.0)
        recall20 = defaultdict(lambda: 0.0)
        recall100 = defaultdict(lambda: 0.0)
        total = 0

        test_data_reps = np.load('encodings/' + test_reps_file)
        kb_reps = np.load('encodings/' + kb_reps_file)
        link_reps = np.load('encodings/' + link_reps_file)
        
        if ipa_test_reps_file:
            ipa_test_data_reps = np.load('encodings/' + ipa_test_reps_file)
            ipa_kb_reps = np.load('encodings/' + ipa_kb_reps_file)
            ipa_link_reps = np.load('encodings/' + ipa_link_reps_file)

        print(test_data_reps.shape)
        print(kb_reps.shape)
        print(link_reps.shape)
        print(ipa_test_data_reps.shape)
        print(ipa_kb_reps.shape)
        print(ipa_link_reps.shape)

        entity_ids = np.concatenate((self.db, self.links))
        
        for idx, id_num in enumerate(self.test_data):
            check, up = self.exact_model.get_match(id_num, g_frac)
            if check:
                if up:
                    recall1['comb'] += 1
                    recall5['comb'] += 1
                    recall10['comb'] += 1
                    recall20['comb'] += 1
                    recall100['comb'] += 1
                    recall_file.write('1' + '\n')
                else:
                    recall_file.write('0' + '\n')
                total += 1 
                continue

            scores = {}
            link_scores = {}
            scores['ortho'] = test_data_reps[idx].dot(kb_reps.T)
            scores['ipa'] = ipa_test_data_reps[idx].dot(ipa_kb_reps.T)
            if pivot:
                link_scores['ortho'] = test_data_reps[idx].dot(link_reps.T)
                link_scores['ipa'] = ipa_test_data_reps[idx].dot(ipa_link_reps.T)
                scores['ortho'] = np.concatenate((scores['ortho'], link_scores['ortho']))
                scores['ipa'] = np.concatenate((scores['ipa'], link_scores['ipa']))

            # for x in ['ortho', 'ipa', 'comb']:
            for x in ['comb']:
                if x == 'comb':
                    cur_scores = g_frac*scores['ortho'] + (1-g_frac)*scores['ipa']
                else:
                    cur_scores = scores[x]
                    
                max_idx = np.argpartition(cur_scores, -200)[-200:]
                ranked_idxs = max_idx[np.argsort(cur_scores[max_idx])]      
                ranked_ids_dup = entity_ids[ranked_idxs][::-1]

                # score_file.write(str(ranked_ids_dup) + '\n')
                # score_file.write(str(cur_scores[ranked_idxs][::-1]) + '\n\n')

                ranked_ids = list(OrderedDict.fromkeys(ranked_ids_dup))

                assert len(ranked_ids) > 100

                if id_num == ranked_ids[0]:
                    recall1[x] += 1
                    recall_file.write('1' + '\n')
                else:
                    recall_file.write('0' + '\n') 
                if id_num in ranked_ids[:5]:
                    recall5[x] += 1
                if id_num in ranked_ids[:10]:
                    recall10[x] += 1
                if id_num in ranked_ids[:20]:
                    recall20[x] += 1
                if id_num in ranked_ids[:100]:
                    recall100[x] += 1  
          
            total += 1

            if total % 10 == 0:
                print(total)

            # recall_file.write('\n')
            # score_file.write('\n')

        # for x in ['ortho', 'ipa', 'comb']:
        for x in ['comb']:
            print(x)        
            print('Recall @1 %.4f' % (recall1[x]/total))
            print('Recall @5 %.4f' % (recall5[x]/total))
            print('Recall @10 %.4f' % (recall10[x]/total))
            print('Recall @20 %.4f' % (recall20[x]/total))
            print('Recall @100 %.4f' % (recall100[x]/total))
            print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb', default='data/en_kb')
    parser.add_argument('--links')
    parser.add_argument('--test') 
    parser.add_argument('--pivot', default='1')
    parser.add_argument('--model')
    parser.add_argument('--frac', default='1.0')
    parser.add_argument('--comb', default='1')
    args, unknown = parser.parse_known_args()

    print(args.links)
    print(args.test)   
    sys.stdout=open(args.test.split('/')[-1] + '_' + args.links.split('/')[-1] + '_pivot' + args.pivot + args.frac + '_final.log', 'w', 0)

    t0 = time.time()
    pivot_model = MultilingualPivot(args.kb, args.links, args.test)
    if int(args.comb):
        print('Combining')
        pivot_model.test_pivot(int(args.pivot), 
            args.model + '/target_'+args.test.split('/')[-1]+'.npy', 
            args.model + '/source_'+args.kb.split('/')[-1]+'.npy', 
            args.model + '/target_'+args.links.split('/')[-1]+'.npy', 
            'ipa_' + args.model + '/target_ipa_'+args.test.split('/')[-1]+'.npy', 
            'ipa_' + args.model + '/source_ipa_'+args.kb.split('/')[-1]+'.npy', 
            'ipa_' + args.model + '/target_ipa_'+args.links.split('/')[-1]+'.npy',
            args.test.split('/')[-1] + args.links.split('/')[-1] + args.pivot + args.frac,
            float(args.frac)
            )
    else:
        pivot_model.test_pivot(int(args.pivot), args.model + '/target_'+args.test.split('/')[-1]+'.npy', args.model + '/source_'+args.kb.split('/')[-1]+'.npy', args.model + '/target_'+args.links.split('/')[-1]+'.npy')
    t1 = time.time()
    print(t1-t0)

    sys.stdout.close()
