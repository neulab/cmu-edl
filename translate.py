import codecs
import argparse
from collections import defaultdict
import sys
import pdb

def get_translation(ent, lex):
    trans_ent = {'':True}

    for w in ent.split():
        new_trans = {}
        if w in lex:
            for x in lex[w]:
                for t in trans_ent.keys():
                    new_trans[t + ' ' + x] = True
        if len(new_trans) > 0:
            trans_ent = new_trans

    return trans_ent

parser = argparse.ArgumentParser()
parser.add_argument('--pivot')
parser.add_argument('--test')
parser.add_argument('--piv', default='1')
args = parser.parse_args()

piv = int(args.piv)

links = {}
kb = {}
ipa_links = {}
ipa_kb = {}
lex = defaultdict(lambda: [])
ipa_lex = defaultdict(lambda: [])

pref = 'data/'

link_file = 'en-' + args.pivot + '_links'
kb_file = 'en_kb'
test_file = 'test_en-' + args.test + '_links'

with codecs.open(pref + link_file, 'r', 'utf8') as f:
    for line in f:
        spl = line.strip().split(' ||| ')
        links[spl[2]] = spl[0]

with codecs.open(pref + kb_file, 'r', 'utf8') as f:
    for line in f:
        spl = line.strip().split(' ||| ')
        kb[spl[1]] = spl[0]

with codecs.open(pref + 'ipa_' + link_file, 'r', 'utf8') as f:
    for line in f:
        spl = line.strip().split(' ||| ')
        ipa_links[spl[2]] = spl[0]

with codecs.open(pref + 'ipa_' + kb_file, 'r', 'utf8') as f:
    for line in f:
        spl = line.strip().split(' ||| ')
        ipa_kb[spl[1]] = spl[0]

with codecs.open('data/en-' + args.pivot + '_links_lex', 'r', 'utf8') as f:
    for line in f:
        spl = line.strip().split(' ||| ')
        lex[spl[1]].append(spl[0])

with codecs.open('data/ipa_en-' + args.pivot + '_links_lex', 'r', 'utf8') as f:
    for line in f:
        spl = line.strip().split(' ||| ')
        ipa_lex[spl[1]].append(spl[0])

acc = defaultdict(lambda: 0)
total = 0.0

with codecs.open(pref + test_file, 'r', 'utf8') as f, codecs.open(pref + 'ipa_' + test_file, 'r', 'utf8') as f2:
    for line in f:
        spl = line.strip().split(' ||| ')
        ent = spl[2]
        ipa_spl = f2.readline().strip().split(' ||| ')
        ipa_ent = spl[2]

        check = False
        ipa_check = False

        if ent in kb:
            if kb[ent] == spl[0]:
                acc['ortho'] += 1
            check = True
        elif ent in links and piv:
            if links[ent] == spl[0]:
                acc['ortho'] += 1
            check = True
        
        if not check:
            for transx in get_translation(ent, lex).keys():
                trans = transx.strip()
                print(trans)
                if trans in kb:
                    if kb[trans] == spl[0]:
                        print('true')
                        acc['ortho'] += 1
                    check = True
                    break        
        
        if ipa_ent in ipa_kb:
            if ipa_kb[ipa_ent] == ipa_spl[0]:
                acc['ipa'] += 1
            ipa_check = True
        elif ipa_ent in ipa_links and piv:
            if ipa_links[ipa_ent] == ipa_spl[0]:
                acc['ipa'] += 1
            ipa_check = True

        if not ipa_check:
            for transx in get_translation(ipa_ent, ipa_lex).keys():
                trans = transx.strip()
                if trans in ipa_kb:
                    if ipa_kb[trans] == ipa_spl[0]:
                        acc['ipa'] += 1
                    ipa_check = True
                    break    

        if check or ipa_check:
            acc['comb'] += 1

        total += 1

sys.stdout=open(args.pivot + '_' + args.test + args.piv + '_trans.log', 'w', 0)
print('Ortho: ' + str(acc['ortho']/total))
print('IPA: ' + str(acc['ipa']/total))
print('Comb: ' +str(acc['comb']/total))
sys.stdout.close()   