import codecs
import argparse
from collections import defaultdict

NIL = 'NIL'

class TranslateLinking(object):
    def __init__(self, kb, lexicon):
        self.kb = {}
        self.lex = defaultdict(lambda: [])

        with codecs.open(kb, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.kb[spl[1]] = spl[0]

        with codecs.open(lexicon, 'r', 'utf8') as f:
            for line in f:
                spl = line.strip().split(' ||| ')
                self.lex[spl[1]].append(spl[0])
        

    def get_translation(self, ent, lex):
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

    def link_entity(self, ent):
        if ent in self.kb:
            return self.kb[ent]            
        
        for transx in self.get_translation(ent, self.lex).keys():
            trans = transx.strip()
            if trans in self.kb:
                return self.kb[trans]

        return NIL   
        

parser = argparse.ArgumentParser()
parser.add_argument('--test_file')
parser.add_argument('--kb', default='en_kb')
parser.add_argument('--lexicon')
parser.add_argument('--output', default='output')
# parser.add_argument('--threshold', default='1.0')
args = parser.parse_args()

# sim_threshold = float(args.threshold)

translate_model = TranslateLinking(args.kb, args.lexicon)

with codecs.open(args.test_file, 'r', 'utf8') as f, codecs.open(args.output, 'w', 'utf8') as out:
    for line in f:
        ent = line.strip()
        link = translate_model.link_entity(ent)
        out.write(link + '\n')
        