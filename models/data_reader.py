from utils import read_tree
from eval.evaluate import evaluate
import os
import torch
import utils
from Vocab import Vocab
import os
class data_manager(object):
    def __init__(self,args):
        self.vocab = None
        self.train_kbest = os.path.join(args.dir,args.train+'.kbest')
        self.train_gold = os.path.join(args.dir,args.train+'.gold')
        self.dev_kbest = os.path.join(args.dir,args.dev+'.kbest')
        self.dev_gold = os.path.join(args.dir,args.dev+'.gold')
        self.test_kbest = os.path.join(args.dir,args.test+'.kbest')
        self.test_gold = os.path.join(args.dir,args.test+'.gold')
        self.vocab_path = os.path.join(args.dir,args.vocab)
        self.model_path = os.path.join(args.dir,args.model_file)
        if os.path.exists(self.vocab_path):
            print 'load vocab'
            self.vocab = torch.load(self.vocab_path)
        else:
            print 'creat vocab'
            self.vocab = Vocab(self.train_gold)
            print 'save dictionary'
            torch.save(self.vocab, self.vocab_path)
        print 'vocab size:' + str(self.vocab.size())
        print 'read dev data'
        self.dev_data = read_data(self.dev_kbest,self.dev_gold,self.vocab)
        print 'number of dev:'+str(len(self.dev_data))
        print 'read train data'
        self.train_data = read_data(self.train_kbest,self.train_gold,self.vocab)
        print 'number of train:' + str(len(self.train_data))


class instance(object):
    def __init__(self,kbest,scores,gold,lines,gold_lines,inputs,gold_inp):
        self.kbest = kbest
        self.scores = scores
        self.gold = gold
        self.gold_lines = gold_lines
        self.lines = lines
        self.f1score = []
        self.inputs = inputs
        self.gold_input = gold_inp
        #self.maxid = self.get_oracle_index()

    def set_f1(self):
        i = 0
        for l in self.lines:
            f1 = evaluate(l, self.gold_lines)[0]
            self.f1score.append(f1)
            i+=1

    def get_oracle_index(self):
        max = 0
        maxid = 0
        i = 0
        for list in self.lines:
            temp = []
            for line in list:
                temp.append(line)
            temp.append('\n')
            res = evaluate(temp, self.gold_lines)[0]
            self.f1score.append(res)
            if res > max:
                max = res
                maxid = i
            i += 1
        return maxid

def read_data(kbest_filename, gold_filename, vocab):
    with open(kbest_filename, 'r') as reader:
        kbest_data = reader.readlines()
    kbest_data.append('PTB_KBEST')
    reader.close()
    kbest = []
    scores = []
    onebest = []
    tree = []
    onescores = []
    lines = []
    onelines = []
    oneinput = []
    inputs = []
    i = 0
    while i < len(kbest_data):
        line = kbest_data[i]
        if line.strip() != 'PTB_KBEST':
            if line.strip() == '':
                res = read_tree(tree, vocab)
                onebest.append(res[0])
                onelines.append(res[1])
                oneinput.append(res[2])
                tree = []
            elif not '_' in line:
                onescores.append(float(line))
            else:
                tree.append(line)
        else:
            if len(onebest) > 1:
                kbest.append(onebest[:])
                scores.append(onescores[:])
                lines.append(onelines)
                inputs.append(oneinput[:])
                onelines = []
                onebest = []
                onescores = []
                oneinput = []
        i += 1

    with open(gold_filename, 'r') as reader:
        data = reader.readlines()
    reader.close()
    list = []
    gold = []
    gold_lines = []
    gold_inp = []
    for line in data:
        if line.strip() == '':
            res = read_tree(list,vocab)
            gold.append(res[0])
            gold_lines.append(res[1])
            gold_inp.append(res[2])
            list = []
        else:
            list.append(line)
    dev_data = []
    for a,b,c,d,e,f,g in zip(kbest, scores, gold, lines, gold_lines,inputs,gold_inp):
        if len(c.children) == 0:
            continue
        dev_data.append(instance(a,b,c,d,e,f,g))
    return dev_data
