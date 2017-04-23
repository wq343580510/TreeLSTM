from utils import read_tree
from eval.evaluate import evaluate
import os
import utils
from Vocab import Vocab

class data_manager(object):
    max_degree = 0
    def __init__(self,batch,train_kbest = None,train_gold = None,dev_kbest = None,dev_gold = None,
                 test_kbest = None,test_gold = None,vocab_path = None):
        self.vocab = None
        self.train_kbest = train_kbest
        self.train_gold = train_gold
        self.dev_kbest = dev_kbest
        self.dev_gold = dev_gold
        self.test_kbest = test_kbest
        self.test_gold = test_gold
        if os.path.exists(vocab_path):
            print 'load vocab'
            self.vocab = utils.load_dict(vocab_path)
        else:
            print 'creat vocab'
            self.vocab = Vocab(self.train_gold)
            print 'save dictionary'
            utils.save_dict(self.vocab, vocab_path)
        print 'vocab size:' + str(self.vocab.size())
        print 'read dev data'
        self.dev_data = read_data(dev_kbest,dev_gold,self.vocab)
        print 'number of dev:'+str(len(self.dev_data))
        print 'read train data'
        self.train_data = read_data(train_kbest,train_gold,self.vocab)
        print 'number of dev:' + str(len(self.train_data))


class instance(object):
    def __init__(self,kbest,scores,gold,lines,gold_lines):
        self.kbest = kbest
        self.scores = scores
        self.gold = gold
        self.gold_lines = gold_lines
        self.lines = lines
        self.f1score = []
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
    i = 0
    while i < len(kbest_data):
        line = kbest_data[i]
        if line.strip() != 'PTB_KBEST':
            if line.strip() == '':
                res = read_tree(tree, vocab)
                onebest.append(res[0])
                onelines.append(res[1])
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
                onelines = []
                onebest = []
                onescores = []
        i += 1

    with open(gold_filename, 'r') as reader:
        data = reader.readlines()
    reader.close()
    list = []
    gold = []
    gold_lines = []
    for line in data:
        if line.strip() == '':
            res = read_tree(list,vocab)
            gold.append(res[0])
            gold_lines.append(res[1])
            list = []
        else:
            list.append(line)
    dev_data = []
    num = 900
    for a,b,c,d,e in zip(kbest, scores, gold, lines, gold_lines):
        if num == 0:
            break
        num = num - 1
        if len(c.children) == 0:
            continue
        dev_data.append(instance(a,b,c,d,e))
    return dev_data
