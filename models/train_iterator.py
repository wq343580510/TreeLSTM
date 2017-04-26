from utils import read_tree
import datetime
from eval.evaluate import evaluate

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

class train_iterator(object):
    def __init__(self, kbest_filename , gold_filename, vocab,batch):
        with open(kbest_filename, 'r') as reader:
            self.data = reader.readlines()
            self.data.append('PTB_KBEST')
        with open(gold_filename, 'r') as reader:
            self.gdata = reader.readlines()
        self.kbest_id = 0
        self.vocab = vocab
        self.index = 0
        self.gindex = 0
        self.batch = batch
        self.length = len(self.data)
        self.glength = len(self.gdata)

    def read_give_tree(self,tree_index):
        scores = []
        tree = []
        ktrees = []
        kbestlines = []
        # read train
        index = 0
        best_num = 0
        while index < self.length:
            line = self.data[index]
            if best_num == tree_index:
                if line.strip() != 'PTB_KBEST':
                    if line.strip() == '':
                        ktrees.append(read_tree(tree, self.vocab))
                        kbestlines.append(tree[:])
                        tree = []
                    elif not '_' in line:
                        scores.append(float(line))
                    else:
                        tree.append(line)
                else:
                    break
            if line.strip() == 'PTB_KBEST':
                best_num += 1
            index += 1
        # read gold
        list = []
        gold = []
        index = 0
        num = 1
        while index < self.glength:
            line = self.gdata[index]
            if num == tree_index:
                if line.strip() == '':
                    root = read_tree(list, self.vocab)
                    gold.append(root)
                    break
                else:
                    list.append(line)
            if line.strip() == '':
                num += 1
            index += 1

        retval = data_util.instance(ktrees, scores, gold,gold_lines=list,lines=kbestlines)
        return retval

    def read_all(self):
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        lines = []
        klines = []
        index = 0
        while index < self.length:
            line = self.data[index]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    ktrees.append(read_tree(tree, self.vocab))
                    lines.append(tree[:])
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    klines.append(lines[:])
                    lines = []
                    ktrees = []
                    scores = []
            index += 1
        # read gold
        list = []
        gold = []
        goldlines = []
        gindex =  0
        while gindex < self.glength:
            line = self.gdata[gindex]
            if line.strip() == '':
                root = read_tree(list, self.vocab)
                goldlines.append(list[:])
                gold.append(root)
                list = []
            else:
                list.append(line)
            gindex += 1
        train_batch = []
        for a, b, c , d, e in zip(kbest, kscores, gold,klines,goldlines):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(data_util.instance(a, b, c, d,e))
        return train_batch

    def read_batch_random(self, batch_size=200):
        total_list = []
        for i in range(20):
            total_list.append(i)
        import random
        random.seed(datetime.datetime.now())
        random.shuffle(total_list)
        sublist = total_list[0:batch_size]
        total_list = sorted(sublist)
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        lines = []
        klines = []
        index = 0
        tree_id = -1
        while index < self.length and tree_id <= 20:
            line = self.data[index]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    ktrees.append(read_tree(tree, self.vocab))
                    lines.append(tree[:])
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if not tree_id in total_list:
                    tree_id += 1
                    lines = []
                    ktrees = []
                    scores = []
                    index += 1
                    continue
                tree_id += 1
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    klines.append(lines[:])
                    lines = []
                    ktrees = []
                    scores = []
            index += 1
        # read gold
        list = []
        gold = []
        goldlines = []
        gindex = 0
        gold_id = 0
        while gindex < self.glength and gold_id <= 20:
            line = self.gdata[gindex]
            if line.strip() == '':
                if not gold_id in total_list:
                    gold_id += 1
                    list = []
                    gindex += 1
                    continue
                gold_id += 1
                root = read_tree(list, self.vocab)
                goldlines.append(list[:])
                gold.append(root)
                list = []
            else:
                list.append(line)
            gindex += 1
        train_batch = []
        for a, b, c, d, e in zip(kbest, kscores, gold, klines, goldlines):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(instance(a, b, c, d, e))
        return train_batch

    def read_batch(self,read_batch = True):
        if self.index == self.length:
            return None
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        #read train
        oneinput = []
        inputs = []
        lines = []
        klines = []
        while self.index < self.length:
            line = self.data[self.index]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    res = read_tree(tree,self.vocab)
                    ktrees.append(res[0])
                    oneinput.append(res[2])
                    lines.append(res[1])
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    klines.append(lines[:])
                    inputs.append(oneinput[:])
                    if read_batch and len(kbest) == self.batch:
                        self.index += 1
                        break
                    ktrees = []
                    oneinput = []
                    scores = []
                    lines = []
            self.index += 1
        #read gold
        list = []
        gold = []
        gold_inp = []
        goldlines = []
        while self.gindex < self.glength:
            line = self.gdata[self.gindex]
            if line.strip() == '':
                res = read_tree(list,self.vocab)
                gold.append(res[0])
                gold_inp.append(res[2])
                goldlines.append(res[1])
                if read_batch and len(gold) == self.batch:
                    self.gindex += 1
                    break
                list = []
            else:
                list.append(line)
            self.gindex += 1
        train_batch = []
        for a,b,c,d,e,f,g in zip(kbest,kscores,gold,klines,goldlines,inputs,gold_inp):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(instance(a,b,c,d,e,f,g))
        return train_batch

    def reset(self):
        self.index = 0
        self.gindex = 0
        self.kbest_id = 0


