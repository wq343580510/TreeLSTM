import torch
from torch.autograd import Variable as Var
import torch.nn as nn
from eval.eval_data import *
from tqdm import tqdm
class Trainer(object):
    def __init__(self,args,model,optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0

    def train_step(self,inst):
        scores = []
        for i in range(len(inst.kbest)):
            scores.append(self.model.predict(inst.kbest[i],inst.inputs[i]))
        max_id = scores.index(max(scores))
        gold_root = inst.gold
        gold_score = self.model.predict(gold_root,inst.gold_input)
        pred_score = scores[max_id]
        loss = pred_score-gold_score
        if loss > 0:
            self.model.train_pair(inst.kbest[max_id],inst.inputs[max_id],inst.gold,inst.gold_input)
            return loss
        else:
            return 0

    def train(self,dataset):
        self.model.train() #set the mode to training
        self.optimizer.zero_grad()
        loss ,k = 0.0,0
        #indices = torch.randperm(len(dataset))
        for idx in tqdm(xrange(len(dataset)),desc = 'Training epoch ' + str(self.epoch+1) +' '):
            err = self.train_step(dataset[idx])
            loss += err
            k += 1
            if k%self.args.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.epoch += 1
        return loss/len(dataset)