import torch
from torch.autograd import Variable as Var
from tqdm import tqdm
class Trainer(object):
    def __init__(self,args,model,criterion,optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train(self,dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss ,k = 0.0,0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(xrange(len(dataset)),desc = 'Training epoch' + str(self.epoch+1) +''):
            input = []
            target = []
            k += 1
            if k%self.args.batchsize==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return loss/len(dataset)