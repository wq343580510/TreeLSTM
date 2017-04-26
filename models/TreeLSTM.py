import torch
import torch.nn as nn

class TreeLSTM(nn.Module):
    def __init__(self,config):
        super(TreeLSTM,self).__init__()
        self.in_dim = config.in_dim
        self.mem_dim = config.mem_dim or 150
        self.mem_zeros = torch.zeros(self.mem_dim)
        self.train = False
    def forward(self,tree,inputs):
        print 'forward'

    def backward(self,tree,inputs,grad):
        print 'backward'

    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False

        # local num_free =  # self[modules]
        # if num_free == 0:
        #     tree[module] = self['new_'..module](self)
        # else:
        #     tree[module] = self[modules][num_free]
        #     self[modules][num_free] = nil
        # necessary for dropout to behave properly
        #if self.train then tree[module]:training() else tree[module]:evaluate()

    def free_module(self,tree,module):
        print "free_module"
        # if tree[module] == nil then return end
        # table.insert(self[module .. 's'], tree[module])
        # tree[module] = nil

    def allocate_module(tree,module):
        print 'adssad'
