# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

from models import Constants


class ChildSumTreeLSTM(nn.Module):
    def __init__(self,cuda,vocab_size,in_dim,h_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.useCuda = cuda
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.emb = nn.Embedding(vocab_size, in_dim, Constants.PAD) # padding idx
        self.ix = nn.Linear(self.in_dim,self.h_dim)
        self.ih = nn.Linear(self.h_dim, self.h_dim)
        self.fx = nn.Linear(self.in_dim, self.h_dim)
        self.fh = nn.Linear(self.h_dim, self.h_dim)
        self.ox = nn.Linear(self.in_dim, self.h_dim)
        self.oh = nn.Linear(self.h_dim, self.h_dim)
        self.ux = nn.Linear(self.in_dim, self.h_dim)
        self.uh = nn.Linear(self.h_dim, self.h_dim)

    def recursive_unit(self,inputs,child_c,child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h,1),0)  # squeeze  在指定维度插入一个假的维度
        # 4 * 3 的tensor 如果torch.squeeze(x,0) 变为 1*4*3 torch.squeeze(x,1) 变为 4*1*3

        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))

        #add extra singleton dimension
        fx = F.torch.unsqueeze(self.ix(inputs),1)
        f = F.torch.cat([self.ih(child_hi)+fx for child_hi in child_h],0)
        f = F.sigmoid(f)

        # removing extra singleton dimension
        f = F.torch.unsqueeze(f,1)

        fc = F.torch.squeeze(F.torch.mul(f.child_c),1)
        c = F.torch.mul(i,u) + F.torch.sum(fc,0)
        h = F.torch.mul(o,F.tanh(c))          # mul element-wise

        return c,h

    def forward(self,tree,inputs):
        # add singleton dimension for future call to node_forward
        embs = F.torch.unsqueeze(self.emb(inputs),1)
        for idx in xrange(tree.num_children):
            _ = self.forward(tree.children[idx],inputs)
        child_c,child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embs[tree.idx-1],child_c,child_h)
        return tree.state

    def get_child_states(self,tree):
        # add extra singleton dimension in middle
        # because pytorch need mini batched
        if tree.num_children == 0:
            child_c = Var(torch.zeros(1,1,self.h_dim))
            child_h = Var(torch.zeros(1, 1, self.h_dim))
            if self.useCuda:
                child_c,child_h = child_c.cuda(),child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children,1,self.h_dim))
            child_h = Var(torch.Tensor(tree.num_children, 1, self.h_dim))
            if self.useCuda:
                child_c,child_h = child_c.cuda(),child_h.cuda()
            for idx in xrange(tree.num_children):
                child_c[idx],child_h[idx] = tree.children[idx].state
            return child_c,child_h