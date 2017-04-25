from ChildSumTreeLSTM import *
from tree import *
import torch
import numpy as np
from torch.autograd import Variable
model = RankerTreeLSTM(False,20,10,10)
node0 = Tree(0,0,0)
node1 = Tree(1,1,1)
node2 = Tree(2,2,2)
node3 = Tree(3,3,3)
node4 = Tree(4,4,4)
node0.add_child(node1)
node0.add_child(node2)
node0.add_child(node3)
node3.add_child(node4)
arr = np.array([0,1,2,3,4])
input = Variable(torch.from_numpy(arr))

import torch.optim as optim

optimizer = optim.Adagrad(model.parameters())

optimizer.zero_grad()
output = model(node0,input)

print output



