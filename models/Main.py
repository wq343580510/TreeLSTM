from __future__ import print_function

from ChildSumTreeLSTM import *
from config import parse_args


def main():
    global args
    args = parse_args()
    args.input_dim,args.hidden_dim = 300,50
    args.mem_dim = 150
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
