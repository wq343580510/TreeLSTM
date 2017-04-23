import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch TreeLSTM for reranking on dependency trees')
    parser.add_argument('--date',default='data/',help = 'directory of data')
    parser.add_argument('--batch_size',default=200,type = int,help = 'batchsize for optimizer updates')
    parser.add_argument('--epochs',default=20,type = int,help='number of total epochs to run')
    parser.add_argument('--LR',default=0.01,type = float,help= 'learning rate')
    parser.add_argument('--wd',default='1e-4',type = float,help = 'weight decay')
    parser.add_argument('--optim', default='adagrad', help='optimizer')
    parser.add_argument('--seed', default='1234', type=int, help='randon seed')
    cuda_parser = parser.add_mutually_exclusive_group(required = False)
    cuda_parser.add_argument('--cuda',dest = 'cuda', action = 'store_true')
    cuda_parser.add_argument('--no-cuda',dest='cuda',action = 'store_false')
    parser.set_defaults(cuda = True)

    args = parser.parse_args()
    return args
