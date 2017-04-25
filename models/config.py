import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch TreeLSTM for reranking on dependency trees')
    parser.add_argument('--dir',default='/Users/qiwang/data/data/',help = 'directory of data')
    parser.add_argument('--train',default='train',help = 'train')
    parser.add_argument('--dev',default='dev',help = 'dev')
    parser.add_argument('--test',default='test',help = 'test')
    parser.add_argument('--model_file',default='model_best.pkl',help = 'model')
    parser.add_argument('--vocab', default='dict.pkl', help='vocab')
    parser.add_argument('--batch_size',default=200,type = int,help = 'batchsize for optimizer updates')
    parser.add_argument('--epochs',default=20,type = int,help='number of total epochs to run')
    parser.add_argument('--lr',default=0.01,type = float,help= 'learning rate')
    parser.add_argument('--wd',default='1e-4',type = float,help = 'weight decay')
    parser.add_argument('--optim', default='adagrad', help='optimizer')
    parser.add_argument('--seed', default='1234', type=int, help='random seed')
    cuda_parser = parser.add_mutually_exclusive_group(required = False)
    cuda_parser.add_argument('--cuda',dest = 'cuda', action = 'store_true')
    cuda_parser.add_argument('--no-cuda',dest='cuda',action = 'store_false')
    parser.set_defaults(cuda = False)

    args = parser.parse_args()
    return args
