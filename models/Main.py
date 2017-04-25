from __future__ import print_function
from eval.eval_data import *
from ChildSumTreeLSTM import *
from config import parse_args
from data_reader import data_manager
import torch.optim as optim
from Trainer import *


def main():
    global args
    args = parse_args()
    args.input_dim,args.hidden_dim = 300,50
    args.mem_dim = 150
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    manager = data_manager(args)
    evaluate_baseline(manager.dev_data)
    evaluate_baseline_random(manager.dev_data)
    evaluate_oracle_worst(manager.dev_data)
    print('build model')
    model = RankerTreeLSTM(args.cuda,manager.vocab.size(),args.input_dim,args.hidden_dim)
    print('model established')
    print(model.__repr__())
    if args.cuda:
        model.cuda()
    optimizer = optim.Adagrad(model.parameters(),lr=args.lr,weight_decay=args.wd)
    trainer = Trainer(args,model,optimizer)
    max_uas = 0
    for epoch in range(args.epochs):
        train_loss = trainer.train(manager.train_data)
        uas = evaluate_dataset_point(model, manager.dev_data)
        print(train_loss)
        if uas > max_uas:
            max_uas = uas
            torch.save(model,manager.model_path)

if __name__ == '__main__':
    main()