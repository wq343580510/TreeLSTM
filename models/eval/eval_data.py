from evaluate import *
import random
import torch
from torch.autograd import Variable as Var
def evaluate_baseline(data):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        for line in inst.lines[len(inst.kbest)-1]:
            pred_trees.append(line)
        for line in inst.gold_lines:
            gold_trees.append(line)
    print 'baseline: %.4f' % (evaluate(pred_trees,gold_trees)[0])

def evaluate_baseline_random(data):
    random.seed(121)
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        rand = int(random.random()*len(inst.kbest))
        for line in inst.lines[rand]:
            pred_trees.append(line)
        for line in inst.gold_lines:
            gold_trees.append(line)
    print 'baseline: %.4f' % (evaluate(pred_trees,gold_trees)[0])


def evaluate_oracle_worst(data):
    oracle_trees = []
    worst_trees = []
    gold_trees = []
    pred_trees = []
    for i, inst in enumerate(data):
        max = 0
        maxid = 0
        min = 1
        minid = 0
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')

        for line in inst.lines[len(inst.kbest)-1]:
            pred_trees.append(line)
        pred_trees.append('\n')

        i = 0
        for list in inst.lines:
            temp = []
            for line in list:
                temp.append(line)
            temp.append('\n')
            res = evaluate(temp, inst.gold_lines)[0]
            if res > max :
                max = res
                maxid = i
            if res < min :
                min = res
                minid = i
            i += 1
        for line in inst.lines[maxid]:
            oracle_trees.append(line)
        oracle_trees.append('\n')
        for line in inst.lines[minid]:
            worst_trees.append(line)
        worst_trees.append('\n')
    print 'f1score: %.4f'  % (evaluate(pred_trees, gold_trees)[0])
    print 'oracle: %.4f'  % (evaluate(oracle_trees, gold_trees)[0])
    print 'worst: %.4f'  % (evaluate(worst_trees, gold_trees)[0])


def evaluate_dataset_point(model, data):
    pred_trees = []
    gold_trees = []
    for inst in data:
        pred_scores = []
        for i in range(len(inst.kbest)):
            pred_scores.append(model.predict(inst.kbest[i],inst.inputs[i]))
        scores = pred_scores
        max_score = max(scores)
        max_id = scores.index(max_score)
        for line in inst.lines[max_id]:
            pred_trees.append(line)
        for line in inst.gold_lines:
            gold_trees.append(line)
    res = evaluate(pred_trees,gold_trees)
    print 'f1score: %.8f' % (res[0])
    return res