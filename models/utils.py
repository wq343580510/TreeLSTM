from tree import *
import pickle

def save_dict(vocab,output_file):
    output = open(output_file, 'wb')
    pickle.dump(vocab,output, protocol=2)
    output.close()

def load_dict(input_file):
    pkl_file = open(input_file, 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    return vocab

def read_tree(list,vocab):
    att_list = []
    nodes = []
    root = None
    retlist = []
    x = []
    for i in range(len(list)):
        att_list.append(list[i].split())
        word = att_list[i][1]
        tag = att_list[i][3]
        val = vocab.index(word)
        tag_idx = vocab.indexoftag(tag)
        nodes.append(Tree(val,i,tag_idx))
        parent = att_list[i][6]
        label = att_list[i][7]
        x.append(val)
        retlist.append(word+'#'+parent+'#'+label)

    for i in range(len(list)):
        parent = int(att_list[i][6]) - 1
        if parent >= 0:
            nodes[parent].add_child(nodes[i])
        elif parent == -1:
            root = nodes[i]

    return root,retlist,x
