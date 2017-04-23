class Tree:
    def __init__(self,word,tag):
        self.parent = None
        self.word = word
        self.tag = tag
        self.num_children = 0
        self.children = {}

    def add_child(self,c):
        c.parent = self
        self.children[self.num_children] = c
        self.num_children += 1

    def size(self):
        if self.size is not None:
            return self.size
        size = 1
        for i in range(0,self.num_children):
            size += self.children[i].size()
        self.size = size
        return size

    def depth(self):
        depth = 0
        if self.num_children > 0:
            for i in range(0,self.num_children):
                depth = max(depth,self.children[i].depth())
            depth += 1
        return depth

    def depth_first_preorder(self):
        nodes = {}
        depth_first_preorder(self, nodes)
        return nodes



def depth_first_preorder(tree,nodes):
    if tree == None:
        return
    nodes.append(tree)
    for i in range(0,tree.num_children):
        depth_first_preorder(tree.children[i],nodes)
