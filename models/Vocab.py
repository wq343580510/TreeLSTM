class Vocab(object):
    def __init__(self,filename):
        self.words = []
        self.tags = []
        self.tag2idx = {}
        self.word2idx = {}
        self.unk_index = -1
        self.unk_token = 'unk'
        self.create_voc(filename)

    def create_voc(self,filename):
        with open(filename) as reader:
            data = reader.readlines()
            for line in data:
                if line.strip() != 'PTB_KBEST' and '_' in line:
                    word = line.split()[1]
                    tag = line.split()[3]
                    if not tag in self.tags:
                        self.tags.append(tag)
                        self.tag2idx[tag] = len(self.tags) - 1
                    if not word in self.words:
                        self.words.append(word)
                        self.word2idx[word]=len(self.words)-1
        reader.close()

    def indexoftag(self,tag):
        return self.tag2idx.get(tag, self.unk_index)
    def index(self,word):
        return self.word2idx.get(word, self.unk_index)
    def size(self):
        return len(self.words)
    def tagsize(self):
        return len(self.tag2idx)

