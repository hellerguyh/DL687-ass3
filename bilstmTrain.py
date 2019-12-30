import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys

FOLDER_PATH =None
DEBUG = True
def DEBUG_PRINT(x):
  if DEBUG:
    print(x)

def list2dict(lst):
    it = iter(lst)
    indexes = range(len(lst))
    res_dct = dict(zip(it, indexes))
    return res_dct

def reverseDict(d):
    vals = ['']*len(d.keys())
    for k in d.keys():
        vals[d[k]] = k
    return vals


class As3Dataset(Dataset):
    def __init__(self, file_path, lower_words = False, is_test_data=False):    
        self.lower_words = lower_words
        self.file_path = file_path
        with open(file_path, "r") as df:
            content = df.read().split('\n')

        dataset = []
        sample_w = []
        sample_t = []
        word_list = []
        tag_list = []
        prefix_list = []
        suffix_list = []
        for line in content:
            if line == "":
                if last_line_is_space == True:
                    pass
                else:
                    dataset.append((sample_w, sample_t))
                    sample_w = []
                    sample_t = []
                    last_line_is_space = True
            else:
                last_line_is_space = False
                splitted_line = line.split()
                label = None if is_test_data else splitted_line[1]
                word = self.lowerWords(splitted_line[0])
                word_list.append(word)
                prefix_list.append(word[:3])
                suffix_list.append(word[-3:])
                tag_list.append(label)
                sample_w.append(self.lowerWords(splitted_line[0]))
                sample_t.append(label if len(splitted_line) > 1 else '')
        
        self.word_set = set(word_list)
        self.prefix_set = set(prefix_list)
        self.suffix_set = set(suffix_list)
        self.tag_set = set(tag_list)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def lowerWords(self, x):
        return x.lower() if self.lower_words else x

    def setWordTranslator(self, tran):
        self.wT = tran
    
    def setLabelTranslator(self, tran):
        self.lT = tran
  
    def __getitem__(self, index):
        return self.wT.translate(self.dataset[index][0]), self.lT.translate(self.dataset[index][1])
        

class Sample2EmbedIndex(object):
    def __init__(self, wordset, prefixset, suffixset, flavor):
        wordset.update('UNKNOWN')
        prefixset.update('UNKOWN')
        suffixset.update('UNKNOWN')
        self.flavor = flavor
        self.wdict = list2dict(list(wordset))
        self.pre_dict = list2dict(list(prefixset))
        self.suf_dict = list2dict(list(suffixset))
        cset = set()
        for word in wordset:
            for c in word:
                cset.update(c)
        self.cdict = list2dict(list(cset))
    
    def _dictHandleExp(self, dic, val):
      try: 
        return dic[val]
      except KeyError:
        return dic['UNKNOWN']

    def _translate1(self, word_list):
        return [[self._dictHandleExp(self.wdict, word)] for word in word_list]

    def _translate2(self, word_list):
        return [[self._dictHandleExp(self.cdict, l) for l in word] for word in word_list]

    def translate(self, word_list):
        if self.flavor == 1:
            return [np.array(self._translate1(word_list))] 
        if self.flavor == 2:
            return [np.array(self._translate2(word_list))]
        if self.flavor == 3:
            w = [self._dictHandleExp(self.wdict, word) for word in word_list]
            p = [self._dictHandleExp(self.pre_dict, word[:3]) for word in word_list]
            s = [self._dictHandleExp(self.suf_dict, word[-3:]) for word in word_list]
            return [np.array([w, p, s])]
        if self.flavor == 4:
            first = self._translate1(word_list)
            second = self._translate2(word_list)
            return [first, second]

    def getLengths(self):
        if self.flavor == 1:
            return {'word': len(self.wdict)}
        if self.flavor == 3:
            return {'word' : len(self.wdict), 'pre' : len(self.pre_dict), 'suf' : len(self.suf_dict)}

class TagTranslator(object):
    def __init__(self, tagset):
        self.tag_dict = list2dict(tagset)
    def translate(self, tag_list):
        return [self.tag_dict[tag] for tag in tag_list]


class MyEmbedding(nn.Module):
    def __init__(self, embedding_dim, translator):
        super(MyEmbedding, self).__init__()
        self.flavor = translator.flavor
        if translator.flavor == 1:
            self.wembeddings = nn.Embedding(translator.getLengths()['word'], embedding_dim)
        if translator.flavor == 3:
            self.wembeddings = nn.Embedding(translator.getLengths()['word'], embedding_dim)
            self.pembeddings = nn.Embedding(translator.getLengths()['pre'], embedding_dim)
            self.sembeddings = nn.Embedding(translator.getLengths()['suf'], embedding_dim)
    
    def forward(self, data):
        if self.flavor == 1:
            try: 
                return self.wembeddings(data)
            except:
                print(data)
                raise Exception()
        if self.flavor == 3:
            return self.wembeddings(data) + self.pembeddings(data) + self.sembeddings(data) 


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_rnn_dim, tagset_size,
                translator, dropout=False):
        super(BiLSTM, self).__init__()
        self.embeddings = MyEmbedding(embedding_dim, translator)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_rnn_dim,
                            bidirectional=True, num_layers=2)
        self.linear1 = nn.Linear(hidden_rnn_dim*2, tagset_size)
    
    def forward(self, data, batch_size):
        embeds = self.embeddings.forward(data)
        lstm_out, hidden1 = self.lstm(embeds.view(len(data[0]), batch_size, -1))
        o_ln1 = [self.linear1(lstm_w) for lstm_w in lstm_out]
        return o_ln1

    def getLabel(self, data):
        _, prediction_argmax = data[0].max(0)
        return prediction_argmax


class Run(object):
    def __init__(self, params):
        self.flavor = params['FLAVOR']
        self.edim = params['EMBEDDING_DIM']
        self.rnn_h_dim = params['RNN_H_DIM']
        self.num_epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']

    def train(self):
        print("Loading data")
        train_dataset = As3Dataset('train_short')
        train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size, shuffle=False)
        print("Done loading data")

        wTran = Sample2EmbedIndex(train_dataset.word_set, train_dataset.prefix_set,
                                  train_dataset.suffix_set, self.flavor)
        lTran = TagTranslator(train_dataset.tag_set)

        train_dataset.setWordTranslator(wTran)
        train_dataset.setLabelTranslator(lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=wTran, tagset_size = len(lTran.tag_dict))

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(tagger.parameters(), lr=0.01)

        print("Starting training")
        for epoch in range(self.num_epochs):
            loss_acc = 0
            for sample in train_dataloader:
                tagger.zero_grad()
                data_list, label_list = sample
                data_list = data_list[0] #since there is only one type of embedding
                tag_score = tagger.forward(data_list, self.batch_size)
                loss = None
                for tag, label in zip(tag_score, label_list):
                    t_label = torch.tensor([label]).long()
                    t = loss_function(tag, t_label)
                    loss = t if loss is None else loss + t
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()
            print("epoch: " + str(epoch) + " " + str(loss_acc))



run = Run({'FLAVOR':1, 'EMBEDDING_DIM' : 3, 'RNN_H_DIM' : 30, 'EPOCHS' : 5, 'BATCH_SIZE' : 1})
run.train()
