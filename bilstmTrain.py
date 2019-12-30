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
                    dataset.append((sample_w, sample_t, len(sample_t)))
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

    def toIndexes(self, wT, lT):
        self.dataset = [(wT.translate(data[0]), lT.translate(data[1]), data[2]) for data in self.dataset]

    def __getitem__(self, index):
        return self.dataset[index]
        return self.dataset[index][0], self.dataset[index][1], self.dataset[index][2]
        #return self.wT.translate(self.dataset[index][0]), self.lT.translate(self.dataset[index][1])
        

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
        return [self._dictHandleExp(self.wdict, word) for word in word_list]

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
    def getLengths(self):
        return {'tag': len(self.tag_dict)}


class MyEmbedding(nn.Module):
    def __init__(self, embedding_dim, translator, padding=False):
        super(MyEmbedding, self).__init__()
        self.flavor = translator.flavor
        p1 = 1 if padding else 0
        if translator.flavor == 1:
            l = translator.getLengths()['word'] 
            padding_idx = l if padding else None
            self.wembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = embedding_dim, padding_idx = l)
        if translator.flavor == 3:
            l = translator.getLengths()['word'] 
            padding_idx = l if padding else None
            self.wembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = embedding_dim, padding_idx = l)
            l = translator.getLengths()['pre'] 
            padding_idx = l if padding else None
            self.pembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = embedding_dim, padding_idx = l)
            l = translator.getLengths()['suf'] 
            padding_idx = l if padding else None
            self.sembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = embedding_dim, padding_idx = l)
    
    def forward(self, data):
        if self.flavor == 1:
            return self.wembeddings(torch.tensor(data).long())
        if self.flavor == 3:
            return self.wembeddings(data) + self.pembeddings(data) + self.sembeddings(data) 

class Padding(object):
    def __init__(self, wT, lT):
        self.wT = wT
        self.lT = lT
        self.wPadIndex = self.wT.getLengths()['word']
        self.lPadIndex = self.lT.getLengths()['tag']

    def padBatch(self, data_b, tag_b, len_b):
        max_l = max(len_b)
        batch_size = len(tag_b)
        padded_data = np.ones((batch_size, max_l))*self.wPadIndex
        padded_tag = np.ones((batch_size, max_l))*self.lPadIndex
        for i, sample in enumerate(zip(data_b, tag_b)):
            data, tag = sample
            padded_data[i][:len_b[i]] = data[0] #first embeddings
            padded_tag[i][:len_b[i]] = np.array(tag)
        return padded_data, padded_tag, len_b

    def collate_fn(self, data):
        data_b = [d[0] for d in data]
        tag_b = [d[1] for d in data]
        len_b = [d[2] for d in data]
        data_b = [b for _,b in sorted(zip(len_b, data_b), reverse=True)]
        tag_b = [b for _,b in sorted(zip(len_b, tag_b), reverse=True)]
        len_b = sorted(len_b, reverse=True)

        return self.padBatch(data_b, tag_b, len_b)


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_rnn_dim, tagset_size,
                translator, dropout=False):
        super(BiLSTM, self).__init__()
        self.embeddings = MyEmbedding(embedding_dim, translator, True)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_rnn_dim,
                            bidirectional=True, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(hidden_rnn_dim*2, tagset_size)
    
    def forward(self, data_list, len_list, batch_size):
        embeds_list = self.embeddings.forward(data_list)
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds_list, len_list, batch_first=True)
        #lstm_out, hidden1 = self.lstm(embeds.view(len(data_list[0]), batch_size, -1))
        lstm_out, _ = self.lstm(packed_embeds)
        unpacked_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first = True)
       
        print("unpacked_lstm_out shape: " + str(unpacked_lstm_out.shape))
        o_ln1 = [[self.linear1(o) for o in seq] for seq in unpacked_lstm_out]
        #o_ln1 = [self.linear1(lstm_w) for lstm_w in unpacked_lstm_out]
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
        print("Done loading data")

        wTran = Sample2EmbedIndex(train_dataset.word_set, train_dataset.prefix_set,
                                  train_dataset.suffix_set, self.flavor)
        lTran = TagTranslator(train_dataset.tag_set)

        #train_dataset.setWordTranslator(wTran)
        #train_dataset.setLabelTranslator(lTran)
        train_dataset.toIndexes(wT = wTran, lT = lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=wTran, tagset_size = len(lTran.tag_dict) + 1)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(tagger.parameters(), lr=0.01)

        padder = Padding(wTran, lTran)

        train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size, shuffle=False,
                          collate_fn = padder.collate_fn)
        print("Starting training")
        for epoch in range(self.num_epochs):
            loss_acc = 0
            for sample in train_dataloader:
                tagger.zero_grad()
                batch_data_list, batch_label_list, batch_len_list = sample
                #data_list = data_list[0] #since there is only one type of embedding
                #padder.padBatch(data_list, label_list, lens_list) 
                batch_tag_score = tagger.forward(batch_data_list, batch_len_list, len(batch_data_list))
                loss = None
                for tag_score, label_list  in zip(batch_tag_score, batch_label_list):
                    for tag, label in zip(tag_score, label_list):
                        t_label = torch.tensor([label]).long()
                        tag = tag.view(1,tag.shape[0])
                        t = loss_function(tag, t_label)
                        loss = t if loss is None else loss + t
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()
            print("epoch: " + str(epoch) + " " + str(loss_acc))



run = Run({'FLAVOR':1, 'EMBEDDING_DIM' : 10, 'RNN_H_DIM' : 30, 'EPOCHS' : 5, 'BATCH_SIZE' : 10})
run.train()
