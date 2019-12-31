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

class WTranslator(object):
    def __init__(self, wordset, prefixset, suffixset, flavor):
        wordset.update('UNKNOWN')
        prefixset.update('UNKOWN')
        suffixset.update('UNKNOWN')
        self.flavor = flavor
        self.wdict = list2dict(list(wordset))
        self.pre_dict = list2dict(list(prefixset))
        self.suf_dict = list2dict(list(suffixset))
        cset = set()
        self.max_word_len = 0#max([len(word) for word in wordset])
        for word in wordset:
            if len(word) > self.max_word_len:
                self.max_word_len = len(word)
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
        letter_trans = [np.array([self._dictHandleExp(self.cdict, l) for l in word]) for word in word_list]
        lengths = [len(word) for word in word_list]
        return [letter_trans, lengths]

    def translate(self, word_list):
        if self.flavor == 1:
            return [np.array(self._translate1(word_list))] 
        if self.flavor == 2:
            return self._translate2(word_list)
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
        return {'word' : len(self.wdict), 'pre' : len(self.pre_dict), 'suf' : len(self.suf_dict), 'c' : len(self.cdict)}

class TagTranslator(object):
    def __init__(self, tagset):
        self.tag_dict = list2dict(tagset)
    def translate(self, tag_list):
        return [self.tag_dict[tag] for tag in tag_list]
    def getLengths(self):
        return {'tag': len(self.tag_dict)}


class MyEmbedding(nn.Module):
    def __init__(self, embedding_dim, translator, c_embedding_dim, padding=False):
        super(MyEmbedding, self).__init__()
        self.flavor = translator.flavor
        p1 = 1 if padding else 0
        if translator.flavor == 1:
            l = translator.getLengths()['word'] 
            padding_idx = l if padding else None
            self.wembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = embedding_dim, padding_idx = l)
        if translator.flavor == 2:
            l = translator.getLengths()['c']
            self.cembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = c_embedding_dim, padding_idx = l)
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
        if self.flavor == 2:
            return self.cembeddings(torch.tensor(data).long())
        if self.flavor == 3:
            return self.wembeddings(data) + self.pembeddings(data) + self.sembeddings(data) 

class Padding(object):
    def __init__(self, wT, lT):
        self.wT = wT
        self.lT = lT
        self.wPadIndex = self.wT.getLengths()['word']
        self.lPadIndex = self.lT.getLengths()['tag']
        self.cPadIndex = self.wT.getLengths()['c']
        self.flavor = wT.flavor

    #def padWord(self, word_list, len_list):
    #    max_l = self.wT.max_word_len
    #    batch_size = len(len_list)

    def padData(self, data_b, len_b, max_l, padIndex):
        batch_size = len(len_b)
        padded_data = np.ones((batch_size, max_l))*padIndex
        for i, data in enumerate(data_b):
            padded_data[i][:len_b[i]] = data[0] #first embeddings
        return padded_data

    def padTag(self, tag_b, len_b, max_l, padIndex):
        batch_size = len(len_b)
        padded_tag = np.ones((batch_size, max_l))*padIndex
        for i,tag in enumerate(tag_b):
            padded_tag[i][:len_b[i]] = np.array(tag)
        return padded_tag
  
    def padList(self, data_b, lens_b,  max_l):
        # Expect data_b shape = <batch_size>, <sentence_len>, [<word_len>, 1]
        # returns: <batch_size>, <max sentence len>, <max word_len>

        batch_size = len(lens_b)
        padded_words = np.ones((batch_size, max_l, self.wT.max_word_len))*self.cPadIndex
        padded_lens = np.ones((batch_size, max_l))
        for i, batch in enumerate(data_b):
            sentence, words_len = batch
            for j, word in enumerate(sentence):
                word_len = words_len[j]
                padded_words[i][j][:word_len] = word
                padded_lens[i][j] = word_len
        
        return padded_words, padded_lens

    def padBatch(self, data_b, tag_b, len_b):
        padded_tag = self.padTag(tag_b, len_b, max(len_b), self.lPadIndex)
        if self.flavor == 1:
            padded_data = self.padData(data_b, len_b, max(len_b), self.wPadIndex)
            padded_sublens = None
        elif self.flavor == 2:
            padded_data, padded_sublens = self.padList(data_b, len_b, max(len_b))
            
        return padded_data, padded_tag, len_b, padded_sublens

    def padBatch_v0(self, data_b, tag_b, len_b):
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
        data.sort(key=lambda x: x[2], reverse=True)

        data_b = [d[0] for d in data]
        tag_b = [d[1] for d in data]
        len_b = [d[2] for d in data]

        return self.padBatch(data_b, tag_b, len_b)


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_rnn_dim, tagset_size,
                translator, c_embedding_dim, dropout=False):
        super(BiLSTM, self).__init__()
        self.flavor = translator.flavor
        self.c_embeds_dim = c_embedding_dim
        self.max_word_len = translator.max_word_len
        self.embeddings = MyEmbedding(embedding_dim, translator, c_embedding_dim, True)
        self.lstmc = nn.LSTM(input_size = c_embedding_dim, hidden_size = embedding_dim,
                            batch_first = True)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_rnn_dim,
                            bidirectional=True, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(hidden_rnn_dim*2, tagset_size)
    
    def forward(self, data_list, len_list, batch_size, padded_sublens):
        embeds_list = self.embeddings.forward(data_list)
        if self.flavor == 2:
            batch_size = data_list.shape[0]
            max_sentence = data_list.shape[1]
            
            reshaped_embeds_list = embeds_list.reshape(-1, self.max_word_len, self.c_embeds_dim)
            reshaped_sublens = padded_sublens.reshape(-1).tolist()
            reshaped_sublens = [int(l) for l in reshaped_sublens]
           
            packed_c_embeds = torch.nn.utils.rnn.pack_padded_sequence(reshaped_embeds_list, reshaped_sublens, batch_first=True, enforce_sorted=False)
            lstm_c_out, _ = self.lstmc(packed_c_embeds)
            unpacked_lstmc_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_c_out, batch_first = True)

            reshaped_indexes = torch.Tensor(np.array(reshaped_sublens) - np.ones(len(reshaped_sublens))).long()
            reshaped_indexes = reshaped_indexes.view(-1,1)
            reshaped_indexes = reshaped_indexes.repeat(1, unpacked_lstmc_out.shape[2])
            reshaped_indexes = reshaped_indexes.view(reshaped_indexes.shape[0], 1, reshaped_indexes.shape[1])
            last_layer = torch.gather(unpacked_lstmc_out,1, reshaped_indexes)
            
            reshaped_last_layer = last_layer.reshape(batch_size, max_sentence, -1)
            
            
            '''reshaped_unpacked_lstmc_out = unpacked_lstmc_out.reshape(batch_size, max_sentence, -1)
            print(reshaped_unpacked_lstmc_out.shape)
            
            
            #print(reshaped_unpacked_lstmc_out.shape)
            #print(padded_sublens.shape)
            ##last_layer = [[[torch.tensor(layers[padded_sublens[b][s]]) for layers in sentence] for s, sentence in enumerate(batch)] for b, batch in enumerate(reshaped_unpacked_lstmc_out)]
            #last_layer = [torch.tensor([reshaped_unpacked_lstmc_out[b][s] 
            #                            for s in range(reshaped_unpacked_lstmc_out.shape[1])]) 
            #                            for b in range(reshaped_unpacked_lstmc_out.shape[0])]

            #indexes = torch.Tensor(padded_sublens).long() - torch.ones(padded_sublens.shape)
            
            t = reshaped_unpacked_lstmc_out.shape
            indexes = padded_sublens - np.ones(padded_sublens.shape)
            indexes = torch.Tensor(indexes).long()
            indexes = indexes.reshape(-1)
            ids = indexes.repeat(1, t[2])
            print(ids.shape)
            ids = ids.view(-1, 1, t[2])
            print(ids.shape)
            last_layer = torch.gather(reshaped_unpacked_lstmc_out, 1, ids)
            print(last_layer.shape)
            #ids = ids.repeat(1, 255).view(-1, 1, 255)
            '''
            embeds_output = reshaped_last_layer 
        else:
            embeds_output = embeds_list

        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds_output, len_list, batch_first=True)
        lstm_out, _ = self.lstm(packed_embeds)
        unpacked_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first = True)

        flatten = unpacked_lstm_out.reshape(-1, unpacked_lstm_out.shape[2])
        o_ln1 = self.linear1(flatten)
        shaped = o_ln1.reshape(batch_size, unpacked_lstm_out.shape[1], o_ln1.shape[1])
        return shaped

    def getLabel(self, data):
        _, prediction_argmax = torch.max(data, 1)
        return prediction_argmax


class Run(object):
    def __init__(self, params):
        self.flavor = params['FLAVOR']
        self.edim = params['EMBEDDING_DIM']
        self.rnn_h_dim = params['RNN_H_DIM']
        self.num_epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']
        self.c_embedding_dim = params['CHAR_EMBEDDING_DIM']

    def train(self):
        print("Loading data")
        train_dataset = As3Dataset('train')
        print("Done loading data")

        self.wTran = WTranslator(train_dataset.word_set, train_dataset.prefix_set,
                                  train_dataset.suffix_set, self.flavor)
        self.lTran = TagTranslator(train_dataset.tag_set)

        train_dataset.toIndexes(wT = self.wTran, lT = self.lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=self.wTran, tagset_size = self.lTran.getLengths()['tag'] + 1,
                c_embedding_dim = self.c_embedding_dim)

        if (sys.argv[1] == 'load') or (sys.argv[1] == 'loadsave'):
            tagger.load_state_dict(torch.load('bilstm_params.pt'))

        loss_function = nn.CrossEntropyLoss() #ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adam(tagger.parameters(), lr=0.01)

        padder = Padding(self.wTran, self.lTran)

        train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size, shuffle=True,
                          collate_fn = padder.collate_fn)
        print("Starting training")
        print("data length = " + str(len(train_dataset)))
        
        for epoch in range(self.num_epochs):
            loss_acc = 0
            progress1 = 0
            progress2 = 0
            correct_cntr = 0
            total_cntr = 0
            for sample in train_dataloader:
                if progress1/1000 == progress2:
                    print("reached " + str(progress2*1000))
                    progress2+=1
                progress1 += self.batch_size
                tagger.zero_grad()
                batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                batch_tag_score = tagger.forward(batch_data_list, batch_len_list, len(batch_data_list), padded_sublens)
               
                flatten_tag = batch_tag_score.reshape(-1, batch_tag_score.shape[2])
                flatten_label = torch.LongTensor(batch_label_list.reshape(-1))

                #calc accuracy
                predicted_tags = tagger.getLabel(flatten_tag)
                diff = predicted_tags - flatten_label
                no_diff = (diff == 0)
                o_mask = (flatten_label == self.lTran.getLengths()['tag'])
                no_diff_and_o_label = no_diff*o_mask
                to_ignore = len(no_diff_and_o_label[no_diff_and_o_label == True])
                tmp = len(diff[diff == 0]) - to_ignore
                if tmp < 0:
                    raise Exception("non valid tmp value")
                correct_cntr += tmp 
                total_cntr += len(predicted_tags) - to_ignore

                loss = loss_function(flatten_tag, flatten_label)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()
            print("epoch: " + str(epoch) + " " + str(loss_acc))
            print("accuracy " + str(correct_cntr/total_cntr))
        
        if (sys.argv[1] == 'save') or (sys.argv[1] == 'loadsave'):
            torch.save(tagger.state_dict(), 'bilstm_params.pt')
       
        '''
        testing_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=1, shuffle=False,
                          collate_fn = padder.collate_fn)
        reversed_dict = reverseDict(lTran.tag_dict)
        reversed_dict.append('UNKNOWN')
        with torch.no_grad():
            with open('tmp_train_res', 'w') as wf:
                for sample in testing_dataloader:
                    batch_data_list, batch_label_list, batch_len_list = sample
                    #print(batch_len_list)
                    batch_tag_score = tagger.forward(batch_data_list, batch_len_list, len(batch_data_list))
                    #print(batch_tag_score.shape)
                    for i, sample_tag_list in enumerate(batch_tag_score):
                        #print(sample_tag_list.shape)
                        predicted_tags = tagger.getLabel(sample_tag_list)
                        #print(predicted_tags.shape)
                        for j in range(batch_len_list[i]):
                            try:
                                t = predicted_tags[j]
                            except:
                                print("j:")
                                print(j)
                                print("t:")
                                print(t)
                                print("predicted_tags:")
                                print(predicted_tags)
                            try:
                                w = reversed_dict[t]
                            except:
                                print("w:")
                                print(w)
                                print("t:")
                                print(t)
                                print("reversed dict:")
                                print(reversed_dict)
                            wf.write(str(w) + "\n")
                        wf.write("\n")
        '''

flavor = sys.argv[2]
run = Run({'FLAVOR':int(flavor), 'EMBEDDING_DIM' : 50, 'RNN_H_DIM' : 50, 'EPOCHS' : 5, 'BATCH_SIZE' : 100, 'CHAR_EMBEDDING_DIM': 5})
run.train()
