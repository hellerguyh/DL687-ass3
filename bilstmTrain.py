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
        self.is_test_data = is_test_data

    def __len__(self):
        return len(self.dataset)

    def lowerWords(self, x):
        return x.lower() if self.lower_words else x

    def toIndexes(self, wT, lT):
        self.dataset = [(wT.translate(data[0]), lT.translate(data[1]) if self.is_test_data==False else None, data[2]) for data in self.dataset]

    def __getitem__(self, index):
        return self.dataset[index]

class WTranslator(object):
    def __init__(self, wordset, prefixset, suffixset, flavor, init=True):
        if init:
            wordset.update(["UNKNOWN"])
            prefixset.update(["UNKNOWN"])
            suffixset.update(["UNKNOWN"])
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
                    cset.update(c.lower())
            cset.update(["UNKNOWN"])
            self.cdict = list2dict(list(cset))
   

    def saveParams(self):
        return {'cdict':self.cdict, 'wdict':self.wdict, 'pre':self.pre_dict, 'suf':self.suf_dict,
                'flavor':self.flavor, 'max_word_len':self.max_word_len}

    def loadParams(self, params):
        self.cdict = params['cdict']
        self.wdict = params['wdict']
        self.pre_dict = params['pre']
        self.suf_dict = params['suf']
        self.max_word_len = params['max_word_len']
        self.flavor = params['flavor']

    def _dictHandleExp(self, dic, val):
      try: 
        return dic[val]
      except KeyError:
        return dic['UNKNOWN']

    def _translate1(self, word_list):
        return [self._dictHandleExp(self.wdict, word) for word in word_list]

    def _translate2(self, word_list):
        letter_trans = [np.array([self._dictHandleExp(self.cdict, l.lower()) for l in word]) for word in word_list]
        lengths = [len(word) for word in word_list]
        return [letter_trans, lengths]

    def translate(self, word_list):
        if self.flavor == 'a':
            return [np.array(self._translate1(word_list))] 
        if self.flavor == 'b':
            return self._translate2(word_list)
        if self.flavor == 'c':
            w = np.array(self._translate1(word_list))
            p = np.array([self._dictHandleExp(self.pre_dict, word[:3]) for word in word_list])
            s = np.array([self._dictHandleExp(self.suf_dict, word[-3:]) for word in word_list])
            return [w, p, s]
        if self.flavor == 'd':
            first = np.array(self._translate1(word_list))
            second = self._translate2(word_list)
            return [first, second]

    def getLengths(self):
        return {'word' : len(self.wdict), 'pre' : len(self.pre_dict), 'suf' : len(self.suf_dict), 'c' : len(self.cdict)}

class TagTranslator(object):
    def __init__(self, tagset, init=True):
        if init:
            self.tag_dict = list2dict(tagset)

    def translate(self, tag_list):
        return [self.tag_dict[tag] for tag in tag_list]

    def getLengths(self):
        return {'tag': len(self.tag_dict)}

    def saveParams(self):
        return {'tag':self.tag_dict}

    def loadParams(self, params):
        self.tag_dict = params['tag']


class MyEmbedding(nn.Module):
    def __init__(self, embedding_dim, translator, c_embedding_dim, padding=False):
        super(MyEmbedding, self).__init__()
        self.flavor = translator.flavor
        p1 = 1 if padding else 0
        if translator.flavor == 'a' or translator.flavor == 'd':
            l = translator.getLengths()['word'] 
            padding_idx = l if padding else None
            self.wembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = embedding_dim, padding_idx = l)
        if translator.flavor == 'b' or translator.flavor == 'd':
            l = translator.getLengths()['c']
            self.cembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = c_embedding_dim, padding_idx = l)
        if translator.flavor == 'c':
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
        if self.flavor == 'a':
            return self.wembeddings(torch.tensor(data).long())
        if self.flavor == 'b':
            return self.cembeddings(torch.tensor(data).long())
        if self.flavor == 'c':
            return (self.wembeddings(torch.tensor(data[0]).long()) + 
                    self.pembeddings(torch.tensor(data[1]).long()) + 
                    self.sembeddings(torch.tensor(data[2]).long()))
        if self.flavor == 'd':
            word_embeds = self.wembeddings(torch.tensor(data[0]).long())
            char_embeds = self.cembeddings(torch.tensor(data[1]).long())
            return (word_embeds, char_embeds)

class Padding(object):
    def __init__(self, wT, lT):
        self.wT = wT
        self.lT = lT
        self.wPadIndex = self.wT.getLengths()['word']
        self.lPadIndex = self.lT.getLengths()['tag']
        self.cPadIndex = self.wT.getLengths()['c']
        self.pPadIndex = self.wT.getLengths()['pre']
        self.sPadIndex = self.wT.getLengths()['suf']
        self.flavor = wT.flavor

    def padData(self, data_b, len_b, max_l, padIndex):
        batch_size = len(len_b)
        padded_data = np.ones((batch_size, max_l))*padIndex
        for i, data in enumerate(data_b):
            padded_data[i][:len_b[i]] = data #first embeddings
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
        padded_sublens = None
        if self.flavor == 'a':
            word_data_b = [b[0] for b in data_b]
            padded_data = self.padData(word_data_b, len_b, max(len_b), self.wPadIndex)
        elif self.flavor == 'b':
            padded_data, padded_sublens = self.padList(data_b, len_b, max(len_b))
        elif self.flavor == 'c':
            word_data_b = [d[0] for d in data_b]
            prefix_data_b = [d[1] for d in data_b]
            suffix_data_b = [d[2] for d in data_b]
            word_padded_data = self.padData(word_data_b, len_b, max(len_b), self.wPadIndex)
            prefix_padded_data = self.padData(prefix_data_b, len_b, max(len_b), self.pPadIndex)
            suffix_padded_data = self.padData(suffix_data_b, len_b, max(len_b), self.sPadIndex)
            padded_data = (word_padded_data, prefix_padded_data, suffix_padded_data)
        elif self.flavor == 'd':
            word_b = [d[0] for d in data_b]
            char_b = [d[1] for d in data_b]
            padded_word_data = self.padData(word_b, len_b, max(len_b), self.wPadIndex)
            padded_char_data, padded_sublens = self.padList(char_b, len_b, max(len_b))
            padded_data = (padded_word_data, padded_char_data)
        
        return padded_data, padded_tag, len_b, padded_sublens

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
        self.embedding_dim = embedding_dim
        self.embeddings = MyEmbedding(embedding_dim, translator, c_embedding_dim, True)
        self.lstmc = nn.LSTM(input_size = c_embedding_dim, hidden_size = embedding_dim,
                            batch_first = True)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_rnn_dim,
                            bidirectional=True, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(hidden_rnn_dim*2, tagset_size)
        self.lineare = nn.Linear(embedding_dim*2, embedding_dim)
   

    def runLSTMc(self, data_list, embeds_list, padded_sublens):
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
        return reshaped_last_layer
            

    def forward(self, data_list, len_list, padded_sublens):
        batch_size = len(len_list)
        embeds_list = self.embeddings.forward(data_list)
        if self.flavor == 'b':
            embeds_char = embeds_list
            char_data_list = data_list
        elif self.flavor == 'd':
            embeds_word = embeds_list[0]
            embeds_char = embeds_list[1]
            char_data_list = data_list[1]

        if self.flavor == 'b' or self.flavor == 'd':
            lstm_embeds_word = self.runLSTMc(char_data_list, embeds_char, padded_sublens)

        if self.flavor == 'b':
            embeds_out = lstm_embeds_word
        elif self.flavor == 'd':
            e_joined = torch.cat((embeds_word, lstm_embeds_word), dim=2)
            flatten = e_joined.reshape(-1, e_joined.shape[2])
            le_out = self.lineare(e_joined)
            embeds_out = le_out.reshape(batch_size, e_joined.shape[1], self.embedding_dim)
        else:
            embeds_out = embeds_list
        
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds_out, len_list, batch_first=True)
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
        self.train_file = params['TRAIN_FILE']
        self.dev_file = params['DEV_FILE']
        self.test_file = params['TEST_FILE']
        self.test_o_file = params['TEST_O_FILE']
        self.tagging_type = params['TAGGING_TYPE']
        self.model_file = params['MODEL_FILE']
        self.save_to_file = params['SAVE_TO_FILE']
        self.run_dev = params['RUN_DEV']
        if self.tagging_type == "ner":
            self.ignore_Os = True 
        elif self.tagging_type == "pos":
            self.ignore_Os = False
        else: 
            raise Exception("Invalid tagging type")

    def _save_model_params(self, tagger, wT, lT):
        try:
            params = torch.load('model_params.pt')
        except FileNotFoundError:
            print("No model params file found - creating new model params")
            params = {}

        flavor_params = {}
        flavor_params.update({'tagger' : tagger.state_dict()})
        flavor_params.update({'wT' : wT.saveParams()})
        flavor_params.update({'lT' : lT.saveParams()})
        params.update({str(self.flavor)+self.tagging_type : flavor_params})
        torch.save(params, self.model_file)

    def _load_translators_params(self, wT, lT):
        params = torch.load(self.model_file)
        flavor_params = params[str(self.flavor)+self.tagging_type]
        wT.loadParams(flavor_params['wT'])
        lT.loadParams(flavor_params['lT'])

    def _load_bilstm_params(self, tagger):
        params = torch.load(self.model_file)
        flavor_params = params[str(self.flavor)+self.tagging_type]
        tagger.load_state_dict(flavor_params['tagger'])

    def _calc_batch_acc(self, tagger, flatten_tag, flatten_label): 
        predicted_tags = tagger.getLabel(flatten_tag)
        diff = predicted_tags - flatten_label
        no_diff = (diff == 0)
        padding_mask = (flatten_label == self.lTran.getLengths()['tag'])
        if self.ignore_Os:
            Os_mask = (flatten_label == self.lTran.tag_dict['O'])
            no_diff_and_padding_label = no_diff*(padding_mask + Os_mask)
            no_diff_and_padding_label = (no_diff_and_padding_label > 0)
        else:
            no_diff_and_padding_label = no_diff*padding_mask

        to_ignore = len(no_diff_and_padding_label[no_diff_and_padding_label == True])
        tmp = len(diff[diff == 0]) - to_ignore
        if tmp < 0:
            raise Exception("non valid tmp value")
        correct_cntr = tmp 
        total_cntr = len(predicted_tags) - to_ignore
        return correct_cntr, total_cntr

    def _flat_vecs(self, batch_tag_score, batch_label_list):
        flatten_tag = batch_tag_score.reshape(-1, batch_tag_score.shape[2])
        flatten_label = torch.LongTensor(batch_label_list.reshape(-1))
        return flatten_tag, flatten_label

    def runOnDev(self, tagger, padder):
        tagger.eval()
        dev_dataset = As3Dataset(self.dev_file)
        dev_dataset.toIndexes(wT = self.wTran, lT = self.lTran)
        dev_dataloader = DataLoader(dataset=dev_dataset,
                                    batch_size=self.batch_size, shuffle=False,
                                    collate_fn = padder.collate_fn)
        with torch.no_grad():
            correct_cntr = 0
            total_cntr = 0
            for sample in dev_dataloader:
                batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample

                batch_tag_score = tagger.forward(batch_data_list, batch_len_list, padded_sublens)
              
                flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                #calc accuracy
                c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                correct_cntr += c 
                total_cntr += t
        
        print("Validation accuracy " + str(correct_cntr/total_cntr))
        tagger.train()

    def test(self):
        test_dataset = As3Dataset(file_path = self.test_file, 
                                  is_test_data = True)

        self.wTran = WTranslator(None, None, None, None, False)
        self.lTran = TagTranslator(None, False)

        self._load_translators_params(self.wTran, self.lTran)
        test_dataset.toIndexes(wT = self.wTran, lT = self.lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=self.wTran, tagset_size = self.lTran.getLengths()['tag'] + 1,
                c_embedding_dim = self.c_embedding_dim)

        self._load_bilstm_params(tagger)
        padder = Padding(self.wTran, self.lTran)
       
        test_dataloader = DataLoader(dataset=test_dataset,
                          batch_size=1, shuffle=False,
                          collate_fn = padder.collate_fn)

        reversed_dict = reverseDict(self.lTran.tag_dict)
        reversed_dict.append('UNKNOWN')
        with torch.no_grad():
            with open(self.test_o_file, 'w') as wf:
                for sample in test_dataloader:
                    batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                    batch_tag_score = tagger.forward(batch_data_list,
                                                     batch_len_list, padded_sublens)
                    for i, sample_tag_list in enumerate(batch_tag_score):
                        predicted_tags = tagger.getLabel(sample_tag_list)
                        for j in range(batch_len_list[i]):
                            t = predicted_tags[j]
                            w = reversed_dict[t]
                            wf.write(str(w) + "\n")
                        wf.write("\n")

        #test_dataset.toIndexes(wT = self.wTran, lT = self.lTran)
        #self.runOnDev(tagger, padder)

    def train(self):
        print("Loading data")
        train_dataset = As3Dataset(self.train_file)
        print("Done loading data")

        self.wTran = WTranslator(train_dataset.word_set, train_dataset.prefix_set,
                                  train_dataset.suffix_set, self.flavor)
        self.lTran = TagTranslator(train_dataset.tag_set)

        train_dataset.toIndexes(wT = self.wTran, lT = self.lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=self.wTran, tagset_size = self.lTran.getLengths()['tag'] + 1,
                c_embedding_dim = self.c_embedding_dim)

        #if (sys.argv[1] == 'load') or (sys.argv[1] == 'loadsave'):
        #    tagger.load_state_dict(torch.load('bilstm_params.pt'))

        loss_function = nn.CrossEntropyLoss() #ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adam(tagger.parameters(), lr=0.01)

        padder = Padding(self.wTran, self.lTran)

        train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size, shuffle=True,
                          collate_fn = padder.collate_fn)
        print("Starting training")
        print("data length = " + str(len(train_dataset)))
       
        if self.run_dev:
            self.runOnDev(tagger, padder) 
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

                batch_tag_score = tagger.forward(batch_data_list, batch_len_list, padded_sublens)
              
                flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                #calc accuracy
                c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                correct_cntr += c 
                total_cntr += t

                loss = loss_function(flatten_tag, flatten_label)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()
            print("epoch: " + str(epoch) + " " + str(loss_acc))
            print("accuracy " + str(correct_cntr/total_cntr))
            if self.run_dev:
                self.runOnDev(tagger, padder) 
        
        if self.save_to_file:
            torch._save_model_params(tagger, self.wTran, self.lTran)
        #if (sys.argv[1] == 'save') or (sys.argv[1] == 'loadsave'):
            #self._save_model_params(tagger, self.wTran, self.lTran)
            #torch.save(tagger.state_dict(), 'bilstm_params.pt')
       
if __name__ == "__main__": 
    flavor = sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]
    tagging_type = sys.argv[4]
    
    run = Run({ 'FLAVOR': flavor, 
                'EMBEDDING_DIM' : 50, 
                'RNN_H_DIM' : 50, 
                'EPOCHS' : 5, 
                'BATCH_SIZE' : 100, 
                'CHAR_EMBEDDING_DIM': 30, 
                'TRAIN_FILE': train_file,
                'DEV_FILE' : None, #dev_file,
                'TAGGING_TYPE' : tagging_type,
                'TEST_FILE': None, #test_file,
                'TEST_O_FILE': None, #test_o_file,
                'MODEL_FILE': model_file,
                'SAVE_TO_FILE': True, 
                'RUN_DEV' : False})

    run.train()

    '''
    folder = sys.argv[3]
    train_file = folder + "train" #sys.argv[3]
    dev_file = folder + "dev" #sys.argv[4]
    test_file = folder + "dev"
    test_o_file = folder + "run_results"
    tagging_type = str(sys.argv[4])
    run_test = sys.argv[5]
    run = Run({ 'FLAVOR': int(flavor), 
                'EMBEDDING_DIM' : 50, 
                'RNN_H_DIM' : 50, 
                'EPOCHS' : 1, 
                'BATCH_SIZE' : 100, 
                'CHAR_EMBEDDING_DIM': 30, 
                'TRAIN_FILE': train_file,
                'DEV_FILE' : dev_file,
                'TAGGING_TYPE' : tagging_type,
                'TEST_FILE': test_file,
                'TEST_O_FILE': test_o_file})
    if run_test == 'Y':
        run.test()
    else:
        run.train()
    '''
