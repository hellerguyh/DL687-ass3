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
    def __init__(self, pos_file_path, neg_file_path, is_test_data=False):    
        with open(pos_file_path, "r") as df:
            content_pos = df.read().split('\n')
        if not is_test_data:
            with open(neg_file_path, "r") as df:
                content_neg = df.read().split('\n')

        dataset = []
        sample_w = []
        sample_t = []
        word_list = []
        if is_test_data == False:
            contents = [content_neg, content_pos]
        else:
            contents = [content_pos]

        for tag, content in enumerate(contents):
            for line in content:
                l = [c for c in line]
                dataset.append((l, tag, len(l)))
                for c in line:
                    word_list.append(c)
                '''if line == "":
                    if last_line_is_space == True:
                        pass
                    else:
                        dataset.append((sample_w, tag, len(sample_w)))
                        sample_w = []
                        last_line_is_space = True
                else:
                    last_line_is_space = False
                    word = line[0]
                    word_list.append(word)
                    sample_w.append(word)'''
        
        self.word_set = set(word_list)
        self.dataset = dataset
        self.is_test_data = is_test_data

    def __len__(self):
        return len(self.dataset)

    def toIndexes(self, wT, lT):
        self.dataset = [(wT.translate(data[0]), data[1], data[2]) for data in self.dataset]

    def __getitem__(self, index):
        return self.dataset[index]

class WTranslator(object):
    def __init__(self, wordset, init=True):
        if init:
            wordset.update(["UNKNOWN"])
            self.wdict = list2dict(list(wordset))

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

    def translate(self, word_list):
        return [np.array(self._translate1(word_list))] 

    def getLengths(self):
        return {'word' : len(self.wdict)}

class TagTranslator(object):
    def __init__(self, tagset, init=True):
        if init:
            self.tag_dict = list2dict([0,1])

    def translate(self, tag_list):
        return [self.tag_dict[tag] for tag in tag_list]

    def getLengths(self):
        return {'tag': len(self.tag_dict)}

    def saveParams(self):
        return {'tag':self.tag_dict}

    def loadParams(self, params):
        self.tag_dict = params['tag']


class MyEmbedding(nn.Module):
    def __init__(self, embedding_dim, translator):
        super(MyEmbedding, self).__init__()
        p1 = 1 
        l = translator.getLengths()['word'] 
        padding_idx = l
        self.wembeddings = nn.Embedding(num_embeddings = l + p1, embedding_dim = embedding_dim, padding_idx = l)
    
    def forward(self, data):
        return self.wembeddings(torch.tensor(data).long())

class Padding(object):
    def __init__(self, wT, lT):
        self.wT = wT
        self.lT = lT
        self.wPadIndex = self.wT.getLengths()['word']
        self.lPadIndex = 2

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
  
    def padBatch(self, data_b, tag_b, len_b):
        #padded_tag = self.padTag(tag_b, len_b, max(len_b), self.lPadIndex)
        padded_tag = tag_b
        word_data_b = [b[0] for b in data_b]
        padded_data = self.padData(word_data_b, len_b, max(len_b), self.wPadIndex)
        
        return padded_data, padded_tag, len_b 

    def collate_fn(self, data):
        data.sort(key=lambda x: x[2], reverse=True)

        data_b = [d[0] for d in data]
        tag_b = [d[1] for d in data]
        len_b = [d[2] for d in data]

        return self.padBatch(data_b, tag_b, len_b)


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_rnn_dim, hidden_mlp_dim, tagset_size, translator, dropout):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = MyEmbedding(embedding_dim, translator)
        self.dropout_0 = nn.Dropout()  
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_rnn_dim,
                            batch_first=True)
        self.linear1 = nn.Linear(hidden_rnn_dim, hidden_mlp_dim)
        self.dropout_1 = nn.Dropout() 
        self.linear2 = nn.Linear(hidden_mlp_dim, tagset_size)
        self.dropout_2 = nn.Dropout()
        self.dropout = dropout
   

    '''def runLSTMc(self, data_list, embeds_list, padded_sublens):
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
        return reshaped_last_layer'''
            

    def forward(self, data_list, len_list):
        batch_size = len(len_list)
        embeds_list = self.embeddings.forward(data_list)
        embeds_out = embeds_list
        
        if self.dropout:
            embeds_out = self.dropout_0(embeds_out)

        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds_out, len_list, batch_first=True)
        lstm_out, _ = self.lstm(packed_embeds)
        unpacked_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first = True)

        reshaped_indexes = torch.Tensor(np.array(len_list) - np.ones(len(len_list))).long()

        reshaped_indexes = reshaped_indexes.view(-1,1)
        reshaped_indexes = reshaped_indexes.repeat(1, unpacked_lstm_out.shape[2])
        reshaped_indexes = reshaped_indexes.view(reshaped_indexes.shape[0], 1, reshaped_indexes.shape[1])
        
        last_layer = torch.gather(unpacked_lstm_out, 1, reshaped_indexes)

        flatten = last_layer.reshape(-1, unpacked_lstm_out.shape[2]) #unpacked_lstm_out.reshape(-1, unpacked_lstm_out.shape[2])
        if self.dropout:
            flatten = self.dropout_1(flatten)

        o_ln1 = self.linear1(flatten)
        o_tanh = torch.tanh(o_ln1)
        o_ln2 = self.linear2(o_tanh)

        shaped = o_ln2.reshape(batch_size, o_ln2.shape[1])
        return shaped

    def getLabel(self, data):
        _, prediction_argmax = torch.max(data, 1)
        return prediction_argmax


class Run(object):
    def __init__(self, params):
        self.edim = params['EMBEDDING_DIM']
        self.rnn_h_dim = params['RNN_H_DIM']
        self.num_epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']
        self.pos_train_file = params['POS_TRAIN_FILE']
        self.neg_train_file = params['NEG_TRAIN_FILE']
        self.dev_file = params['DEV_FILE']
        self.test_file = params['TEST_FILE']
        self.test_o_file = params['TEST_O_FILE']
        self.save_to_file = params['SAVE_TO_FILE']
        self.learning_rate = params['LEARNING_RATE']
        self.dropout = params['DROPOUT']
        self.hidden_mlp_dim = params['HIDDEN_MLP_DIM']
        self.run_dev = params['RUN_DEV']
        self.acc_data_list = []

    def _save_model_params(self, tagger, wT, lT):
        try:
            params = torch.load(self.model_file)
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
       
        acc = correct_cntr/total_cntr
        self.acc_data_list.append(acc)
        print("Validation accuracy " + str(acc))
        
        tagger.train()


    def _saveAccData(self):
        try:
            acc_data = torch.load('accuracy_graphs_data')
        except FileNotFoundError:
            print("No accuracy data file found - creating new")
            acc_data = {}

        acc_data.update({self.tagging_type+str(self.flavor): self.acc_data_list})
        torch.save(acc_data, 'accuracy_graphs_data')

    def test(self):
        test_dataset = As3Dataset(file_path = self.test_file, 
                                  is_test_data = True)

        self.wTran = WTranslator(None, None, None, None, False)
        self.lTran = TagTranslator(None, False)

        self._load_translators_params(self.wTran, self.lTran)
        test_dataset.toIndexes(wT = self.wTran, lT = self.lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=self.wTran, tagset_size = self.lTran.getLengths()['tag'] + 1,
                c_embedding_dim = self.c_embedding_dim, dropout = self.dropout)

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

    def train(self):
        print("Loading data")
        train_dataset = As3Dataset(self.pos_train_file, self.neg_train_file)
        print("Done loading data")

        self.wTran = WTranslator(train_dataset.word_set)
        self.lTran = TagTranslator((0,1))

        train_dataset.toIndexes(wT = self.wTran, lT = self.lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                tagset_size = self.lTran.getLengths()['tag'] + 1,
                hidden_mlp_dim=self.hidden_mlp_dim, translator=self.wTran, 
                dropout = self.dropout)

        #if (sys.argv[1] == 'load') or (sys.argv[1] == 'loadsave'):
        #    tagger.load_state_dict(torch.load('bilstm_params.pt'))

        loss_function = nn.CrossEntropyLoss() #ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adam(tagger.parameters(), lr=self.learning_rate) #0.01)

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
            sentences_seen = 0
            for sample in train_dataloader:
                if progress1/1000 == progress2:
                    print("reached " + str(progress2*1000))
                    progress2+=1
                progress1 += self.batch_size
                sentences_seen += self.batch_size

                tagger.zero_grad()
                batch_data_list, batch_label_list, batch_len_list = sample

                batch_tag_score = tagger.forward(batch_data_list, batch_len_list)
              
                #flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)
                flatten_tag, flatten_label = batch_tag_score, torch.LongTensor(batch_label_list)

                #calc accuracy
                c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                correct_cntr += c 
                total_cntr += t

                loss = loss_function(flatten_tag, flatten_label)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()

                if sentences_seen >= 500:
                    sentences_seen = 0
                    if self.run_dev:
                        self.runOnDev(tagger, padder) 
            
                print("accuracy " + str(correct_cntr/total_cntr))

            print("epoch: " + str(epoch) + " " + str(loss_acc))
            print("accuracy " + str(correct_cntr/total_cntr))
        
        if self.save_to_file:
            self._save_model_params(tagger, self.wTran, self.lTran)

        if self.run_dev:
            self._saveAccData()
        #if (sys.argv[1] == 'save') or (sys.argv[1] == 'loadsave'):
            #self._save_model_params(tagger, self.wTran, self.lTran)
            #torch.save(tagger.state_dict(), 'bilstm_params.pt')


FAVORITE_RUN_PARAMS = { 
                'EMBEDDING_DIM' : 10, 
                'RNN_H_DIM' : 10, 
                'EPOCHS' : 20, 
                'BATCH_SIZE' : 50, 
                'CHAR_EMBEDDING_DIM': 50,#30,
                'LEARNING_RATE' : 0.01
                }

if __name__ == "__main__": 
   
    RUN_PARAMS = FAVORITE_RUN_PARAMS
    RUN_PARAMS.update({ 
                'POS_TRAIN_FILE': './pos_examples_dev' ,
                'NEG_TRAIN_FILE': './neg_examples_dev' ,
                'DEV_FILE' : None, #dev_file,
                'TEST_FILE': None, #test_file,
                'TEST_O_FILE': None, #test_o_file,
                'SAVE_TO_FILE': False, 
                'RUN_DEV' : False,
                'EPOCHS' : 30, 
                'DROPOUT' : False,
                'HIDDEN_MLP_DIM': 20})
    
    run = Run(RUN_PARAMS)

    run.train()
