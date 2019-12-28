import torch
import torch.nn as nn 
import numpy as np
import random

def printToGraph(x_axis_label, y_axis_label, x, y):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()

class RnnAcceptor(nn.Module):
    def __init__(self, embedding_dim, hidden_rnn_dim, hidden_mlp_dim, vocab_size, tagset_size,
                 dropout=False, evecs = None):
        super(RnnAcceptor, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_rnn_dim)
        self.linear1 = nn.Linear(hidden_rnn_dim, hidden_mlp_dim)
        self.linear2 = nn.Linear(hidden_mlp_dim, tagset_size)

    def forward(self, data):
        embeds = self.embeddings(data)
        lstm_out, self.hidden = self.lstm(embeds.view(len(data[0]), 1, -1))
        lstm_last = lstm_out[-1]
        o_ln1 = self.linear1(lstm_last)
        o_tng = torch.tanh(o_ln1)
        o_ln2 = self.linear2(o_tng)
        return o_ln2

    def getLabel(self, data):
        _, prediction_argmax = data[0].max(0)
        return prediction_argmax

def list2dict(lst):
    it = iter(lst)
    indexes = range(len(lst))
    res_dct = dict(zip(it, indexes))
    return res_dct

d_vocab_lst = ['1','2','3','4','5','6','7','8','9','10','a','b','c','d']
d_vocab = list2dict(d_vocab_lst)

good = ['a','b','c','d']
bad = ['a','c','b','d']

model = RnnAcceptor(14, 10, 10, len(d_vocab_lst), 2)
loss_function = nn.CrossEntropyLoss()#nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
loss_acc_lst = []
dev_acc_lst = []
import gen_examples
from gen_examples import LanguageGen

SUB_SEQ_LIM = 10

lg = LanguageGen(None,None, SUB_SEQ_LIM)
examples = lg.genExamples(250)

devlg = LanguageGen(None,None, SUB_SEQ_LIM)
devexamples = lg.genExamples(25)

BATCH_SIZE = 50

for i in range(40):
    random.shuffle(examples)
    loss_acc = 0
    batch_cntr = 0 
    model.zero_grad()
    first = True
    for sample in examples:
        batch_cntr+=1
        #if (bool(random.getrandbits(1))):
        #    sample = good
        #    label = torch.tensor([0]).long()
        #else:
        #    sample = bad
        #    label = torch.tensor([1]).long()
        data, label = sample

        t_data = torch.LongTensor([[d_vocab[s] for s in data]])
        t_label = torch.tensor([label]).long()
        tag_score = model(t_data)
        
        if first == True:
            loss = loss_function(tag_score, t_label)
        else:
            loss += loss_function(tag_score, t_label)
        first = False

        if batch_cntr%(BATCH_SIZE) == 0:
            loss.backward()
            loss_acc += loss.item()
            optimizer.step()
            batch_cntr = 0
            model.zero_grad()
            first = True
    print(str(i) + " " + str(loss_acc))
    loss_acc_lst.append(loss_acc/len(examples))
    with torch.no_grad():
        dev_cntr = 0
        for sample in devexamples:
            data, label = sample
            t_data = torch.LongTensor([[d_vocab[s] for s in data]])
            t_label = torch.tensor([label]).long()
            tag_score = model(t_data)
            prediction = model.getLabel(tag_score)
            if int(prediction) == int(label):
                dev_cntr += 1
        dev_acc = dev_cntr/len(devexamples)
        dev_acc_lst.append(dev_acc)
        print("dev acc = " + str(dev_acc))


printToGraph("time", "train loss", [i for i in range(len(loss_acc_lst))], loss_acc_lst) 
printToGraph("time", "dev acc", [i for i in range(len(dev_acc_lst))], dev_acc_lst) 

with torch.no_grad():
    for d in [good, bad]:
        sample = torch.LongTensor([[d_vocab[s] for s in d]])
        tag_score = model(sample)
        label = model.getLabel(tag_score)
        print(label)
        
