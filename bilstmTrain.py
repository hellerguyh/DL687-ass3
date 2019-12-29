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


class As2Dataset(Dataset):
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
                dataset.append((sample_w, sample_t))
                sample_w = []
                sample_t = []
            else:
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

    def translate(self, word_list):
        if self.flavor == 1:
            return [[self._dictHandleExp(self.wdict, word)] for word in word_list]
        if self.flavor == 2:
            return [[[self._dictHandleExp(self.cdict, l) for l in word]] for word in word_list]
        if self.flavor == 3:
            return [[self._dictHandleExp(self.wdict, word),self._dictHandleExp(self.pre_dict, word[:3]), 
                self._dictHandleExp(self.suf_dict, word[-3:])] for word in word_list]
        if self.flavor == 4:
            return [[self._dictHandleExp(self.wdict, word), [self._dictHandleExp(self.cdict, l) for l in word]] for word in word_list]

class TagTranslator(object):
    def __init__(self, tagset):
        self.tag_dict = list2dict(tagset)
    def translate(self, tag_list):
        return [self.tag_dict[tag] for tag in tag_list]


class Run(object):
    def __init__(self, params):
        self.flavor = params['FLAVOR']

    def train(self):
        train_dataset = As2Dataset('train_short')

        wTran = Sample2EmbedIndex(train_dataset.word_set, train_dataset.prefix_set,
                                  train_dataset.suffix_set, self.flavor)
        lTran = TagTranslator(train_dataset.tag_set)

        train_dataset.setWordTranslator(wTran)
        train_dataset.setLabelTranslator(lTran)

        for sample in train_dataset:
            pass

run = Run({'FLAVOR':3})
run.train()
'''


            DEBUG_PRINT("building space-widened content (content with edges)")
            space_widened_content = [("\sb", "SP"), ("\sb", "SP")]

            for line in content:
                if line == "":
                    space_widened_content.append(("\se", "SP"))
                    space_widened_content.append(("\se", "SP"))
                    space_widened_content.append(("\sb", "SP"))
                    space_widened_content.append(("\sb", "SP"))
                else:
                    splitted_line = line.split()
                    label = None if is_test_data else splitted_line[1]
                    space_widened_content.append((self.lowerWords(splitted_line[0]), label if len(splitted_line) > 1 else ''))

            space_widened_content.append(("\sb", "SP"))
            space_widened_content.append(("\sb", "SP"))

            DEBUG_PRINT("Building N-Grams")
            self.data_set = [([space_widened_contest[i] 



            self.tagged_offset = tagged_offset
            self.ngrams_set = [([self.lowerWords(w[0]) for w in space_widened_content[i:i + NGRAM_SIZE]], space_widened_content[i + self.tagged_offset][1]) for i in
                           range(len(space_widened_content) - NGRAM_SIZE) if space_widened_content[i + self.tagged_offset][1] != 'SP']

            if include_subword_features:
              self.prefix_ngrams_set = [([self.lowerWords(w[0][:3]) for w in space_widened_content[i:i + NGRAM_SIZE]], space_widened_content[i + self.tagged_offset][1]) for i in
                          range(len(space_widened_content) - NGRAM_SIZE) if space_widened_content[i + self.tagged_offset][1] != 'SP']

              self.suffix_ngrams_set = [([self.lowerWords(w[0][-3:]) for w in space_widened_content[i:i + NGRAM_SIZE]], space_widened_content[i + self.tagged_offset][1]) for i in
                          range(len(space_widened_content) - NGRAM_SIZE) if space_widened_content[i + self.tagged_offset][1] != 'SP']


            self.length = len(self.ngrams_set)

            if include_subword_features:
              if len(self.prefix_ngrams_set) != self.length:
                print("problem - prefix dataset is not the same size as word dataset")
                sys.exit(1)

              if len(self.suffix_ngrams_set) != self.length:
                print("problem - suffix dataset is not the same size as word dataset")
                sys.exit(1)

            self.learned_vocab_lst = list(set([self.lowerWords(w[0]) for w in self.space_widened_content]))
            self.learned_vocab_lst.append('UNMAPPED_DEFAULT')

            if include_subword_features:
              self.learned_prefix_vocab_lst = list(set([self.lowerWords(w[0][:3]) for w in self.space_widened_content]))
              self.learned_prefix_vocab_lst.append('UNMAPPED_DEFAULT')
              self.learned_suffix_vocab_lst = list(set([self.lowerWords(w[0][-3:]) for w in self.space_widened_content]))
              self.learned_suffix_vocab_lst.append('UNMAPPED_DEFAULT')

            if data_vocab is None:
              self.vocab = list2dict(self.learned_vocab_lst)
            else:
              self.vocab = data_vocab

            if include_subword_features:
              if prefix_data_vocab is None:
                self.prefix_vocab = list2dict(self.learned_prefix_vocab_lst)
              else:
                self.prefix_vocab = prefix_data_vocab

              if suffix_data_vocab is None:
                self.suffix_vocab = list2dict(self.learned_suffix_vocab_lst)
              else:
                self.suffix_vocab = suffix_data_vocab

            if label_vocab is None:
              labels = [w[1] for w in space_widened_content if w[1] != 'SP']
              labels_set = list(set(labels))
              self.labels_vocab = list2dict(labels_set)
            else:
              self.labels_vocab = label_vocab

  def lowerWords(self, x):
      return x.lower() if self.lower_words else x

  def updateVocab(self, new_vocab):
    self.vocab = new_vocab

  def __getitem__(self, index):
    def _dictHandleExp(dic, val):
      try: 
        return dic[val]
      except KeyError:
        self.exp_cntr += 1
        self.not_found.append(val)
        return dic['UNMAPPED_DEFAULT']
    if self.include_subword_features:
      data = np.array([_dictHandleExp(self.vocab,self.ngrams_set[index][0][i]) for i in range(NGRAM_SIZE)])
      data_pre = np.array([_dictHandleExp(self.prefix_vocab,self.prefix_ngrams_set[index][0][i]) for i in range(NGRAM_SIZE)])
      data_suf = np.array([_dictHandleExp(self.suffix_vocab,self.suffix_ngrams_set[index][0][i]) for i in range(NGRAM_SIZE)])
      label = self.labels_vocab[self.ngrams_set[index][1]]
      return np.array([data, data_pre, data_suf]) , label
    else:
      data = np.array([_dictHandleExp(self.vocab,self.ngrams_set[index][0][i]) for i in range(NGRAM_SIZE)])
      label = self.labels_vocab[self.ngrams_set[index][1]]
      return data, label
    
  def __len__(self):
    return self.length

  def getVocab(self):
    return self.vocab

  def getLabelsVocab(self):
    return self.labels_vocab

  def getPrefixVocab(self):
    if self.include_subword_features:
      return self.prefix_vocab
    else:
      return []

  def getSuffixVocab(self):
    if self.include_subword_features:
      return self.suffix_vocab
    else:
      return []



'''
