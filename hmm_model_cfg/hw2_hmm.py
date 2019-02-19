########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log

import numpy as np
# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###

        # initial dist
        self.pi = defaultdict(float)

        # tags counts
        self.tot_tags = 0
        self.tag_cnt = defaultdict(float)
        
        # all words counts before unkown handling
        self.orig_words = defaultdict(int)

        # words counts after unkown handling
        self.new_words = defaultdict(float)

        # transition and emmission probability
        self.trans_prob = defaultdict(float)
        self.emit_prob = defaultdict(float)


    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        print("Your first task is to train a bigram HMM tagger from an input file of POS-tagged text")

        # count words before and after UNK handling
        for sent in data:
            for tok in sent:
                self.orig_words[tok.word] += 1

        for sent2 in data:
            for tok2 in sent2:
                if self.orig_words[tok2.word] < self.minFreq:
                    tok2.word=UNK

        # counts
        for sent3 in data:
            # initialize
            self.pi[sent3[0].tag] += 1.0   # float
            self.new_words[sent3[0].word] += 1.0
            self.tag_cnt[sent3[0].tag] += 1.0
            self.emit_prob[(sent3[0].word, sent3[0].tag)] += 1.0
            for i in range(1, len(sent3)):
                self.new_words[sent3[i].word] += 1.0
                self.tag_cnt[sent3[i].tag] += 1.0
                self.trans_prob[(sent3[i-1].tag, sent3[i].tag)] +=1.0
                self.emit_prob[(sent3[i].word, sent3[i].tag)] +=1.0
            
        # total tags
        self.tot_tags = len(self.tag_cnt)
        # initialize
        for ti in self.pi.keys():
            self.pi[ti] /= len(data)    # normalize

        # cal prob

        # prevent your tagger from assigning zero probability to a test sentence, use add-one smoothing on the transition dist
        # to ensure that a sequence with non-zero probability exists.
        for ti in self.tag_cnt.keys():
            for tj in self.tag_cnt.keys():
                self.trans_prob[(ti,tj)] = (1.0 + self.trans_prob[(ti,tj)])/(self.tag_cnt[ti]+self.tot_tags)

        for (wi, ti) in self.emit_prob.keys():
            self.emit_prob[(wi, ti)] /= self.tag_cnt[ti]


    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString)
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        # returns the list of Viterbi POS tags (strings)
        #return ["NULL"]*len(words) # this returns a dummy list of "NULL", equal in length to words

        # handle UNK
        cl_words = list(words)
        for i in range(len(cl_words)):
            if cl_words[i] not in self.new_words:
                cl_words[i] = UNK

        # all the tags
        all_tags = list(self.tag_cnt.keys())

        # create the trellis, num of tags X num of cl_words, use log prob
        tre = np.zeros((len(all_tags), len(cl_words)))

        # start from the first col
        for i in range(len(all_tags)):
            if self.pi[all_tags[i]]==0.0 or self.emit_prob[(cl_words[0], all_tags[i])]==0.0:
                tre[i][0]=-float("inf")
            else:
                tre[i][0]=log(self.pi[all_tags[i]]*self.emit_prob[(cl_words[0],all_tags[i])])

        # matrix to store the indices for the best tags
        bk_ptr = -np.ones((len(all_tags), len(cl_words)))

        # enumerate the rest of the cols
        for i in range(1, len(cl_words)):

            for j in range(len(all_tags)):
                if self.emit_prob[(cl_words[i], all_tags[j])] == 0.0:
                    tre[j][i] = -float("inf")
                else:
                    this_col = np.zeros(len(all_tags))

                    for k in range(len(all_tags)):   # enumerate all possible tags

                        this_col[k] = tre[k][i-1] + log(self.trans_prob[(all_tags[k],all_tags[j])])

                    prev_best = np.argmax(this_col)
                    
                    tre[j][i] = this_col[prev_best] + log(self.emit_prob[(cl_words[i],all_tags[j])])

                    bk_ptr[j][i] = prev_best


        # start from last column
        largest_idx = np.argmax(tre[:, -1])
        bk_ptr_tags = []
        bk_ptr_tags.append(all_tags[int(largest_idx)])

        # keep going back to prev cols
        for i in range(len(cl_words)-1, 0, -1):

            largest_idx = bk_ptr[int(largest_idx), i]

            bk_ptr_tags.append(all_tags[int(largest_idx)])

        bk_ptr_tags.reverse()

        return bk_ptr_tags
# end viterbi


if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
