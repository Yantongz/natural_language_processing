##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
import numpy as np

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]


# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        print("Your task is to implement an evaluation program for POS tagging")
        # read the two files in with a function from hw2_hmm
        self.gold_file = self.readLabeledData(goldFile)
        self.test_file = self.readLabeledData(testFile)
	
        self.tot_tok = 0

        # unique tags in files
        self.all_unique_tags = []
        for sent in self.gold_file:
            self.tot_tok += len(sent)
            for tok in sent:
                if tok.tag not in self.all_unique_tags:
                    self.all_unique_tags.append(tok.tag)

        # comfusion matrix: actual tags(gold) X predictted tags(test)
        self.cnf_matrix = np.zeros((len(self.all_unique_tags), len(self.all_unique_tags)), dtype = int)    # count each tag's appearance
        for i in range(len(self.gold_file)):
            for j in range(len(self.gold_file[i])):
                irow = self.all_unique_tags.index(self.gold_file[i][j].tag)    # find positions of tags
                icol = self.all_unique_tags.index(self.test_file[i][j].tag)
                self.cnf_matrix[irow, icol] += 1

    # from hw2_hmm
    @staticmethod
    def readLabeledData(inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r")  # open the input file in read-only mode
            sens = []
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence)  # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit()  # exit the script
    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        print("Return the percentage of correctly-labeled tokens")

        acc_cnt = np.trace(self.cnf_matrix)
        acc = acc_cnt/self.tot_tok
        return acc

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        print("Return the percentage of sentences where every word is correctly labeled")

        acc_cnt = np.ones(len(self.gold_file))    # number of sentence
        for i in range(len(self.gold_file)):
            for j in range(len(self.gold_file[i])):
                if self.gold_file[i][j].tag!= self.test_file[i][j].tag:
                    acc_cnt[i] = 0

        acc = (np.sum(acc_cnt)*1.0)/len(self.gold_file)
        return acc

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        print("Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)")

        f = open(outFile, "w")
        f.write('    '.join(self.all_unique_tags) + '\n') # column names
        for i in range(len(self.cnf_matrix)):
            line = self.cnf_matrix[i].astype(str)
            f.write(self.all_unique_tags[i] + '    ' + '    '.join(line) + '\n')
        f.close()

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        print("Return the tagger's precision when predicting tag t_i")

        ith = self.all_unique_tags.index(tagTi)    # find position
        prec = (self.cnf_matrix[ith, ith]*1.0)/np.sum(self.cnf_matrix[:,ith])
        return prec

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        print("Return the tagger's recall for correctly predicting gold tag t_j")

        jth = self.all_unique_tags.index(tagTj)
        rec = (self.cnf_matrix[jth, jth]*1.0)/np.sum(self.cnf_matrix[jth])
        return rec


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and out.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("conf_matrix.txt")
