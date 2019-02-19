##
## Part 1:
## Use pointwise mutual information to compare words in the movie corpora
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict

import math # for log2
import heapq

# ----------------------------------------
#  Data input 
# ----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r")  # open the input file in read-only mode
        i = 0  # this is just a counter to keep track of the sentence numbers
        corpus = []  # this will become a list of sentences
        print("Reading file", f, "...")
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            corpus.append(sentence) # append this list as an element to the list of sentences
            # if i % 1000 == 0:
            #    sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
        return corpus
    else:
        print("Error: corpus file", f, "does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit()  # exit the script

# --------------------------------------------------------------
# PMI data structure
# --------------------------------------------------------------
class PMI:
    # Given a corpus of sentences, store observations so that PMI can be calculated efficiently
    def __init__(self, corpus):
        #print("\nYour task is to add the data structures and implement the methods necessary to efficiently get the pairwise PMI of words from a corpus")

        # f(w), f(c) and f(w,c)
        fw = defaultdict(bool)
        fc = defaultdict(bool)
        fwc = defaultdict(lambda: defaultdict(int))

        # total tokens contained in corpus
        total_n = 0.0   # "N" in slides; float calculation later
        for sent_i in corpus:
            sent = sorted(list(set(sent_i)))
            total_n += len(sent)

            cnt_dict = defaultdict(bool)
            for i in range(len(sent)):
                fc[sent[i]] += len(sent)
                for j in range(i+1, len(sent)):
                    fwc[sent[i]][sent[j]] += 1
                if not cnt_dict[sent[i]]:
                    fw[sent[i]] += 1
                    cnt_dict[sent[i]] =True

        self.total_n = total_n
        self.fw = fw
        self.fc = fc
        self.fwc = fwc


    # Return the pointwise mutual information (based on sentence (co-)occurrence frequency) for w1 and w2
    def getPMI(self, w1, w2):
        #print("\nSubtask 1: calculate the PMI for a pair of words")
        
        w1, w2 = self.pair(w1,w2)   # min, max
        pr_wc = self.fwc[w1][w2]/self.total_n
        pr_w = self.fw[w1]/self.total_n
        pr_c = self.fc[w2]/self.total_n
        ratio = pr_wc/(pr_w*pr_c)

        try:
            return math.log(ratio, 2)
        except:
            return float("-inf")


    # Given a frequency cutoff k, return the list of observed words that appear in at least k sentences
    def getVocabulary(self, k):
        #print("\nSubtask 2: return the list of words where a word is in the list iff it occurs in at least k sentences")

        corpus_vocab = []
        for wrd in self.fw:
            if self.fw[wrd]>k:
                corpus_vocab.append(wrd)
        return corpus_vocab


    # Given a list of words and a number N, return a list of N pairs of words that have the highest PMI
    # (without repeated pairs, and without duplicate pairs (wi, wj) and (wj, wi)).
    # Each entry in the list should be a triple (pmiValue, w1, w2), where pmiValue is the
    # PMI of the pair of words (w1, w2)
    def getPairsWithMaximumPMI(self, words, N):
        #print("\nSubtask 3: given a list of words and a number N, find N pairs of words with the greatest PMI")

        pmi_list = []
        sorted_wrds = sorted(words)
        for i, wrd_i in enumerate(sorted_wrds):
            wrd_i = sorted_wrds[i]
            
            for j, wrd_j in enumerate(sorted_wrds[i+1:]):
                wrd_j = sorted_wrds[j+i+1]
                heapq.heappush(pmi_list, (self.getPMI(wrd_i, wrd_j), wrd_i, wrd_j))
                if len(pmi_list)>N:
                    heapq.heappop(pmi_list)

        max_pmi = heapq.nlargest(N,pmi_list, key=itemgetter(0))  # sorted by pmi value
        return max_pmi


    #-------------------------------------------
    # Provided PMI methods
    #-------------------------------------------
    # Writes the first numPairs entries in the list of wordPairs to a file, along with each pair's PMI
    def writePairsToFile(self, numPairs, wordPairs, filename): 
        f=open(filename, 'w+')
        count = 0
        for (pmiValue, wi, wj) in wordPairs:
            if count > numPairs:
                break
            count += 1
            print("%f %s %s" % (pmiValue, wi, wj), end = "\n", file=f)


    # Helper method: given two words w1 and w2, returns the pair of words in sorted order
    # That is: pair(w1, w2) == pair(w2, w1)
    def pair(self, w1, w2):
        return (min(w1, w2), max(w1, w2))

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    corpus = readFileToCorpus('movies.txt')
    pmi = PMI(corpus)
    lv_pmi = pmi.getPMI("luke", "vader")
    print("  PMI of \"luke\" and \"vader\": ", lv_pmi)
    numPairs = 100
    k = 2
    # for k in 2, 5, 10, 50, 100, 200:
    commonWords = pmi.getVocabulary(k)    # words must appear in least k sentences
    wordPairsWithGreatestPMI = pmi.getPairsWithMaximumPMI(commonWords, numPairs)
    pmi.writePairsToFile(numPairs, wordPairsWithGreatestPMI, "pairs_minFreq="+str(k)+".txt")
