from collections import defaultdict

# Constant for NULL word at position zero in target sentence
NULL = "NULL"

# Your task is to finish implementing IBM Model 1 in this class
class IBMModel1:

    def __init__(self, trainingCorpusFile):
        # Initialize data structures for storing training data
        self.fCorpus = []                   # fCorpus is a list of foreign (e.g. Spanish) sentences

        self.tCorpus = []                   # tCorpus is a list of target (e.g. English) sentences

        self.trans = {}                     # trans[e_i][f_j] is initialized with a count of how often target word e_i and foreign word f_j appeared together.
        # Initialize any additional data structures here (e.g. for probability model)
        
        # Model params: 
        # q(m|n)
        self.length_prob = defaultdict(lambda: defaultdict(float))
        # the prob that English word e_x generates non-English word f_y
        self.trans_prob = defaultdict(lambda: defaultdict(float))
        # EM algo
        self.counts = defaultdict(lambda: defaultdict( ))
        
        self.initialize(trainingCorpusFile);


    # Reads a corpus of parallel sentences from a text file (you shouldn't need to modify this method)
    def initialize(self, fileName):
        f = open(fileName)
        i = 0
        j = 0;
        tTokenized = ();
        fTokenized = ();
        for s in f:
            if i == 0:
                tTokenized = s.split()
                # Add null word in position zero
                tTokenized.insert(0, NULL)
                self.tCorpus.append(tTokenized)
            elif i == 1:
                fTokenized = s.split()
                self.fCorpus.append(fTokenized)
                for tw in tTokenized:
                    if tw not in self.trans:
                        self.trans[tw] = {};
                    for fw in fTokenized:
                        if fw not in self.trans[tw]:
                             self.trans[tw][fw] = 1
                        else:
                            self.trans[tw][fw] =  self.trans[tw][fw] +1
            else:
                i = -1
                j += 1
            i +=1
        f.close()
        return

    # Uses the EM algorithm to learn the model's parameters
    def trainUsingEM(self, numIterations=10, writeModel=False, convergenceEpsilon=0.01):
        ###
        # Part 1: Train the model using the EM algorithm
        #
        # <you need to finish implementing this method's sub-methods>
        #
        ###

        # Compute translation length probabilities q(m|n)
        self.computeTranslationLengthProbabilities()         

        # Set initial values for the translation probabilities p(f|e)
        self.initializeWordTranslationProbabilities()        

        # Write the initial distributions to file
        if writeModel:
            self.printModel('initial_model.txt')                 

        for i in range(numIterations):
            print ("Starting training iteration "+str(i))
            # Run E-step: calculate expected counts using current set of parameters
            self.computecounts()                     
            # Run M-step: use the expected counts to re-estimate the parameters
            self.updateTranslationProbabilities()            

            # Write model distributions after iteration i to file
            if writeModel:
                self.printModel('model_iter='+str(i)+'.txt')     


    # Compute translation length probabilities q(m|n)
    def computeTranslationLengthProbabilities(self):

        # pairwise lengths count dict
        length_cnts = defaultdict(lambda: defaultdict(float))

        for i in range(len(self.fCorpus)):
            n = len(self.tCorpus[i])
            m = len(self.fCorpus[i])

            length_cnts[n][m] += 1.0

        # prob
        self.length_prob = defaultdict(lambda: defaultdict(float))
        for e, e_dict in length_cnts.items():
            for f, cnt in e_dict.items():
                self.length_prob[e][f] = cnt/sum(e_dict.values())


    # Set initial values for the translation probabilities p(f|e)
    def initializeWordTranslationProbabilities(self):
        
        self.trans_prob = defaultdict(lambda: defaultdict(float))

        for ei, ei_dict in self.trans.items():
            for fj in ei_dict.keys():
                self.trans_prob[ei][fj] = float(1 / float(len(ei_dict.keys())))


    # Run E-step: calculate expected counts using current set of parameters
    def computecounts(self):
        print('E step.')
        
        self.counts = defaultdict(lambda: defaultdict())

        for sent in range(len(self.fCorpus)):
            for fj in self.fCorpus[sent]:
                total = 0
                for ei in self.tCorpus[sent]:
                    total += self.trans_prob[ei][fj]
                for ei in self.tCorpus[sent]:
                    if fj not in self.counts[ei]:
                        self.counts[ei][fj] = []
                    self.counts[ei][fj].append({'sent_id': sent, 'cnt' : self.trans_prob[ei][fj]/total})


    # Run M-step: use the expected counts to re-estimate the parameters
    def updateTranslationProbabilities(self):
        print('M step.')
        
        for ex in self.trans.keys():
            tot_cnt_f = {}
            for fy in self.trans[ex].keys():
                tot_cnt_f[fy] = 0
                for sent_idx in self.counts[ex][fy]:
                    tot_cnt_f[fy] += sent_idx['cnt']

            z = sum(tot_cnt_f.values())

            for fy in self.trans[ex].keys():
                self.trans_prob[ex][fy] = tot_cnt_f[fy]/z


    # Returns the best alignment between fSen and tSen, according to your model
    def align(self, fSen, tSen):
        ###
        # Part 2: Find and return the best alignment
        # <you need to finish implementing this method>
        # Remove the following code (a placeholder return that aligns each foreign word with the null word in position zero of the target sentence)
        ###

        # check null word in pos 0
        if tSen[0]!=NULL:
            tSen.insert(0,NULL)

        dummyAlignment = []

        for j_idx, fj in enumerate(fSen):   # go through all f words
            max_prob = -1       # best prob
            max_idx  = 0        # pos
            for idx, ei in enumerate(tSen):
                if self.trans_prob[ei][fj] > max_prob:
                    max_prob = self.trans_prob[ei][fj]
                    max_idx = idx
            dummyAlignment.append(max_idx)

        return dummyAlignment

    # CORRECTION : Return q(fLength | tLength) --------------------------------
    # (Can either return log probability or regular probability)
    def getTranslationLengthProbability(self, fLength, tLength):

        try:
            return self.length_prob[tLength][fLength]
        except KeyError:
            return 0.0

    # Return p(f_j | e_i), the probability that English word e_i generates non-English word f_j
    # (Can either return log probability or regular probability)
    def getWordTranslationProbability(self, f_j, e_i):
        return self.trans_prob[e_i][f_j]


    # Write this model's probability distributions to file
    def printModel(self, filename):

        lengthFile = open(filename+'_lengthprobs.txt', 'w')         # Write q(m|n) for all m,n to this file
        
        for e_n, e_dict in self.length_prob.items():
            for f_m, f_prob in e_dict.items():
                lengthFile.write('q({}|{}) = {}\n'.format(e_n, f_m, f_prob))

        translateProbFile = open(filename+'_translationprobs.txt', 'w') # Write p(f_j | e_i) for all f_j, e_i to this file
        for ei, e_dict in self.trans_prob.items():
            for fj, f_prob in e_dict.items():
                translateProbFile.write('p({}|{}) = {}\n'.format(fj, ei, f_prob))

        lengthFile.close();
        translateProbFile.close()


# utility method to pretty-print an alignment
# You don't have to modify this function unless you don't think it's that pretty...
def prettyAlignment(fSen, tSen, alignment):
    pretty = ''
    for j in range(len(fSen)):
        pretty += str(j)+'  '+fSen[j].ljust(20)+'==>    '+tSen[alignment[j]]+'\n';
    return pretty

if __name__ == "__main__":
    # Initialize model
    model = IBMModel1('eng-spa.txt')
    # Train model
    model.trainUsingEM(20);
    model.printModel('after_training')
    # Use model to get an alignment
    fSen = 'No pierdas el tiempo por el camino .'.split()
    tSen = 'Don\' t dawdle on the way'.split()
    alignment = model.align(fSen, tSen);
    print (prettyAlignment(fSen, tSen, alignment))
