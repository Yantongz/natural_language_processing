import sys
import os
import math

# The start symbol for the grammar
TOP = "TOP"

'''
A grammatical Rule has a probability and a parent category, and is
extended by UnaryRule and BinaryRule
'''


class Rule:

    def __init__(self, probability, parent):
        self.prob = probability
        self.parent = parent

    # Factory method for making unary or binary rules (returns None otherwise)
    @staticmethod
    def createRule(probability, parent, childList):
        if len(childList) == 1:
            return UnaryRule(probability, parent, childList[0])
        elif len(childList) == 2:
            return BinaryRule(probability, parent, childList[0], childList[1])
        return None

    # Returns a tuple containing the rule's children
    def children(self):
        return ()

'''
A UnaryRule has a probability, a parent category, and a child category/word
'''


class UnaryRule(Rule):

    def __init__(self, probability, parent, child):
        Rule.__init__(self, probability, parent)
        self.child = child

    # Returns a singleton (tuple) containing the rule's child
    def children(self):
        return (self.child,)  # note the comma; (self.child) is not a tuple

'''
A BinaryRule has a probability, a parent category, and two children
'''


class BinaryRule(Rule):

    def __init__(self, probability, parent, leftChild, rightChild):
        Rule.__init__(self, probability, parent)
        self.leftChild = leftChild
        self.rightChild = rightChild

    # Returns a pair (tuple) containing the rule's children
    def children(self):
        return (self.leftChild, self.rightChild)

'''
An Item stores the label and Viterbi probability for a node in a parse tree
'''


class Item:

    def __init__(self, label, prob, numParses):
        self.label = label
        self.prob = prob
        self.numParses = numParses

    # Returns the node's label
    def toString(self):
        return self.label

'''
A LeafItem is an Item that represents a leaf (word) in the parse tree (ie, it
doesn't have children, and it has a Viterbi probability of 1.0)
'''


class LeafItem(Item):

    def __init__(self, word):
        # using log probabilities, this is the default value (0.0 = log(1.0))
        Item.__init__(self, word, 0.0, 1)	# label, prob, numParses

'''
An InternalNode stores an internal node in a parse tree (ie, it also
stores pointers to the node's child[ren])
'''


class InternalItem(Item):

    def __init__(self, category, prob, children=()):
        Item.__init__(self, category, prob, 0)
        self.children = children
        # Your task is to update the number of parses for this InternalItem
        # to reflect how many possible parses are rooted at this label
        # for the string spanned by this item in a chart
        # self.numParses = -1  # dummy numParses value; this should not be -1!
        if len(self.children) > 2:
            print("Warning: adding a node with more than two children (CKY may not work correctly)")

    # For an internal node, we want to recurse through the labels of the
    # subtree rooted at this node
    def toString(self):
        ret = "( " + self.label + " "
        for child in self.children:
            ret += child.toString() + " "
        return ret + ")"

'''
A Cell stores all of the parse tree nodes that share a common span

Your task is to implement the stubs provided in this class
'''


class Cell:

    def __init__(self):
        self.items = {}

    def addItem(self, item):
        # Add an Item to this cell
        pass

    def getItem(self, label):
        # Return the cell Item with the given label
        pass

    def getItems(self):
        # Return the items in this cell
        pass

'''
A Chart stores a Cell for every possible (contiguous) span of a sentence

Your task is to implement the stubs provided in this class
'''


class Chart:

    def __init__(self, sentence):
        # Initialize the chart, given a sentence
        pass

    def getRoot(self):
        # Return the item from the top cell in the chart with
        # the label TOP
        pass

    def getCell(self, i, j):
        # Return the chart cell at position i, j
        pass

'''
A PCFG stores grammatical rules (with probabilities), and can be used to
produce a Viterbi parse for a sentence if one exists
'''


class PCFG:

    def __init__(self, grammarFile, debug=False):
        # in ckyRules, keys are the rule's RHS (the rule's children, stored in
        # a tuple), and values are the parent categories
        self.ckyRules = {}
        self.debug = debug                  # boolean flag for debugging
        # reads the probabilistic rules for this grammar
        self.readGrammar(grammarFile)
        # checks that the grammar at least matches the start symbol defined at
        # the beginning of this file (TOP)
        self.topCheck()

    '''
    Reads the rules for this grammar from an input file
    '''

    def readGrammar(self, grammarFile):
        if os.path.isfile(grammarFile):
            file = open(grammarFile, "r")
            for line in file:
                raw = line.split()
                # reminder, we're using log probabilities
                prob = math.log(float(raw[0]))
                parent = raw[1]
                children = raw[
                    3:]   # Note: here, children is a list; below, rule.children() is a tuple
                rule = Rule.createRule(prob, parent, children)
                if rule.children() not in self.ckyRules:
                    self.ckyRules[rule.children()] = set([])
                self.ckyRules[rule.children()].add(rule)

    '''
    Checks that the grammar at least matches the start symbol (TOP)
    '''

    def topCheck(self):
        for rhs in self.ckyRules:
            for rule in self.ckyRules[rhs]:
                if rule.parent == TOP:
                    return  # TOP generates at least one other symbol
        if self.debug:
            print("Warning: TOP symbol does not generate any children (grammar will always fail)")

    '''
    Your task is to implement this method according to the specification. You may define helper methods as needed.

    Input:        sentence, a list of word strings
    Returns:      The root of the Viterbi parse tree, i.e. an InternalItem with label "TOP" whose probability is the Viterbi probability.
                   By recursing on the children of this node, we should be able to get the complete Viterbi tree.
                   If no such tree exists, return None\
    '''

    def CKY(self, sentence):

        prob_cky = {}   # probabilistic cky
        rl = {} # rule
        prs = {}    # of parsing
        self.bkptr = {} # back pointer

        for j in range(1,len(sentence)+1):
            
            for i_rl in self.ckyRules[sentence[j-1],]:
                prob_cky[j-1,j,i_rl.parent] = i_rl.prob
                rl[j-1,j,i_rl.parent] = InternalItem(i_rl.parent,i_rl.prob,(LeafItem(sentence[j-1]),))
                prs[j-1,j,i_rl.parent] = 1.0
            
            for i in range(j-2,-1,-1):
                for k in range(i+1,j):
                    for rhs in self.ckyRules:
                        if (len(rhs) == 2 and (i,k,rhs[0]) in prob_cky and (k,j,rhs[1]) in prob_cky):
                            if (prob_cky[i,k,rhs[0]] >float('-inf') and prob_cky[k,j,rhs[1]] >float('-inf')):
                                for rule in self.ckyRules[rhs]:

                                    tmp = rule.prob + prob_cky[i,k,rhs[0]] +prob_cky[k,j,rhs[1]]
                                    if (i,j,rule.parent) in prob_cky:
                                        prs[i,j,rule.parent] +=  prs[i,k,rhs[0]] *prs[k,j,rhs[1]]
                                        if (prob_cky[i,j,rule.parent] < tmp):
                                            prob_cky[i,j,rule.parent] = tmp
                                            
                                    else:
                                        prob_cky[i,j,rule.parent] = tmp
                                        prs[i,j,rule.parent] = prs[i,k,rhs[0]] * prs[k,j,rhs[1]]

                                    self.bkptr[i,j,rule.parent] = [k,rhs[0],rhs[1]]
                                    rl[i,j,rule.parent] = InternalItem(rule.parent,rule.prob,(rl[i,k,rhs[0]],rl[k,j,rhs[1]]))

        # no such tree exis, a parse failure
        if (0,len(sentence),'S') not in self.bkptr:
            return None 
        else:
            # return the root of the viterbi parse tree

            # find children of 'S'
            c1 = rl[0,self.bkptr[0,len(sentence),'S'][0],self.bkptr[0,len(sentence),'S'][1]]
            c2 = rl[self.bkptr[0,len(sentence),'S'][0],len(sentence),self.bkptr[0,len(sentence),'S'][2]]
            
            # TOP -> S
            top_s = InternalItem('S', float(prob_cky[0,len(sentence),'S']),(c1,c2)) 
            
            # root of parse tree
            i_root = InternalItem('TOP', float(prob_cky[0,len(sentence),'S']),(top_s,))
            i_root.numParses = prs[0,len(sentence),'S']

            return (i_root)


if __name__ == "__main__":
    pcfg = PCFG('toygrammar.pcfg')
    sen = "the man eats the tuna with a fork and some sushi with the chopsticks".split()

    tree = pcfg.CKY(sen)

    if tree is not None:
        print(tree.toString())
        print("Probability: " + str(math.exp(tree.prob)))
        print("Num parses: " + str(tree.numParses))
    else:
        print("Parse failure!")
