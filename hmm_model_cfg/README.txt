CS 447, Fall 2018
Homework 2

This directory contains 11 files:

 (1)    README.txt:             This file
 (2)    hw2.pdf:                PDF handout with description of the assignment

Part 1: Hidden Markov Model POS Taggers (6 points)
 Code:
 (3)    hmm_sanity_check.py:    Test script for Part 1 (provided to validate your code)
 (4)    hw2_hmm.py:             Stub Python module for bigram HMM tagger
 (5)    hw2_hmm_eval.py:        Stub Python module for evaluating your tagger

 Data:
 (6)    train.txt:              Tagged POS training data
 (7)    test.text:              Raw POS test data
 (8)    gold.text:              Tagged POS test data

Part 2: Writing a Context-Free Grammar
 Code:
 (9)    hw2_nltkcfg.py:         Python module for parsing using CFG using NLTK and your grammar (provided code)

 Data:
 (10)   sentences.txt:          The list of sentences that are parsed by hw2_nltkcfg.py (provided data)
 (11)   mygrammar.cfg:          The grammar being used by the parsing in hw2_nltkcfg.py (your solution)

 Output:
 (12)   hw2_cfg_out.txt:        Successful parse trees in tabbed, bracketed format, generated from hw2_nltkcfg.py (generated file)


***********************************************************************************************
*  Please note that you are not allowed to share the data with people outside of this class.  *
*  Furthermore, you must delete the three data files at the end of this semester.             *
***********************************************************************************************

Summary:
For part 1, I used dictionares with tuples as keys to store the probabilities for transition and emission probabilities.
The confusion matrix is of the format actual tags(from gold) X predictted tags(from test).

For part 2, I defined NP and VP similar to examples in slides, add ADJ which stands for adjective phrases. I also included clauses in implementing the rules.


