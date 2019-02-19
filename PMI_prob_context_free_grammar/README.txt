CS 447, Fall 2018
Homework 3

This directory contains 9 files:

  (1)  README.txt:             This file
  (2)  hw3.pdf:                PDF handout with the description of the assignment

Part 1: Pointwise Mutual Information (4 points)
  Code:
  (3)  hw3_pmi.py:             Stub Python module for calculating PMI of movies.txt corpus
  (4)  pmi_sanity_check.py:    Test script for Part 1 (provided to validate your code)

  Data:
  (5)  movies.txt:             Movie review corpus for PMI calculation

Part 2: Parsing with Probabilistic Context Free Grammars (6 points)
  Code:
  (6)  hw3_pcfg.py:            Python module for parsing using CKY (your solution)
  (7)  hw3_pcfg_test.py:       Python module for evaluating your hw3_pcfg.py solution (provided code)

  Data:
  (8)  toygrammar.cfg:         The grammar being used for the parsing in hw3_pcfg.py (provided data)
  (9)  pcfg_test_gold.txt:     The set of gold parses that hw3_pcfg_test.py uses to evaluate your solution (provided data)


***********************************************************************************************
*  Please note that you are not allowed to share the data with people outside of this class.  *
*  Furthermore, you must delete the three data files at the end of this semester.             *
***********************************************************************************************

For PMI part:
Used heap data structure with heapq library as mentioned, use nlargest sorted by PMI value for the maximum k pairs. The runtime for k (cutoff frequency) = 2 or 5 is longer than others...

For PCFG part:
Not sure what the Cell class or the Chart class is for, didn't use it; looks like it can be used in filling in the table of probabilistic CKY, I just don't know how or why we need it. 
