import os
import hw3_pcfg


def readFile(inputFile):
    if os.path.isfile(inputFile):
        file = open(inputFile, "r")
        lines = []
        for line in file:
            lines.append(line.strip())
        return lines


def readViterbiScores(inputFile):
    if os.path.isfile(inputFile):
        file = open(inputFile, "r")
        scores = []
        for line in file:
            scores.append(float(line.strip()))
        return scores


if __name__ == "__main__":
    pcfg = hw3_pcfg.PCFG('toygrammar.pcfg')

    # parse the sentences, trees, scores, and parses from the
    # gold file
    lineList = readFile('pcfg_test_gold.txt')
    sentences = list()
    goldTrees = list()
    scores = list()
    parses = list()
    for line in lineList:
        lineArr = line.split('|')
        scores.append(round(float(lineArr[0].strip()), 5))
        parses.append(int(lineArr[1].strip()))
        sentences.append(lineArr[2].strip())
        goldTrees.append(lineArr[3].strip())

    l = len(sentences)
    parsedAll = True
    matchedGold = True
    matchedScore = True
    matchedParses = True
    for i in range(l):
        vitTree = pcfg.CKY(sentences[i].split())
        if vitTree is None:
            parsedAll = False
            print("PARSE FAILURE: ".ljust(25) + sentences[i])
        else:
            vitTreePrint = vitTree.toString()
            vitTreeScore = round(vitTree.prob, 5)
            vitTreeParses = vitTree.numParses

            matchedGold = True
            matchedScore = True
            matchedParses = True

            if vitTreePrint != goldTrees[i]:
                matchedGold = False
            if vitTreeScore != scores[i]:
                matchedScore = False
            if vitTreeParses != parses[i]:
                matchedParses = False

            if not matchedGold and matchedScore:
                print("WARNING - wrong viterbi parse but correct score:".ljust(25) + sentences[i])
                print("  Correct parse: " + goldTrees[i])
                print("     Your parse: " + vitTreePrint)
            else:
                if not matchedGold or not matchedScore or not matchedParses:
                    print("ERROR: " + sentences[i])

                if not matchedGold:
                    print("  Correct parse: " + goldTrees[i])
                    print("     Your parse: " + vitTreePrint)
                if not matchedScore:
                    print(" Correct score: " + str(scores[i]))
                    print("     Your score: " + str(vitTreeScore))
                if not matchedParses:
                    print("  Correct parses: " + str(parses[i]))
                    print("     Your parses: " + str(vitTreeParses))

            if matchedGold and matchedScore and matchedParses:
                print("Nice job, CKY produces correct viterbi parses and score!")
