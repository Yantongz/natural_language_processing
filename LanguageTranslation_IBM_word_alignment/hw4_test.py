from hw4_translate_ec import IBMModel1

gold_alignments = []
with open('./gold_alignments.txt','r') as gold_f:
    gold_f = gold_f.readlines()
    N = int(len(gold_f)/4)
    for i in range(N):
        align = [int(x) for x in gold_f[i*4+2].split()]
        gold_alignments.append(align)

model = IBMModel1('./eng-spa_small.txt')

print(model.getTranslationLengthProbability(14,12))
model.trainUsingEM(20)

correct = 0
for i in range(len(model.fCorpus)):
    fSen = model.fCorpus[i]
    tSen = model.tCorpus[i]
    alignment = model.align(fSen, tSen)
    gold = gold_alignments[i]
    if alignment == gold:
        correct += 1
print ('Your accuracy is %.2f' % (correct/N))
