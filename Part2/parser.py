import spacy
from spacy.symbols import *
import sys
import statistics

def main():
    model = Model()
    nlp = spacy.load("en_core_web_sm")
    PARAM_NUM = 3
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        return
    with open(sys.argv[1], 'r') as corpus:
        for count, line in enumerate(corpus):
            if count % 100 == 0 and count > 1:
                print("Parsed about " + str(count) + " summaries")
                break #################################################################################
            doc = nlp(line.split('\t')[1].split('\n')[0])
            model.parse(doc)

    print("Parsing done, now printing model as a text file")

    model.printToFile(sys.argv[2])

    print("Complete")

class Help:
    def increDict(dic, key):
        if key in dic:
            dic[key] += 1
        else:
            dic[key] = 1


class Model:
    def __init__(self):
        self.states = dict()
        self.stats = dict()
        self.wordLengthsOfSen = []
        self.senLengthsOfDoc = []

    def printToFile(self, outFile):
        # WORD LENGTH FOR SENS ########################################
        mean = statistics.mean(self.senLengthsOfDoc)
        median = statistics.median(self.senLengthsOfDoc)
        stdev = statistics.stdev(self.senLengthsOfDoc)
        self.addStat("DOC_SEN_LEN_MEAN", mean)
        self.addStat("DOC_SEN_LEN_MEDIAN", median)
        self.addStat("DOC_SEN_LEN_STDEV", stdev)
        # SENS LENGTH FOR DOCS ########################################
        mean = statistics.mean(self.wordLengthsOfSen)
        median = statistics.median(self.wordLengthsOfSen)
        stdev = statistics.stdev(self.wordLengthsOfSen)
        self.addStat("SEN_WORD_LEN_MEAN", mean)
        self.addStat("SEN_WORD_LEN_MEDIAN", median)
        self.addStat("SEN_WORD_LEN_STDEV", stdev)
        ###############################################################
        with open(outFile, 'w') as out:
            out.write("@STATS\n")
            for stat in sorted(self.stats.keys()):
                out.write("\t" + stat + "\t" + str(self.stats[stat]) + "\n")
            for s in self.states:
                self.states[s].writeToFile(out)

    def addStat(self, name, value):
        self.stats[name] = value

    def parse(self, doc):
        sents = list(doc.sents)
        self.senLengthsOfDoc.append(len(sents))
        for sent in doc.sents:
            if len(sent) <= 2:
                continue
            self.wordLengthsOfSen.append(len(sent))
            self.parseRec(sent.root)

    def parseRec(self, node):
        left = ""
        right = ""
        ancestor = "<root>"
        ancestors = list(node.ancestors)
        if len(ancestors) > 0:
            ancestor = ancestors[0].tag_
        pair = (ancestor, node.tag_)
        if not(pair in self.states):
            self.states[pair] = State(pair)
        self.states[pair].addWord(node)
        
        for child in node.lefts:
            left = left + "|" + child.tag_
            self.parseRec(child)
        if len(left) > 0:
            left = left[1:]

        for child in node.rights:
            right = right + "|" + child.tag_
            self.parseRec(child)
        if len(right) > 0:
            right = right[1:]

        self.states[pair].addProd((left, right))
        
class State:
    def __init__(self, pair):
        self.pair = pair
        self.wordcount = dict()
        self.nextcount = dict()
        self.size = 0
        self.productions = dict()
        self.prodsize = 0

    def addWord(self, token):
        self.size += 1
        word = token.text
        if word in self.wordcount:
            self.wordcount[word] += 1
        else:
            self.wordcount[word] = 1

    def addProd(self, prod):
        self.prodsize += 1
        if prod in self.productions:
            self.productions[prod] += 1
        else:
            self.productions[prod] = 1

    def writeToFile(self, ofile):
        ofile.write("@STATE\t" + str(self.pair) + "\n")
        ofile.write("\t@SIZE\t" + str(self.size) + "\n")
        ofile.write("\t@EMITS\n")
        for word in sorted(self.wordcount.keys()):
            ofile.write("\t\t" + word + "\t" + str(self.wordcount[word]) + "\n")
        ofile.write("\t@PRODUCTIONS\n")
        for prod in sorted(self.productions.keys()):
            ofile.write("\t\t" + str(prod) + "\t" + str(self.productions[prod]) + "\n")
        
    
if __name__ == '__main__':
    main()
