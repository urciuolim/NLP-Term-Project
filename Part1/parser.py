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
            if count % 100 == 0:
                print("Parsed about " + str(count) + " summaries")
            doc = nlp(line.split('\t')[1].split('\n')[0])
            model.parse(doc)

    print("Parsing done, now printing model as a text file")

    model.printToFile(sys.argv[2])

    print("Complete")


class Model:
    def __init__(self):
        self.states = dict()
        self.stats = dict()
        self.wordLengthsOfDoc = []

    def printToFile(self, outFile):
        self.wordLengthsOfDoc = sorted(self.wordLengthsOfDoc)
        l = int(len(self.wordLengthsOfDoc)*.1)
        # Only considered word lengths of middle 90% to try and help std dev
        # this might only be a problem when considering overall word length,
        # not with sentence length / overall num of sentences in doc
        middle = self.wordLengthsOfDoc[l:len(self.wordLengthsOfDoc)-l]
        mean = statistics.mean(middle)
        median = statistics.median(middle)
        stdev = statistics.stdev(middle)
        self.addStat("DOC_WORD_LEN_MEAN", mean)
        self.addStat("DOC_WORD_LEN_MEDIAN", median)
        self.addStat("DOC_WORD_LEN_STDEV", stdev)
        with open(outFile, 'w') as out:
            out.write("@STATS\n")
            for stat in sorted(self.stats.keys()):
                out.write("\t" + stat + "\t" + str(self.stats[stat]) + "\n")
            for s in self.states:
                self.states[s].writeToFile(out)

    def addStat(self, name, value):
        self.stats[name] = value

    def parse(self, doc):
        wordNum = 0
        for sent in doc.sents:
            for token in sent:
                wordNum += 1
                pos = 1
                pos_ = 'dummy'
                if not (pos in self.states):
                    self.states[pos] = State(pos, pos_)
                self.states[pos].addWord(token)
        self.wordLengthsOfDoc.append(wordNum)
        
class State:
    def __init__(self, pos, pos_):
        self.pos = pos
        self.pos_ = pos_
        self.wordcount = dict()
        self.totalcount = 0

    def addWord(self, token):
        self.totalcount += 1
        if token.text in self.wordcount:
            self.wordcount[token.text] += 1
        else:
            self.wordcount[token.text] = 1

    def writeToFile(self, ofile):
        ofile.write("@STATE\t" + str(self.pos) + "\n")
        ofile.write("\t@POS\t" + self.pos_ + "\n")
        ofile.write("\t@TOTAL_COUNT\t" + str(self.totalcount) + "\n")
        ofile.write("\t@EMITS\n")
        for word in sorted(self.wordcount.keys()):
            ofile.write("\t\t" + word + "\t" + str(self.wordcount[word]) + "\n")
        
    
if __name__ == '__main__':
    main()
