from numpy import random
import sys

EMITS = "@EMITS"

def main():
    stats = dict()
    PARAM_NUM = 3
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        return
    model = Model(sys.argv[1])
    mean = model.stats["DOC_WORD_LEN_MEAN"]
    stdev = model.stats["DOC_WORD_LEN_STDEV"]

    wordlen = int(getNormRand(mean, stdev))
    print("Writing story with word length of around " + str(wordlen))

    count = 0

    with open(sys.argv[2], 'w') as story:
        while count < wordlen:
            sen = model.nextSen()
            count += len(sen)
            for word in sen:
                story.write(word + " ")
        

def getNormRand(mean, stdev):
    num = random.normal(mean, stdev, 1)[0]
    while num < 0:
        num = random.normal(mean, stdev)[0]
    return num

class Model:
    def __init__(self, mfile):
        self.stats = dict()
        self.states = dict()

        with open(mfile, 'r') as mod:
            lines = mod.readlines()
            i = 0
            while i < len(lines):
                line = self.clean(lines, i)
                if line[0] == "@STATS":
                    i += 1
                    line = self.clean(lines, i)
                    while line[0] == '':
                        self.stats[line[1]] = float(line[2])
                        i += 1
                        line = self.clean(lines, i)
                if line[0] == "@STATE":
                    args = dict()
                    pos = int(line[1])
                    args["pos"] = pos
                    i += 1
                    line = self.clean(lines, i)
                    while line[0] == '' and line[1] != EMITS:
                        args[line[1]] = line[2]
                        i += 1
                        line = self.clean(lines, i)
                    print(i)
                    self.states[pos] = State(args)
                    if len(line) > 1 and line[1] == EMITS:
                        i += 1
                        line = self.clean(lines, i)
                        # reverse counts, so that we can get a distribution
                        # for a random num -> word function
                        total = 0
                        while line[0] == '' and line[1] == '':
                            total += int(line[3])
                            self.states[pos].addWord(total, line[2])
                            i += 1
                            line = self.clean(lines, i)
                        if total != self.states[pos].totalcount:
                            print("Mismatch in counts",
                                  total,
                                  self.states[pos].totalcount)

    def clean(self, lines, i):
        if not(i < len(lines)):
            return "#EOF"
        return lines[i][:len(lines[i])-1].split('\t')

    def nextSen(self):
        lim = self.states[1].totalcount
        num = random.randint(1, lim)
        return [self.states[1].getWord(num)]
        
                        
class State:
    def __init__(self, args):
        self.pos = args["pos"]
        self.pos_ = args["@POS"]
        self.totalcount = int(args["@TOTAL_COUNT"])
        self.words = dict()
        self.sortedkeys = None
        print(str(self.pos) + " State created")

    def addWord(self, num, word):
        self.words[num] = word

    def getWord(self, num):
        if self.sortedkeys is None:
            self.sortedkeys = sorted(self.words.keys())
        for key in self.sortedkeys:
            if num <= key:
                return self.words[key]
    
if __name__ == '__main__':
    main()
