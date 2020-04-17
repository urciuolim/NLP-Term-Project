from numpy import random
import sys
from ast import literal_eval as make_tuple

EMITS = "@EMITS"
PRODUCTIONS = "@PRODUCTIONS"
START = "<start>"
SIZE = "@SIZE"

def main():
    stats = dict()
    PARAM_NUM = 3
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        return
    model = Model(sys.argv[1])
    mean = model.stats["DOC_SEN_LEN_MEAN"]
    stdev = model.stats["DOC_SEN_LEN_STDEV"]
    doclen = int(getNormRand(mean, stdev))
    #mean = model.stats["SEN_WORD_LEN_MEAN"]
    #stdev = model.stats["SEN_WORD_LEN_STDEV"]
    
    print("Writing story with " + str(doclen) + " sentences")
    #print("with about " + int(mean) + " words per sentence")

    count = 0

    with open(sys.argv[2], 'w') as story:
        while count < doclen:
            root = model.nextSen()
            count += 1
            for word in root.sentence:
                story.write(word + " ")
        

def getNormRand(mean, stdev):
    num = random.normal(mean, stdev, 1)[0]
    while num < 0:
        num = random.normal(mean, stdev, 1)[0]
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
                # Collect all global stats
                if line[0] == "@STATS":
                    i += 1
                    line = self.clean(lines, i)
                    while line[0] == '':
                        self.stats[line[1]] = float(line[2])
                        i += 1
                        line = self.clean(lines, i)
                # Start creating a new state
                if line[0] == "@STATE":
                    args = dict()
                    pair = make_tuple(line[1])
                    args["pair"] = pair
                    i += 1
                    line = self.clean(lines, i)
                    # collect all stats about this state
                    while line[0] == '' and line[1] != EMITS:
                        args[line[1]] = line[2]
                        i += 1
                        line = self.clean(lines, i)
                    # insert this state into self.states, which is a dictionary of dictionaries
                    # indexed first by current pos tag, then by next pos tag (returning a number
                    # used in a probability distrobution)
                    if not (pair[0] in self.states):
                        self.states[pair[0]] = dict()
                    self.states[pair[0]][pair[1]] = State(args)
                    if not (SIZE in self.states[pair[0]]):
                        self.states[pair[0]][SIZE] = 0
                    self.states[pair[0]][SIZE] += int(args[SIZE])
                    # Read all words that this state emits
                    if len(line) > 1 and line[1] == EMITS:
                        i += 1
                        line = self.clean(lines, i)
                        # reverse counts, so that we can get a distribution
                        # for a random num -> word function
                        total = 0
                        while line[0] == '' and line[1] != PRODUCTIONS:
                            total += int(line[3])
                            self.getstate(pair).addWord(total, line[2])
                            i += 1
                            line = self.clean(lines, i)
                        if total != self.getstate(pair).size:
                            print("Mismatch in counts for emits in " + str(pair),
                                  total,
                                  self.getstate(pair).size)
                    # Read all productions that this state emits, formatted in a tree
                    # where the current state is the root and has left/right children
                    # in the read in production tuples
                    if len(line) > 1 and line[1] == PRODUCTIONS:
                        i += 1
                        line = self.clean(lines, i)
                        total = 0
                        while line[0] == '' and line[1] == '':
                            total += int(line[3])
                            self.getstate(pair).addProd(total, line[2])
                            i += 1
                            line = self.clean(lines, i)
                        if total != self.getstate(pair).size:
                            print("Mismatch in counts for productions in " + str(pair),
                                  total,
                                  self.getstate(pair).size)
                # End of creating State
        #Model read in
        #end of constructor

    def getstate(self, state_tuple):
        return self.states[state_tuple[0]][state_tuple[1]]
                    

    def clean(self, lines, i):
        if not(i < len(lines)):
            return "#EOF"
        return lines[i][:len(lines[i])-1].split('\t')

    def nextSen(self):
        # assign root of sentence
        lim = self.states[START][SIZE]+1
        while True:
            num = random.randint(1, lim)
            for key in sorted(self.states[START].keys()):
                if key == SIZE or not(key[0] == 'V'):
                    continue
                num -= self.states[START][key].size
                if num <= 0:
                    return SentenceNode(self.states, self.states[START][key])

class SentenceNode:
    def __init__(self, states, state):
        LEFT = 0
        RIGHT = 1
        self.state = state
        # Assign emission of this state
        lim = state.size+1
        num = random.randint(1, lim)
        self.word = state.getWord(num)
        # Assign production of this state
        num = random.randint(1, lim)
        self.production = state.getProd(num)
        # Instantiate children and form sentence array
        currState = state.pair[1]
        self.sentence = []
        # Starting with left
        print(str(self.production))
        for nextState in self.production[LEFT].split(":"):
            if nextState == '':
                continue
            node = SentenceNode(states, states[currState][nextState])
            self.sentence.extend(node.sentence)
        self.sentence.append(self.word)
        # and then right
        for nextState in self.production[RIGHT].split(":"):
            if nextState == '':
                continue
            node = SentenceNode(states, states[currState][nextState])
            self.sentence.extend(node.sentence)
                        
class State:
    def __init__(self, args):
        self.pair = args["pair"]
        self.size = int(args["@SIZE"])
        self.words = dict()
        self.productions = dict()
        self.sorted_w_keys = None
        self.sorted_p_keys = None
        print(str(self.pair) + " State created")

    def addWord(self, num, word):
        self.words[num] = word

    def addProd(self, num, prod):
        self.productions[num] = make_tuple(prod)

    def getWord(self, num):
        if self.sorted_w_keys is None:
            self.sorted_w_keys = sorted(self.words.keys())
        for key in self.sorted_w_keys:
            if num <= key:
                return self.words[key]

    def getProd(self, num):
        if self.sorted_p_keys is None:
            self.sorted_p_keys = sorted(self.productions.keys())
        for key in self.sorted_p_keys:
            if num <= key:
                return self.productions[key]
    
if __name__ == '__main__':
    main()
