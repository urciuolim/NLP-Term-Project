from statistics import mean, median, stdev, variance
from random import choice
import os
from ner_getter import name_match
START = "@START"
END = "~END"

def load_model(base_dir):
    model = Model()
    if base_dir[-1] != "/":
        base_dir += "/"
    for f in os.listdir(base_dir):
        if f == "STATS":
            model.loadStats(base_dir+f)
        elif f == "UNKNOWN":
            # Unknown is a category of sentences where POS tagger
            # was unable to categorize the root of the sentence
            # as a verb. I'm choosing to not use those since
            # there are a lots more sentences to choose from
            continue
        else:
            model.states[f] = load_state(base_dir, f)
    return model

def load_state(base_dir, vp):
    state = State(vp)
    file_path = base_dir + vp
    sections = ["@LOCS", "@EMITS", "@NEXT"]
    section = None
    with open(file_path, 'r') as sfile:
        for line in sfile:
            line = line.strip("\n")
            if line in sections:
                section = line
                continue
            tmp = line.split("\t")
            if section == sections[0]:
                state.locInDocCount[int(tmp[1])] = int(tmp[2])
            elif section == sections[1]:
                if tmp[1] == "UNKNOWN":
                    continue
                else:
                    state.emits.append(tmp[1])
            elif section == sections[2]:
                if tmp[1] == "UNKNOWN":
                    continue
                else:
                    state.nextState.append(tmp[1])
    return state        

class Model:
    def __init__(self):
        self.stats = dict()
        self.states = dict()
        self.states[START] = State(START)
        self.states[END] = State(END)
        self.lastState = START
        self.docLength = []

    def genSentence(self, endFlag):
        count = 0
        while True:
            count += 1
            try:
                possibleStates = self.states[self.lastState].nextState
                if endFlag and END in possibleStates:
                    return END
                while END in possibleStates:
                    possibleStates.remove(END)
                if len(possibleStates) == 0:
                    return END
                state = choice(possibleStates)
                emissions = self.states[state].emits
                self.lastState = state
                return choice(emissions)
            except KeyError as e:
                # Known bug: At least one url address got past the cleaning phase
                # and is listed as a VP and thus is a possible state. Although the
                # probability of reaching that state is small, it might happen and
                # cause a key error. So just try generating another sentence. If this
                # error is caused 10 times in a row then it might be another issue,
                # so also checking for that.
                if count <= 10:
                    continue
                else:
                    raise e
        

    def loadStats(self, file_path):
        with open(file_path, 'r') as sfile:
            for line in sfile:
                tmp = line.split("\t")
                try:
                    self.stats[tmp[0]] = float(tmp[1])
                except ValueError:
                    self.stats[tmp[0]] = tmp[1]

    def printToFile(self, base_dir):
        doc_mean = mean(self.docLength)
        doc_median = median(self.docLength)
        doc_stdev = stdev(self.docLength)
        doc_variance = variance(self.docLength)
        
        if base_dir[-1] != "/":
            base_dir += "/"
        with open(base_dir + "STATS", 'w') as sfile:
            print("DOC_LEN_MEAN\t" + str(round(doc_mean, 1)), file=sfile)
            print("DOC_LEN_MEDIAN\t" + str(doc_median), file=sfile)
            print("DOC_LEN_STDEV\t" + str(round(doc_stdev, 1)), file=sfile)
            print("DOC_LEN_VARIANCE\t" + str(round(doc_variance, 1)), file=sfile)

        count = 0
        for state in self.states:
            count += 1
            if state != END:
                try:
                    self.states[state].printToFile(base_dir)
                except:
                    continue

        print("Printed out", count, "verb files")

    def parse(self, sent, i, nlp, masked_sent):
        if '\t' in sent:
            if '@' in sent:
                self.lastState = START
                return
            elif '~' in sent:
                self.states[self.lastState].addNext(END)
                return
        spacy_doc = nlp(sent)
        # This loop should only execute once, unless two sentences
        # got merged together during cleaning process
        for sent in spacy_doc.sents:
            root = sent.root
            tokens = list(sent)
            s = -1
            e = -1
            for i in range(len(tokens)):
                if (tokens[i].pos_ == "VERB" or
                    tokens[i].pos_ == "ADV" or
                    tokens[i].pos_ == "AUX" or
                    tokens[i].pos_ == "PART"):
                    if s == -1:
                        s = i
                else:
                    s = -1
                if tokens[i] == root:
                    e = i
                    for j in range(i+1, len(tokens)):
                        if tokens[j].pos_ == "PART":
                            e = j
                        else:
                            break
                    break;
            # end of: for i in range(len(tokens)):
            root_vp = "UNKNOWN"
            if s != -1 and e != -1 and s <= e:
                root_vp = ""
                for word in tokens[s:e+1]:
                    root_vp += " " + word.text
            root_vp = root_vp.strip()

            if not root_vp in self.states:
                self.states[root_vp] = State(root_vp)
            self.states[root_vp].addEmit(masted_sent, i)
            self.states[self.lastState].addNext(root_vp)
            self.lastState = root_vp
        # end of: for sent in spacy_doc.sents:

class State:
    def __init__(self, vp):
        self.verb_phrase = vp
        self.emits = []
        self.nextState = []
        self.locInDocCount = dict()

    def addEmit(self, sent, i):
        self.emits.append(sent)
        if not i in self.locInDocCount:
            self.locInDocCount[i] = 0
        self.locInDocCount[i] += 1

    def addNext(self, vp):
        self.nextState.append(vp)
        
    def printToFile(self, base_dir):
        if base_dir[-1] != "/":
            base_dir += "/"
        with open(base_dir + self.verb_phrase, 'w') as ofile:
            print("@LOCS", file=ofile)
            for loc,cnt in self.locInDocCount.items():
                print("\t" + str(loc) + "\t" + str(cnt), file=ofile)
            print("@EMITS", file=ofile)
            for emit in self.emits:
                print("\t" + emit, file=ofile)
            print("@NEXT", file=ofile)
            for state in self.nextState:
                print("\t" + state, file=ofile)
