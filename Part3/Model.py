from statistics import mean, median, stdev, variance
from random import choice, random, shuffle
import numpy
from numpy import zeros
import os
from ner_getter import name_match
import re
from math import floor, ceil

START = "@START"
END = "~END"

def load_model(base_dir):
    model = Model()
    if base_dir[-1] != "/":
        base_dir += "/"
    for f in os.listdir(base_dir):
        if f == "STATS":
            model.loadStats(base_dir+f)
        elif f == "ne.stats":
            model.loadNEStats(base_dir+f)
        elif ".ne" in f:
            model.loadNENames(base_dir+f)
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
        self.ne_names = dict()
        self.ne_stats = dict()
        self.docLength = []
        self.ne_by_typ = dict()
        self.intd_length = 0
        self.nes_left = None
        self.typ2indx = dict()
        self.last_emis = ""

    # Based on HMM concept, at each given state (VP), and with respect to enumerated
    # named entites that should appear in the story, probabilisitically sample states
    # until a suitable sentence is found in that states list of emissions.
    # If given the end flag, will generate an end state as soon as it finds one.
    # If the current state only transitions to an end state, it will return the end state
    # If at the current state there is no next state that fulfills the named entity requirements
    # at this stage, then raise an error indicating a dead end
    def genSentence(self, curr_line, endFlag):
        count = 0
        while True:
            count += 1
            try:
                self.calc_nes_left_at_step(curr_line)
                possibleStates = self.states[self.lastState].nextState
                if endFlag and END in possibleStates:
                    return self.ne_emit_check(END)
                while END in possibleStates:
                    possibleStates.remove(END)
                if len(possibleStates) == 0:
                    return self.ne_emit_check(END)
                shuffle(possibleStates)
                for state in possibleStates:
                    emissions = self.states[state].emits
                    shuffle(emissions)
                    emissions = sorted(emissions, key=lambda x: self.dist_from_needed_ne_emits(x))
                    for e in emissions:
                        fit = self.good_fit(e, curr_line)
                        if fit:
                            #print(fit)
                            self.lastState = state
                            return fit
                raise ValueError("Reached dead end")
            except KeyError as e:
                # Known bug: At least one url address got past the cleaning phase
                # and is listed as a VP and thus is a possible state. Although the
                # probability of reaching that state is small, it might happen and
                # cause a key error. So just try generating another sentence. If this
                # error is caused 10 times in a row then it might be another issue,
                # so also checking for that.
                #print("!", end="", flush=True)
                if count <= 10:
                    continue
                else:
                    raise e

    # Each time step has a number of named entities (by type) that need to be emitted.
    # This function calculates how close a candidate emission is to fulfilling that need
    def dist_from_needed_ne_emits(self, emis):
        typs = []
        nes_needed = zeros((len(self.ne_by_typ),), dtype="int32")
        
        for token in emis.split(" "):
            if "<" in token:
                typs.append(re.sub("[0-9]", "", token.split(">")[0].split("<")[-1]))

        for typ in typs:
            if not typ in self.ne_by_typ:
                return float("inf")
            nes_needed[self.typ2indx[typ]] += 1

        return numpy.linalg.norm(self.nes_left-nes_needed)

    # For a given time step, calculate how many named entities (by type) need to be emitted
    # Taking into account that some named entities will not appear before/after a certain line
    def calc_nes_left_at_step(self, curr_line):
        S,E = 2,3
        self.nes_left = zeros((len(self.ne_by_typ)),dtype="int32")
        self.typ2indx = dict()
        for i,typ in enumerate(list(self.ne_by_typ.keys())):
            self.typ2indx[typ] = i
            for ne in self.ne_by_typ[typ]:
                s_lim = floor(ne[S]*self.intd_length)
                e_lim = ceil(ne[E]*self.intd_length)
                if curr_line >= s_lim and curr_line <= e_lim:
                    if ne[0] > 0:
                        self.nes_left[self.typ2indx[typ]] += 1

    # Checks to make sure enough emissions have occured, if not raises an error
    def ne_emit_check(self, END):
        for typ in self.ne_by_typ:
            for ne in self.ne_by_typ[typ]:
                if ne[0] > 1:
                    raise ValueError("End reached before all NEs could be utilized")
        return END

    # Returns an emission with masked named entities filled in with "real" (filler) named entities
    # returns False if the emission requires too many named entities (for a given type)
    def good_fit(self, emis, curr_line):
        # Check to make sure not to emit the same sentence twice, in case a state can loop to itself
        if self.last_emis == emis:
            return False
        original_emis = emis
        S,E = 2,3
        # Sort each list of enumerated named entities by how many occurances they still require
        for typ in self.ne_by_typ:
            self.ne_by_typ[typ] = sorted(self.ne_by_typ[typ], reverse=True)

        # Split all masked named entity tags, so they can be "filled in" by enumerated NEs
        params = []
        for token in emis.split(" "):
            if "<" in token:
                params.append(token.split(">")[0].split("<")[-1])
                
        chosen = []
        for p in params:
            typ = re.sub("[0-9]", "", p)
            # If the sentence requires a named entity that we have not enumerated,
            # return False, since all enumerations are done at the start
            if not typ in self.ne_by_typ:
                return False
            # If named entities of this type of already been used up by this sentence,
            # and the sentence requires more, then return False, it can't be satisfied
            if len(self.ne_by_typ[typ]) == 0:
                return False
            c = None
            # For each enumerated named entity
            for i in range(len(self.ne_by_typ[typ])):
                # Calulate the lines where it will min/max appear on
                s_lim = floor(self.ne_by_typ[typ][i][S]*self.intd_length)
                e_lim = ceil(self.ne_by_typ[typ][i][E]*self.intd_length)
                # If the current line is between these limits (inclusive)
                # then this named entity can fulfill a parameter of the sentence
                if curr_line >= s_lim and curr_line <= e_lim:
                    c = self.ne_by_typ[typ].pop(i)
                    break
            # If there are no named entities that can fulfill this parameter at this
            # time step, then this sentence can't be fulfilled, return False
            if c == None:
                return False
            # Otherwise chose this enumereated named entity to fulfil this parameter
            chosen.append((p, c))

        # At this point, all parameters of the sentence have been fulfilled
        for c in chosen:
            # Update the sentence, filling in the masked entity with an enumerated one
            typ = re.sub("[0-9]", "", c[0])
            updated_ne = (c[1][0]-1, c[1][1], c[1][2], c[1][3])
            self.ne_by_typ[typ].append(updated_ne)
            emis = emis.replace("<" + c[0] + ">", c[1][1])

        # Update the last emission (masked version)
        self.last_emis = original_emis

        return emis

    # With a given intended length, enumerate which named entities will
    # appear in the story, how frequently (relative number of lines) and
    # between what limits
    def enumerate_nes(self, intd_length):
        self.intd_length = intd_length
        # Named entity statistics are calculated in 10 categories,
        # The number of lines / 5 = category, with a max category of 9
        cat = min(int(intd_length / 5), 9)
        enum_dict = dict()
        # For each named entity type that has appeared in this category
        for typ in self.ne_stats[cat]:
            # get relavent statistics
            prob,data = self.ne_stats[cat][typ]
            # Produce a random betwee, 0 and 1
            r = random()
            n = 0
            # Randomly sample from a distribution of number of entities of
            # this type, normalized to have a probability between 0 and 1
            while n < len(prob)-1:
                if r <= prob[n]:
                    break
                r -= prob[n]
                n += 1
            # If an amount of named entities are chosen more than 0
            if n > 0:
                # Store their stats, will enumerate their names below
                enums = data[0:n]
                enum_dict[typ] = enums

        # For each type that has at least one enumerated named entity
        for typ in enum_dict:
            self.ne_by_typ[typ] = list()
            for i,ne in enumerate(enum_dict[typ]):
                # Number of lines this NE should appear
                num_lines = round(ne[0] * intd_length, 1)
                # Randomly pick a name out of this category of NEs
                # <I'm doing this in lieu of a "name generator" or generic tags>
                r = int(round(random() * len(self.ne_names[typ])))
                name = self.ne_names[typ].pop(r)
                self.ne_by_typ[typ].append((num_lines, "<" + name + ">", ne[1], ne[2]))

    # Load stats file for doc length
    def loadStats(self, file_path):
        with open(file_path, 'r') as sfile:
            for line in sfile:
                tmp = line.split("\t")
                try:
                    self.stats[tmp[0]] = float(tmp[1])
                except ValueError:
                    self.stats[tmp[0]] = tmp[1]

    # Load named entity stats
    def loadNEStats(self, file_path):
        CAT,TCOUNT,TYP,MAXLEN,PROB,DATA = 0,1,2,3,4,5
        STATE = CAT
        cat,tcount,typ,maxlen,prob,data,dcount = None,None,None,None,None,None,None
        with open(file_path, 'r') as ifile:
            lines = ifile.readlines()
        for line in lines:
            if "#" in line:
                continue
            tabs = line.split("\t")
            if STATE == CAT:
                cat = int(tabs[0].strip())
                self.ne_stats[cat] = dict()
                STATE = TCOUNT
            elif STATE == TCOUNT:
                tcount = int(tabs[1].strip())
                STATE = TYP
            elif STATE == TYP:
                typ = tabs[1].strip()
                STATE = MAXLEN
            elif STATE == MAXLEN:
                maxlen = int(tabs[2].strip())
                prob = zeros((maxlen+1,), dtype="float32")
                data = zeros((maxlen,3), dtype="float32")
                STATE = PROB
            elif STATE == PROB:
                for i in range(2,2+maxlen+1):
                    prob[i-2] = float(tabs[i].strip())
                STATE = DATA
                count = 0
            elif STATE == DATA:
                for i in range(2, 5):
                    data[count][i-2] = float(tabs[i].strip())
                count += 1
                if count >= maxlen:
                    self.ne_stats[cat][typ] = (prob,data)
                    tcount -= 1
                    if tcount > 0:
                        STATE = TYP
                    else:
                        STATE = CAT
                        cat,typ,maxlen,prob,data,count = None,None,None,None,None,None

    # Load list of names for each seen type
    def loadNENames(self, file_path):
        typ = file_path.split("/")[-1].split(".")[0]
        with open(file_path, 'r') as nfile:
            self.ne_names[typ] = [line.strip("\n") for line in nfile.readlines()]

    # Print the model to file with document statistics
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

    # Given a sentence and it's masked (for named entities) version,
    # and line number within the original summary, incorporate the sentence
    # into the model
    def parse(self, sent, masked, i, nlp):
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
            # Find all VPs
            for i in range(len(tokens)):
                if (tokens[i].pos_ == "VERB" or
                    tokens[i].pos_ == "ADV" or
                    tokens[i].pos_ == "AUX" or
                    tokens[i].pos_ == "PART"):
                    if s == -1:
                        s = i
                else:
                    s = -1
                # Found the root VP
                if tokens[i] == root:
                    e = i
                    for j in range(i+1, len(tokens)):
                        if tokens[j].pos_ == "PART":
                            e = j
                        else:
                            break
                    break;
            # end of: for i in range(len(tokens)):

            # Start with "Unknown" VP, which will serve as an
            # ignore tag. If we can't determine if this is a
            # legit sentence, then throw it out. We have plenty
            # of others in the corpus
            root_vp = "UNKNOWN"
            if s != -1 and e != -1 and s <= e:
                root_vp = ""
                for word in tokens[s:e+1]:
                    root_vp += " " + word.text
            root_vp = root_vp.strip()

            # Seperate the first sentence of a summary into a seperate category
            # so START only transitions to these specific sentences
            if self.lastState == START:
                root_vp = "|"+root_vp

            # Record emission and next state occurances
            # typical of bi-gram HMM
            if not root_vp in self.states:
                self.states[root_vp] = State(root_vp)
            self.states[root_vp].addEmit(masked, i)
            self.states[self.lastState].addNext(root_vp)
            self.lastState = root_vp
        # end of: for sent in spacy_doc.sents:

# State class used to keep track of bi-gram sentence level
# states
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
