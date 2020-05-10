import spacy
import sys
import numpy
from numpy import array, zeros
from Model import Model, State
from ner_getter import get_all_NEs, load_name_base, name_match
import statistics
from statistics import stdev

START = "@START"
END = "~END"
master_list = dict()
ne_stats = dict()
COUNT = 0
FIRST = 1
LAST = 2

def main():
    params = "<clean_text> <bank_dir> <name_file>"
    if len(sys.argv)-1 < len(params.split(" ")):
        print(len(sys.argv))
        print(len(params.split(" ")))
        print("Need " + str(len(params.split(" "))-(len(sys.argv)-1)) + " more args")
        print("all params: " + params)
        return

    clean_text = sys.argv[1]
    bank_dir = sys.argv[2]
    name_file = sys.argv[3]
    
    model = Model()
    nlp = spacy.load("en_core_web_sm")
    name_base = load_name_base(name_file)
    
    with open(clean_text, 'r') as corpus:
        cleanlines = corpus.readlines()

    summary_list = []
    for line in cleanlines:
        line = line.strip("\n")
        if '\t' in line:
            if '@' in line:
                summary = [line]
            elif '-' in line:
                summary.append("~END\t")
                summary_list.append(summary)
                summary = []
        else:
            summary.append(line)

    print("Starting parse")

    progressBar(int(len(summary_list)/600)+1)

    for count,summary in enumerate(summary_list):
        MovieID = summary[0].split("\t")[1]
        masked_list = mask(summary, name_base, nlp)
        i = -1         
        for sentence,masked in zip(summary,masked_list):
            model.parse(sentence, masked, i, nlp)
            i += 1
        model.docLength.append(i)

        if count % 600 == 0 and count > 1:
            print(".", end="", flush=True)

    print("Parsing done, now printing model as a text file")

    model.printToFile(bank_dir)
    print("Model printed")
    printMasterNEList(bank_dir)
    print("NEs printed")
    printNEStats(bank_dir)
    print("NE stats printed")
    
    print("Complete")

def printNEStats(bank_dir):
    global ne_stats

    if bank_dir[0] != "/":
        bank_dir += "/"
    
    with open(bank_dir + "ne.stats", 'w') as sfile:
        for cat in sorted(list(ne_stats.keys())):
            print(cat, file=sfile)
            tot_num_cat = ne_stats[cat]["@TOTAL"]
            tot_num_typ = len(ne_stats[cat])-1
            print("\t", "#" + str(tot_num_cat), file=sfile)
            print("\t", tot_num_typ, file=sfile)
            for typ in sorted(list(ne_stats[cat].keys())):
                if typ == "@TOTAL":
                    continue
                print("\t", typ, file=sfile)
                num_inst = len(ne_stats[cat][typ])
                lens = [len(inst) for inst in ne_stats[cat][typ]]
                max_len = max(lens)
                len_prob = numpy.zeros((max_len+1,), dtype = "float32")
                for l in lens:
                    len_prob[l] += 1
                len_prob[0] = tot_num_cat-sum(len_prob)
                # Laplacian smoothing
                for i in range(0, len(len_prob)):
                    len_prob[i] += 1
                len_prob /= tot_num_cat + len(len_prob)
                len_prob = numpy.absolute(numpy.round(len_prob, 3))
                    
                print("\t\t", max_len, file=sfile)
                print("\t", end="", file=sfile)
                for p in len_prob:
                    print("\t", p, end="", file=sfile)
                print("", file=sfile)

                typ_ne_arr = numpy.zeros((max_len,3), dtype="float32")
                for inst in ne_stats[cat][typ]:
                    for i in range(min(max_len, len(inst))):
                        for j in range(3):
                            typ_ne_arr[i][j] += inst[i][j]
                            
                for i in range(max_len):
                    print("\t", end="", file=sfile)
                    for j in range(3):
                        typ_ne_arr[i][j] = round(float(typ_ne_arr[i][j]/num_inst),3)
                        print("\t", typ_ne_arr[i][j], end="", file=sfile)
                    print("", file=sfile)

def remove_punc(word):
    c = 0
    while True:
        if c >= len(word) or word[c] in [" ", ",", ".", "?", "!"]:
            break
        elif c < len(word)-1 and word[c:c+2] == "\'s":
            break
        else:
            c += 1
    return word[0:c]

def partial_name_lookup(word, NEs, nlp):
    name_list = [x[1] for x in NEs]
    n_match = name_match(word, name_list)
    if not -1 in n_match and n_match:
        if len(n_match) > 1:
            print("THIS HAPPENED")
            print(word)
            print(NEs)
        else:
            n = n_match[0]
            for ne in NEs:
                if n == ne[1]:
                    return ne + (word,)
    elif not n_match:
        for ne in NEs:
            if word in ne[1]:
                return ne + (word,)
    if nlp == None:
        return None
    token = [t for t in nlp(word)][0]

    if token.lemma_ != "-PRON-" and token.lemma_ != word:
        newword = token.lemma_
        if word[0].isupper():
            newword = newword[0].upper() + newword[1:]
        return partial_name_lookup(newword, NEs, None)
    return None

def mask(summary, name_base, nlp):
    MovieID = summary[0].split("\t")[1]
    masked_sents = [[]]
    NEs, Nums = get_all_NEs(MovieID, summary[1:-1], name_base, nlp)

    tmp = summary[1:-1]
    if type(tmp) != list:
        tmp = list(tmp)

    summaryTagMention = dict()
        
    i = 0
    for sent in tmp:
        i += 1
        tagCounter = dict()
        tagEnum = dict()
        tagMention = dict()
        for ne in NEs:
            if scrollingWindowSearch(ne[1], sent):
                add_to_master_list(ne)
                update_mention(ne, i, tagMention)
                c = str(increCounter(ne, tagCounter, tagEnum))
                sent = sent.replace(ne[1], "<" + ne[0] + c + ">")
        for num in Nums:
            if scrollingWindowSearch(num[1], sent):
                add_to_master_list(num)
                #update_mention(num, tagMention)
                c = str(increCounter(num, tagCounter, tagEnum))
                sent = sent.replace(num[1], "<" + num[0] + c + ">")
        for word in sent.split(" "):
            if "<" not in word and word != word.lower():
                word = [t.text for t in nlp(word)][0]
                name_entry = partial_name_lookup(word, NEs, nlp)
                if type(name_entry) == tuple:
                    add_to_master_list(name_entry)
                    update_mention(name_entry, i, tagMention)
                    c = str(increCounter(name_entry, tagCounter, tagEnum))
                    sent = sent.replace(name_entry[2], "<" + name_entry[0] + c + ">")
        masked_sents.append(sent)

        for typ in tagMention:
            if not typ in summaryTagMention:
                summaryTagMention[typ] = dict()
            for ne in tagMention[typ]:
                if ne not in summaryTagMention[typ]:
                    summaryTagMention[typ][ne] = [0, tagMention[typ][ne][FIRST], 0]
                summaryTagMention[typ][ne][COUNT] += 1
                summaryTagMention[typ][ne][LAST] = tagMention[typ][ne][LAST]
    masked_sents.append([])

    update_ne_stats(i, summaryTagMention)
    
    return masked_sents

def update_ne_stats(num_lines, tagMention):
    global ne_stats
    category = int(num_lines / 5)
    # 10 distinct categories of summary length
    if category > 9:
        category = 9
    normalized_mentions = dict()
    for typ in tagMention:
        normalized_mentions[typ] = list()
        for m in sorted(list(tagMention[typ].values()), reverse=True):
            normalized_mentions[typ].append((float(m[0]/num_lines),
                                             float(m[1]/num_lines),
                                             float(m[2]/num_lines)))
    if not category in ne_stats:
        ne_stats[category] = dict()
        ne_stats[category]["@TOTAL"] = 0
    ne_stats[category]["@TOTAL"] += 1
    for typ in normalized_mentions:
        if not typ in ne_stats[category]:
            ne_stats[category][typ] = list()
        ne_stats[category][typ].append(normalized_mentions[typ])

def update_mention(entry, lnum, tagMention):
    if not entry[0] in tagMention:
        tagMention[entry[0]] = dict()
    if not entry[1] in tagMention[entry[0]]:
        #tagMention[entry[1]] = [count, first, last]
        tagMention[entry[0]][entry[1]] = [0,lnum,0]
    tagMention[entry[0]][entry[1]][COUNT] += 1
    tagMention[entry[0]][entry[1]][LAST] = lnum

def add_to_master_list(ne):
    global master_list
    if not ne[0] in master_list:
        master_list[ne[0]] = set()
    master_list[ne[0]].add(ne[1])

def printMasterNEList(bank_dir):
    global master_list
    if bank_dir[0] != "/":
        bank_dir += "/"
    for typ in master_list:
        with open(bank_dir + typ + ".ne", 'w') as ofile:
            for ne in master_list[typ]:
                print(ne, file=ofile)

def scrollingWindowSearch(name, sent):
    s = 0
    while s+len(name) <= len(sent):
        if name == sent[s:s+len(name)]:
            if (s+len(name) == len(sent) or
                sent[s+len(name)] == " " or
                sent[s+len(name)] == "." or
                sent[s+len(name)] == "," or
                sent[s+len(name)] == "?" or
                sent[s+len(name)] == "\'" or
                sent[s+len(name)] == "!"):
                return True
        s+= 1
    return False

def increCounter(entry, tagCounter, tagEnum):
    if not entry[1] in tagEnum:
        if not entry[0] in tagCounter:
            tagCounter[entry[0]] = 0
        tagEnum[entry[1]] = tagCounter[entry[0]]
        tagCounter[entry[0]] += 1
    return tagEnum[entry[1]]
    
def progressBar(num):
    for _ in range(num):
        print("_", end="")
    print("")
    
if __name__ == '__main__':
    main()
