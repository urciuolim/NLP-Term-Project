import spacy
import nltk
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import sys

def main():
    params = "<clean_text> <name_file>"
    if len(sys.argv)-1 < len(params.split(" ")):
        print(len(sys.argv))
        print(len(params.split(" ")))
        print("Need " + str(len(params.split(" "))-(len(sys.argv)-1)) + " more args")
        print("all params: " + params)
        return

    clean_text = sys.argv[1]
    name_file = sys.argv[2]

    with open(clean_text, 'r') as corpus:
        cleanlines = corpus.readlines()
    cleanlines = [line.strip("\n") for line in cleanlines]

    summary_list = []
    for line in cleanlines:
        if '\t' in line:
            if '@' in line:
                summary = [line]
            elif '-' in line:
                summary.append("~END\t")
                summary_list.append(summary)
                summary = []
        else:
            summary.append(line)

    nlp = spacy.load("en_core_web_sm")
    name_base = load_name_base(name_file)

    for count,summary in enumerate(summary_list):
        MovieID = summary[0].split("\t")[1]
        get_all_NEs(MovieID, summary[1:-1:], name_base, nlp)
        if count % 100 == 0 and count > 1:
            print(".", end="", flush=True)

def get_all_NEs(MovieID, summary, name_base, nlp):
    if type(summary) != list:
        summary = list(summary)
    text = ""
    for line in summary:
        text += line + " "

    nltk_ne = []
    st_char = 0
    end_char = 0
    for sent in nltk.sent_tokenize(text[:-1]):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            end_char = st_char
            word = chunk[0]
            if hasattr(chunk, 'label'):
                word = chunk[0][0]
                end_char += len(word)
                nltk_ne.append(('nltk', chunk.label(), (' '.join(c[0] for c in chunk)),
                                (st_char, end_char-st_char)))
            else:
                end_char += len(word)
            if ((word[0].isalpha() or word[0].isdigit()) and not word.lower() == "\'s"):
                end_char += 1
            st_char  = end_char
    
    doc = nlp(text[:-1])
    spacy_ne = []
    for ent in doc.ents:
        spacy_ne.append(('spacy', ent.label_, ent.text, (ent.start_char, ent.end_char-ent.start_char)))

    nltk_ne = list(dict.fromkeys(nltk_ne))
    spacy_ne = list(dict.fromkeys(spacy_ne))

    NEs = dict()
    Nums = dict()

    if MovieID in name_base:
        for name in name_base[MovieID]:
            if name == "":
                continue
            NEs[name] = "GIVENPERSON"

    for nne in nltk_ne:
        n_match = name_match(nne[2], NEs.keys())
        typ = nne[1]
        if typ == "ORGANIZATION":
            typ = "ORG"
        if -1 in n_match:
            NEs[nne[2]] = typ
        elif n_match:
            for n in n_match:
                NEs[n] += "_" + typ

    for sne in spacy_ne:
        if sne[1] == "ORDINAL" or sne[1] == "CARDINAL" or sne[1] == "DATE" or sne[1] == "TIME":
            parse_nums(sne, Nums)
            continue
        n_match = name_match(sne[2], NEs.keys())
        if -1 in n_match:
            NEs[sne[2]] = sne[1]
        elif n_match:
            if len(n_match) > 1:
                for n in list(n_match):
                    reverse_n_match = name_match(n, [sne[2]])
                    if not -1 in reverse_n_match:
                        typ = NEs[n]
                        NEs[sne[2]] = sne[1]
                        for t in typ.split("_"):
                            NEs[sne[2]] += "_" + t
                        del NEs[n]
                        break
                else:
                    NEs[sne[2]] = sne[1]      
            else:
                for n in n_match:
                    NEs[n] += "_" + sne[1]
                    
    final_list = []
    final_num_list = []
    for ne in NEs:
        n_match = name_match(ne, NEs.keys())
        if n_match:
            final_list.append(reconcile_votes((NEs[ne], ne)))

    for num in Nums:
        typ = Nums[num]
        final_num_list.append((typ, num))

    return final_list, final_num_list

def parse_nums(num_tup, Nums):
    typ = num_tup[1]
    if typ == "DATE":
        if "age" in num_tup[2] or "old" in num_tup[2]:
            typ = "AGE"
            Nums[num_tup[2]] = typ
            return
    Nums[num_tup[2]] = typ

def reconcile_votes(name_tup):
    old_votes = name_tup[0].split("_")
    if "GIVENPERSON" in old_votes:
        return ("PERSON", name_tup[1])
    votes = name_tup[0].split("_")
    while True:
        votes_set = set(votes)
        for v in votes_set:
            votes.remove(v)
        if len(votes) <= 0:
            break
        old_votes = [x for x in votes]
    typ = ""
    for v in sorted(old_votes):
        typ += "_" + v
    return (typ[1:], name_tup[1])

def name_match(qname, name_list):
    AMBIGUOUS = -2
    NONE = -1
    match = NONE
    name_split = qname.split(" ")
    split_match = []
    for split in name_split:
        for i,bank_name in enumerate(name_list):
            bank_name_split = bank_name.split(" ")
            for bank_split in bank_name_split:
                if split == bank_split:
                    if match == NONE:
                        match = i
                    elif match == i:
                        continue
                    else:
                        match = AMBIGUOUS
        # end for i,bank_name...
        split_match.append(match)
        match = NONE

    if -1 in split_match:
        return [-1]
    match_set = set(split_match)
    if -2 in match_set:
        match_set.remove(-2)
    match_return = []
    for m in match_set:
        match_return.append(list(name_list)[m])
    return match_return     

def load_name_base(file_path):
    name_base = dict()
    with open(file_path, 'r') as name_file:
        name_lines = name_file.readlines()
    name_lines = [line.strip("\n").split("\t") for line in name_lines]

    ID = 0
    NAME = 3

    for line in name_lines:
        if not line[ID] in name_base:
            name_base[line[ID]] = list()
        name_base[line[ID]].append(replace_mil_prefixes(line[NAME]))

    return name_base

def replace_mil_prefixes(word):
    word = word.replace("Pvt.", "PVT")
    word = word.replace("Pfc.", "PFC")
    word = word.replace("Cpl.", "CPL")
    word = word.replace("Sgt.", "SGT")
    word = word.replace("SSgt.", "SSG")
    word = word.replace("Lt.", "LT")
    word = word.replace("Cpt.", "CPT")
    word = word.replace("Capt.", "CPT")
    word = word.replace("Maj.", "MAJ")
    word = word.replace("Col.", "COL")
    word = word.replace("Gen.", "GEN")
    word = word.replace("Adm.", "ADM")
    word = word.replace("Ens.", "ENS")
    word = word.replace("Ft.", "Fort")
    word = word.replace("St.", "Saint")
    return word
    
if __name__ == "__main__":
    main()
