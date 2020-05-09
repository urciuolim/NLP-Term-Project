import spacy
import sys
from Model import Model, State
from ner_getter import get_all_NEs, load_name_base, name_match

START = "@START"
END = "~END"

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
        if count <= 0:
            continue
        MovieID = summary[0].split("\t")[1]
        #NEs, Nums = get_all_NEs(MovieID, summary[1:-1], name_base, nlp)
        masked = mask(summary, name_base, nlp)
        #for ne in NEs:
            #print(ne)
        #print(Nums)
        i = -1
        text = ""
        for line in summary:
            text += line + " "

        s = 0
        for sentence in summary:
            e = s + len(sentence)
            if (text[s:e] != sentence):
                print("uh oh")
            s = e + 1
        exit()
            
        for sentence in summary:
            model.parse(sentence, i, nlp, masked_sent)
            i += 1
        model.docLength.append(i)
        if count > 0:
            exit()
        if count % 600 == 0 and count > 1:
            print(".", end="", flush=True)

    print("Parsing done, now printing model as a text file")

    model.printToFile(bank_dir)

    print("Complete")

def partial_name_lookup(word, NEs):
    word = word.replace(".", "")
    word = word.replace(",", "")
    word = word.replace("?", "")
    word = word.replace("!", "")
    word = word.replace("\'s", "")
    word = word.replace("\'S", "")
    name_list = [x[1] for x in NEs]
    n_match = name_match(word, name_list)
    if not -1 in n_match:
        if len(n_match) > 1:
            raise ValueError("This happened")
        else:
            n = n_match[0]
            for ne in NEs:
                if n == ne[1]:
                    return ne
    return word

def mask(summary, name_base, nlp):
    MovieID = summary[0].split("\t")[1]
    masked_sents = [[]]
    NEs, Nums = get_all_NEs(MovieID, summary[1:-1], name_base, nlp)

    tmp = summary[1:-1]
    if type(tmp) != list:
        tmp = list(tmp)
        ############################## NEED TO ADD ENUMERATION OF MASKS

    for sent in tmp:
        for ne in NEs:
            sent = sent.replace(ne[1], "<" + ne[0] + ">")
        for num in Nums:
            sent = sent.replace(num[1], "<" + num[0] + ">")
        for word in sent.split(" "):
            if "<" not in word and word != word.lower():
                name_entry = partial_name_lookup(word, NEs)
                if type(name_entry) == tuple:
                    sent = sent.replace(word, "<" + name_entry[0] + ">")
        print(sent)

    exit()
    
    sent_start = 0
    for sent in tmp:
        sent_end = sent_start + len(sent)
        this_ne_list = []
        for ne in ne_list:
            if ne[0] >= sent_start and ne[0] < sent_end:
                this_ne_list.append(ne)
        masked = ""
        curr = 0
        for ne in this_ne_list:
            s,e = ne[0:2]
            s -= sent_start
            if curr != s+e:
                masked += sent[curr:s]
                masked += "<" + ne[2] + ">"
                curr = s + e
        # end for
        masked += sent[curr:]
        masked_sents.append(masked)
        sent_start = sent_end + 1
    masked_sents.append([])


    exit()
        
    
    
def progressBar(num):
    for _ in range(num):
        print("_", end="")
    print("")
    
if __name__ == '__main__':
    main()
