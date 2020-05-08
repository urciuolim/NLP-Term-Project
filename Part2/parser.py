import spacy
import sys
from Model import Model, State

START = "@START"
END = "~END"

def main():
    params = "<clean_text> <bank_dir>"
    if len(sys.argv)-1 < len(params.split(" ")):
        print(len(sys.argv))
        print(len(params.split(" ")))
        print("Need " + str(len(params.split(" "))-(len(sys.argv)-1)) + " more args")
        print("all params: " + params)
        return

    clean_text = sys.argv[1]
    bank_dir = sys.argv[2]
    
    model = Model()
    nlp = spacy.load("en_core_web_sm")
    
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
        # Do stat stuff here
        i = -1
        for sentence in summary:
            model.parse(sentence, i, nlp)
            i += 1
        model.docLength.append(i)
        if count % 600 == 0 and count > 1:
            print(".", end="", flush=True)

    print("Parsing done, now printing model as a text file")

    model.printToFile(bank_dir)

    print("Complete")

def progressBar(num):
    for _ in range(num):
        print("_", end="")
    print("")
    
if __name__ == '__main__':
    main()
