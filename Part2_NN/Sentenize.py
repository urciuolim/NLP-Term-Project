import sys
import spacy
import time

def main():
    PARAM_NUM = 3
    
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        print("<corpus> <output file>")
        return

    print(" ", end="", flush=True)
    for i in range(42):
        print("_", end="", flush=True)
    print(" ")

    start = time.time()
    nlp = spacy.load("en_core_web_sm")
    print("|", end="", flush=True)
    sentences = []
    sentences_pos = []
    with open(sys.argv[1], 'r') as corpus:
        for count, line in enumerate(corpus):
            if count % 1000 == 0 and count > 1:
                print(".", end="", flush=True)
            doc = nlp(line.split('\t')[1].split('\n')[0])
            for sentence in doc.sents:
                sen = []
                sen_pos = []
                for token in sentence:
                    if token.tag_ == "_SP" or token.text == "%s":
                        continue
                    else:
                        sen.append(token.text.lower())
                        sen_pos.append(token.tag_.lower())
                if len(sen) >= 3:
                    sen.append("|")
                    sen.extend(sen_pos)
                    sentences.append(sen)
                else:
                    continue
    print("|")

    with open(sys.argv[2], 'w') as output:
        for sent in sentences:
            for word in sent:
                output.write(word + '\t')
            output.write('\n')

    end = time.time()

    print("Sentences written to " + sys.argv[2])
    print("It took", str((end-start)/60.0)[:6], "minutes")

if __name__ == "__main__":
    main()
