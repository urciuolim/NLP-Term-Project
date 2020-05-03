import sys
import spacy
import time

def main():
    PARAM_NUM = 4
    
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        print("<metadata> <corpus> <output file base>")
        return

    MD = dict()
    with open(sys.argv[1], 'r') as metadata:
        for lines in metadata:
            line = lines.split("\t")
            ID = int(line[0])
            MD[ID] = []
            Genres = line[8]
            Genres = Genres.replace('{', '').replace('}', '').replace('\n', '').split(",")
            if Genres[0] != '':
                for i in range(0, len(Genres)):
                    Genres[i] = Genres[i].split(":")[1].replace('"', '').replace(' ', '').replace("/", "|")
                    MD[ID].append(Genres[i])

    GenreFilePaths = dict()
    base = sys.argv[3]
    for key in MD:
        for genre in MD[key]:
            if not genre in GenreFilePaths:
                GenreFilePaths[genre] = base + genre + ".txt"
                with open(GenreFilePaths[genre], 'w') as f:
                    print("Created " + GenreFilePaths[genre])
                     
    print(" ", end="", flush=True)
    for i in range(42):
        print("_", end="", flush=True)
    print(" ")

    start = time.time()
    nlp = spacy.load("en_core_web_sm")
    print("|", end="", flush=True)
    with open(sys.argv[2], 'r') as corpus:
        for count, line in enumerate(corpus):
            if count % 1000 == 0 and count > 1:
                print(".", end="", flush=True)
            ID = int(line.split('\t')[0])
            if not ID in MD:
                continue
            doc = nlp(line.split('\t')[1].split('\n')[0])
            for sentence in doc.sents:
                sen = []
                for token in sentence:
                    sen.append(token.text.lower())
                for genre in MD[ID]:
                    with open(GenreFilePaths[genre], 'a') as output:
                        for word in sen:
                            output.write(word + '\t')
                        output.write("\n")
    print("|")


if __name__ == "__main__":
    main()
