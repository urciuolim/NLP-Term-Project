# Source: https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import sys
import os
import time

def main():
    PARAM_NUM = 3
    
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        print("<base folder> <min_count>")
        print("Corpus must have one sentence per line")
        return

    for filename in sorted(os.listdir(sys.argv[1])):
        start = time.time()
        sentences = []
        with open(sys.argv[1] + filename, 'r') as corpus:
            print(sys.argv[1] + filename)
            for count, line in enumerate(corpus):
                if count % 1000 == 0 and count > 1:
                    print(".", end="", flush=True)
                sent = line.split('\t')[:-1]
                while ' ' in sent:
                    sent.remove(' ')
                if len(sent) < 3:
                    continue
                sentences.append(sent)

        print("\nSentences loaded")
        newpath = filename.replace("Sents_", "Movie_WE_100D_")
        newpath = newpath.replace(".txt", ".model")
        path = get_tmpfile(newpath)
        minNum = int(sys.argv[2])
        
        model = Word2Vec(sentences, size=100, window = 5, min_count = minNum, sg=1, workers = 4, iter=30)
        end = time.time()
        model.save(newpath)
        
        print(newpath + " created")
        print("It took", str((end-start)/60.0)[:6], "minutes")
    
if __name__ == "__main__":
    main()
