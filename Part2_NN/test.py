from gensim.models import Word2Vec
import sys

def main():
    PARAM_NUM = 2
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        print("<word2vec model>")
        return
    
    model = Word2Vec.load(sys.argv[1])

    trials = ["katniss", "taxi", "indiana", "sword", "whiskey", "united", "jersey", "NO WAY THIS IN THERE", "computer"]
    for t in trials:
        try:
            print("-------------------------")
            print(t)
            print(model.wv.most_similar(t))
        except KeyError:
            print("not found")
            continue
    


if __name__ == "__main__":
    main()
