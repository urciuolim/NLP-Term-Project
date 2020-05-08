from numpy import random
import sys
from Model import Model, State, load_model

START = "@START"
END = "~END"

def main():
    params = "<model_dir> <length> <output_file>"
    if len(sys.argv)-1 < len(params.split(" ")):
        print(len(sys.argv))
        print(len(params.split(" ")))
        print("Need " + str(len(params.split(" "))-(len(sys.argv)-1)) + " more args")
        print("all params: " + params)
        return

    model_dir = sys.argv[1]
    length = int(sys.argv[2])
    output_file = sys.argv[3]

    model = load_model(model_dir)

    print("Writing story with at least", length, "sentences")
    
    count = 0
    lastSentence = START
    story = ""

    while True:
        flag = count >= length-1
        sen = model.genSentence(flag)
        count += 1
        if sen == END:
            if count < length:
                story = ""
                # Story too short, try again
                continue
            else:
                break
        else:
            story += sen + "\n"

    with open(output_file, 'w') as ofile:
        print(story, file=ofile)
    print("File written")
    
if __name__ == '__main__':
    main()
