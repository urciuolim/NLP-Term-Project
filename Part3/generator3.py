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

    print("Writing story with at least", length, "sentences")

    # Generate a story, keeping track of number of lines currently at
    # Relient on the model generating sentences until it reaches an end state
    # If an early end state is reached, try again
    # Will also try again if the model reports that it has reached a dead end state
    while True:
        try:
            model = load_model(model_dir)
            model.enumerate_nes(length)
            count = -1
            lastSentence = START
            story = ""
            bCount = 0

            while True:
                flag = count >= length-1
                sen = model.genSentence(count, flag)
                count += 1
                if sen == END:
                    if count < length:
                        bCount += 1
                        if bCount > 10:
                            raise ValueError("Dead end detected on outer level")
                        count -= 1
                        story = ""
                        # Story too short, try again
                    else:
                        break
                else:
                    story += sen + "\n"
            # End while, story constructed
            break
        except ValueError as e:
            continue

    print(story)

    with open(output_file, 'w') as ofile:
        print(story, file=ofile)
    print("File written")
    
if __name__ == '__main__':
    main()
