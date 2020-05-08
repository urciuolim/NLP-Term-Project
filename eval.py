import sys
from random import shuffle, randint
sys.path.insert(0, "./Part1/")
from generator1 import main as main1
sys.path.insert(0, "./Part2/")
from generator2 import main as main2

def main():
    params = "<model1_file> <model2_dir> <output_dir>"
    if len(sys.argv)-1 < len(params.split(" ")):
        print(len(sys.argv))
        print(len(params.split(" ")))
        print("Need " + str(len(params.split(" "))-(len(sys.argv)-1)) + " more args")
        print("all params: " + params)
        return

    model1_file = sys.argv[1]
    model2_dir = sys.argv[2]
    output_dir = sys.argv[3]
    answer_file = output_dir + "answers.txt"
    
    if output_dir[-1] != "/":
        output_dir == "/"

    GEN_MAIN = 0
    GEN_NUM = 1
    GEN_INPUT = 2

    generators = [(main1, 1, model1_file), (main2, 2, model2_dir)]

    with open(answer_file, 'w') as afile:
        for i in range(10):
            shuffle(generators)
            length = randint(5, 30)
            print("Iteration:", i)
            for j in range(len(generators)):
                output_path = output_dir + "story" + str(i) + "_" + str(j) + ".txt"
                sys.argv = ["", generators[j][GEN_INPUT], str(length), output_path]
                generators[j][GEN_MAIN]()
                print(output_path, file=afile)
                print("Was actually", generators[j][GEN_NUM], file=afile)
            print("------------------------------------------------", file=afile)
    
if __name__ == '__main__':
    main()
