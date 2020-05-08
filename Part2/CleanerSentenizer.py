import sys
import spacy

def main():

    params = "<corpus_text> <output_file>"
    if len(sys.argv)-1 < len(params.split(" ")):
        print(len(sys.argv))
        print(len(params.split(" ")))
        print("Need " + str(len(params.split(" "))-(len(sys.argv)-1)) + " more args")
        print("all params: " + params)
        return

    corpus_text = sys.argv[1]
    output_file = sys.argv[2]

    nlp = spacy.load("en_core_web_sm")
    
    print("Starting to clean text")

    progressBar(42)
    
    cleanlines = []
    with open(corpus_text, 'r') as in_file:
        for count, line in enumerate(in_file):
            cline = []
            # Movie ID is before summary, both on one line,
            # seperated by a single '\t'
            tmp = line.split()
            cleanlines.append("@BEGIN\t" + tmp[0])
            # Split whole summary by spaces
            for word in tmp[1:]:
                # Replace some strings within the word
                word = replace_func(word)
                # Test for a bad word (markup lang, etc)
                if badwordtest(word):
                    continue
                # Finally test to make sure there is still a word left
                # and if so, add it to cleanline
                if len(word.strip()) > 0:
                    cline.append(word)
            # One last check for null strings or whitespaces (indicating erased words)
            cline = [word.strip() for word in list(filter(None, cline))]
            # Reconstruct the line using each clean word
            cleanlinestring = cline[0]
            for cleanword in cline[1:]:
                cleanlinestring += " " + cleanword
            # Finally append the line
            cleanlines.append(cleanlinestring)
            # Also indicate summary is complete
            cleanlines.append("-\t-")

            # Show progress
            if count % 1000 == 0 and count > 1:
                print(".", end="", flush=True)
    print("")

    print("Lines cleaned, splitting by sentence")

    progressBar(int(len(cleanlines) / 3000) + 1)

    sentences = []
    apos = set()
    for count, line in enumerate(cleanlines):
        if '\t' in line:
            sentences.append(line)
        else:
            doc = nlp(line)
            sentence = ""
            first = True
            for sent in doc.sents:
                for word in sent:
                    if ((word.text[0].isalpha() or word.text[0].isdigit()) and
                        not first and not word.text.lower() == "\'s"):
                        sentence += " "
                    sentence += word.text
                    first = False
                if not word.text[-1] == ',':
                    sentences.append(sentence)
                    sentence = ""
                    first = True
        # Show progress
        if count % 3000 == 0 and count > 1:
            print(".", end="", flush=True)
            
    print("")

    print("Sentences split, printing to file")

    with open(output_file, 'w') as output:
        for s in sentences:
            print(s, file=output)
    
    print("Complete")

def progressBar(num):
    for _ in range(num):
        print("_", end="")
    print("")

def replace_func(word):
    if word == "@":
        word = "at"
        return word
    word = word.replace("([[", "")
    word = word.replace('("', '')
    word = word.replace("[[", "")
    word = word.replace('<>(("', "")
    word = word.replace("&#39;", "\'")
    word = word.replace("&#34;", '')
    word = word.replace("&#60;", "<")
    word = word.replace("&#62;", ">")
    word = word.replace("&#8209;", "-")
    word = word.replace("&mdash;", "-")
    word = word.replace("&shy;", "-")
    word = word.replace("&mdahs;", "-")
    word = word.replace("&ndash;", "-")
    word = word.replace("&nbsp;", " ")
    word = word.replace("&ldquo;", '')
    word = word.replace("&rdquo;", '')
    word = word.replace("&rsquo;", "\'")
    word = word.replace("&lsquo;", "\'")
    word = word.replace("&hellip;", "...")
    word = word.replace("&frac12;", "one half")
    word = word.replace("#", "")
    word = word.replace("*", "")
    word = word.replace("+", "")
    word = word.replace("\n", "")
    word = word.replace("~", "")
    word = word.replace("`", "\'")
    word = word.replace('"', "")
    word = word.replace("(", "")
    word = word.replace("-", " ")
    word = word.replace(";", ".")
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
    word = word.replace("n\'t", " not")
    word = word.replace("\'ll", " will")
    word = word.replace("\'ve", " have")
    word = word.replace("\'re", " are")
    word = word.replace("\'d", " would")
    word = word.replace("ol\'", "old")
    word = word.replace("\'em", " them")
    word = word.replace("Goin\'", "Going")
    word = word.replace("goin\'", "going")
    word = word.replace("I\'VE", "I HAVE")
    word = word.replace("lovin\'", "loving")
    word = word.replace("...", ",")
    if ":" in word:
        if word[0].isupper():
            word = ""
        else:
            word = word.replace(":", ",")
    if "%." in word or "%," in word:
        word.replace("%", " percent")
    return word

def badwordtest(word):
    if len(word) > len("Antidisestablishmentarianism"):
        return True
    elif word == "":
        return True
    elif '@' in word:
        return True
    elif '#' in word:
        return True
    elif '%' in word:
        return True
    elif '*' in word:
        return True
    elif '_' in word:
        return True
    elif '+' in word:
        return True
    elif '=' in word:
        return True
    elif '[' in word:
        return True
    elif ']' in word:
        return True
    elif '{' in word:
        return True
    elif '}' in word:
        return True
    elif '\\' in word:
        return True
    elif '|' in word:
        return True
    elif '<' in word:
        return True
    elif '>' in word:
        return True
    elif "--" in word:
        return True
    else:
        return False

if __name__ == '__main__':
    main()
