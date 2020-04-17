import spacy
from spacy import symbols

def printRec(node, depth):
    s = ""
    for i in range(0, depth):
        s = s + "\t"
    print(s + "-----")
    for child in node.lefts:
        printRec(child, depth+1)
    ancestors = list(node.ancestors)
    ancestor = "<start>"
    if len(ancestors) > 0:
        ancestor = ancestors[0].tag_
    print(s + str((ancestor, node.tag_)) + "\t" + node.text)
    for child in node.rights:
        printRec(child, depth+1)
    print(s + "-----")

nlp = spacy.load("en_core_web_sm")
doc = nlp("Shlykov, a hard-working taxi driver and Lyosha, a saxophonist, develop a bizarre love-hate relationship, and despite their prejudices, realize they aren't so different after all.")
print(len(doc))
for sent in doc.sents:
    print(len(sent))
    print(sent)
    root = sent.root
    printRec(root, 0)
    print("--------------------------------------------")
