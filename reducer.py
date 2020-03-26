import spacy
import wikipedia
from wikipedia import WikipediaPage
from helper import getName
from spacy.symbols import PROPN, CCONJ, NOUN, VERB, ADJ
import re

names = dict()
nouns = dict()
verbs = dict()
adjs = dict()
verbose = False
wikiCharsSet = False
wikiChars = None
nameSummaries = dict()
namesToTry = list()

def main():
    nlp = spacy.load("en_core_web_sm")
    with open('test.txt', 'r') as file:
        data = file.read().replace('\n','')

    doc = nlp(data)
    
    if verbose: #
        for sent in doc.sents:
            bigline()
            l = list()
            for token in sent:
                l.append([token.text, token.pos_, token.tag_])
            print(l)

    for sent in doc.sents:
        if verbose: bigline() #
        for token in sent:
            if token.is_alpha:
                s = getName(token, verbose)
                if s == None:
                    if token.pos == PROPN:
                        continue
                    if token.pos == NOUN and not token.is_stop:
                        s = token.lemma_
                        mapIncre(nouns, s)
                    elif token.pos == VERB and not token.is_stop:
                        s = token.lemma_
                        mapIncre(verbs, s)
                    elif token.pos == ADJ and not token.is_stop:
                        s = token.lemma_
                        mapIncre(adjs, s)
                else:
                    if verbose: print("--", s) #
                    flag = True
                    for name in names:
                        if compareNames(s, name):
                            flag = False
                            s = name
                            break
                    if flag:
                        names[s] = s
                        findSummaries(s, wikiCharsSet)
                    mapIncre(nouns, s)

    bigline()
    
    for k in nouns:
        if nouns[k] > 1:
            print(k, nouns[k])

    bigline()

    for k in verbs:
        if verbs[k] > 1:
            print(k, verbs[k])

    bigline()

    for k in adjs:
        if adjs[k] > 1:
            print(k, adjs[k])

    bigline()

    for n in names:
        print(n)

    bigline()

    for n in nameSummaries:
        print(n)
        bigline()
        print(nameSummaries[n])
        bigline()

def findSummaries(s, wikiCharsSet):
    sum = ""
    try:
        wiki = WikipediaPage(s)
        if "characters" in wiki.title:
            if not wikiCharsSet:
                wikiCharsSet = True
                wikiChars = wiki
                catchUpOnSummaries()
            nameSummaries[s] = wiki.section(s)
        else:
            nameSummaries[s] = wiki.summary
    except:
        if wikiCharsSet:
            nameSummaries[s] = wiki.section(s)
        else:
            namesToTry.append(s)
            

def catchUpOnSummaries():
    sum = ""
    for name in namesToTry:
        sum = wikiChars.section(name)
        nameSummaries[name] = sum

def mapIncre(map, key):
    if key in map:
        map[key] += 1
    else:
        map[key] = 1

def compareNames(qName, fullName):
    listQ = qName.split(' ')
    listF = fullName.split(' ')
    if len(listQ) == len(listF):
        for i in range(0,len(listF)):
            if listQ[i] != listF[i]:
                return False
        return True
    elif len(listQ) == 1:
        if listQ[0] == listF[0] or listQ[0] == listF[-1]:
            return True
    elif len(listQ) == 2 and len(listF) == 3:
        if listQ[0] == listF[0] and listQ[-1] == listF[-1]:
            return True
    return False

def bigline():
    print("-------------------------------------------")


main()
