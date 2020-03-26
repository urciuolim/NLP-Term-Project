import spacy
import wikipedia
from wikipedia import WikipediaPage
from bs4 import BeautifulSoup
from helper import getName
from spacy.symbols import PROPN, CCONJ, NOUN, VERB, ADJ
import inspect

names = dict()
nicknames = dict()
nouns = dict()
verbs = dict()
adjs = dict()
verbose = True
silent = False
wikiFlag = False
wikiChars = None
soupChars = None
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

    count = 0
    sents = [sent for sent in doc.sents]
    tokenBefore = None
    for sent in sents:
        count+=1
        if not silent and count % 10 == 0:
            print("Finished ", count, " sentences...")
        if verbose: bigline() #
        for token in sent:
            if token.is_alpha:
                s = getName(token, verbose, tokenBefore)
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
                        for nick in nicknames:
                            if compareNames(s, nick):
                                flag = False
                                s = nicknames[nick]
                    if flag:
                        names[s] = s
                        findSummaries(s)
                    #mapIncre(nouns, s)
            tokenBefore = token

#------------------------------------
    if verbose:
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
        
        for n in nicknames:
            print(n, " -> ", nicknames[n])
            
        bigline()

        for n in nameSummaries:
            print(n)
            bigline()
            print(nameSummaries[n])
            bigline()
#-------------------------------------

def findSummaries(s):
    global wikiFlag
    global wikiChars
    global nameSummaries
    global soupChars
    sum = ""
    try:
        if wikiFlag:
            s1 = uniqueMatchAgainstLastName(s)
            if s != s1:
                nicknames[s] = s1
            s = s1
        wiki = WikipediaPage(s)
        if "characters" in wiki.title:
            # name brought up a list of characters wiki page
            if not wikiFlag:
                # this is the first time we've found this page
                wikiFlag = True
                wikiChars = wiki
                soupChars = BeautifulSoup(wikiChars.html(), 'html.parser').find_all('span', class_='mw-headline')
                catchUpOnSummaries()
            summary = getSomeText(wiki.section(uniqueMatchAgainstLastName(s)))
            if summary != None:
                # the name is a section, so assign that summary
                nameSummaries[s] = summary
                return
            s2 = uniqueMatchNameAgainstFirstName(s)
            if s2 != s:
                # the name (uniquely) matches to a first name, so grab that summary
                summary = getSomeText(wiki.section(s2))
                if summary != None:
                    nameSummaries[s] = summary
                    return
            s3 = uniqueMatchNameAgainstLastName(s.split(' ')[-1])
            if s3 != s:
                nameSummaries[s] = getSomeText(wiki.section(s3))
            else:
                nameSummaries[s] = None
            return
        else:
            nameSummaries[s] = getSomeText(wiki.summary)
            findNicknames(s, wiki)
            return
    except wikipedia.exceptions.DisambiguationError as e:
        if not wikiFlag:
            namesToTry.append(s)
            return
        if verbose: print("-e1", s) #
        # name matched by last name is causing wiki ambiguation
        # try matching by first name
        s2 = uniqueMatchAgainstFirstName(s)
        if s2 != s:
            if verbose: print("-e2", s2)
            nicknames[s] = s2
            findSummaries(s2)
            return
        # first name did not uniquely match, so try just s's last name
        # and matching against another last name
        s3 = uniqueMatchAgainstLastName(s.split(' ')[-1])
        if s3 != s:
            if verbose: print("-e3", s3) #
            nicknames[s] = s3
            findSummaries(s3)
            return
        # nothing is matching, so just take the first entry on wikipedia
        if verbose: print("-ed", e.options[0]) #
        nicknames[s] = e.options[0]
        findSummaries(e.options[0])
        return
    except:
        if not wikiFlag:
            namesToTry.append(s)
            return
        if verbose: print("-m1", s) #
        summary = getSomeText(wikiChars.section(s))
        if summary != None:
            nameSummaries[s] = summary
            return
        s2 = uniqueMatchAgainstFirstName(s)
        if s2 != s:
            nicknames[s] = s2
            if verbose: print("-m2", s2) #
            summary = getSomeText(wikiChars.section(s2))
            if summary != None:
                nameSummaries[s] = summary
                return
        s3 = uniqueMatchAgainstLastName(s.split(' ')[-1])
        if s3 != s:
            nicknames[s] = s3
            if verbose: print("-m3", s3) #
            summary = getSomeText(wikiChars.section(s3))
            if summary != None:
                nameSummaries[s] = summary
                return
    # Everything failed, description of this must be complex to find
    if verbose: print("None", s)
    nameSummaries[s] = None
            

def findNicknames(s, wiki):
    try:
        soup = BeautifulSoup(wiki.html(), 'html.parser')
        table = soup.find('table', class_='infobox')
        table = soup.find_all('th')
        nicks = None
        for th in table:
            if th.string == "Nickname":
                nicks = th.next_sibling
                break
        if nicks != None:
            f = True
            for br in nicks.find_all('br'):
                if f:
                    f = False
                    nicknames[br.previous_sibling.string] = s
                nicknames[br.next_sibling.string] = s      
    except:
        if verbose <= 1: print("Could not find nicknames for", s)

def getSomeText(s):
    if s == None:
        return None
    limit = 500
    paragraph = ""
    for tok in s.split():
        if len(paragraph) > limit:
            break
        paragraph += tok + " "
    return paragraph
            

def catchUpOnSummaries():
    for s in namesToTry:
        s2 = uniqueMatchAgainstFirstName(s)
        if s2 != s:
            summary = getSomeText(wikiChars.section(s2))
            if summary != None:
                nameSummaries[s] = summary
                return
        s3 = uniqueMatchAgainstLastName(s.split(' ')[-1])
        if s3 != s:
            summary = getSomeText(wikiChars.section(s3))
            if summary != None:
                nameSummaries[s] = summary
                return
        nameSummaries[s] = None
        

def uniqueMatchAgainstLastName(name):
    val = name
    counter = 0
    for tag in soupChars:
        if (compareNames(name, tag.text.split(' ')[-1])):
            val = tag.text
            counter += 1
    if counter <= 1:
        return val
    return name

def uniqueMatchAgainstFirstName(name):
    val = name
    counter = 0
    for tag in soupChars:
        if (compareNames(name, tag.text.split(' ')[0])):
            val = tag.text
            counter += 1
    if counter <= 1:
        return val
    return name

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
