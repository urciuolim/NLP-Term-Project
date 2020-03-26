import spacy
from spacy.symbols import PROPN, CCONJ, ADP, NOUN

def func(token):
	if (token.pos != PROPN):
		return token.lemma_
	ancestors = [t for t in token.ancestors]
	if len(ancestors) == 0:
		return token.text
	if ancestors[0].pos != PROPN:
		name = ""
		for t in reversed(list(token.lefts)):
			if (t.pos == CCONJ or t.is_punct):
				break
			if (t.pos == PROPN):
				name = t.text + " " + name
		name += token.text
		for t in token.rights:
			if (t.pos == CCONJ or t.is_punct):
				break
			if (t.pos == PROPN):
				name += " " + t.text
		return name
	return None

def getName(token, verbose):
    if token.pos == PROPN:
        ancestors = [t for t in token.ancestors]
        par = None
        gpar = None
        if len(ancestors) > 0:
            par = ancestors[0]
        if len(ancestors) > 1:
            gpar = ancestors[1]
        if verbose: #
            print("*****", token.text, par.text, par.pos_, par.right_edge.text, par.right_edge.pos_)
            if gpar != None:
                print("**********", gpar.text, gpar.pos_, gpar.right_edge.text, gpar.right_edge.pos_)
        if par.pos == PROPN:
            if par.right_edge == token:
                if verbose: #
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GOT HERE")
                return token.text
            return None
        else: #par.pos != PROPN and gpar.pos != PROPN
            name = ""
            for t in reversed(list(token.lefts)):
                if t.pos == CCONJ or t.is_punct:
                    break
                if t.pos == PROPN:
                    name = t.text + " " + name
            name += token.text
            for t in token.rights:
                if t.pos == CCONJ or t.is_punct:
                    break
                elif t.pos == PROPN:
                    name += " " + t.text
                elif t.pos == ADP:
                    name += " " + t.text
                    for child in t.children:
                        name += " " + child.text
            return name
    elif token.pos == NOUN and token.is_title:
        name = token.text
        flag = False
        for t in token.rights:
            if t.pos == CCONJ or t.is_punct:
                break
            elif t.pos == ADP and t.lemma_ == "of":
                name += " " + t.text
                for child in t.children:
                    s = child.text
                    if child.pos == PROPN:
                        flag = True
                        s = getName(child, verbose)
                    name += " " + s
        if flag:
            return name
    return None
