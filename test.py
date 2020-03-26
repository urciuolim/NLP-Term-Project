import spacy
import wikipedia
from wikipedia import WikipediaPage
from helper import getName
from spacy.symbols import NOUN, PROPN, CCONJ

names = list()

def main():
  nlp = spacy.load("en_core_web_sm")
  with open('test.txt', 'r') as file:
    data = file.read().replace('\n','')
	
  doc = nlp(data)
  wiki = WikipediaPage("Indiana Jones and the Raiders of the Lost Ark")

  map = dict()
	
  for sent in doc.sents:
    for token in sent:
      if (token.is_alpha and not token.is_stop):
        s = getName(token)
        if (s = None):
          s = token.lemma_
        if (token.pos == PROPN):
		    flag = True
		    for name in names:
			if (s in name):
			    flag = False
			    s = name
                            break
		    if (flag):
			names.append(s)
		if (s in map):
		    map[s] += 1
		else:
                    map[s] = 1

	print("---------------------------------------")

	for k in map:
		if(map[k] > 1):
			print(k, map[k])

	print("---------------------------------------")
	print(names)
	
	print("---------------------------------------")
	
	for sent in doc.sents:
		root = sent.root
		line = ""
		for token in root.lefts:
			s = func(token)
			print(s)
			if(s != None and s in map and map[s] > 1):
				line += " " + s
		line += " " + root.lemma_
		for token in root.rights:
			s = token.lemma_
			if (token.pos == PROPN):
				s = str(token)
			if (s in map and map[s] > 1):
				line += " " + s
		print(line)
	
	


def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

main()
