import spacy
from spacy.symbols import NOUN, PROPN, CCONJ

names = list()

def main():
	nlp = spacy.load("en_core_web_sm")
	with open('test.txt', 'r') as file:
		data = file.read().replace('\n','')
	
	doc = nlp(data)
	#i = 0
	#for sent in doc.sents:
	#	for token in sent:
	#		print(token.text, token.lemma_, token.pos_, token.tag_)
	#	if (i > 1):
	#		break
	#	i += 1
	#return
	#print("-----------------------------------------")
	map = dict()
	
	for sent in doc.sents:
		for token in sent:
			if (token.is_alpha and not token.is_stop):
				s = func(token)
				if (s != None):
					if (token.pos == PROPN):
						flag = True
						for name in names:
							if (s in name):
								flag = False
								s = name
						if (flag):
							names.append(s)
					if (s in map):
						map[s] += 1
					else:
						map[s] = 1
				else:
					i = -1
					for x in range(0,len(names)):
						if (token.text in names[x]):
							i = x
							break
					if (i > -1):
						if(names[i] in map):
							map[names[i]] += 1
						else:
							map[names[i]] = 1
					else:
						if (token.text in map):
							map[token.text] += 1
						else:
							map[token.text] = 1

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
				break;
			if (t.pos == PROPN):
				name += " " + t.text
		return name
	return None

def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

main()
