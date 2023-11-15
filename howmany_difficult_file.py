import spacy
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import English
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.parse import CoreNLPParser
import nltk
import nltk.data
parser = StanfordCoreNLP(r'/mnt/c/users/visha/onedrive/Documents/college/sem5/nlp/stanford-corenlp-full-2018-10-05')


nlp = spacy.load("en_core_web_sm")
s = u"The city features 30 skyscrapers, two inclined railways, a pre-revolutionary fortification, and the Point State Park at the confluence of the rivers."
s2 = u"John walked to the park."
s3 = u"Apple is looking at buying U.K. startup for $1 billion."


def treeToStr(t):
	return " ".join(t.leaves())

const_tree = Tree.fromstring(parser.parse(s))[0]



spacy_nlp = spacy.load('en')
try:
    nertags = parser.ner(s)
except:
    nertags = []
    s1 = spacy_nlp(s) 
    for w in s1:
        nertags.append((str(w), w.ent_type_))



def getPhrase(tree, phrase):
	i = 0
	while i < len(tree) and tree[i].label() != phrase:
	    i += 1
	if i == len(tree):
	    return None
	else:
		return tree[i]

def searchAndRem(const_tree, phraseType):
	phrases = []
	q = [const_tree]
	while (len(q) > 0):
		t = q.pop(0)
		if not t:
			break
		for subtree in t:
			if len(subtree) > 1:
				q.append(subtree)
			if subtree.label() == phraseType:
				phrases.append(subtree)
				t.remove(subtree)
	return const_tree, phrases

def howMany(tree, ner):
	tree1 = tree.copy(deep=True)
	np = getPhrase(tree, "NP")
	vp = getPhrase(tree, "VP")
	(tree2, preps) = searchAndRem(tree1, "PP")
	npInVp = getPhrase(vp, "NP")

	verb = treeToStr(vp[0]).split()[0]
	verb2 = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(verb,'v')
	print(verb2)

	for i in range(len(npInVp)):
		if npInVp[i].label() == "NP":
			numFlag = False
			num = ""
			for j in range(len(npInVp[i])):
				if(npInVp[i][j].label() == "CD"):
					numFlag = True
					num = npInVp[i][j][0]
			if(numFlag):
				base = treeToStr(npInVp[i]).replace(num,"")
				print("How many" + base + " does " + treeToStr(np).lower() + " " + verb2 + "?")

howMany(const_tree, nertags)




def firstword(tree):
    t = tree
    while isinstance(t[0], Tree):
        t = t[0]
    return t

def undoCapital(tree):
    first = firstword(tree)
    if first[0] == 'I' or first.label() == 'NNP' or first.label() == 'NNPS':
        return tree
    first[0] = first[0].lower()
    return tree

#doc = nlp(s2)

'''
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
for chunk in doc.noun_chunks:
	print(chunk.text)

'''

def findPhrase(tree, phrase):
	phrase_tags = ['S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
	if tree.label() not in phrase_tags:
		return []
	all = []
	l = [(tree, None)]
	while len(l) > 0:
		(curr_t, noun) = l.pop(0)
		for t in curr_t:
			if t.label() == phrase:
				all.append(t)
			if t.label() in phrase_tags:
				l.append((t, noun))
	return all


def searchPhrase(const_tree, phraseType):
    phrases = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
    if const_tree.label() not in phrases:
        return []
    phrases = []
    q = [const_tree]
    while (len(q) > 0):
        t = q.pop(0)
        for subtree in t:
            if subtree in phrases:
                q.append(subtree)
            if subtree.label() == phraseType:
                phrases.append(subtree)
    return phrases




#print(searchAndRem(const_tree, "PP")[0])
#print(searchAndRem(const_tree, "VP")[0])

#print(searchPhrase(const_tree, "VP"))


def howMany2(sentence):
	doc = nlp(sentence)
	for chunk in doc.noun_chunks:
		numFlag = False
		for word in chunk:
			if(word.pos_ == "NUM"):
				numFlag = True
		if numFlag:
			print(chunk.text, chunk.root.text, chunk.root.head.text)
			print ("How many " + chunk.root.text)

def who(sentence):
	doc = nlp(sentence)
	for chunk in doc.noun_chunks:
		perFlag = False
		for word in chunk:
			if(word.pos_ == "NUM"):
				numFlag = True
		if numFlag:
			print(chunk.text, chunk.root.text, chunk.root.head.text)
			print ("How many " + chunk.root.text)

#howMany(s)
#print(const_tree.leaves)
		