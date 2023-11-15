from __future__ import unicode_literals
from named_entity import *
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import textacy
from nltk.parse import CoreNLPParser
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
import pdb
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import English
import nltk
import nltk.data

nlp = spacy.load("en_core_web_sm")
parser = StanfordCoreNLP(r'/Users/mashiau/Downloads/stanford-corenlp-full-2018-10-05')

def getEntities(text):
    doc=nlp(text)
    l=[(str(X), X.ent_type_) for X in doc]
    return l

#returns the large NP/VP if it exists in the tree, otherwise returns None 
def getPhrase(tree, phrase):
    if tree is None:
        return None
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

def treeToStr(t):
	return " ".join(t.leaves())

phrase_tags = ['SBAR', 'S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']
            
def findPhrase(tree, phrase):
    if tree.label() not in phrase_tags:
        return []
    all = []
    l = [(tree, None)]
    while len(l) > 0:
        (curr_t, noun) = l.pop(0)
        for t in curr_t:
            if t.label() == phrase:
                all.append((t, curr_t))
            if t.label() == phrase:
                l.append((t, noun))
    return all

def findPhrase2(tree, phrase):
    if tree.label() not in phrase_tags:
        return []
    all = []
    l = [(tree, None)]
    while len(l) > 0:
        (curr_t, noun) = l.pop(0)
        for t in curr_t:
            if t.label() == phrase:
                all.append((t, curr_t))
            if t.label() in phrase_tags:
                l.append((t, noun))
    return all

def findPhrase_del(tree, phrase):
    #pdb.set_trace()
    if tree.label() not in phrase_tags:
        return tree, []
    all = []
    l = [(tree, None, False)]
    while(len(l) > 0):
        (curr_t, verb, fromcc) = l.pop(0)
        newcc=False
        for t in curr_t:
            if t.label() == 'CC':
                newcc = True
                break
        if curr_t.label() == "VP" and not fromcc:
            verb = curr_t
        toremove = []
        for t in curr_t:
            if t.label() == phrase:
                all.append((t, verb))
                toremove.append(t)
            elif t.label() in phrase_tags:
                if newcc:
                    l.append((t, t, True))
                else:
                    l.append((t,verb, fromcc))
        for t in toremove:
            curr_t.remove(t)
    return tree, all

locationtags = ['ORG', 'LOC', 'GPE']
timetags = ['DATE', 'TIME']
def findTYPE_PP_del(tree, ner, tags):
    if tree.label() not in phrase_tags:
        return tree, []
    all = []
    l = [(tree, getPhrase(tree, 'VP'))]
    while(len(l) > 0):
        (curr_t, verb) = l.pop(0)
        newcc=False
        for t in curr_t:
            if t.label() == 'CC':
                newcc = True
                break
        toremove = []
        for t in curr_t:
            if t.label() == 'PP' and checkifTYPE(t, ner, tags):
                all.append((t, verb))
                toremove.append(t)
            elif t.label() == 'PP' and ',' in t.leaves():
                toremove.append(t)
                continue
            elif t.label() in phrase_tags:
                if newcc and t.label() == 'VP':
                    l.append((t, t))
                else:
                    l.append((t,verb))
        for t in toremove:
            curr_t.remove(t)
    return tree, all

def checkifTYPE(p, ner, tags):
    noun = findPhrase(p, "NP")
    if len(noun) == 0:
        return False
    #pdb.set_trace()
    noun = noun[0][0].leaves()
    for i in range(len(ner)):
        if ner[i][0] in noun and ner[i][1] in tags:
            return True
    if "DATE" in tags and "age" in p.leaves():
        return True
    

def delByPhrase(tree):
    verb = getPhrase(tree, 'VP')
    byphrase = None
    parent = None
    pps = findPhrase2(tree, 'PP')
    for (pp, p) in pps:
        if firstword(pp)[0].lower() == 'by':
            byphrase = pp
            parent = p
            break
    
    if byphrase is None:
        return None
    #pdb.set_trace()
    for phrase in byphrase:
        if phrase.label() == 'NP':
            if firstword(phrase).label() == 'VBG':
                parent.remove(byphrase)
                return tree
            if getPhrase(phrase, 'PP') is not None:
                parent.remove(byphrase)
                return tree
            else:
                return None
        if phrase.label() == 'VP':
            parent.remove(byphrase)
            return tree
        if phrase.label() == 'S' and phrase[0].label() == 'VP':
            parent.remove(byphrase)
            return tree
    return None
        
def delAdv(tree):
    for subtree in tree:
        if subtree.label() == 'ADVP' and len(subtree) == 1:
            tree.remove(subtree)
            return tree, subtree.leaves()[0]
    verb = getPhrase(tree, 'VP')
    if verb is None:
        return None, None
    for subtree in verb:
        if subtree.label() == 'ADVP' and len(subtree) == 1:
            verb.remove(subtree)
            return tree, subtree.leaves()[0]
    return None, None
        

#returns the first word in the sentence/tree (left most in the tree)
def firstword(tree):
    ptr = tree
    while isinstance(ptr[0], Tree):
        ptr = ptr[0]
    return ptr

#undos the capital letter in the first word of the sentence
def undoCapital(tree, ner):
    first = firstword(tree)
    if first[0] == 'I' or first.label() == 'NNP' or first.label() == 'NNPS':
        return tree
    for (word, label) in ner:
        if word == first[0]:
            if label == '':
                break
            else:
                return tree
    first[0] = first[0].lower()
    return tree

lem = English.Defaults.create_lemmatizer()#Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
#returns the lemma of the first word in the tree 
def lemma(tree):
    verb = firstword(tree)
    return lem(u'' + verb[0], u'VERB')[0]

#inserts/replaced the lemmatized word in the first word of the tree
#returns the corresponding "do" word to the verb for yes/no question generation
def fix_tenses(verb, verbverb, lemmatized):
    ptr = verb
    while isinstance(ptr[0], Tree):
        ptr = ptr[0]
    ptr[0] = lemmatized
    if verbverb.label() == "VBZ":
        return "Does"
    elif verbverb.label() == "VBP":
        return "Do"
    else:
        return "Did"