from __future__ import unicode_literals
from named_entity import *
from question_gen_helper import *
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
from stanfordnlp.server import CoreNLPClient
import neuralcoref
import sys
import re

nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)
parser = StanfordCoreNLP(r'/Users/mashiau/Downloads/stanford-corenlp-full-2018-10-05')


def wrapper_og(s):
    for phrase in re.findall('"([^"]*)"', s):
        s = s.replace('"{}"'.format(phrase), phrase.replace(' ', '_'))
    doc = nlp(s)
    s = doc._.coref_resolved
    sentences=nltk.sent_tokenize(s)
    trees = []
    for sentence in sentences:
        trees.append((Tree.fromstring(parser.parse(sentence))[0], sentence))
    return trees

def wrapper(s):
    import os
    trees = []
    os.environ["CORENLP_HOME"] = 'stanford-corenlp-full-2018-10-05'
    with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=30000, memory='16G') as client:
        ann = client.annotate(s)
        coref = ann.corefChain
        s = coref_rephrase2(coref, s)
        sentences=nltk.sent_tokenize(s)
        for sentence in sentences:
            ann = client.annotate(sentence)
            constituency_parse = ann.sentence[0].parseTree
            if (constituency_parse.score) > -250:
                trees.append((wrapper_help(constituency_parse), sentence))
        return trees
        
def wrapper_help(tree):
    children = tree.child
    if len(children) == 1 and len(children[0].child) == 0:
        return Tree(tree.value, [children[0].value])
    c = []
    for kid in children:
        c.append(wrapper_help(kid))
    if tree.value == 'ROOT':
        return c[0]
    return Tree(tree.value, c)

def tokenize_text(text):
    token_sen = nltk.sent_tokenize(text)
    word = []
    for i in range(len(token_sen)):
        word.append(nltk.word_tokenize(token_sen[i]))
    return word

def coref_rephrase2(corefChains, text):
    process_text = tokenize_text(text)
    for coref_entity in corefChains:
        proper=""
        for mention in coref_entity.mention:
            sen=mention.sentenceIndex
            st=mention.headIndex
            if mention.mentionType=="PROPER":
                proper=process_text[sen][st]
            if mention.mentionType=="PRONOMINAL" and proper!="":
                process_text[sen][st]=proper

    rephrase = [[' '.join(word) for word in process_text]]
    rerephrase=""
    for sentence in rephrase[0]:
        rerephrase+=sentence+" "
    return rerephrase


def yesno_q(tree1, ner):
    tree = tree1.copy(deep=True)
    start = undoCapital(tree[0], ner)
    noun = getPhrase(tree, 'NP')
    verb = getPhrase(tree, 'VP')
    return yesno_qhelp(start, noun, verb)

def yesno_qhelp(start, noun, verb):
    if(noun is None or verb is None):
        return None
    verbverb = firstword(verb)
    x = " ".join(verbverb.leaves())
    lemmatized = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(x,'v')
    if len(verb) > 1:
        if verb[1].label() == 'VP':
            pp = True
        else:
            for word in verb[1:]:
                if word.label() == 'VP':
                    pp = True
                elif word.label() != 'RB' and word.label() != 'ADVP':
                    break
    auxverb = [u'be', u'have', u'do']
    if verb[0].label() == 'MD' or lemmatized in auxverb:
        if verb[0].label() != "VP":
            verb = verb[1:]
        else:
            verb = verb[0][1:]
        question = ""
        noun_part = ""
        for word in verb:
            if word.label() == "CC" or word.label() == "PP" or word.label() == ",":
                break
            question += (" " + " ".join(word.leaves()))
        for word in noun:
            noun_part += (" " + " ".join(word.leaves()))
        return(verbverb[0].capitalize() +noun_part+ question + "?")
    else:
        do = fix_tenses(verb,verbverb, lemmatized)
        question = ""
        noun_part = ""
        for word in verb:
            if word.label() == "CC" or word.label() == ",":
                break
            if word.label() == 'VP':
                for word2 in word:
                    if word2.label() == "CC" or word.label() == ",":
                        break
                    question += (" " + " ".join(word2.leaves()))
                continue      
            question += (" " + " ".join(word.leaves()))
        for word in noun:
            noun_part += (" " + " ".join(word.leaves()))
        return(do +noun_part+ question + "?")

locationtags = ['ORG', 'LOC', 'GPE']
timetags = ['DATE', 'TIME']
whotags = ['PERSON', 'NORP']

def where_q(tree1, ner):
    tree = tree1.copy(deep=True)
    (tree, pps) = findTYPE_PP_del(tree, ner, locationtags)
    for (p, parentverb) in pps:
        start = undoCapital(tree[0], ner)
        if start.label() == "," or start.label() == ";":
            start = undoCapital(tree[1], ner)
        noun = getPhrase(tree, 'NP')
        if parentverb is None:
            parentverb = getPhrase(tree, 'VP')
        q = yesno_qhelp(start, noun, parentverb)
        if q is None:
            continue
        return("Where " + q[:1].lower() + q[1:])
    return None

def when_q(tree1, ner):
    tree = tree1.copy(deep=True)
    (tree, pps) = findTYPE_PP_del(tree, ner, timetags)
    for (p, parentverb) in pps:
        start = undoCapital(tree[0], ner)
        if start.label() == "," or start.label() == ";":
            start = undoCapital(tree[1],ner)
        noun = getPhrase(tree, 'NP')
        if parentverb is None:
            parentverb = getPhrase(tree, 'VP')
        q = yesno_qhelp(start, noun, parentverb)
        if q is None:
            continue
        return("When " + q[:1].lower() + q[1:])
    return None

phrase_tags = ['S', 'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']

def who_q(tree1, ner):
    tree = tree1.copy(deep=True)
    verb = getPhrase(tree, 'VP')
    noun = getPhrase(tree, 'NP')
    if noun is None or verb is None:
        return None
    
    allNPs = findPhrase(tree, 'NP')
    parent = None
    np_found = None
    is_poss = False
    for (np, p) in allNPs:
        n = np.leaves()
        found = False
        for i in range(len(ner)):
            if ner[i][0] in n and ner[i][1] in whotags:
                np_found = np
                parent = p
                found = True
                break
        if found:
            #check if possessive
            poss = False
            for word in np_found:
                if word.label() == 'POS':
                    poss = True
                    break
            is_poss = poss

    if np_found is None:
        return None
    
    verb_q = ""
    for word in verb:
            if word.label() == ",":
                break
            if word.label() in phrase_tags:
                leave = False
                for wordword in word:
                    if wordword.label() == ",":
                        leave = True
                        break
                    verb_q += (" " + " ".join(wordword.leaves()))
                if leave: break
            else:
                verb_q += (" " + " ".join(word.leaves()))

    if is_poss:
        parent.remove(np_found)
        return 'Whose ' + ' '.join(noun.leaves()) + verb_q + '?'
    else:
        return 'Who' + verb_q + '?'


def why_q(tree1, ner):
    tree = tree1.copy(deep=True)
    reason = None
    for subtree in tree:
        phrase = ' '.join(subtree.leaves())
        if subtree.label() == 'SBAR' and 'because' in phrase:
            tree.remove(subtree)
            reason = subtree
    verb = getPhrase(tree, 'VP')
    if verb is None:
        return None
    for subtree in verb:
        if subtree.label() == 'SBAR' or subtree.label() == 'ADJP':
            phrase = ' '.join(subtree.leaves())
            if 'because' in phrase or 'due to' in phrase or 'as a result of' in phrase:
                verb.remove(subtree)
                reason = subtree
    if reason is None:
        return None
    q = yesno_q(tree, ner)
    if q is None:
        return None
    return "Why " + q[:1].lower() + q[1:]

fakes = ['all', 'almost', 'already', 'also', 'basically', 'further', 'finally', 'generally', 'greatly', 'however', 'initially', 'just', 'later', 'largely', 'longer', 'mostly', 'meanwhile', 'often', 'only', 'perhaps', 'now', 'then', 'typically']
def how_q(tree1, ner):
    tree = tree1.copy(deep=True)
    #verb = getPhrase(tree, 'VP')
    #pdb.set_trace()
    removed_tree = delByPhrase(tree)
    if removed_tree is None:
        (removed_tree, adv) = delAdv(tree)
        if adv in fakes:
            return None
    if removed_tree is None:
        return None
    if removed_tree[0].label() == ',':
        removed_tree.remove(removed_tree[0])
    q = yesno_q(removed_tree, ner)
    if q is None:
        return None
    return "How " + q[:1].lower() + q[1:]


def howMany(tree, ner):
    tree1 = tree.copy(deep=True)
    np = getPhrase(tree, "NP")
    vp = getPhrase(tree, "VP")
    (tree2, preps) = searchAndRem(tree1, "PP")
    npInVp = getPhrase(vp, "NP")
    if npInVp is None:
        return None
    verb = treeToStr(vp[0]).split()[0]
    verb2 = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(verb,'v')
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
                return("How many" + base + " does " + treeToStr(np).lower() + " " + verb2 + "?")
    return None


def readfile(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def filterq(questions, n):
    good = {'yesno': [], 'howmany': [], 'where': [], 'when': [], 'who': [], 'why': [], 'how': [], 'what': []}
    other = {'yesno': [], 'howmany': [], 'where': [], 'when': [], 'who': [], 'why': [], 'how': [], 'what': []}
    for label in questions:
        while len(questions[label]) > 0:
            q = questions[label].pop(0)
            if len(q.split()) > 6 and len(q.split()) < 22:
                good[label].append(q)
            else:
                other[label].append(q)
    finalq = []
    goodlabels = ['yesno', 'howmany', 'where', 'when', 'who', 'why', 'how']
    while len(finalq) < n:
        if len(good['yesno']) + len(good['howmany']) + len(good["where"]) + len(good["when"]) + len(good["who"]) + len(good["why"]) + len(good["how"]) == 0:
            break
        for l in goodlabels:
            if len(good[l]) > 0:
                finalq.append(good[l].pop(0))
    while len(finalq) < n:
        for l in goodlabels:
            if len(other[l]) > 0:
                finalq.append(other[l].pop(0))
    
    while len(finalq) < n:
        finalq.append(finalq[0])
    while len(finalq) > n:
        finalq.pop()
    return finalq

def question_generator(text, n):
    trees = wrapper_og(text)
    questions = {'yesno': [], 'howmany': [], 'where': [], 'when': [], 'who': [], 'why': [], 'how': [], 'what': []}
    for (const_tree, sentence) in trees:
        NERtags = getEntities(sentence)
        
        q = (howMany(const_tree, NERtags))
        if q is not None:
            questions['howmany'].append(q)
        q = (yesno_q(const_tree, NERtags))
        if q is not None:
            questions['yesno'].append(q)
        q = (where_q(const_tree, NERtags))
        if q is not None:
            questions['where'].append(q)
        q = (when_q(const_tree, NERtags))
        if q is not None:
            questions['when'].append(q)
        q = (who_q(const_tree, NERtags))
        if q is not None:
            questions['who'].append(q)
        q = (why_q(const_tree, NERtags))
        if q is not None:
            questions['why'].append(q)
        q = (how_q(const_tree, NERtags))
        if q is not None:
            questions['how'].append(q)
    
    final = filterq(questions, n)
    i = 1
    for q in final:
        print("Q" + str(i) + " " + q.replace('-LRB- ', '(').replace(' -RRB-', ')').replace(" ,", ",").replace( " '", "'").replace(" .", "").replace("  ", " "))
        i += 1
        
if __name__ == '__main__':
    filename = sys.argv[1]
    n = int(sys.argv[2])
    s = open(filename, "rt").read()
    question_generator(s, n)
