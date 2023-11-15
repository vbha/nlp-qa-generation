#https://www.rangakrish.com/index.php/2019/02/03/coreference-resolution-using-spacy/
#pronoun resolution using spaCy

import spacy
import neuralcoref
nlp = spacy.load('en_coref_sm')
#neuralcoref.add_to_pipe(nlp)

def printMentions(doc):
    print('All the mentions in the given text:')
    for cluster in doc._.coref_clusters:
        print(cluster.mentions)

def printPronounReferences(doc):
    print('Pronouns and their references:')
    for token in doc:
        if token.pos_ == 'PRON' and token._.in_coref:
            for cluster in token._.coref_clusters:
                print(token.text + '=>' + cluster.main.text)

def processDoc(text):
    doc = nlp(text)
    if doc._.has_coref:
        print("Given text: " + text)
        printMentions(doc)
        printPronounReferences(doc)
processDoc('My sister has a dog and she loves him.')

#using wordnet to get similiar/relevant words
'''from nltk.corpus import wordnet
syn = wordnet.synsets('fire')[0]

print ("Synset name :  ", syn.name()) 
  
print ("\nSynset abstract term :  ", syn.hypernyms()) 
  
print ("\nSynset specific term :  ",  
       syn.hypernyms()[0].hyponyms()) 
  
syn.root_hypernyms() 
  
print ("\nSynset root hypernerm :  ", syn.root_hypernyms())'''


#named entity recognization using 
'''
import spacy
from pprint import pprint
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
doc = nlp('Who is Andrew Carnegie?')
pprint([(X.text, X.label_) for X in doc.ents])
print("")
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])'''