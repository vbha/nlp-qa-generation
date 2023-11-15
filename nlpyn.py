import sys
import nltk
import nltk.data
import spacy

nlp = spacy.load("en_core_web_sm")


def answerYN(q, sentences):
    q=q[:len(q)-1]
    question=nlp(q)
    allKeywords=[]
    for w in question:
        if w.pos_ =="NOUN" or w.pos_=="PROPN":
            allKeywords+=[w.text]
    print(allKeywords)
    
    answer=False
    finalAnswer=False
    flip=False

    for sentence1 in sentences:
        sentence=sentence1[:len(sentence1)-1].lower()
        sentence=sentence.split()
        allKey=True
        keywordCopy=allKeywords+[]
        for keyword in allKeywords:
            key = nltk.stem.wordnet.WordNetLemmatizer().lemmatize(keyword.lower(),'v')
            for sentenceWord in sentence:
                sentWord=nltk.stem.wordnet.WordNetLemmatizer().lemmatize(sentenceWord.lower(),'v')
                if keyword == sentenceWord or key==sentWord or key==sentenceWord:
                    answer=True
                    if keyword in keywordCopy:
                        keywordCopy.remove(keyword)
                if sentenceWord=="not" or sentWord=="not":
                    flip=True
            if keywordCopy!=[]:
                allKey=False
            else:
                allKey=True
        if allKey==True:
            if flip:
                finalAnswer=not answer
            else:
                finalAnswer=answer

    if finalAnswer:
        return "yes"
    else:
        return "no"

L=["Pittsburgh is a city in the state of Pennsylvania in the United States, and is the county seat of Allegheny County.", "A population of about 301,048 residents live within the city limits, making it the 66th-largest city in the U.S.", "The metropolitan population of 2,324,743 is the largest in both the Ohio Valley and Appalachia, the second-largest in Pennsylvania (behind Philadelphia), and the 27th-largest in the U.S. Pittsburgh is located in the southwest of the state, at the confluence of the Allegheny, Monongahela, and Ohio rivers.", "Pittsburgh is known both as the Steel City for its more than 300 steel-related businesses and as the City of Bridges for its 446 bridges.",  "The city features 30 skyscrapers, two inclined railways, a pre-revolutionary fortification and the Point State Park at the confluence of the rivers.", "The city developed as a vital link of the Atlantic coast and Midwest, as the mineral-rich Allegheny Mountains made the area coveted by the French and British empires, Virginians, Whiskey Rebels, and Civil War raiders.Aside from steel, Pittsburgh has led in manufacturing of aluminum, glass, shipbuilding, petroleum, foods, sports, transportation, computing, autos, and electronics.", "For part of the 20th century, Pittsburgh was behind only New York City and Chicago in corporate headquarters employment; it had the most U.S. stockholders per capita.", "America's 1980s deindustrialization laid off area blue-collar workers and thousands of downtown white-collar workers when the longtime Pittsburgh-based world headquarters moved out.", "This heritage left the area with renowned museums, medical centers, parks, research centers, and a diverse cultural district."]

L1=["Pittsburgh is a city in the state of Pennsylvania in the United States, and is the county seat of Allegheny County.", "A population of about 301,048 residents live within the city limits, making it the 66th-largest city in the U.S."]



q="Are there bridges in Pittsburgh?"
q2="Is Pittsburgh in Florida?"
q3="Is Pittsburgh in Pennsylvania?"
q4="Does Pittsburgh have skyscrapers?"

print(answerYN(q,L))
print(answerYN(q2,L))
print(answerYN(q3,L))
print(answerYN(q4, L))

         
