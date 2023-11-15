import spacy
import nltk
import math
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import cmp_to_key

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def createFreq(doc):
    total = 0
    ret = []
    i = 0
    for sent in doc:
        d={}
        sentence = word_tokenize(sent)
        for word in sentence:
            word = word.lower()
            if word not in stop_words:
                total += 1
                if word in d:
                    d[word] += 1
                else:
                    d[word] = 1
        ret.append({"id": i, "freq": d})
        i += 1
    return (ret, total)

def calc_TF(doc_info, freq_info):
    (freqs, total) = freq_info
    TFscores = []
    for x in freqs:
        id = x["id"]
        d=x["freq"]
        for word in d:
            TFscores.append({"TFscore" : d[word] / doc_info[id]["length"], "word": word, "id": id})
    return TFscores
    
def calc_IDF(doc_info, freq_info):
    (freqs, total) = freq_info
    IDFscores = []
    index = 0
    for x in freqs:
        id = x["id"]
        d=x["freq"]
        for word in d:
            c = sum([word in x["freq"] for x in freqs])
            IDFscores.append({"IDFscore": math.log(len(doc_info)/c), "id": id, "word": word})
        index += 1
    return IDFscores

def compute_combined(TFscores, IDFscores):
    combined = []
    for j in IDFscores:
        for i in TFscores:
            if j["word"] == i["word"] and j["id"] == i["id"]:
                combined.append({"id" : j["id"], "score": j["IDFscore"]*i["TFscore"], "word": j["word"]})
    return combined


nlp = spacy.load("en_core_web_sm")

def readFile(path):
    # This makes a very modest attempt to deal with unicode if present
    with open(path, 'rt', errors='surrogateescape') as f:
        return f.read()

doc = nlp(readFile("testfile.txt"))

text1 = """
I like eating frozen bananas.
It's really not as bad as it sounds.
Do the Tooth Fairy or Santa or the Easter Bunny exist? 
Pittsburgh is a city in the state of Pennsylvania in the United States, and is the county seat of Allegheny County. 
A population of about 301,048 residents live within the city limits, making it the 66th-largest city in the U.S. 
The metropolitan population of 2,324,743 is the largest in both the Ohio Valley and Appalachia, the second-largest in Pennsylvania (behind Philadelphia), and the 27th-largest in the U.S.
Pittsburgh is located in the southwest of the state, at the confluence of the Allegheny, Monongahela, and Ohio rivers. 
Pittsburgh is known both as "the Steel City" for its more than 300 steel-related businesses and as the "City of Bridges" for its 446 bridges. 
The city features 30 skyscrapers, two inclined railways, a pre-revolutionary fortification and the Point State Park at the confluence of the rivers. 
The city developed as a vital link of the Atlantic coast and Midwest, as the mineral-rich Allegheny Mountains made the area coveted by the French and British empires, Virginians, Whiskey Rebels, and Civil War raiders.
Aside from steel, Pittsburgh has led in manufacturing of aluminum, glass, shipbuilding, petroleum, foods, sports, transportation, computing, autos, and electronics.
For part of the 20th century, Pittsburgh was behind only New York City and Chicago in corporate headquarters employment; it had the most U.S. stockholders per capita.
America's 1980s deindustrialization laid off area blue-collar workers and thousands of downtown white-collar workers when the longtime Pittsburgh-based world headquarters moved out. 
This heritage left the area with renowned museums, medical centers, parks, research centers, and a diverse cultural district.
I go to school in Pittsburgh at Carnegie Mellon University; I study computer science in the School of Computer Science, where I learn about computer science.
"""
def clean_string(s):
    clean = re.sub('[^\w\s]', '', s)
    clean = re.sub('[^A-Za-z .-]+', ' ', clean)
    clean = clean.replace('-', '')
    clean = clean.replace('...', '')
    clean = clean.replace('_', '')
    clean = clean.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')
    return clean.strip()

def get_docinfo(doc):
    doc_info = []
    id = 0
    for sent in doc:
        count = 0
        for word in word_tokenize(sent):
            count += 1
        doc_info.append({"id": id, "length": count})
        id+= 1
    return doc_info

def sort_fn(x,y):
    (score, ind) = x
    (score2, ind2) = y
    if score > score2:
        return 1
    elif score == score2:
        return 0
    else:
        return -1

def everything():
    text = sent_tokenize(text1)
    clean = [clean_string(s) for s in text]
    doc_info = get_docinfo(clean)
    #print(doc_info)
    freq_info = createFreq(clean)
    #print(freq_info)

    tf = calc_TF(doc_info, freq_info)
    
    idf = calc_IDF(doc_info, freq_info)

    combed = compute_combined(tf, idf)
    '''for x in combed:
        print(x)'''
    
    average = [(0,0)] * len(clean)
    for entry in combed:
        (score, num) = average[entry['id']]
        score += entry['score']
        num += 1
        average[entry['id']] = (score, num)
    
    average2 = []
    for i in range(len(average)):
        entry = average[i]
        (a,b) = entry
        average2.append((a/b, i))
    
    average2.sort(key=cmp_to_key(sort_fn))
    i = 0
    while i < len(average2) and i < 3:
        (entry, index) = average2[i]
        print(clean[index])
        i+=1
    
everything()



