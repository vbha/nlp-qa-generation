import spacy
import nltk

nlp = spacy.load("en_core_web_sm")

def readFile(path):
    # This makes a very modest attempt to deal with unicode if present
    with open(path, 'rt', errors='surrogateescape') as f:
        return f.read()

doc = nlp(readFile("pittsburgh.txt"))
print(nltk.word_tokenize(doc))
for i in doc.sents:
	print(i)
# for token in doc:
	# print((token.text, token.pos_) )
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            # token.shape_, token.is_alpha, token.is_stop)
def preProcessQuestion(question, nlp):
	token_question = nlp(question)
	pos_tags = []
	for token in token_question:
		pos_tags.append((token.text, token.pos_))
	return pos_tags

# Figure out what kind of question it is 
# PERSON (who)
# PLACE (where)
# TIME (when)
# QUANTITY (how many)
# IDEA (what) how are we gonna deal w like "what time?"
# go off the naiive assumption that the question word will be the first word
# still haven't dealt with posession
def questionType(token_questions):
	if len(token_questions) < 0: return "ERR"
	(word, pos) = token_questions[0]
	word = word.lower() 
	if word == "who":
		q_type = "PERSON"
	elif word == "where":
		q_type = "LOCATION"
	elif word == "when":
		q_type = "TIME"
	elif word == "how":
		(word2, _) = token_questions[1]
		if word2 == "much" or word2 == "many":
			q_type = "QUANTITY" 
	elif word == "what":
		q_type = "IDEA"
	elif word == "is":
		q_type = "BINARY"
	else:
		q_type = "UNK"
	return q_type

#what's left is token_question[1:]

token_question = preProcessQuestion("where is your pittsburgh?", nlp)
print(questionType(token_question))

doc = nlp(readFile("seattle.txt"))
def answerBinary(article, question_tokens):
	num_common_words = dict()
	doc_words = set()
	for sentence in article.sents:
		sentence = set(nltk.word_tokenize(sentence))

		# num_common_words[sent] = 



# answerBinary()

