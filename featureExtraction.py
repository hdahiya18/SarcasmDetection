import numpy
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import slangReplace as sr
from textblob import TextBlob
import string

stemmer = SnowballStemmer("english", ignore_stopwords=True)

def modelFeatures(tweet, topicModeler):
	finalFeatures = {}
	newTweet = re.sub('[!,.?]', '', tweet)
	tweet = newTweet
	#ngrams(finalFeatures, tweet)	# n-grams as features
	posFeat(finalFeatures, tweet)	#pos as features
	capFeat(finalFeatures, tweet)	#capitalization feature
	sentimentScore(finalFeatures, tweet) #senitment analysis
	topicFeature(finalFeatures, tweet, topicModeler)
	
	return finalFeatures



#ngrams features
def ngrams(finalFeatures, tweet):
	newTweet = sr.repGeneral(tweet)
	newTweet = sr.repEmoti(newTweet)

	stop = set(stopwords.words('english') + ['.','!','?','"','...','\\',"''",'[',']','~',"'m","'s",';',':','..','$'])
	tokens = nltk.word_tokenize(newTweet)
	unigrams = []
	for each in tokens:
		if each not in stop:
			unigrams.append(each)

	#unigrams = [each if each not in stop]
	unigrams = [stemmer.stem(tok.lower()) for tok in unigrams]	#stem the words to inlcude in dictionary
	bigrams = nltk.bigrams(unigrams)
	bigrams = [words[0]+' '+words[1] for words in bigrams]	#generate all bigrams
	
	ngrams = unigrams #+ bigrams	#final feature contains both uni and bi grams
	for each in ngrams:
		finalFeatures['contains(%s)' % each] = 1.0	#binary dictionary



#parts of speech features including nn(noun), jj(adjective), vb(verb), rb(adverb) 
def posFeat(finalFeatures, tweet):
	tweet = tweet.lower()
	newTweet = sr.repGeneral(tweet)
	newTweet = sr.repEmoti(newTweet)

	tokens = nltk.word_tokenize(newTweet)
	#tokens = [tok.lower() for tok in nltk.word_tokenize(newTweet)]
	posTags = nltk.pos_tag(tokens)

	posFeatVector = numpy.zeros(4)	# we are taking broadly four pos as mentioned above

	for i in range(len(tokens)):
		pos = posTags[i][1]
		if pos[0:2] == 'NN':
			posFeatVector[0] += 1
		elif pos[0:2] == 'JJ':
			posFeatVector[1] += 1
		elif pos[0:2] == 'VB':
			posFeatVector[2] += 1
		elif pos[0:2] == 'RB':
			posFeatVector[3] += 1

	for i in range(len(posFeatVector)):
		finalFeatures['POS' + str(i+1)] = posFeatVector[i]

#at least 4 tokens starting with a capitalization
def capFeat(finalFeatures, tweet):
    current = 0
    threshold = 3
    #tokens = nltk.word_tokenize(tweet)

    for j in range(len(tweet)):
        current += int(tweet[j].isupper())
    
    finalFeatures['Capitalization'] = int(current>=threshold)
    #finalFeatures['Capitalization'] = int(current)


def sentimentScore(finalFeatures, tweet):
	tweet = tweet.lower()
	newTweet = sr.repGeneral(tweet)
	newTweet = sr.repSenti(newTweet)

	
	#print newTweet
	tokens = nltk.word_tokenize(newTweet)
	#tokens = [each.lower() for each in tokens]
	stop = set(stopwords.words('english') + ['.','!','?','"','...','\\',"''",'[',']','~',"'m","'s",';',':','..','$'])
	finalTok = []
	for each in tokens:
		if each not in stop:
			finalTok.append(each)

	tokens = finalTok
	#sentiment score for overall tweet
	try:
		blob = TextBlob("".join([" "+tk if not tk.startswith("'") and tk not in string.punctuation else tk for tk in tokens]).strip())
		finalFeatures['Blob sentiment'] = blob.sentiment.polarity
		finalFeatures['Blob subjectivity'] = blob.sentiment.subjectivity
	except:
		finalFeatures['Blob sentiment'] = 0.0
		finalFeatures['Blob subjectivity'] = 0.0

	#sentiment score tweet divided into 2 parts
	if len(tokens) == 1:
		tokens += ['.']
	fHalf = tokens[0:len(tokens)/2]
	sHalf = tokens[len(tokens)/2:]

	#print fHalf
	#print sHalf
	try:
		blob = TextBlob("".join([" "+tk if not tk.startswith("'") and tk not in string.punctuation else tk for tk in fHalf]).strip())
		finalFeatures['Blob sentiment 1/2'] = blob.sentiment.polarity
		finalFeatures['Blob subjectivity 1/2'] = blob.sentiment.subjectivity
	except:
		finalFeatures['Blob sentiment 1/2'] = 0.0
		finalFeatures['Blob subjectivity 1/2'] = 0.0
	
	try:
		blob = TextBlob("".join([" "+tk if not tk.startswith("'") and tk not in string.punctuation else tk for tk in sHalf]).strip())
		finalFeatures['Blob sentiment 2/2'] = blob.sentiment.polarity
		finalFeatures['Blob subjectivity 2/2'] = blob.sentiment.subjectivity
	except:
		finalFeatures['Blob sentiment 2/2'] = 0.0
		finalFeatures['Blob subjectivity 2/2'] = 0.0

	finalFeatures['Blob sentiment contrast 2'] = numpy.abs(finalFeatures['Blob sentiment 1/2'] - finalFeatures['Blob sentiment 2/2'])


	#sentiment score when tweet divided into 3 parts
	if len(tokens) == 2:
		tokens += ['.']

	fHalf = tokens[0:len(tokens)/3]
	sHalf = tokens[len(tokens)/3:2*len(tokens)/3]
	tHalf = tokens[2*len(tokens)/3:]
	
	#print fHalf
	#print sHalf
	#print tHalf

	try:
		blob = TextBlob("".join([" "+tk if not tk.startswith("'") and tk not in string.punctuation else tk for tk in fHalf]).strip())
		finalFeatures['Blob sentiment 1/3'] = blob.sentiment.polarity
		finalFeatures['Blob subjectivity 1/3'] = blob.sentiment.subjectivity
	except:
		finalFeatures['Blob sentiment 1/3'] = 0.0
		finalFeatures['Blob subjectivity 1/3'] = 0.0
    
	try:
		blob = TextBlob("".join([" "+tk if not tk.startswith("'") and tk not in string.punctuation else tk for tk in sHalf]).strip())
		finalFeatures['Blob sentiment 2/3'] = blob.sentiment.polarity
		finalFeatures['Blob subjectivity 2/3'] = blob.sentiment.subjectivity
	except:
		finalFeatures['Blob sentiment 2/3'] = 0.0
		finalFeatures['Blob subjectivity 2/3'] = 0.0
    
	try:
		blob = TextBlob("".join([" "+tk if not tk.startswith("'") and tk not in string.punctuation else tk for tk in tHalf]).strip())
		finalFeatures['Blob sentiment 3/3'] = blob.sentiment.polarity
		finalFeatures['Blob subjectivity 3/3'] = blob.sentiment.subjectivity
	except:
		finalFeatures['Blob sentiment 3/3'] = 0.0
		finalFeatures['Blob subjectivity 3/3'] = 0.0

	maxSenti = max([finalFeatures['Blob sentiment 1/3'],finalFeatures['Blob sentiment 2/3'],finalFeatures['Blob sentiment 3/3']])
	minSenti = min([finalFeatures['Blob sentiment 1/3'],finalFeatures['Blob sentiment 2/3'],finalFeatures['Blob sentiment 3/3']])
	finalFeatures['Blob sentiment contrast 31'] = numpy.abs(maxSenti - minSenti)
	finalFeatures['Blob sentiment contrast 32'] = numpy.abs(finalFeatures['Blob sentiment 1/3'] - finalFeatures['Blob sentiment 3/3'])


def topicFeature(finalFeatures,tweet,topicModeler):

	topics = topicModeler.transform(tweet)
	for j in range(len(topics)):
		finalFeatures['Topic :' +str(topics[j][0])] = topics[j][1]



