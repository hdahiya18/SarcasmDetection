import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sa
import slangReplace as sr
from textblob import TextBlob
import string

stemmer = SnowballStemmer("english", ignore_stopwords=True)


def modelFeatures(tweet, topicModeler, mostFreq):
	finalFeatures = {}
	newTweet = re.sub('[!,.?]', '', tweet)
	tweet = newTweet
	ngrams(finalFeatures, tweet, mostFreq)	# n-grams as features
	posFeat(finalFeatures, tweet)	#pos as features
	capFeat(finalFeatures, tweet)	#capitalization feature
	sentimentScore(finalFeatures, tweet) #senitment analysis
	#topicFeature(finalFeatures, tweet, topicModeler)
	
	return finalFeatures



#ngrams features
def ngrams(finalFeatures, tweet, mostFreq):
	newTweet = sr.repGeneral(tweet)
	newTweet = sr.repEmoti(newTweet)

	#stop = set(stopwords.words('english') + ['.','!','?','"','...','\\',"''",'[',']','~',"'m","'s",';',':','..','$'])
	stop = ['a','an','the','to','of','on','and','with','our','your','so','in','my','for','i','you','we','this','the','that','those','is','was','has','have','had','will','would','can','could','shall','should','they','them','then','their']
	stop = stop + ['.','!','?','"','...','\\',"''",'[',']','~',"'m","'s",';',':','..','$']
	tokens = nltk.word_tokenize(newTweet)
	unigrams = []
	for each in tokens:
		if each not in stop and each in mostFreq:
			unigrams.append(each)

	#unigrams = [each if each not in stop]
	unigrams = [stemmer.stem(tok.lower()) for tok in unigrams]	#stem the words to inlcude in dictionary
	bigrams = nltk.bigrams(unigrams)
	bigrams = [words[0]+' '+words[1] for words in bigrams]	#generate all bigrams
	
	ngrams = unigrams #+ bigrams	#final feature contains both uni and bi grams
	for each in mostFreq:
		if each in ngrams:
			finalFeatures['contains(%s)' % each] = 1.0	#binary dictionary
		else:
			finalFeatures['contains(%s)' % each] = 0.0


#parts of speech features including nn(noun), jj(adjective), vb(verb), rb(adverb) 
def posFeat(finalFeatures, tweet):
	tweet = tweet.lower()
	newTweet = sr.repGeneral(tweet)
	newTweet = sr.repEmoti(newTweet)

	tokens = nltk.word_tokenize(newTweet)
	#tokens = [tok.lower() for tok in nltk.word_tokenize(newTweet)]
	posTags = nltk.pos_tag(tokens)

	posFeatVector = np.zeros(4)	# we are taking broadly four pos as mentioned above

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
	sid = sa()

	
	#print sid.polarity_scores(newTweet)
	#print newTweet
	
	tokens = nltk.word_tokenize(newTweet)
	
	#tokens = [each.lower() for each in tokens]
	stop = set(stopwords.words('english') + ['.','!','?','"','...','\\',"''",'[',']','~',"'m","'s",';',':','..','$'])
	finalTok = []
	for each in tokens:
		if each not in stop:
			finalTok.append(each)

	tokens = finalTok
	try:
		score = sid.polarity_scores(' '.join(tokens))
		finalFeatures['Compound'] = score['compound']
		finalFeatures['Positive'] = score['pos']
		finalFeatures['Negative'] = score['neg']
		finalFeatures['Neutral'] = score['neu']
	except:
		finalFeatures['Compound'] = 0.0
		finalFeatures['Positive'] = 0.0
		finalFeatures['Negative'] = 0.0
		finalFeatures['Neutral'] = 0.0
	
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
		score = sid.polarity_scores(' '.join(fHalf))
		finalFeatures['Compound 1/2'] = score['compound']
		finalFeatures['Positive 1/2'] = score['pos']
		finalFeatures['Negative 1/2'] = score['neg']
		finalFeatures['Neutral 1/2'] = score['neu']
	except:
		finalFeatures['Compound 1/2'] = 0.0
		finalFeatures['Positive 1/2'] = 0.0
		finalFeatures['Negative 1/2'] = 0.0
		finalFeatures['Neutral 1/2'] = 0.0
	
	try:
		score = sid.polarity_scores(' '.join(sHalf))
		finalFeatures['Compound 2/2'] = score['compound']
		finalFeatures['Positive 2/2'] = score['pos']
		finalFeatures['Negative 2/2'] = score['neg']
		finalFeatures['Neutral 2/2'] = score['neu']
	except:
		finalFeatures['Compound 2/2'] = 0.0
		finalFeatures['Positive 2/2'] = 0.0
		finalFeatures['Negative 2/2'] = 0.0
		finalFeatures['Neutral 2/2'] = 0.0

	finalFeatures['Contrast in 2'] = np.abs(finalFeatures['Compound 1/2'] - finalFeatures['Compound 2/2'])

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

	finalFeatures['Blob sentiment contrast 2'] = np.abs(finalFeatures['Blob sentiment 1/2'] - finalFeatures['Blob sentiment 2/2'])


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
		score = sid.polarity_scores(' '.join(fHalf))
		finalFeatures['Compound 1/3'] = score['compound']
		finalFeatures['Positive 1/3'] = score['pos']
		finalFeatures['Negative 1/3'] = score['neg']
		finalFeatures['Neutral 1/3'] = score['neu']
	except:
		finalFeatures['Compound 1/3'] = 0.0
		finalFeatures['Positive 1/3'] = 0.0
		finalFeatures['Negative 1/3'] = 0.0
		finalFeatures['Neutral 1/3'] = 0.0

	try:
		score = sid.polarity_scores(' '.join(sHalf))
		finalFeatures['Compound 2/3'] = score['compound']
		finalFeatures['Positive 2/3'] = score['pos']
		finalFeatures['Negative 2/3'] = score['neg']
		finalFeatures['Neutral 2/3'] = score['neu']
	except:
		finalFeatures['Compound 2/3'] = 0.0
		finalFeatures['Positive 2/3'] = 0.0
		finalFeatures['Negative 2/3'] = 0.0
		finalFeatures['Neutral 2/3'] = 0.0


	try:
		score = sid.polarity_scores(' '.join(tHalf))
		finalFeatures['Compound 3/3'] = score['compound']
		finalFeatures['Positive 3/3'] = score['pos']
		finalFeatures['Negative 3/3'] = score['neg']
		finalFeatures['Neutral 3/3'] = score['neu']
	except:
		finalFeatures['Compound 3/3'] = 0.0
		finalFeatures['Positive 3/3'] = 0.0
		finalFeatures['Negative 3/3'] = 0.0
		finalFeatures['Neutral 3/3'] = 0.0


	maxSenti = max([finalFeatures['Compound 1/3'],finalFeatures['Compound 2/3'],finalFeatures['Compound 3/3']])
	minSenti = min([finalFeatures['Compound 1/3'],finalFeatures['Compound 2/3'],finalFeatures['Compound 3/3']])
	finalFeatures['Contrast in 31'] = np.abs(maxSenti - minSenti)
	finalFeatures['Contrast in 32'] = np.abs(finalFeatures['Compound 1/3'] - finalFeatures['Compound 3/3'])

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
	finalFeatures['Blob sentiment contrast 31'] = np.abs(maxSenti - minSenti)
	finalFeatures['Blob sentiment contrast 32'] = np.abs(finalFeatures['Blob sentiment 1/3'] - finalFeatures['Blob sentiment 3/3'])


def topicFeature(finalFeatures,tweet,topicModeler):

	topics = topicModeler.transform(tweet)
	for j in range(len(topics)):
		finalFeatures['Topic :' +str(topics[j][0])] = topics[j][1]



