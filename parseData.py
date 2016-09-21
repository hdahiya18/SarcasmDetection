import csv	#data as .csv files
import numpy
import re	#regular expressions extraction

def cleanData(dataObj):
	data = csv.reader(dataObj, delimiter = '\n')	#assuming each line is a new tweet

	finalData = []
	hashTags = re.compile(r'#\w+\s?')	#1 or more chars after hashtag and then 0 or 1 space
	friendTags = re.compile(r'@\w+\s?')	#removing tagged persons
	sarcasmWord = re.compile(r'sarcasm', re.I)	#remove the word 'sarcasm'
	rtWord = re.compile(r'RT : ') #remove 'retweet' identifier

	for tweet in data:
		if len(tweet) > 0:
			tweet = tweet[0]	#list item as sentence
			#if tweet has link or uni-code ignore it and if tweet starts with '@' that could mean it's contextual and so ignore it
			if 'http' not in tweet and '\u' not in tweet and (len(tweet)>0 and tweet[0] != '@'):
				tweet = re.sub(hashTags, '', tweet)
				tweet = re.sub(friendTags, '', tweet)
				tweet = re.sub(sarcasmWord, '', tweet)
				tweet = re.sub(rtWord, '', tweet)
				tweet = " ".join(tweet.split())	#removing any multiple spaces
				if (len(tweet.split()) >= 3):	#tweets with atleast 3 words
					finalData = finalData + [tweet]

		
	return finalData


#with open("sarcasticData.txt","r") as sarcDataObj:
with open("smallSar.txt","r") as sarcDataObj:
	sarcData = cleanData(sarcDataObj)

#with open("normalData.txt","r") as normDataObj:
with open("smallNor.txt","r") as normDataObj:
	normData = cleanData(normDataObj)

numpy.save('posData', sarcData)
numpy.save('negData', normData)

print "Number of sarcastic tweets in dataset : " + str(len(sarcData))
print "Number of non-sarcastic tweets in dataset : " + str(len(normData))