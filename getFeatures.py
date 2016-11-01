import numpy as np
import scipy as sp
import nltk
import sys
import featureExtraction
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle
import topic
import random
import operator
import slangReplace as sr
import svm
#from sklearn.svm import LinearSVC
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier

print "Loading data...."

posData = np.load('posData.npy')
negData = np.load('negData.npy')

print "Total Sarcastic tweets: ", len(posData)
print "Total Non-Sarcastic tweets: ", len(negData)

output = ['Non-Sarcastic', 'Sarcastic']

featureSet = []

select = 10000

print "Randomly selecting ", select, " tweets from each set..."
posData = random.sample(posData,select)#[0:2000]
negData = random.sample(negData,select)#[0:2000]

#topicMod = topic.topic(nbtopic=200,alpha='symmetric')
#topicMod.fit(numpy.concatenate((posData,negData)))
topicMod = []

print "Getting most common words for n-grams feature..."
mostFreq = {}
stop = ['a','an','the','to','of','it','at','be','&','as','on','and','with','our','your','so','in','are','am','me','my','for','i','you','we','this','the','that','those','is','was','has','have','had','will','would','can','could','shall','should','they','them','then','their']
stop = stop + ['.','!','?','"','...','\\',"''",'[',']','~',"'m","'s",';',':','..','$',"'t"]

for each in posData+negData:
	each = each.lower()
	each = sr.repGeneral(each)
	each = sr.repEmoti(each)
	each = nltk.word_tokenize(each)
	for word in each:
		if word not in stop and mostFreq.has_key(word):
			mostFreq[word] += 1
		elif word not in stop:
			mostFreq[word] = 1

sortedWords = sorted(mostFreq.items(), key=operator.itemgetter(1))
sortedWords.reverse()
common = []
for i in range(len(sortedWords)):
	if (sortedWords[i][1] > 3):
		common.append(sortedWords[i][0])

###################

print "Extracting features...."

index = 0
for tweet in posData:
	percent = float(index) / len(posData)
	hashes = '#' * int(round(percent * 20))
	spaces = ' ' * (20 - len(hashes))
	sys.stdout.write("\rPositive data processed: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
	sys.stdout.flush()
	featureSet.append((featureExtraction.modelFeatures(tweet, topicMod, common), output[1]))
	index += 1

print ''

index = 0
for tweet in negData:
	percent = float(index) / len(negData)
	hashes = '#' * int(round(percent * 20))
	spaces = ' ' * (20 - len(hashes))
	sys.stdout.write("\rNegative data processed: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
	sys.stdout.flush()
	featureSet.append((featureExtraction.modelFeatures(tweet, topicMod, common), output[0]))
	index += 1

print ''

featureSet = np.array(featureSet)

targets = (featureSet[0::,1]=='Sarcastic').astype(int)

for i in range(len(targets)):
	if targets[i] == 0:
		targets[i] = -1

vec = DictVectorizer()
featureVec = vec.fit_transform(featureSet[0::,0])

order = shuffle(range(len(featureSet)))
targets = targets[order]

featureVec = featureVec[order, 0::]
#featureVecNaive = featureVec[order, 0::].toarray().astype(float)
featureVecNaive = featureVec.toarray()#.astype(float)

vocab = vec.get_feature_names()

size = int(len(featureVecNaive) * 0.3)	# 30% will be used for testing

trainVec = featureVecNaive[size:]
trainTargets = targets[size:]

testVec = featureVecNaive[:size]
testTargets = targets[:size]

np.save('trainData', trainVec)
np.save('trainLabel', trainTargets)
np.save('testData', testVec)
np.save('testLabel', testTargets)
"""

featList = []
index = 0
for each in featureVecNaive:
	featDict = {}
	for i in range(0,len(vocab)):
		featDict[vocab[i]] = featureVecNaive[0][i]
	
	featList.append((featDict,targets[index]))
	index += 1

print featList[0]

size = int(len(featureSet) * 0.3)	# 30% will be used for testing

trainVec = featList[size:]
trainTargets = targets[size:]

testVec = featList[:size]
testTargets = targets[:size]

print "Naive Bayes Classifier"
classifier = nltk.NaiveBayesClassifier.train(trainVec)
classifier.classify()
print 'Accuracy Naive Bayes Classifier: ', nltk.classify.accuracy(classifier, testVec)*100,' %'
#print ' '
#print classifier.show_most_informative_features()
"""

"""
print "SVM Classifier"
trainVec = featureVec[size:,0::]
trainTargets = targets[size:]

testVec = featureVec[:size,0::]
testTargets = targets[:size]

classifier = LinearSVC(C=0.1,penalty='l2',dual=True)
classifier.fit(trainVec,trainTargets)

output = classifier.predict(testVec)
#print classification_report(test_targets, output, target_names=cls_set)

correct = 0
for i in range(len(output)):
	if output[i] == testTargets[i]:
		correct += 1

print 'Accuracy Linear SVM Classifier: ', float(correct)/len(output)*100,  ' %'

rbfClassifier = svm.SVC(kernel='rbf')
rbfClassifier.fit(trainVec,trainTargets)

output = rbfClassifier.predict(testVec)
#print classification_report(test_targets, output, target_names=cls_set)

correct = 0
for i in range(len(output)):
	if output[i] == testTargets[i]:
		correct += 1

print 'Accuracy kernel SVM Classifier: ', float(correct)/len(output)*100,  ' %'
"""







