import numpy
import scipy
import nltk
import sys
import featureExtraction
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.utils import shuffle
import topic
import random

print "Loading data...."

posData = numpy.load('posData.npy')
negData = numpy.load('negData.npy')

print "Total Sarcastic tweets: ", len(posData)
print "Total Non-Sarcastic tweets: ", len(negData)

output = ['Non-Sarcastic', 'Sarcastic']

featureSet = []

select = 5000
print "Randomly selecting ", select, " tweets from each set"
posData = random.sample(posData,select)#[0:2000]
negData = random.sample(negData,select)#[0:2000]

topicMod = topic.topic(nbtopic=200,alpha='symmetric')
topicMod.fit(numpy.concatenate((posData,negData)))
#topicMod = []

print "Extracting features...."

index = 0
for tweet in posData:
	percent = float(index) / len(posData)
	hashes = '#' * int(round(percent * 20))
	spaces = ' ' * (20 - len(hashes))
	sys.stdout.write("\rPositive data processed: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
	sys.stdout.flush()
	featureSet.append((featureExtraction.modelFeatures(tweet, topicMod), output[1]))
	index += 1

print ''

index = 0
for tweet in negData:
	percent = float(index) / len(negData)
	hashes = '#' * int(round(percent * 20))
	spaces = ' ' * (20 - len(hashes))
	sys.stdout.write("\rNegative data processed: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
	sys.stdout.flush()
	featureSet.append((featureExtraction.modelFeatures(tweet, topicMod), output[0]))
	index += 1

print ''

featureSet = numpy.array(featureSet)

targets=(featureSet[0::,1]=='Sarcastic').astype(int)

vec = DictVectorizer()
featureVec = vec.fit_transform(featureSet[0::,0])

order = shuffle(range(len(featureSet)))
targets = targets[order]

featureVec = featureVec[order, 0::]
featureVecNaive = featureVec[order, 0::].toarray().astype(float)
vocab = vec.get_feature_names()


featList = []
index = 0
for each in featureVecNaive:
	featDict = {}
	for i in range(0,len(vocab)):
		featDict[vocab[i]] = featureVecNaive[0][i]
	
	featList.append((featDict,targets[index]))
	index += 1

size = int(len(featureSet) * 0.3)	# 30% will be used for testing

trainVec = featList[size:]
trainTargets = targets[size:]

testVec = featList[:size]
testTargets = targets[:size]

print "Naive Bayes Classifier"
classifier = nltk.NaiveBayesClassifier.train(trainVec)

print 'Accuracy Naive Bayes Classifier: ', nltk.classify.accuracy(classifier, testVec)*100,' %'
#print ' '
#print classifier.show_most_informative_features()

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








