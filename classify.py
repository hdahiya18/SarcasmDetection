import numpy as np
import svm
from sklearn.ensemble import RandomForestClassifier

trainVec = np.load('trainData.npy')
trainTargets = np.load('trainLabel.npy')
testVec = np.load('testData.npy')
testTargets = np.load('testLabel.npy')

#print trainVec[0]
#print trainTargets[0]


print "SVM Classifier:"

classifier = svm.SVM()
classifier.fit(trainVec, trainTargets)

predTargets = classifier.predict(testVec)
#correct = np.sum(predTargets == testTargets)

correct = 0
corrpos = 0
corrneg = 0
totalPos = 0
totalNeg = 0
for i in range(len(predTargets)):
	if predTargets[i] == testTargets[i]:
		correct += 1
		if testTargets[i] == 1:
			corrpos += 1
		else:
			corrneg += 1
	if testTargets[i] == 1:
		totalPos += 1
	else:
		totalNeg += 1
print correct, 'correctly classified out of ', totalPos+totalNeg
print corrpos, 'correctly classified as Sarcastic out of total ', totalPos, ' Sarcastic'
print corrneg, 'correctly classified as Non-Sarcastic out of total ', totalNeg, ' Non-Sarcastic'

print 'SVM Overall Accuracy:', correct/float(len(testTargets)) * 100, '%'


print ''
print "Random Forest Classifier:"


classifier = RandomForestClassifier(n_estimators=10)
classifier = classifier.fit(trainVec, trainTargets)

predTargets = classifier.predict(testVec)
#correct = np.sum(predTargets == testTargets)

correct = 0
corrpos = 0
corrneg = 0
totalPos = 0
totalNeg = 0
for i in range(len(predTargets)):
	if predTargets[i] == testTargets[i]:
		correct += 1
		if testTargets[i] == 1:
			corrpos += 1
		else:
			corrneg += 1
	if testTargets[i] == 1:
		totalPos += 1
	else:
		totalNeg += 1
print correct, 'correctly classified out of ', totalPos+totalNeg
print corrpos, 'correctly classified as Sarcastic out of total ', totalPos, ' Sarcastic'
print corrneg, 'correctly classified as Non-Sarcastic out of total ', totalNeg, ' Non-Sarcastic'

print 'Random Forest Overall Accuracy:', correct/float(len(testTargets)) * 100, '%'























