import re

#general exp
genSlangs = {
	#chat acronyms
	r"\bv\b" : "we",
    r"\br\b" : "are",
    r"\bu\b" : "you",
    r"\bc\b" : "see",
    r"\by\b" : "why",
    r"\bb\b" : "be",
    r"\bda\b" : "the",
    r"\bhaha\b" : "ha",
    r"\bhahaha\b" : "ha",
    #apostrophe opened
    r"\bdon't\b" : "do not",
    r"\bdoesn't\b" : "does not",
    r"\bdidn't\b" : "did not",
    r"\bhasn't\b" : "has not",
    r"\bhaven't\b" : "have not",
    r"\bhadn't\b" : "had not",
    r"\bwon't\b" : "will not",
    r"\bwouldn't\b" : "would not",
    r"\bcan't\b" : "can not",
    r"\bcannot\b" : "can not", 
    r"\bi'll\b" : "i will",
    r"\bwe'll\b" : "we will",
    r"\byou'll\b" : "you will",
    r"\bisn't\b" : "is not",
    r"\bthat's\b" : "that is",
    #common short forms (http://www.webopedia.com/quick_ref/Twitter_Dictionary_Guide.asp)
    r"\bidk\b" : "i do not know",
    r"\btbh\b" : "to be honest",
    r"\bic\b" : "i see",
    r"\bbtw\b" : "by the way",
    r"\blol\b" : "laughing",
    r"\bimo\b" : "in my opinion"
}

#general emoticons

genEmo = {
    #good emotions
    "&lt;3" : " heart ",
    ":D" : " smile ",
    ":d" : " smile ",
    ":dd" : " smile ",
    ":P" : " smile ",
    ":p" : " smile ",
    "8)" : " smile ",
	"8-)" : " smile ",
    ":-)" : " smile ",
    ":)" : " smile ",
    ";)" : " smile ",
    "(-:" : " smile ",
    "(:" : " smile ",
    ":')" : " smile ",
    "xD" : " smile ",
    "XD" : " smile ",
    #bad emotions
    ":/" : " worry ",
    "&gt;" : " angry ",
    ":'(" : " sad ",
    ":-(" : " sad ",
    ":(" : " sad ",
    ":s" : " sad ",
    ":-s" : " sad ",
    "-_-" : " bad ",
    "-.-" : " bad "
}

#emoticons replacement for sentiment analysis
sentiEmo = {
	
	#positive emoticons
	"&lt;3" : " good ",
	":D" : " good ",
	":d" : " good ",
    ":dd" : " good ",
    ":P" : " good ",
    ":p" : " good ",
    "8)" : " good ",
	"8-)" : " good ",
    ":-)" : " good ",
    ":)" : " good ",
    ";)" : " good ",
    "(-:" : " good ",
    "(:" : " good ",
    ":')" : " good ",
    "xD" : " good ",
    "XD" : " good ",
    #positive sentiment exclaimations
    "yay!" : " good ",
    "yay" : " good ",
    "yaay" : " good ",
    "yaaay" : " good ",
    "yaaaay" : " good ",
    "yaaaaay" : " good ",
    "Yay!" : " good ",
    "Yay" : " good ",
    "Yaay" : " good ",
    "Yaaay" : " good ",
    "Yaaaay" : " good ",
    "Yaaaaay" : " good ", 
    #bad emotions
    ":/" : " bad ",
    "&gt;" : " sad ",
    ":'(" : " sad ",
    ":-(" : " sad ",
    ":(" : " sad ",
    ":s" : " bad ",
	":-s" : " bad ",
	"-_-" : " bad ",
    "-.-" : " bad "
}


def repGeneral(tweet):
	newTweet = tweet
	for r, formal in genSlangs.iteritems():
		newTweet = re.sub(r, formal, newTweet)

	return newTweet

def repEmoti(tweet):
	newTweet = tweet
	for key, value in genEmo.iteritems():
		newTweet = newTweet.replace(key, value)

	return newTweet

def repSenti(tweet):
	newTweet = tweet
	for key, value in sentiEmo.iteritems():
		newTweet = newTweet.replace(key, value)

	return newTweet
