import sys, os, getopt, io
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import random

from nltk.corpus import brown

def printUsage():
  print 'usage: 20news_preproc.py [-i <inputfile>] [-p <prob>]'  

def main(argv):
  try:
    opts, args = getopt.getopt(argv,"i:p:",["ifile=","ofile="])
  except getopt.GetoptError:
    printUsage()
    sys.exit(2)

  targetdir = ""
  prob = -1
  #targetfile = ""
  #trainrootdir = ""
  #testtargetsfile = ""
  debug = False

  for opt, arg in opts:
    if opt == '-i':
      targetdir = arg
    if opt == '-p':
      prob = float(arg)

  
  if targetdir == "" or prob == -1:
    printUsage()
    sys.exit(2)

  tokenizer = RegexpTokenizer(r'[a-z]+')
  
  if debug: print "Rate :"+str(prob)
  else: print str(prob)+",",

   
  categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
  #categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
  twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
  twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
  documents = []
  targets = []
  #documents = twenty_train.data
  #targets = twenty_train.target

  if not os.path.exists(os.path.dirname(targetdir+"/data/labeled/")): os.makedirs(targetdir+"/data/labeled/")
  if not os.path.exists(os.path.dirname(targetdir+"/data/unlabeled/")): os.makedirs(targetdir+"/data/unlabeled/")

  lab_targetsoutput = open(targetdir+"/lab_twenty_train.dat","w")
  unlab_targetsoutput = open(targetdir+"/unlab_twenty_train.dat","w")
  ntraindocs = 0
  for i in range(0,len(twenty_train.data)):
    rand = random.random()
    if rand < prob:
      documents.append(twenty_train.data[i])
      targets.append(twenty_train.target[i])
      trainoutput = io.open(targetdir+"/data/labeled/"+str(i)+".dat","w")
      trainoutput.write(twenty_train.data[i]);
      trainoutput.close()
      lab_targetsoutput.write(str(i)+","+str(twenty_train.target[i])+"\n")
      ntraindocs +=1
    else:
      trainoutput = io.open(targetdir+"/data/unlabeled/"+str(i)+".dat","w")
      trainoutput.write(twenty_train.data[i]);
      unlab_targetsoutput.write(str(i)+","+str(twenty_train.target[i])+"\n")
      trainoutput.close()
     
  docs_test = twenty_test.data
  train_targets = twenty_test.target

  
  lab_targetsoutput.close()
  unlab_targetsoutput.close() 
  #print( twenty_train.data[1] )


  # Train and test the system 

  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(documents)
  tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  X_train_tf = tf_transformer.transform(X_train_counts)

  clf = MultinomialNB().fit(X_train_tf, targets)

  X_prediction = count_vect.transform(docs_test)
  X_prediction_tfidf = tf_transformer.transform(X_prediction)
  predicted = clf.predict(X_prediction_tfidf) 


  if debug: print "Training examples: "+str(ntraindocs)
  else: print str(ntraindocs)+",",

  if debug: print("NB Accuracy: "+ str(np.mean(predicted == train_targets)))
  else: print str(np.mean(predicted == train_targets))+",",

  clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tf, targets)
  predicted = clf.predict(X_prediction_tfidf)

  if debug: print("SVM Accuracy: "+ str(np.mean(predicted == train_targets)))
  else: print str(np.mean(predicted == train_targets))+",",


if __name__ == "__main__":
  main(sys.argv[1:])
