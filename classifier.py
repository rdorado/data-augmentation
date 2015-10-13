import sys, os, getopt, io
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import random

from nltk.corpus import brown

def printUsage():
  print 'usage: classifier.py [-i <inputfile>] [-t <targetsfile>]'  

def main(argv):
  try:
    opts, args = getopt.getopt(argv,"i:t:e:u:",["ifile=","ofile="])
  except getopt.GetoptError:
    printUsage()
    sys.exit(2)

  rootdir = ""
  targetfile = ""
  trainrootdir = ""
  testtargetsfile = ""

  for opt, arg in opts:
    if opt == '-i':
      rootdir = arg
    if opt == '-t':
      targetfile = arg
    if opt == '-e':
      trainrootdir = arg
    if opt == '-u':
      testtargetsfile = arg
  
  if rootdir == "":
    printUsage()
    sys.exit(2)

  tokenizer = RegexpTokenizer(r'[a-z]+')
  
  documents = []

  for dirname, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:

      inpfile = os.path.join(dirname,filename)

      with io.open(inpfile, "r", errors='ignore') as fp:
        lines = fp.readlines()
      fp.close()
      document = ""
      for line in lines:
        line = line.lower()
        document = document+" "+line
      
      documents.append(document)

  docs_test = []
  for dirname, dirnames, filenames in os.walk(trainrootdir):
    for filename in filenames:

      inpfile = os.path.join(dirname,filename)
      with io.open(inpfile, "r", errors='ignore') as fp:
        lines = fp.readlines()
      fp.close()
      document = ""
      for line in lines:
        line = line.lower()
        document = document+" "+line
      
      docs_test.append(document)

  targets = []  
  with io.open(targetfile, "r", errors='ignore') as fp:
    lines = fp.readlines()
  
  for val in lines:
    targets.append(int(val))

  train_targets = []
  with io.open(testtargetsfile, "r", errors='ignore') as fp:
    lines = fp.readlines()

  for val in lines:
    train_targets.append(int(val))



   
  categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
  #categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
  twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
  twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
  documents = []
  targets = []
  #documents = twenty_train.data
  #targets = twenty_train.target

  if not os.path.exists(os.path.dirname(os.getenv("HOME")+"/tmp/topics/twenty_train/data/labeled/")): os.makedirs(os.path.dirname(os.getenv("HOME")+"/tmp/topics/twenty_train/data/labeled/"))
  if not os.path.exists(os.path.dirname(os.getenv("HOME")+"/tmp/topics/twenty_train/data/unlabeled/")): os.makedirs(os.path.dirname(os.getenv("HOME")+"/tmp/topics/twenty_train/data/unlabeled/"))

  targetsoutput = open(os.getenv("HOME")+"/tmp/topics/twenty_train/twenty_train.dat","w")

  for i in range(0,len(twenty_train.data)):
    rand = random.random()
    if rand < 0.05:
      documents.append(twenty_train.data[i])
      targets.append(twenty_train.target[i])
      trainoutput = io.open(os.getenv("HOME")+"/tmp/topics/twenty_train/data/labeled/"+str(i)+".dat","w")
      trainoutput.write(twenty_train.data[i]);
      trainoutput.close()
      targetsoutput.write(str(twenty_train.target[i])+"\n")
    else:
      trainoutput = io.open(os.getenv("HOME")+"/tmp/topics/twenty_train/data/unlabeled/"+str(i)+".dat","w")
      trainoutput.write(twenty_train.data[i]);
      trainoutput.close()
     
  docs_test = twenty_test.data
  train_targets = twenty_test.target

  
  targetsoutput.close()
  #print( twenty_train.data[1] )


  count_vect = CountVectorizer()

  X_train_counts = count_vect.fit_transform(documents)
  tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  X_train_tf = tf_transformer.transform(X_train_counts)

  print( X_train_tf.shape )
  print( len(targets) )

  clf = MultinomialNB().fit(X_train_tf, targets)

  X_prediction = count_vect.transform(docs_test)
  X_prediction_tfidf = tf_transformer.transform(X_prediction)
  predicted = clf.predict(X_prediction_tfidf)
  
  print( len(train_targets) ) 
  print(str(predicted[0])+" == "+str(train_targets[0]))

  print("Accuracy: "+ str(np.mean(predicted == train_targets)) )

if __name__ == "__main__":
  main(sys.argv[1:])
