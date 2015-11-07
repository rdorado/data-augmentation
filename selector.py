import sys, os, getopt, io
import numpy as np
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
#from rpy2.robjects.packages import importr

def tostr(num):
  if num<10:
    return " "+str(int(num))
  return str(int(num))

def topicstr(num):
   if len(str(num)) == 1: return "00"+str(num)
   elif len(str(num)) == 2: return "0"+str(num)
   else: return str(num)

def printUsage():
  print 'usage: selector.py [-i <inputfile>] [-t <targetsfile>]'    

def tobinary(array):
  resp = np.zeros(len(array))
  for i in range(0,len(array)):
    if array[i] > 0: resp[i] = 1
  return resp

def removeNans(array):
  for i in range(0,len(array)): 
    if math.isnan(array[i]):
      array[i] = 0
  return array  
 
def toR(array):
  resp = ""
  for val in array:
    resp=resp+str(val)+","
  return resp[:-1]

def dotprodcut(array1, array2):
  resp = np.zeros(len(array1))
  for i in range(0,len(array1)): 
    resp[i]=array1[i]*array2[i]
  return resp

def logsum(array):
  resp = 0
  for i in range(0,len(array)): 
    resp+=math.log(max(array[i],0.001))
  return resp

# **************************
# main
# **************************

def main(argv):
  try:
    opts, args = getopt.getopt(argv,"i:t:g:u:e:d",["ifile=","ofile="])
  except getopt.GetoptError:
    printUsage()
    sys.exit(2)

  debug = False
  keywordsfile = ""
  traindatadir = ""
  unlabeleddatadir = ""
  classfile = ""
  uclassfile = ""
  ntopics = 0
  tokenizer = RegexpTokenizer(r'[a-z]+')
  validatedocs=0

  for opt, arg in opts:
    if opt == '-i':
      keywordsfile = arg
    elif opt == '-t':
      traindatadir = arg
    elif opt == '-g':
      classfile = arg
    elif opt == '-e':
      uclassfile = arg
    elif opt == '-u':
      unlabeleddatadir = arg
    elif opt == '-d':
      debug = True

  train_documents = [] 
  train_targets = []

  #load cluster info from file
  clusters = []  
  with io.open(keywordsfile, "r", errors='ignore') as fp:
    lines = fp.readlines()
  fp.close()

  vocab = {}
  nterms = 0
  cluster = []
  for val in lines:
    val = val.strip(" \n")
    if val == "topic "+topicstr(ntopics):
      ntopics = ntopics+1
    elif val != "":   
      cluster.append( val )
      try:
        id = vocab[val]
      except KeyError:
        vocab[val] = nterms
        nterms = nterms + 1
    elif len(cluster) > 0: 
        clusters.append(cluster)
        cluster = []


  dictValues = {}
  with io.open(classfile, "r", errors='ignore') as fp:
    lines = fp.readlines()
  fp.close()
  for val in lines:
    key = val[:val.index(",")]
    value =  val[val.index(",")+1:]
    dictValues[key] = int(value)

  udictValues = {}
  with io.open(uclassfile, "r", errors='ignore') as fp:
    lines = fp.readlines()
  fp.close()
  for val in lines:
    key = val[:val.index(",")]
    value =  val[val.index(",")+1:]
    udictValues[key] = int(value)

  categories = []
  counts = [] 
  iddoc = 0
  for dirname, dirnames, filenames in os.walk(traindatadir):
    for filename in filenames:
      inpfile = os.path.join(dirname,filename)
      with io.open(inpfile, "r", errors='ignore') as fp:
        lines = fp.readlines()
      fp.close()
      with io.open(inpfile, "r", errors='ignore') as fp:
        train_documents.append(fp.read())

      categories.append( int(dictValues[filename[:-4]]) )
      train_targets.append( int(dictValues[filename[:-4]]) )
      #count = np.(nterms)
      count = np.zeros(nterms)
      for line in lines:
        if line.startswith("From:") or line.startswith("Subject:") or line.startswith("Reply-To:") or line.startswith("Organization:") or line.startswith("Lines:") or line.lower().startswith("Nntp-Posting-Host:") or line.startswith("X-Newsreader:") or line.startswith("Distribution:") or line.startswith("Keywords:") or line.startswith("Article-I.D.:") or line.startswith("Supersedes:") or line.startswith("Expires:") or line.startswith("NNTP-Posting-Host:") or line.startswith("Summary:") or line.startswith("Originator:") : continue;
        line = line.lower()
        splits = tokenizer.tokenize(line)
        filtered_words = [word for word in splits if word not in stopwords.words('english')]
        filtered_words = [word for word in filtered_words if len(word) > 2]
        filtered_words = [word for word in filtered_words if word not in ["edu","com","subject","writes","mil", "subject"]]
        for word in filtered_words:
           
           try:
             id_word = vocab[word]
             count[id_word] += 1
           except:
             pass
      counts.append(count)
      iddoc += 1 
  ncat = len( set(categories) )

  countcat = [0 for x in range(ncat)]
  for i in categories:
    countcat[i]+=1


  if debug: 
    print "\n**********************************\n  Loaded info:\n**********************************\n"
    print "Categories loaded: "+str(ncat)+"\n"
    print "Topics loaded: "+str(ntopics)+"\n"
    print "Dictionary:"
    for val in vocab: print "  "+str(vocab[val])+": "+val
    print "\n**********************************\n  Document counts:\n**********************************\n"
    i=1
    for count in counts: 
      print "Document "+str(i)+", (cat:"+str(categories[i-1])+"):" 
      print "   "+str(count)  
      i=i+1
    print( "Training with " + str(len(train_documents))+" docs" )
  # Finished loading categories and vocabulary



  # PModel training 

  sumscounts = [] 
  sumall = np.zeros(nterms)
  for i in range(ncat):
    sumscounts.append( np.zeros(nterms) )

  logcats = [math.log(x/float(len(categories))) for x in countcat]
  sumcats = [0 for x in range(ncat)]
  for i in range(0,len(categories)-validatedocs):
    sumscounts[categories[i]] += counts[i]
    sumcats[categories[i]]+= sum(counts[i])
    sumall+=counts[i]

  conditionals = []
  for i in range(ncat):
     tmp = [x/float(sumcats[i]) for x in sumscounts[i]]
     conditionals.append(tmp)
  
  probs = []
  for j in range(ncat):
    with np.errstate(invalid='ignore'):
      probs.append( removeNans(sumscounts[j]/sumall) )
    
  if debug: 
    print "\n**********************************\n  Model parameters acquisition:\n**********************************\n"
    print "Sum counts:"
    for i in range(len(sumscounts)):
      print "  Category "+str(i)+":" 
      print "    "+str(sumscounts[i])
    print "\nAggregate vector: "
    print "    "+str(sumall)
    print "\nProbabilities p(c|term):"
    for i in range(ncat):
      print "  "+str(probs[i])


  # Model test

  if debug: 
    print "\n**********************************\n  Testing:\n**********************************\n"
    print "Testing with "+str(validatedocs)+" documents\n"

  for i in range(len(categories)-validatedocs,len(categories)):
    best = -10000000
    bestid = -1
    for j in range(ncat):
      prob = logsum( dotprodcut(tobinary(counts[i]),probs[j]) )
      if best < prob:
        best = prob
        bestid = j
    if debug: 
      print "Document ("+i+"), predicted:"+str(bestid)+", real:"+str(categories[i])


#  print(counts[114])
#  print(vocab)   

  if debug: 
    print "\n**********************************\n  Predicting unsupervised examples:\n**********************************\n"

  # data augmentation
  ucategories = []
  ucounts = []
  for dirname, dirnames, filenames in os.walk(unlabeleddatadir):
    for filename in filenames:
      inpfile = os.path.join(dirname,filename)
      with io.open(inpfile, "r", errors='ignore') as fp:
        lines = fp.readlines()
      fp.close()
      with io.open(inpfile, "r", errors='ignore') as fp:
        train_documents.append(fp.read())
      ucategories.append( int(udictValues[filename[:-4]]) )
      #count = np.(nterms)
      count = np.zeros(nterms)
      for line in lines:
        if line.startswith("From:") or line.startswith("Subject:") or line.startswith("Reply-To:") or line.startswith("Organization:") or line.startswith("Lines:") or line.lower().startswith("Nntp-Posting-Host:") or line.startswith("X-Newsreader:") or line.startswith("Distribution:") or line.startswith("Keywords:") or line.startswith("Article-I.D.:") or line.startswith("Supersedes:") or line.startswith("Expires:") or line.startswith("NNTP-Posting-Host:") or line.startswith("Summary:") or line.startswith("Originator:") : continue;
        line = line.lower()
        splits = tokenizer.tokenize(line)
        filtered_words = [word for word in splits if word not in stopwords.words('english')]
        filtered_words = [word for word in filtered_words if len(word) > 2]
        filtered_words = [word for word in filtered_words if word not in ["edu","com","subject","writes","mil", "subject"]]
        for word in filtered_words:
          try:
             id_word = vocab[word]
             count[id_word] += 1
          except:
             pass
      ucounts.append(count)          


  predicted = []
  i=0
  ucorr=0
  ucorr2=0
  ucorr3=0
  for vector in ucounts:
    best = -10000000
    bestid = -1

    best2 = -10000000
    bestid2 = -1

    best3 = -10000000
    bestid3 = -1
    for j in range(ncat):
      prob = logsum( dotprodcut(tobinary(vector),probs[j]) )
      prob2 = logsum( dotprodcut(tobinary(vector),conditionals[j]) )
      
      if best < prob:
        best = prob
        bestid = j

      if best2 < prob2:
        best2 = prob2
        bestid2 = j

      prob3 = logcats[j] + prob2

      if best3 < prob3:
        best3 = prob3
        bestid3 = j

    if bestid==ucategories[i]: ucorr+=1 
    if bestid2==ucategories[i]: ucorr2+=1 
    if bestid3==ucategories[i]: ucorr3+=1 

    predicted.append(bestid3)
    train_targets.append(bestid3)
    i+=1

  if debug: print("Unsupervised Prediction Accuracy: "+ str(ucorr/float(i)) )
  else: print str(ucorr/float(i))+",",

  if debug: print("Unsupervised Prediction Accuracy: "+ str(ucorr2/float(i)) )
  else: print str(ucorr2/float(i))+",",

  if debug: print("Unsupervised Prediction Accuracy: "+ str(ucorr3/float(i)) )
  else: print str(ucorr3/float(i))+",",


   #Micro measures calculations

  relevant = Counter(ucategories) 
  retrieved = Counter(predicted)

  successful_array = [] 
  for i in range(len(predicted)):
    if predicted[i] == ucategories[i]:
      successful_array.append(predicted[i])

  successful = Counter(successful_array)
  pmacro = 0
  rmacro = 0
  for cat in range(len(categories)):
    if retrieved[cat]!=0 : pmacro += float(successful[cat])/retrieved[cat]
    if relevant[cat]!=0 : rmacro += float(successful[cat])/relevant[cat]

  pmacro = pmacro/len(categories)
  rmacro = rmacro/len(categories)

  print str(pmacro)+",",
  print str(rmacro)+",",
  print str(2*(pmacro*rmacro)/(pmacro+rmacro))+",",






  if debug: 
    print "\n**********************************\n  Classifier train and evaluation:\n**********************************\n"

  if debug: 
    print( "Training classifier with " + str(len(train_documents))+" docs" )
  #print(counts[114])
  #print(vocab)   
  #print("Unsupervised accuracy: "+ str(np.mean(predicted == ucategories)) )

  # Train and test the system 

  #categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
  #twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
  twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
  test_docs = twenty_test.data
  test_targets =  twenty_test.target


   
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(train_documents)
  tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  X_train_tf = tf_transformer.transform(X_train_counts)

  clf = MultinomialNB().fit(X_train_tf, train_targets)

  #print(test_docs)
  #print(test_targets)

  X_prediction = count_vect.transform(test_docs)
  X_prediction_tfidf = tf_transformer.transform(X_prediction)
  predicted = clf.predict(X_prediction_tfidf)
  
  if debug: print("NB Accuracy: "+ str(np.mean(predicted == test_targets)) )
  else: print str(np.mean(predicted == test_targets))+",",

   #Micro measures calculations
  relevant = Counter(test_targets) 
  retrieved = Counter(predicted)

  successful_array = [] 
  for i in range(len(predicted)):
    if predicted[i] == test_targets[i]:
      successful_array.append(predicted[i])

  successful = Counter(successful_array)
  pmacro = 0
  rmacro = 0
  for cat in range(len(categories)):
    if retrieved[cat]!=0 : pmacro += float(successful[cat])/retrieved[cat]
    if relevant[cat]!=0 : rmacro += float(successful[cat])/relevant[cat]

  pmacro = pmacro/len(categories)
  rmacro = rmacro/len(categories)
  print str(pmacro)+",",
  print str(rmacro)+",",
  print str(2*(pmacro*rmacro)/(pmacro+rmacro))+",",



  clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tf, train_targets)
  predicted = clf.predict(X_prediction_tfidf)

  if debug: print("SVM Accuracy: "+ str(np.mean(predicted == test_targets)) )
  else: print str(np.mean(predicted == test_targets))+",",

  #Micro measures calculations
  relevant = Counter(test_targets) 
  retrieved = Counter(predicted)

  successful_array = [] 
  for i in range(len(predicted)):
    if predicted[i] == test_targets[i]:
      successful_array.append(predicted[i])

  successful = Counter(successful_array)
  pmacro = 0
  rmacro = 0
  for cat in range(len(categories)):
    if retrieved[cat]!=0 : pmacro += float(successful[cat])/retrieved[cat]
    if relevant[cat]!=0 : rmacro += float(successful[cat])/relevant[cat]

  pmacro = pmacro/len(categories)
  rmacro = rmacro/len(categories)
  print str(pmacro)+",",
  print str(rmacro)+",",
  print str(2*(pmacro*rmacro)/(pmacro+rmacro))+",",

  '''
  clf = RandomForestClassifier(n_estimators = 100)
  predicted = clf.predict(X_prediction_tfidf)

  if debug: print("Random Forest Accuracy: "+ str(np.mean(predicted == test_targets)) )
  else: print str(np.mean(predicted == test_targets))+",",

  #Micro measures calculations
  relevant = Counter(test_targets) 
  retrieved = Counter(predicted)

  successful_array = [] 
  for i in range(len(predicted)):
    if predicted[i] == test_targets[i]:
      successful_array.append(predicted[i])

  successful = Counter(successful_array)
  pmacro = 0
  rmacro = 0
  for cat in range(len(categories)):
    if retrieved[cat]!=0 : pmacro += float(successful[cat])/retrieved[cat]
    if relevant[cat]!=0 : rmacro += float(successful[cat])/relevant[cat]

  pmacro = pmacro/len(categories)
  rmacro = rmacro/len(categories)
  print str(pmacro)+",",
  print str(rmacro)+",",
  print str(2*(pmacro*rmacro)/(pmacro+rmacro))+",",




  clf = SVC(gamma=2, C=1)
  predicted = clf.predict(X_prediction_tfidf)

  if debug: print("SVM Accuracy: "+ str(np.mean(predicted == test_targets)) )
  else: print str(np.mean(predicted == test_targets))+",",

  #Micro measures calculations
  relevant = Counter(test_targets) 
  retrieved = Counter(predicted)

  successful_array = [] 
  for i in range(len(predicted)):
    if predicted[i] == test_targets[i]:
      successful_array.append(predicted[i])

  successful = Counter(successful_array)
  pmacro = 0
  rmacro = 0
  for cat in range(len(categories)):
    if retrieved[cat]!=0 : pmacro += float(successful[cat])/retrieved[cat]
    if relevant[cat]!=0 : rmacro += float(successful[cat])/relevant[cat]

  pmacro = pmacro/len(categories)
  rmacro = rmacro/len(categories)
  print str(pmacro)+",",
  print str(rmacro)+",",
  print str(2*(pmacro*rmacro)/(pmacro+rmacro))+",",



  counts = []
  transp =[]
  j=0
  for dirname, dirnames, filenames in os.walk(traindatadir):
    for filename in filenames:
      inpfile = os.path.join(dirname,filename)
      with io.open(inpfile, "r", errors='ignore') as fp:
        lines = fp.readlines()
      fp.close()
      linecount = np.zeros(ntopics)
      for line in lines:
        line = line.lower()
        splits = tokenizer.tokenize(line) 
        for word in splits:
          for i in range(0,len(clusters)):
            if word in clusters[i]:
              linecount[i] = linecount[i]+1 
      counts.append( linecount )
#      print(counts) 





  for j in range(0,len(counts[0])):
    transp.append([])

  for i in range(0,len(counts)):
    if categories[i] != 0: continue
    for j in range(0,len(counts[i])):
      transp[j].append(counts[i][j])    
  
  for j in range(0,len(transp)):
    print str(np.std(transp[j]))+" "+str(np.mean(transp[j]))
     



  #print( str(len(counts)) +" "+ str(len(categories)) ) 
  #print( [categories.count(i) for i in set(categories)])

  probs = []
  for i in range(0,ncat):
    probs.append( np.zeros(ndim) ) 

  pdoc = np.zeros(ncat)
  for i in range(0,len(counts)-20):
    probs[categories[i]] = probs[categories[i]] + (counts[i]/sum(counts[i])) 
    pdoc[categories[i]] = pdoc[categories[i]]+1

  for i in range(0,ncat):
    probs[i] = probs[i]/pdoc[i]
    print(probs[i])
    print( sum( probs[i] ) )

  score = 0
  total = 0
  for i in range(len(counts)-20, len(counts)):
    maxp = 0
    best = -1
    for j in range(0,len(probs)):
      pclass = ro.r("dmultinom( c("+toR(counts[i])+"),prob=c("+toR(probs[j])+"),log=F)")
      #print( str(j)+" "+str(pclass[0])+" "+str(math.log(pclass[0])))
      if pclass[0] > maxp:
        maxp = pclass[0] 
        best = j 
    total = total+1
    if best == categories[i]:
      score = score+1   
    #print( str(best)+" "+str(categories[i]))
  print("Score: "+str(score)+", "+str(score/float(total)))
  #print( categories[i] )

  #ro.r("x <- matrix(c(0.86, 0.09, -0.85, 0.87, -0.44, -0.43, -1.1, 0.4, -0.96, 0.17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 10)")
  #ro.r("y <- matrix(c(2.49, 0.83, -0.25, 3.10,  0.87,  0.02, -0.12,1.81, -0.83, 0.43),10)")

  #print(ro.r('solve( t(x) %*% x ) %*% t(x) %*%y'))
  '''


if __name__ == "__main__":
  main(sys.argv[1:])
