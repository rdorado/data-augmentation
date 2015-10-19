import sys, os, getopt, io
import numpy as np
#import rpy2.robjects as ro
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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
  resp = [0 for x in range(len(array))] 
  for i in range(0,len(array)):
    if array[i] > 0: resp[i] = 1
  return resp

def tobinary_matrix(matrix):
  resp = []
  for i in range(len(matrix)):
    row = [0 for x in range(len(matrix[i]))]
    for j in range(len(matrix[i])):
      if matrix[i][j] > 0: row[j] = 1
    resp.append(row) 
  return resp

def removeNans(matrix):
  for array in matrix:
    
    print len(array)
  return matrix  
 
def toR(array):
  resp = ""
  for val in array:
    resp=resp+str(val)+","
  return resp[:-1]

def dotproduct(array1, array2):
  resp = np.zeros(len(array1))
  for i in range(0,len(array1)): 
    resp[i]=array1[i]*array2[i]
  return resp

def logsum(array):
  resp = 0
  for i in range(0,len(array)): 
    resp+=math.log(max(array[i],0.01))
  return resp


def multlog_probs(matrix1, matrix2):
  resp = []
  for i in range(len(matrix1)):
    resp.append( logsum( dotproduct( matrix1[i], matrix2[i] ) ) ) 
  return resp

def multlog(array1, array2):
  if len(array1)!=len(array2): raise Exception("Arrays must be equal in length. arr1:"+str(len(array1))+", arr2:"+str(len(array2)))
  resp = 0
  for i in range(len(array1)):
    resp += math.log( max(array1[i]*array2[i],0.0001) ) 
  return resp
  

def divide(matrix1, matrix2):
  resp = []
  for i in range(len(matrix1)):
    row = []
    for j in range(len(matrix1[i])):
      if matrix1[i][j] == 0 or matrix2[i][j] == 0: row.append(0)
      else: row.append( float(matrix1[i][j]) / matrix2[i][j] )
    resp.append(row)
  return resp  


def nan_to_num(matrix):
  resp = []
  for i in range(len(matrix)):
    row = []   
    for j in range(len(matrix[i])):
      if matrix[i][j]>0 : row.append(matrix[i][j])
      else: row.append(0) 
    resp.append(row)
  return matrix

def sumlog_probs(array):
  maxlog = max(array)
  return maxlog + math.log( sum([math.exp(x - maxlog) for x in array])  )

def sumlog_probs_vect(array1, array2):
  resp = []
  for i in range(len(array1)):
    maxlog = max(array1[i],array2[i])
    resp.append( maxlog + math.log(   math.exp(array1[i]+maxlog) +  math.exp(array2[i]+maxlog)    ) )
  return resp

def sum_matrix(matrix1, matrix2):
  resp = []
  for i in range(len(matrix1)):
    row = []
    for j in range(len(matrix1[i])):
      row.append(matrix1[i][j] + matrix2[i][j])
    resp.append(row)
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
  tokenizer = RegexpTokenizer(r'[a-z]+')
  validatedocs=0
  uclassfile = ""

  for opt, arg in opts:
    if opt == '-i':
      keywordsfile = arg
    elif opt == '-t':
      traindatadir = arg
    elif opt == '-g':
      classfile = arg
    elif opt == '-u':
      unlabeleddatadir = arg
    elif opt == '-d':
      debug = True
    elif opt == '-e':
      uclassfile = arg


  ntopics = 0
  nterms = 0
  vocab = {}                 #Vocabulary
  clusters = []              #keyword clusters
  categoryIDdoc = {}         #Map IDdoc vs category
  nclusters = 0

  categories = []
  counts = [] 


  train_documents = [] 
  train_targets = []
  #load cluster info from file
    
  with io.open(keywordsfile, "r", errors='ignore') as fp:
    lines = fp.readlines()
  fp.close()

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
  nclusters = len(clusters) 

  with io.open(classfile, "r", errors='ignore') as fp:
    lines = fp.readlines()
  fp.close()
  for val in lines:
    key = val[:val.index(",")]
    value =  val[val.index(",")+1:]
    categoryIDdoc[int(key)] = int(value)

  udictValues = {}
  with io.open(uclassfile, "r", errors='ignore') as fp:
    lines = fp.readlines()
  fp.close()
  for val in lines:
    key = val[:val.index(",")]
    value =  val[val.index(",")+1:]
    udictValues[int(key)] = int(value)

  if debug: print "\n**********************************\n  Reading documents:\n**********************************\n"

  iddoc = 0
  ruled_out=0 
  totaldocs=0 
  for dirname, dirnames, filenames in os.walk(traindatadir):
    for filename in filenames:
      inpfile = os.path.join(dirname,filename)
      with io.open(inpfile, "r", errors='ignore') as fp:
        lines = fp.readlines()
      fp.close()
     
      count = [[0 for x in range(nterms)] for x in range(nclusters)]
      doc_counts = 0
      totaldocs+=1

      for line in lines:
        if line.startswith("From:") or line.startswith("Subject:") or line.startswith("Reply-To:") or line.startswith("Organization:") or line.startswith("Lines:") or line.lower().startswith("Nntp-Posting-Host:") or line.startswith("X-Newsreader:") or line.startswith("Distribution:") or line.startswith("Keywords:") or line.startswith("Article-I.D.:") or line.startswith("Supersedes:") or line.startswith("Expires:") or line.startswith("NNTP-Posting-Host:") or line.startswith("Summary:") or line.startswith("Originator:") : continue;
        line = line.lower()
        splits = tokenizer.tokenize(line)
        filtered_words = [word for word in splits if word not in stopwords.words('english')]
        filtered_words = [word for word in filtered_words if len(word) > 2]
        filtered_words = [word for word in filtered_words if word not in ["edu","com","subject","writes","mil"]]
        for word in filtered_words:
           
           try:
    
             id_word = vocab[word]
              
             for i in range(nclusters):
               if word in clusters[i]:
                  count[i][id_word] += 1
                  doc_counts+=1                 
           except:
             pass

      if doc_counts>0:
        #for i in range(len(count)):
        #count = tobinary(count)
        counts.append(count)
        iddoc += 1
  
        doc_cat = int(categoryIDdoc[int(filename[:-4])])
        categories.append( doc_cat )
        train_targets.append( doc_cat ) 
        
        with io.open(inpfile, "r", errors='ignore') as fp:
          train_documents.append(fp.read())   
      else: 
        ruled_out+=1
        if debug : print "  Document "+str(totaldocs)+" ruled out"

  ncat = len( set(categories) )


  if debug : print " "+str(totaldocs)+" documents read"

  if debug: 
    print "\n**********************************\n  Loaded info:\n**********************************\n"
    print "Topics loaded: "+str(ntopics)+"\n"
    print "Number of terms: "+str(nterms)+""


    print "\n**********************************\n  Clusters info:\n**********************************\n"
    for vector in clusters:
      print "  "+str(vector)

   
    print "Categories loaded: "+str(ncat)+"\n"
    print "Dictionary:"
    for val in vocab: print "  "+str(vocab[val])+": "+val
    print "\n**********************************\n  Document counts:\n**********************************\n"
    i=1
    for count in counts: 
      print "Document "+str(i)+", (cat:"+str(categories[i-1])+"):" 
      for j in range(ntopics): 
        print "  Topic "+str(j+1)+":" 
        print "   "+str(count[j])  
      i=i+1
    print "Ruled out documents: "+str(ruled_out)+"\n"
    
  # Finished loading categories and vocabulary
 # if debug: print( "Train docs:" + str(len(train_documents)) )



 # Acquiring parameters of Model 

  #sumscounts =[[0 for x in range(nterms)] for x in range(ntopics)] for x in range(ncat)]	
  sumall = [[0 for x in range(nterms)] for x in range(ntopics)]
  sumtopic = [0 for x in range(ntopics)]
  sumcatterm = [[0 for x in range(ntopics)] for x in range(ncat)]
  sumcats = [0 for x in range(ncat)]
  Ptc = [[0 for x in range(ntopics)] for x in range(ncat)]
  Pc = [0 for x in range(ncat)]

  for i in range(len(categories)-validatedocs):
    #sumscounts[categories[i]] = sum_matrix( sumscounts[categories[i]], counts[i] )
    counts[i] = tobinary_matrix(counts[i])
    sumall = sum_matrix(sumall, counts[i])
    #sumcatterm[categories[i]]

    tmp = [0 for x in range(ntopics)]
    for j in range(ntopics):
      tmp[j] = sum(counts[i][j])

    for j in range(ntopics):
      Ptc[categories[i]][j] += tmp[j]/float(sum(tmp))
    sumcats[categories[i]]+=1

    tmp = [0 for x in range(ntopics)]
    for j in range(ntopics):
      tmp[j] = sum(counts[i][j])
    for j in range(ntopics):
      Ptc[categories[i]][j] += tmp[j]/float(sum(tmp))
    sumcats[categories[i]]+=1

  for i in range(ncat):
    for j in range(ntopics):
      Ptc[i][j] = Ptc[i][j]/sumcats[i]
    Pc = [float(x)/sum(sumcats) for x in sumcats]

  for i in range(ntopics):
    sumtopic[i] = float( sum(sumall[i]) )
 
  Pwt=[[sumall[i][j]/sumtopic[i] for j in range(nterms)] for i in range(ntopics)]

  if debug: 
    print "\n**********************************\n  Model parameters acquisition:\n**********************************\n"
    #print "Sum counts:"
    #for i in range(len(sumscounts)):
    #  print "  Category "+str(i)+":" 
    #  for j in range(len(sumscounts[i])):      
    #    print "    Topic "+str(j)+":"       
    #    print "    "+str(sumscounts[i][j])

    print "\nCount topic, term:"
    for i in range(len(sumall)):      
      print "    Topic "+str(i)+":"       
      print "    "+str(sumall[i])

    print "\nTerm aggregates:"
    for i in range(ntopics):    
      print "    Topic "+str(i)+":"       
      print "    "+str(sumtopic[i])

    print "\np(term|topic):"
    for i in range(len(sumall)):      
      print "    Topic "+str(i)+":"       
      print "    "+str(Pwt[i])

    print "\nCount topic, category:"
    for i in range(len(sumcatterm)):      
      print "    Category "+str(i)+":"       
      print "    "+str(sumcatterm[i])

    print "\np( topic|category):"
    print Ptc

    print "\p(category):"
    print Pc



  '''

  #nterms x ntopics x ncat
 

  probs = [[[0 for x in range(nterms)] for x in range(ntopics)] for x in range(ncat)]
  lambdas = [[0 for x in range(ntopics)] for x in range(ncat)]

  for i in range(0,len(categories)-validatedocs):
    sumscounts[categories[i]] = sum_matrix( sumscounts[categories[i]], counts[i] )
    sumall = sum_matrix(sumall, counts[i])

  for i in range(ncat):
    probs[i] = divide(sumscounts[i], sumall)

  for j in range(0,len(categories)-validatedocs):
    for i in range(len(counts[j])):
      lambdas[categories[j]][i] += sum(counts[j][i])

  for i in range(len(lambdas)):
    sum_lambdas = float(sum(lambdas[i]))
    for j in range(len(lambdas[i])):
      lambdas[i][j] = lambdas[i][j]/sum_lambdas  

  sumTopics = [0 for x in range(ntopics)]
  for i in range(ntopics):
    sumTopics[i] = sum(sumall[i]) 

  pDt = [[float(sumall[j][i])/sumTopics[j] for i in range(nterms)] for j in range(ntopics)]

      for k in range(ntopics):
        probDT = multlog(vector, pDt[k])
  
  print "p(t): "+str(sumTopics)
  print "log p(w|t): "+str(pDt)
   '''



  '''
  if debug: 
    print "\n**********************************\n  Calculating acceptance threshold:\n**********************************\n"

  correct_docs =0
  for i in range(len(categories)-validatedocs):
    best = -10000000
    bestid = -1
    for j in range(ncat):
      logprobs = multlog_probs( counts[i],probs[j] )
      for k in range(len(probs)):
        logprobs[k] = logprobs[k] + lambdas[j][k] 
      prob = sumlog_probs(logprobs)
      if debug: print "log p("+str(j)+"|d) = "+str(prob)
      if best < prob:
        best = prob
        bestid = j
    if bestid == categories[i]: correct_docs+=1
    if debug: print "Predicted: "+(str(bestid)+", truth: "+str(categories[i])+"\n")
  if debug: print "Self predicted accuracy: "+str(float(correct_docs)/(len(categories)-validatedocs))

  if debug: 
    print "\n**********************************\n  Testing:\n**********************************\n"
  '''
  ''' 

  # Model test
  
  for i in range(len(categories)-validatedocs,len(categories)):
    best = -10000000
    bestid = -1
    for j in range(ncat):
      logprobs = multlog_probs( counts[i],probs[j] )
      for k in range(len(probs)):
        logprobs[k] = logprobs[k] + lambdas[j][k] 
      prob = sumlog_probs(logprobs)
      if debug: print "log p("+str(j)+"|d) = "+str(prob)
      if best < prob:
        best = prob
        bestid = j
    if debug: print "Predicted: "+(str(bestid)+", truth: "+str(categories[i])+"\n")
      

#  print(counts[114])
#  print(vocab)   
  '''

  if debug: 
    print "\n**********************************\n  Data augmentation:\n**********************************\n"

  
  # data augmentation
  ucounts = []
  word_counts = []
  ucategories = []
  document_count=0
  for dirname, dirnames, filenames in os.walk(unlabeleddatadir):
    for filename in filenames:
      keyword_count=0
      inpfile = os.path.join(dirname,filename)
      with io.open(inpfile, "r", errors='ignore') as fp:
        lines = fp.readlines()
      fp.close()
      document_count+=1

      word_count = [0 for x in range(nterms)] 
      count = []
      doc_counts = 0
      for i in range(nclusters):
         count.append([0 for x in range(nterms)])
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
            word_count[id_word]+=1
            keyword_count+=1  
          except:
            pass
      
      #if keyword_count==0:
      #  print str(keyword_count)+" keywords found in document "+str(document_count)+", skipping it"
      #  continue
      ucounts.append(count)          
      word_counts.append(word_count)
      with io.open(inpfile, "r", errors='ignore') as fp:
        train_documents.append(fp.read())
      ucategories.append( int(udictValues[int(filename[:-4])]) )
     

  ucorr=0
  predicted = []
  i=0
  for vector in word_counts:
    vector = tobinary(vector)
    best = -10000000
    bestid = -1

    if debug: print "  Example "+str(i+1)+":" 
    logptD = [0 for x in range(ntopics)] 
    for k in range(ntopics):
      if debug: print "   log Prob(Topic="+str(k)+"|Doc):"+str(multlog(vector,Pwt[k]))
      logptD[k] = multlog(vector,Pwt[k])

    for j in range(ncat):
      if debug: print "   Category "+str(j)+":"
      if debug: print "   P(Cat="+str(j)+"): "+str(Pc[j])

      probs = [0 for x in range(ntopics)]
      for k in range(ntopics):
        if debug: print "    log Prob(Topic="+str(k)+"|Cat="+str(j)+"): "+str(Ptc[j][k])+", log="+str(math.log(Ptc[j][k]))
        if debug: print "    log Prob(Topic="+str(k)+"|Cat="+str(j)+") + log Prob(Topic="+str(k)+"|Doc): "+str(logptD[k] + math.log(Ptc[j][k]))
        probs[k] = logptD[k] + math.log(Ptc[j][k])
      if debug: print  "    sumlog Prob(Topic="+str(k)+"|Cat="+str(j)+"): "+str(sumlog_probs(probs))
      prob = sumlog_probs(probs) + math.log(Pc[j])
      #print prob    
        
      #prob2 =  log p(c)  +  sum_log p(t|c) p(D=vector|t)   

      if best < prob:
        best = prob
        bestid = j

    if debug: print "   Result: "+str(bestid)+", Expected: "+str(ucategories[i])+"\n\n"
    if bestid==ucategories[i]: ucorr+=1 

    predicted.append(bestid)
    train_targets.append(bestid)
    i+=1


  if debug: print("Unsupervised Prediction Accuracy: "+ str(ucorr/float(i)) )
  else: print str(ucorr/float(i))+",",


  if debug: 
    print "\n**********************************\n  Classifier train and evaluation:\n**********************************\n"

  if debug: 
    print( "Training classifier with " + str(len(train_documents))+" docs" )
  #print(counts[114])
  #print(vocab)   
  #print("Unsupervised accuracy: "+ str(np.mean(predicted == ucategories)) )

  # Train and test the system 
   
  count_vect = CountVectorizer()
  X_train_counts = count_vect.fit_transform(train_documents)
  tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  X_train_tf = tf_transformer.transform(X_train_counts)

  clf = MultinomialNB().fit(X_train_tf, train_targets)

  categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
  twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
  test_docs = twenty_test.data
  test_targets =  twenty_test.target

  #print(test_docs)
  #print(test_targets)

  X_prediction = count_vect.transform(test_docs)
  X_prediction_tfidf = tf_transformer.transform(X_prediction)
  predicted = clf.predict(X_prediction_tfidf)
  
  if debug: print("NB Accuracy: "+ str(np.mean(predicted == test_targets)) )
  else: print str(np.mean(predicted == test_targets))+",",

  clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tf, train_targets)
  predicted = clf.predict(X_prediction_tfidf)

  if debug: print("SVM Accuracy: "+ str(np.mean(predicted == test_targets)) )
  else: print str(np.mean(predicted == test_targets))+"",


  '''
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
