'''
This tool written in python is a handler to work with the reuters-21578 dataset. It has several functions:

  1. Preprocess the reuters-21578 dataset to adjust it to the desired settings of the experiments.
  2. Select portions of the documents to avoid full data processing in the development state.

This python script loads the reuters data from a tar.gz compressed file.

'''

import gzip
import tarfile
import io
import sys, getopt, os
import random


def getAttribute(string, paramname):
  indx = string.index(paramname)
  indx = indx+len(paramname)+2
  return string[indx:string.index('"',indx)]

def getTopic(string):
  return string[string.index("<D>")+3:string.index("</D>")]
 
def main(argv):
  try:
    opts, args = getopt.getopt(argv,"i:o:t:c:u:p:",["ifile=","ofile="])
  except getopt.GetoptError:
    print 'reuters-handler.py [-i <inputfile>] [-o <outputdirectory>] [-t <ouputtopics>] [-c <train_classification>] [-u <test_classification>] [-p <selrate>]'
    sys.exit(2)

  filename = "../../data/reuters21578.tar.gz"
  topicnamefile = "topics.vocab"
  topicdocfile = "class.dat"
  testdocfile = "test-class.dat"
  dirout = "output"
  prob = 1.0

  for opt, arg in opts:
    if opt == '-i':
      filename = arg
    elif opt == '-o':
      dirout =arg
    elif opt == '-p':
      prob = float(arg)
    elif opt == '-t':
      topicnamefile =arg
    elif opt == '-c':
      topicdocfile = arg
    elif opt == '-u':
      testdocfile = arg

  reading = False
  count = {}
  count["test"]=0
  count["train/labeled"]=0
  count["train/unlabeled"]=0

  topic_dict = {}
  topic_list = []
  ntopics = 0
  topicdocclass = []
  testtopicdocclass = []

  with tarfile.open(filename,'r:gz') as gzipfile:
      for filename in gzipfile:
        if not ".sgm" in filename.name: continue
        textfile = gzipfile.extractfile(filename)
        lines = textfile.readlines()

        for line in lines:
          if "<TOPICS>" in line:
            doctopics = line.count("</D>")
            if doctopics > 0:
              strtopic = getTopic(line)
  
          elif "<REUTERS" in line:
              iddoc = getAttribute(line,"NEWID")  
              classdoc = getAttribute(line,"CGISPLIT")
              if classdoc == "PUBLISHED-TESTSET":
                classdoc = "test"
              else:
                rand = random.random()
                if rand < prob:
                   classdoc = "train/labeled"
                else:      
                   classdoc = "train/unlabeled"
          elif "<BODY>" in line:
            reading = True
            outline = line[line.index("<BODY>")+6:]
          elif "</BODY>" in line and reading:
            reading = False
            if len(outline) > 0 and doctopics==1:
              filename = dirout+"/"+classdoc+"/"+iddoc+".txt"
              count[classdoc] = count[classdoc] + 1
              if not os.path.exists(os.path.dirname(filename)): os.makedirs(os.path.dirname(filename))
              output = open(filename,"w")
              output.write(outline)
              output.close()
 
              try: 
                idtop = topic_dict[strtopic]
              except KeyError:
                  
                topic_dict[strtopic] = ntopics
                idtop = ntopics  
                ntopics = ntopics+1
                topic_list.append(strtopic)

              
              if classdoc=="train/labeled":
                topicdocclass.append(idtop)
              elif classdoc=="test":
                testtopicdocclass.append(idtop)
          elif reading:
            outline = outline+line
            

  print("N docs = "+str(count))
  print("K topics = "+str(len(topic_list)))

  output = open(topicnamefile,"w")
  for var in topic_list:
    output.write(var+"\n")
  output.close()

  output = open(topicdocfile,"w")
  for var in topicdocclass:
    output.write(str(var)+"\n")
  output.close()

  output = open(testdocfile,"w")
  for var in testtopicdocclass:
    output.write(str(var)+"\n")
  output.close()



if __name__ == "__main__":
   main(sys.argv[1:])
