import os, os.path
import sys, getopt
import io
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def main(argv):

  try:
    opts, args = getopt.getopt(argv,"o:v:",["ifile=","ofile="])
  except getopt.GetoptError:
    print ' [-o <datafile>] [-v <vocabfile>]'
    sys.exit(2)

  outputfile = "trainging.dat"
  vocabfile = "vocab.txt"


  for opt, arg in opts:
    if opt == '-o':
      outputfile = arg
    elif opt == '-v':
      vocabfile = arg

  tokenizer = RegexpTokenizer(r'[a-z]+')
  id_dict = {}
  nterms = 0;
  wordlist = []

  categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
  twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

  with io.open(outputfile, "wb") as output:
    for i in range(0,len(twenty_train.data)):
        lines = twenty_train.data[i].split('\n')

        fd = {}
        termsdoc=0
        for line in lines:
          if line.startswith("From:") or line.startswith("Subject:") or line.startswith("Reply-To:") or line.startswith("Organization:") or line.startswith("Lines:") or line.lower().startswith("Nntp-Posting-Host:") or line.startswith("X-Newsreader:") or line.startswith("Distribution:") or line.startswith("Keywords:") or line.startswith("Article-I.D.:") or line.startswith("Supersedes:") or line.startswith("Expires:") or line.startswith("NNTP-Posting-Host:") or line.startswith("Summary:") or line.startswith("Originator:") : continue;
          line = line.lower()
          splits = tokenizer.tokenize(line)
          filtered_words = [word for word in splits if word not in stopwords.words('english')]
          filtered_words = [word for word in filtered_words if len(word) > 2]
          filtered_words = [word for word in filtered_words if word not in ["edu","com","subject","writes","mil", "subject"]]
          for word in filtered_words:

            try:
              id = id_dict[word]
            except KeyError:
              id_dict[word] = nterms
              id = nterms
              nterms = nterms+1
              wordlist.append(word)

            try:
              fd[id] = fd[id]+1
            except KeyError:
              fd[id] = 1
              termsdoc = termsdoc+1



        outline = str(termsdoc)         
        for idterm in fd:
          outline = outline+" "+str(idterm)+":"+str(fd[idterm])

        output.write(outline+"\n")
  output.close()
 
  output = open(vocabfile,"w")
  for val in wordlist:
    output.write(str(val)+"\n")
  output.close()


if __name__ == "__main__":
   main(sys.argv[1:])



