import os, os.path
import sys, getopt
import io


def main(argv):

  try:
    opts, args = getopt.getopt(argv,"i:o:l:",["ifile=","ofile="])
  except getopt.GetoptError:
    print 'preproc.py [-i <inputfile>] [-o <datafile>]'
    sys.exit(2)

  inputfile = ""
  outputfile = ""
  task = "fix"
  LINES = 4

  for opt, arg in opts:
    if opt == '-i':
      inputfile = arg
    elif opt == '-o':
      outputfile = arg
    elif opt == '-l':
      LINES = int(arg)

  with open(inputfile, "r") as fp:
      lines = fp.readlines()
  fp.close()
  
  i=1
  with io.open(outputfile, "wb") as output:
    
    for line in lines:
      line = line.strip("\n")
      #print "'"+line+"'"
      if i==LINES: 
        output.write(line+"\n")
        i=0 
      else:
        output.write(line)  
      i+=1
  output.close()

if __name__ == "__main__":
  main(sys.argv[1:])
