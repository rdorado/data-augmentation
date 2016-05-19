This code contains the experimentations performed for the paper "Semisupervised Text Classification Using Unsupervised Topic Information" presented at FLAIRS-29. 

The porpose of this code is to execute the following experiment:

1. Obtain the corpus to experiment 20newsgroups.
 The training data should be divided in two subsets: labeled and non-labeled. 
2. Execute an LDA analysis


Execution:

This experiment performs an unsupervised analysis of a corpus and uses the results to augment a dataset. Posteriorly, a classifier is trained 

The training data is composed by a set of labeled examples L, and a set of non labeled examples U.

1. Prepare all the training data (L+U) to perform an LDA analysis. The tool lda-c-dist requires to have the input data in a specific format. To prepare the data you have to execute the script "./corpus_to_lda.py" providing three parameters: -o "output_file", where "output_file" is the file to be used as input of the lda-c-dist, -v "vocab_file" the vocabulary file related to the data, and -c "corpus", the corpus to be used. For the moment, only 20newsgroups is working. 

Example: 

python2 ./corpus_to_lda.py -o ~/tmp/topics/data2/20newsgroups5/lda_data.dat -v ~/tmp/topics/data2/20newsgroups5/lda_data.vocab -c 20newsgroups


2. Perform an LDA analysis using lda-c-dist. We want to execute the LDA analysis several times with different parameters and register the results of the analysis. This is made by a shell script called "exec_lda.sh" that receives several parameters to control de programatic executions. First,  it must receive a working directory, that must contain the result of the previous step, the file "lda_data.dat". Second, a secuence containing the number of topics to execute the lda, for example the secuence "3 2 9" will perform an LDA analysis for 3, 5, 7, and 9 topics. The third parameter, also a secuence, will store the number of keywords in a file, for example "10 5 20" will store 10, 15, and 20 keywords for execution. 

The result of each execution is stored in a file in the working directory with the following structure: "lda_data_"$i"_"$j".keywords", where $i will be replaced by the number of topics and $j will be replaced with the number of kewords to store.

Example:

./exec_lda.sh /home/rdorado/tmp/topics/data/20newsgroups5 "5 5 25" "20 10 100"


3.a Next, it is required to separate the original data into two different datasets: one to be used as labeled data L, and the other part to be used as non-labeled data. The script separate_data.py perform this operation, and also train the supervised systems that are going to be used as a baseline. The parameters for this script are: -i "work_directory", the directory where the files are going to be saved, and -p "rate", the rate of examples to be used as labeled data. The script reports in the standard output the results for the supervised measures: precision, recall, and F1-score.

3.b The experimentation using data augmentation is performed by the "data_augmentation.py" script. This script receives the set of keywords selected from the LDA, the training data labeled and non-labeled, and their respective targets. 

As in point two, it is required to execute this tasks several times programatically to obtain the comparative information. This is performed though a shell script called exec_exp.sh. This scripts execute the previous python scripts and collect the results in a file. This script receives four parameters: 1. the percentage of labeled examples, 2. the sequence of topics to use (already in the result file of the previous step), 3. the number of keywords to use, (already in the result file of the previous step), and 4. the number of time the experiment should be repeated. 

Example:

./exec_exp.sh 045 "5 1 5" "20 1 20" 10


4. Aditionally, to collect the result file, it is required to execute a post-script. This script fixes some errors during the execution and 
collection of the results. This is important to make the graphics and analyse the data in R. This script receives the result of the previous step, a file to store the result of the processing, and third parameter that tells the script the number of rows per example (should be allways 3).

Example:

python2 result_processor.py -i ~/tmp/topics/data/20newsgroups5/result005.dat -o ~/tmp/topics/data/20newsgroups5/result005.table.dat -l 3



Example of a complete experiment:


python2 ./corpus_to_lda.py -o ~/tmp/topics/data/20newsgroups5/lda_data.dat -v ~/tmp/topics/data/20newsgroups5/lda_data.vocab -c 20newsgroups

./exec_lda.sh /home/rdorado/tmp/topics/data/20newsgroups5 "10 5 25" "20 10 100"

./exec_exp.sh 045 "5 1 5" "20 1 20" 10

python2 result_processor.py -i ~/tmp/topics/data/20newsgroups5/result005.dat -o ~/tmp/topics/data/20newsgroups5/result005.table.dat -l 3



