
WORK_DIRECTORY=$HOME"/tmp/topics/twenty_train05"
LDA_DIRECTORY=$HOME"/twentynews_lda"

if [ ! -d "$WORK_DIRECTORY" ]; then
  mkdir -p $WORK_DIRECTORY
else
  rm -fr $WORK_DIRECTORY/*
fi

python2 20news_preproc.py -i $WORK_DIRECTORY -p 0.05
python2 selector.py -i $LDA_DIRECTORY/lda_data_4_20.keywords -t $WORK_DIRECTORY/data/labeled/ -g $WORK_DIRECTORY/lab_twenty_train.dat -u $WORK_DIRECTORY/data/unlabeled/ -e $WORK_DIRECTORY/unlab_twenty_train.dat
python2 selector_clusters.py -i $LDA_DIRECTORY/lda_data_4_20.keywords -t $WORK_DIRECTORY/data/labeled/ -g $WORK_DIRECTORY/lab_twenty_train.dat -u $WORK_DIRECTORY/data/unlabeled/ -e $WORK_DIRECTORY/unlab_twenty_train.dat

#python2 20news_preproc.py -i $WORK_DIRECTORY -p 0.05
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train10/ -p 0.1
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train15/ -p 0.15
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train20/ -p 0.2
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train25/ -p 0.25
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train30/ -p 0.3
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train35/ -p 0.35
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train40/ -p 0.4
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train45/ -p 0.45
#python2 20news_preproc.py -i ~/tmp/topics/twenty_train50/ -p 0.5


#python2 preproc.py -i $WORK_DIRECTORY/data/ -o $WORK_DIRECTORY/data05.dat -v $WORK_DIRECTORY/data05.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train10/data/ -o ~/tmp/topics/twenty_train/data10.dat -v ~/tmp/topics/twenty_train/data10.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train15/data/ -o ~/tmp/topics/twenty_train/data15.dat -v ~/tmp/topics/twenty_train/data15.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train20/data/ -o ~/tmp/topics/twenty_train/data20.dat -v ~/tmp/topics/twenty_train/data20.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train25/data/ -o ~/tmp/topics/twenty_train/data25.dat -v ~/tmp/topics/twenty_train/data25.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train30/data/ -o ~/tmp/topics/twenty_train/data30.dat -v ~/tmp/topics/twenty_train/data30.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train35/data/ -o ~/tmp/topics/twenty_train/data35.dat -v ~/tmp/topics/twenty_train/data35.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train40/data/ -o ~/tmp/topics/twenty_train/data40.dat -v ~/tmp/topics/twenty_train/data40.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train45/data/ -o ~/tmp/topics/twenty_train/data45.dat -v ~/tmp/topics/twenty_train/data45.vocab
#python2 preproc.py -i ~/tmp/topics/twenty_train50/data/ -o ~/tmp/topics/twenty_train/data50.dat -v ~/tmp/topics/twenty_train/data50.vocab


#./lda-c-dist/lda est 0.1 4 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_4g
#./lda-c-dist/lda est 0.1 5 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_5g
#./lda-c-dist/lda est 0.1 6 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_6g
#./lda-c-dist/lda est 0.1 7 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_7g
#./lda-c-dist/lda est 0.1 8 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_8g
#./lda-c-dist/lda est 0.1 9 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_9g
#./lda-c-dist/lda est 0.1 10 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_10g
#./lda-c-dist/lda est 0.1 11 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_11g
#./lda-c-dist/lda est 0.1 12 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_12g
#./lda-c-dist/lda est 0.1 13 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data05.dat random ~/tmp/topics/twenty_train/lda_data05_13g
#./lda-c-dist/lda est 0.1 10 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data10.dat random ~/tmp/topics/twenty_train/data10_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data15.dat random ~/tmp/topics/twenty_train/data15_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data20.dat random ~/tmp/topics/twenty_train/data20_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data25.dat random ~/tmp/topics/twenty_train/data25_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data30.dat random ~/tmp/topics/twenty_train/data30_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data35.dat random ~/tmp/topics/twenty_train/data35_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data40.dat random ~/tmp/topics/twenty_train/data40_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data45.dat random ~/tmp/topics/twenty_train/data45_lda
#./lda-c-dist/lda est 0.1 20 ./lda-c-dist/settings.txt ~/tmp/topics/twenty_train/data50.dat random ~/tmp/topics/twenty_train/data50_lda


#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_4g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_4g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_5g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_5g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_6g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_6g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_7g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_7g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_8g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_8g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_9g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_9g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_10g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_10g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_11g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_11g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_12g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_12g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/lda_data05_13g/final.beta ~/tmp/topics/twenty_train/data05.vocab 20 > ~/tmp/topics/twenty_train/lda_data05_13g.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data10_lda/final.beta ~/tmp/topics/twenty_train/data10.vocab 20 > ~/tmp/topics/twenty_train/lda_data10_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data15_lda/final.beta ~/tmp/topics/twenty_train/data15.vocab 20 > ~/tmp/topics/twenty_train/lda_data15_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data20_lda/final.beta ~/tmp/topics/twenty_train/data20.vocab 20 > ~/tmp/topics/twenty_train/lda_data20_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data25_lda/final.beta ~/tmp/topics/twenty_train/data25.vocab 20 > ~/tmp/topics/twenty_train/lda_data25_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data30_lda/final.beta ~/tmp/topics/twenty_train/data30.vocab 20 > ~/tmp/topics/twenty_train/lda_data30_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data35_lda/final.beta ~/tmp/topics/twenty_train/data35.vocab 20 > ~/tmp/topics/twenty_train/lda_data35_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data40_lda/final.beta ~/tmp/topics/twenty_train/data40.vocab 20 > ~/tmp/topics/twenty_train/lda_data40_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data45_lda/final.beta ~/tmp/topics/twenty_train/data45.vocab 20 > ~/tmp/topics/twenty_train/lda_data45_20.keywords
#./lda-c-dist/topics.py ~/tmp/topics/twenty_train/data50_lda/final.beta ~/tmp/topics/twenty_train/data50.vocab 20 > ~/tmp/topics/twenty_train/lda_data50_20.keywords


#python2 selector.py -i ~/tmp/topics/twenty_train/lda_data05_4g.keywords -t ~/tmp/topics/twenty_train05/data/labeled/ -g ~/tmp/topics/twenty_train05/lab_twenty_train.dat -u ~/tmp/topics/twenty_train05/data/unlabeled/ -e ~/tmp/topics/twenty_train05/unlab_twenty_train.dat -d


#python2 selector_clusters.py -i ~/tmp/topics/twenty_train/lda_data20_20.keywords -t ~/tmp/topics/twenty_train50/data/labeled/ -g ~/tmp/topics/twenty_train50/twenty_train.dat -u ~/tmp/topics/twenty_train50/data/unlabeled/ -d

#for i in {1..5}
#do
#  echo "archivo $i.dat"
#done


# ++++++++++++++++++++++ TEST +++++++++++++++++++++++++
#python2 preproc.py -i ~/tmp/topics/test/data/ -o ~/tmp/topics/test/data.dat -v ~/tmp/topics/test/data.vocab
#./lda-c-dist/lda est 0.1 3 ./lda-c-dist/settings.txt ~/tmp/topics/test/data.dat random ~/tmp/topics/test/data_lda
#./lda-c-dist/topics.py ~/tmp/topics/test/data_lda/final.beta ~/tmp/topics/test/data.vocab 4 > ~/tmp/topics/test/lda_data.keywords
#python2 selector.py -i ~/tmp/topics/test/lda_data.keywords -t ~/tmp/topics/test/data/labeled/ -g ~/tmp/topics/test/labels.dat -u ~/tmp/topics/test/data/unlabeled/ 
#python2 selector_clusters.py -i ~/tmp/topics/test/lda_data.keywords -t ~/tmp/topics/test/data/labeled/ -g ~/tmp/topics/test/labels.dat -u ~/tmp/topics/test/data/unlabeled/



