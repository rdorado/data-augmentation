
PROB=$1
WORK_DIRECTORY=$HOME"/tmp/topics/twenty_train"$PROB
LDA_DIRECTORY=$HOME"/tmp/topics/data/20newsgroups5"
OUTPUT_DIRECTORY=$HOME"/tmp/topics/data/20newsgroups5"
OUTPUT_FILE="result_t"$2"_k"$3"_p"$PROB".dat"	


# 20 40 200
# 50 50 1000
for n in `seq $2`
do
  for m in `seq $3`
  do
    for j in `seq $4`
    do
      OUTPUT_FILE="result_t"$n"_k"$m"_p"$PROB".dat" 
      if [ ! -d "$WORK_DIRECTORY" ]; then
        mkdir -p $WORK_DIRECTORY
      else
        rm -fr $WORK_DIRECTORY/*
      fi
      echo $n","$m"," >> $OUTPUT_DIRECTORY"/"$OUTPUT_FILE
      python2 separate_data.py -i $WORK_DIRECTORY -p "0."$PROB >> $OUTPUT_DIRECTORY"/"$OUTPUT_FILE
      python2 data_augmentation.py -i $LDA_DIRECTORY"/lda_data_"$n"_"$m".keywords" -t $WORK_DIRECTORY/data/labeled/ -g $WORK_DIRECTORY/lab_twenty_train.dat -u $WORK_DIRECTORY/data/unlabeled/ -e $WORK_DIRECTORY/unlab_twenty_train.dat $5 >> $OUTPUT_DIRECTORY"/"$OUTPUT_FILE
      #python2 selector_clusters.py -i $LDA_DIRECTORY"/lda_data_"$n"_"$m".keywords" -t $WORK_DIRECTORY/data/labeled/ -g $WORK_DIRECTORY/lab_twenty_train.dat -u $WORK_DIRECTORY/data/unlabeled/ -e $WORK_DIRECTORY/unlab_twenty_train.dat >> $OUTPUT_DIRECTORY"/"$OUTPUT_FILE

    done
  done
done

