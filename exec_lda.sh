
WORK_DIRECTORY=$HOME"../data/twentynews_lda"

for i in {4..20}
do
  ./lda-c-dist/lda est 0.1 $i ./lda-c-dist/settings.txt "$WORK_DIRECTORY/lda_data.dat" random $WORK_DIRECTORY"/lda_data_"$i
  for j in {10..100}
  do
    ./lda-c-dist/topics.py "$WORK_DIRECTORY/lda_data_$i/final.beta" "$WORK_DIRECTORY/lda_data.vocab" $j > $WORK_DIRECTORY"/lda_data_"$i"_"$j".keywords" 
  done
done


