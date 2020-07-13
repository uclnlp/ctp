head -n 1 1.2,1.3_train.csv > head.csv

cat 1.2,1.3_train.csv | tail -n 10094 | shuf --random-source=/dev/zero > shuf.csv

cat head.csv > dev.csv
cat head.csv > train.csv

cat shuf.csv | head -n 1009 >> dev.csv
cat shuf.csv | tail -n $(echo 10094 - 1009 | bc) >> train.csv

rm -f head.csv shuf.csv
