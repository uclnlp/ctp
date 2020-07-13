cat 1.2,1.3_train.csv | tr ")" "\n" | tr "[" "\n" | grep ",," | grep -v train | sort  | uniq | tr -d "\"" | tr -d "," | sort | uniq | wc -l
