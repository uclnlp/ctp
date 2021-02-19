# Conditional Theorem Provers

### Reproducing Results on CLUTRR

```bash
./bin/clutrr-cli.py --train data/clutrr-emnlp/data_db9b8f04/1.2,1.3,1.4_train.csv --test data/clutrr-emnlp/data_db9b8f04/1.10_test.csv data/clutrr-emnlp/data_db9b8f04/1.2_test.csv data/clutrr-emnlp/data_db9b8f04/1.3_test.csv data/clutrr-emnlp/data_db9b8f04/1.4_test.csv data/clutrr-emnlp/data_db9b8f04/1.5_test.csv data/clutrr-emnlp/data_db9b8f04/1.6_test.csv data/clutrr-emnlp/data_db9b8f04/1.7_test.csv data/clutrr-emnlp/data_db9b8f04/1.8_test.csv data/clutrr-emnlp/data_db9b8f04/1.9_test.csv -S 1 -s concat -V 128 -b 32 -d 2 --test-max-depth 4 --hops 2 2 2 -e 5 -o adagrad -l 0.1 --init random --ref-init random -m 5 -t min -k 20 -i 1.0 -r linear -R 0 --seed 1
[..]
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.10_test.csv: 89.344262
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.2_test.csv: 100.000000
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.3_test.csv: 71.962617
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.4_test.csv: 74.025974
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.5_test.csv: 97.837838
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.6_test.csv: 96.190476
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.7_test.csv: 98.064516
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.8_test.csv: 97.777778
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.9_test.csv: 94.354839
[..]
```

### Reproducing Link Prediction Results

#### UMLS
```bash
./bin/hoppy-cli.py --train data/umls/train.tsv --dev data/umls/dev.tsv --test data/umls/test.tsv -k 50 -K 4 -b 8 -e 100 -f 0 -l 0.1 -o adagrad -I standard -V 3 -q --hops 0 2 --nb-negatives 3 -r linear -R 0 --init uniform --ref-init random --refresh 100 --index nms
```

#### Kinship
```bash
./bin/hoppy-cli.py --train data/kinship/train.tsv --dev data/kinship/dev.tsv --test data/kinship/test.tsv -k 50 -K 4 -b 8 -e 100 -f 0 -l 0.1 -o adagrad -I standard -V 3 -q --hops 0 1 1R --nb-negatives 3 -r attentive -R 0 --init uniform --ref-init random --refresh 100 --index nms
```

#### Nations
```bash
./bin/hoppy-cli.py --train data/nations/train.tsv --dev data/nations/dev.tsv --test data/nations/test.tsv -k 50 -K 4 -b 16 -e 100 -f 0 -l 0.1 -o adagrad -I standard -V 3 -q --hops 0 2 1R --nb-negatives 3 -r linear -R 0 --init uniform --ref-init random --refresh 100 --index nms
```

Note that the reasoning depth here is always 1, and the rule templates are given by the `--hops` parameter - e.g. `0 2 1R` means there are three reformulation modules, one for rules in the form `p(X,Y) :- q(X,Z), r(Z,Y)` (2 hops from the subject to the object) and one for rules in the form `p(X,Y) :- q(Y,X)` (1 hop from the object to the subject). Here 0 means no hops is also a legal choice (just match the query with the facts in the KB). For matching with facts in the KB, we use a nearest neighbour search engine for finding the most similar facts to the query (otherwise we should have to compute the Gaussian kernel between each sub-goal and all facts in the KB all the time, and backpropagate through all these comparisons - https://arxiv.org/abs/1912.10824)

### Citing

```
@inproceedings{minervini20icml,
  author    = {Pasquale Minervini and
               Sebastian Riedel and
               Pontus Stenetorp and
               Edward Grefenstette and
               Tim Rockt{\"{a}}schel},
  title     = {Learning Reasoning Strategies in End-to-End Differentiable Proving},
  booktitle = {{ICML}},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {6938--6949},
  publisher = {{PMLR}},
  year      = {2020}
}
```