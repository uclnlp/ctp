# Conditional Theorem Provers

## Example

Sample usage:

```
$ ./bin/clutrr-cli.py --train data/clutrr-emnlp/data_db9b8f04/1.2,1.3,1.4_train.csv --test data/clutrr-emnlp/data_db9b8f04/1.10_test.csv data/clutrr-emnlp/data_db9b8f04/1.2_test.csv data/clutrr-emnlp/data_db9b8f04/1.3_test.csv data/clutrr-emnlp/data_db9b8f04/1.4_test.csv data/clutrr-emnlp/data_db9b8f04/1.5_test.csv data/clutrr-emnlp/data_db9b8f04/1.6_test.csv data/clutrr-emnlp/data_db9b8f04/1.7_test.csv data/clutrr-emnlp/data_db9b8f04/1.8_test.csv data/clutrr-emnlp/data_db9b8f04/1.9_test.csv -S 1 -s concat -V 128 -b 32 -d 2 --test-max-depth 4 --hops 2 2 2 2 -e 5 -o adagrad -l 0.1 --init random --ref-init random -m 5 -t min -k 20 -i 1.0 -r memory -R 256 --seed 1
INFO:clutrr-cli.py:Device: cpu
INFO:kbcr.clutrr.models.model:Hoppy(k=5, depth=2, hops_lst=['MemoryReformulator', 'MemoryReformulator', 'MemoryReformulator', 'MemoryReformulator'])

[..]

INFO:clutrr-cli.py:Epoch 1/5	Batch 254/472	Loss 0.0939 ± 0.1072
INFO:clutrr-cli.py:Epoch 1/5	Batch 255/472	Loss 0.0751 ± 0.0926
INFO:clutrr-cli.py:Epoch 1/5	Batch 256/472	Loss 0.0732 ± 0.1006
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.2_test.csv: 100.000000
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.3_test.csv: 95.327103
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.4_test.csv: 94.805195
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.5_test.csv: 99.459459
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.6_test.csv: 100.000000
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.7_test.csv: 98.064516
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.8_test.csv: 93.333333
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.9_test.csv: 98.387097
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.10_test.csv: 94.262295
INFO:clutrr-cli.py:Epoch 1/5	Batch 257/472	Loss 0.0740 ± 0.0775
INFO:clutrr-cli.py:Epoch 1/5	Batch 258/472	Loss 0.0813 ± 0.1155
INFO:clutrr-cli.py:Epoch 1/5	Batch 259/472	Loss 0.0822 ± 0.0891

[..]
```

Another example:

```
$ ./bin/clutrr-cli.py --train data/clutrr-emnlp/data_db9b8f04/1.2,1.3,1.4_train.csv --test data/clutrr-emnlp/data_db9b8f04/1.10_test.csv data/clutrr-emnlp/data_db9b8f04/1.2_test.csv data/clutrr-emnlp/data_db9b8f04/1.3_test.csv data/clutrr-emnlp/data_db9b8f04/1.4_test.csv data/clutrr-emnlp/data_db9b8f04/1.5_test.csv data/clutrr-emnlp/data_db9b8f04/1.6_test.csv data/clutrr-emnlp/data_db9b8f04/1.7_test.csv data/clutrr-emnlp/data_db9b8f04/1.8_test.csv data/clutrr-emnlp/data_db9b8f04/1.9_test.csv -S 1 -s concat -V 128 -b 32 -d 2 --test-max-depth 4 --hops 2 2 2 2 -e 5 -o adagrad -l 0.1 --init uniform --ref-init random -m 5 -t min -k 20 -i 1.0 -r linear -R 0 --seed 1
INFO:clutrr-cli.py:Device: cpu
INFO:kbcr.clutrr.models.model:Hoppy(k=5, depth=2, hops_lst=['LinearReformulator', 'LinearReformulator', 'LinearReformulator', 'LinearReformulator'])

[..]

INFO:clutrr-cli.py:Epoch 1/5	Batch 254/472	Loss 0.0991 ± 0.1065
INFO:clutrr-cli.py:Epoch 1/5	Batch 255/472	Loss 0.0787 ± 0.0969
INFO:clutrr-cli.py:Epoch 1/5	Batch 256/472	Loss 0.0779 ± 0.1097
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.2_test.csv: 100.000000
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.3_test.csv: 85.981308
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.4_test.csv: 87.012987
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.5_test.csv: 98.918919
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.6_test.csv: 98.095238
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.7_test.csv: 98.709677
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.8_test.csv: 97.777778
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.9_test.csv: 96.774194
INFO:clutrr-cli.py:Test Accuracy on data/clutrr-emnlp/data_db9b8f04/1.10_test.csv: 94.262295
INFO:clutrr-cli.py:Epoch 1/5	Batch 257/472	Loss 0.0762 ± 0.0905
INFO:clutrr-cli.py:Epoch 1/5	Batch 258/472	Loss 0.0825 ± 0.0969
INFO:clutrr-cli.py:Epoch 1/5	Batch 259/472	Loss 0.0865 ± 0.1010

[..]
```
