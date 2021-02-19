# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


def run_model(model: str):
    _run_model(f'{model}')
    _run_model(f'{model} --predicate')


def _run_model(model: str):
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = f'python3 ./bin/geometric-cli.py --train data/clutrr-emnlp/data_test/32.csv ' \
        f'--test data/clutrr-emnlp/data_test/32.csv -m {model} -e 100'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    print(out)
    print(err)

    lines = out.decode("utf-8").split("\n")

    accuracy_value = None
    sanity_check_flag = False

    for line in lines:
        if 'Accuracy' in line:
            tokens = line.split(' ')

            accuracy_value = float(tokens[-1])
            sanity_check_flag = True

    assert sanity_check_flag
    np.testing.assert_allclose(accuracy_value, 1.0)


@pytest.mark.light
def test_gometric_gat_v1():
    run_model('gat')


@pytest.mark.light
def test_gometric_gcn_v1():
    run_model('gcn')


@pytest.mark.light
def test_gometric_cnn_v1():
    # run_model('cnn')
    run_model('cnn --v2')


@pytest.mark.light
def test_gometric_cnnh_v1():
    # run_model('cnnh')
    run_model('cnnh --v2')


@pytest.mark.light
def test_gometric_rnn_v1():
    run_model('rnn')
    run_model('rnn --v2')


@pytest.mark.light
def test_gometric_lstm_v1():
    run_model('lstm')
    run_model('lstm --v2')


@pytest.mark.light
def test_gometric_gru_v1():
    run_model('gru')
    run_model('gru --v2')


@pytest.mark.light
def test_gometric_intra_v1():
    # run_model('intra')
    # run_model('intra --v2')
    pass


if __name__ == '__main__':
    pytest.main([__file__])
