# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


@pytest.mark.light
def disabled_test_complex_cli_v1():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = 'python3 ./bin/kbc-cli.py --train data/wn18rr/dev.tsv --dev data/wn18rr/dev.tsv ' \
              '--test data/wn18rr/test.tsv -m complex -k 100 -b 100 -e 1 --N3 0.0001 -l 0.1 -V 1 -o adagrad -B 2000'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    sanity_check_flag_1 = False

    for line in lines:
        if 'Batch 1/31' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 18.311768, atol=1e-3, rtol=1e-3)
        if 'Batch 10/31' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 18.273418, atol=1e-3, rtol=1e-3)
        if 'Final' in line and 'dev results' in line:
            value = float(line.split()[4])
            np.testing.assert_allclose(value, 0.116451, atol=1e-3, rtol=1e-3)

            sanity_check_flag_1 = True

    assert sanity_check_flag_1


if __name__ == '__main__':
    pytest.main([__file__])
