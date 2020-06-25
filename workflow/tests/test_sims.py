import pytest
import math
import numpy as np
from workflow.scripts.dna_sudoku import dna_sudoku
from workflow.scripts.two_d_pooling import two_dimensional_pooling
from workflow.scripts.s_stage import s_stage
from workflow.scripts.modified_3_stage import modified_3_stage
from workflow.scripts.binary_splitting_by_halving import binary_splitting_by_halving
from workflow.scripts.generalized_binary_splitting import generalized_binary_splitting


def get_sample_array(N, positive_positions):
    arr = np.zeros(N)
    arr[positive_positions] = 1
    return arr

@pytest.mark.parametrize("window,sample,expected", [
    ([10, 11], get_sample_array(96, [40, 82]), {'sample_size': 96,
    'weight': 2,
    'windows': [10, 11],
    'obs_positives': 4,
    'exp_positives': 1,
    'true_positives': 2,
    'ambiguous': True,
    'n_tests': 25,
    'n_steps': 2,
    'n_pipettes': 196}),
    ([10, 11], get_sample_array(96, [82]), {'sample_size': 96,
                                            'weight': 2,
                                            'windows': [10, 11],
                                            'obs_positives': 1,
                                            'exp_positives': 1,
                                            'true_positives': 1,
                                            'ambiguous': False,
                                            'n_tests': 21,
                                            'n_steps': 1,
                                            'n_pipettes': 192}),
    ([10, 11], np.zeros(96), {'sample_size': 96,
                                            'weight': 2,
                                            'windows': [10, 11],
                                            'obs_positives': 0,
                                            'exp_positives': 1,
                                            'true_positives': 0,
                                            'ambiguous': False,
                                            'n_tests': 21,
                                            'n_steps': 1,
                                            'n_pipettes': 192}),
    ([10, 11], np.ones(96), {'sample_size': 96,
                              'weight': 2,
                              'windows': [10, 11],
                              'obs_positives': 96,
                              'exp_positives': 1,
                              'true_positives': 96,
                              'ambiguous': True,
                              'n_tests': 117,
                              'n_steps': 2,
                              'n_pipettes': 288})
])
def test_dna_sudoku(window, sample, expected):
    assert dna_sudoku(window, sample) == expected


@pytest.mark.parametrize("window,sample,pipette,expected", [
    (10, get_sample_array(96, [1]), [1, 8, 16], {'sample_size': 96,
                                                'windows': 10,
                                                'n_arrays': 1,
                                                'obs_positives': 1,
                                                'true_positives': 1,
                                                'ambiguous': False,
                                                'n_tests': 20,
                                                'n_steps': 1,
                                                '1_channel_pipette': 192,
                                                '8_channel_pipette': 39,
                                                '16_channel_pipette': 20}),
    (10, get_sample_array(100, [1]), [1, 8, 16], {'sample_size': 100,
                                                 'windows': 10,
                                                 'n_arrays': 1,
                                                 'obs_positives': 1,
                                                 'true_positives': 1,
                                                 'ambiguous': False,
                                                 'n_tests': 20,
                                                 'n_steps': 1,
                                                 '1_channel_pipette': 200,
                                                 '8_channel_pipette': 40,
                                                 '16_channel_pipette': 20}),
    (5, np.zeros(100), [1, 8, 16], {'sample_size': 100,
                                                  'windows': 5,
                                                  'n_arrays': 4,
                                                  'obs_positives': 0,
                                                  'true_positives': 0,
                                                  'ambiguous': False,
                                                  'n_tests': 40,
                                                  'n_steps': 1,
                                                  '1_channel_pipette': 200,
                                                  '8_channel_pipette': 40,
                                                  '16_channel_pipette': 40}),
    (2, np.ones(100), [1, 8, 16], {'sample_size': 100,
                                                  'windows': 2,
                                                  'n_arrays': 25,
                                                  'obs_positives': 100,
                                                  'true_positives': 100,
                                                  'ambiguous': True,
                                                  'n_tests': 200,
                                                  'n_steps': 2,
                                                  '1_channel_pipette': 300,
                                                  '8_channel_pipette': 200,
                                                  '16_channel_pipette': 200})

])
def test_2D_pooling(window, sample, pipette, expected):
    assert two_dimensional_pooling(window, sample, pipette) == expected



def get_S(n, d):
    """
    Calculate the number of steps
    """
    return math.ceil(np.log(n/d))

@pytest.mark.parametrize("exp,sample,pipette,expected", [
    (1, get_sample_array(96, [1]), [1, 8, 16], {'sample_size': 96,
                                                'group_size': [48, 16, 8, 4, 1],
                                                 'true_positives': 1,
                                                 'exp_positives': 1,
                                                 'n_tests': 13,
                                                 'n_steps': get_S(96, 1),
                                                 '1_channel_pipette': 172,
                                                 '8_channel_pipette': 26,
                                                 '16_channel_pipette': 17}),
    (10, get_sample_array(96, [1]), [1, 8, 16], {'sample_size': 96,
                                                'group_size': [6, 3, 1],
                                                'true_positives': 1,
                                                'exp_positives': 10,
                                                'n_tests': 24,
                                                'n_steps': get_S(96, 10),
                                                '1_channel_pipette': 105,
                                                '8_channel_pipette': 24,
                                                '16_channel_pipette': 24}),
    (1, get_sample_array(96, [1,2]), [1, 8, 16], {'sample_size': 96,
                                                'group_size': [48, 16, 8, 4, 1],
                                                'true_positives': 2,
                                                'exp_positives': 1,
                                                'n_tests': 13,
                                                'n_steps': get_S(96, 1),
                                                '1_channel_pipette': 172,
                                                '8_channel_pipette': 26,
                                                '16_channel_pipette': 17}),

])
def test_s_stage(exp, sample, pipette, expected):
    assert s_stage(exp, sample, pipette) == expected


@pytest.mark.parametrize("exp,sample,pipette,expected", [
    (1, get_sample_array(96, [1]), [1, 8, 16], {'sample_size': 96,
                                                'group_size': [24, 6, 1],
                                                'true_positives': 1,
                                                'exp_positives': 1,
                                                'n_tests': 14,
                                                'n_steps': 3,
                                                'max_samples': 24,
                                                '1_channel_pipette': 126,
                                                '8_channel_pipette': 22,
                                                '16_channel_pipette': 18}),
    (10, get_sample_array(96, [1]), [1, 8, 16], {'sample_size': 96,
                                                 'group_size': [6, 3, 1],
                                                 'true_positives': 1,
                                                 'exp_positives': 10,
                                                 'n_tests': 24,
                                                 'n_steps': 3,
                                                 'max_samples': 6,
                                                 '1_channel_pipette': 105,
                                                 '8_channel_pipette': 24,
                                                 '16_channel_pipette': 24}),
    (1, get_sample_array(96, [1, 2]), [1, 8, 16], {'sample_size': 96,
                                                   'group_size': [24, 6, 1],
                                                   'true_positives': 2,
                                                   'exp_positives': 1,
                                                   'n_tests': 14,
                                                   'n_steps': 3,
                                                   'max_samples': 24,
                                                   '1_channel_pipette': 126,
                                                   '8_channel_pipette': 22,
                                                   '16_channel_pipette': 18}),

])
def test_modified_3_stage(exp, sample, pipette, expected):
    assert modified_3_stage(exp, sample, pipette) == expected


@pytest.mark.parametrize("sample,pipette,expected", [
    (get_sample_array(96, [0]), [1, 8, 16], {'sample_size': 96,
                                                'true_positives': 1,
                                                'n_tests': 9,
                                                'n_steps': 9,
                                                '1_channel_pipette': 287,
                                                '8_channel_pipette': 39,
                                                '16_channel_pipette': 22}),
])
def test_binary_splitting_by_halving(sample, pipette, expected):
    assert binary_splitting_by_halving(sample, pipette) == expected


@pytest.mark.parametrize("exp,sample,pipette,expected", [
    (1, get_sample_array(96, [0]), [1, 8, 16], {'sample_size': 96,
                                             'true_positives': 1,
                                             'exp_positives': 1,
                                             'n_tests': 8,
                                             'n_steps': 8,
                                             '1_channel_pipette': 222,
                                             '8_channel_pipette': 30,
                                             '16_channel_pipette': 17}),
])
def test_generalized_binary_splitting(exp, sample, pipette, expected):
    assert generalized_binary_splitting(exp, sample, pipette) == expected
