import math
import numpy as np
import pandas as pd
try:
    from workflow.scripts.utils import get_random_sample_set
except ImportError:
    from utils import get_random_sample_set


def binary_search(sample, n_channel_pipette):
    """
    Run binary search on subgroup until a single positive sample
    is identified. Returns the untested samples, the number of tests
    and the number of steps required to identify the positive sample.
    """
    not_tested = np.array([], dtype=bool)
    n_tests = 0
    n_steps = 0
    n_pipettes = {
        "{}_channel_pipette".format(channel): 0
        for channel in n_channel_pipette}
    while len(sample) > 1:
        l = len(sample)
        half = math.ceil(l/2)
        first_half = sample[:half]
        second_half = sample[half:]
        n_tests += 1
        n_steps += 1
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(
                channel)] += math.ceil(len(first_half) / channel)
        if np.any(first_half):
            sample = first_half
            not_tested = np.append(not_tested, second_half)
        else:
            sample = second_half
    return {
        'not_tested': not_tested,
        "n_tests": n_tests,
        'n_steps': n_steps,
        **n_pipettes}


def binary_splitting_by_halving(sample, n_channel_pipette):
    """
    Runs binary splitting by halving 
    simuliation on sample.
    Returns number of tests, steps and,
    individual pipettes required to
    fully recover all defectives in the
    sample.
    Input
    -----
    sample: sample: (arr) A binary/boolean array.
    n_channel_pipette: (arr) calculate number of pipettes
                        assuming n_channel_pipettes.
    Output
    ------
    Dictionary with values for
    sample_size: Number of samples provided
    true_positives: The number of defectives (True) in sample (sum(sample))
    n_tests: Total number of tests. 
    n_steps: Number of steps in the pooling scheme.
    <N>_pipettes: Number of pipettes to set up the pools using an n_channel_pipette.
    """
    true_positives = sum(sample)
    sample_size = len(sample)
    n_steps = 0
    n_tests = 0
    n_pipettes = {
        "{}_channel_pipette".format(channel): 0
        for channel in n_channel_pipette}
    not_tested = sample
    while len(not_tested) > 1:
        # count initial test to see if any positives
        # are in remaining untested samples
        n_tests += 1
        n_steps += 1
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(
                channel)] += math.ceil(len(not_tested) / channel)
        if not np.any(not_tested):
            # stop when the remaining samples
            # are no longer postive
            not_tested = []
            break
        b_search_result = binary_search(
            not_tested, n_channel_pipette)
        not_tested = b_search_result['not_tested']
        n_tests += b_search_result['n_tests']
        n_steps += b_search_result['n_steps']
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(
                channel)] += b_search_result["{}_channel_pipette".format(
                    channel)]
    if len(not_tested):
        # add an additional test if a single sample remains
        # in untested.
        n_steps += 1
        n_tests += 1
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(
                channel)] += 1    

    return {
        'sample_size': sample_size,
        'n_steps': n_steps,
        'n_tests': n_tests,
        'true_positives': true_positives,
        **n_pipettes
    }


def run_binary_splitting_by_halving_sims(N, K, R, P):
    data = {}
    data[0] = binary_splitting_by_halving(
        get_random_sample_set(N, K),
        P)
    data[0]['rep'] = R
    return pd.DataFrame.from_dict(data).T



def snakemake_main(outfile, N, K, R, P):
    df = run_binary_splitting_by_halving_sims(N, K, R, P)
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    snakemake_main(
        snakemake.output[0],
        int(snakemake.params.N),
        int(snakemake.params.K),
        int(snakemake.params.R),
        snakemake.params.P)
