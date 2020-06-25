import math
import numpy as np
import pandas as pd

try:
    from workflow.scripts.utils import get_random_sample_set
except ImportError:
    from utils import get_random_sample_set

def binary_search(subsample, n_channel_pipette):
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
    while len(subsample) > 1:
        l = len(subsample)
        half = math.ceil(l/2)
        first_half = subsample[:half]
        second_half = subsample[half:]
        n_tests += 1
        n_steps += 1
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(
                channel)] += math.ceil(len(first_half) / channel)
        if np.any(first_half):
            subsample = first_half
            not_tested = np.append(not_tested, second_half)
        else:
            subsample = second_half
    return {
        'not_tested': not_tested,
        "n_tests": n_tests,
        'n_steps': n_steps,
        **n_pipettes}



def generalized_binary_splitting(exp_d, sample, n_channel_pipette):
    """
    Runs Hwang's generalized binary splitting algorithm 
    simuliation on sample
    given an expected number of defectives.
    Returns number of tests, steps and,
    individual pipettes required to 
    fully recover all defectives in the
    sample.
    Input
    -----
    exp_d: (int) Expected number of defectives
    sample: sample: (arr) A binary/boolean array.
    n_channel_pipette: (arr) calculate number of pipettes
                        assuming n_channel_pipettes.
    Output
    ------
    Dictionary with values for
    sample_size: Number of samples provided
    true_positives: The number of defectives (True) in sample (sum(sample))
    n_tests: Total number of tests. 
    n_steps: Number of steps in the pooling scheme 
    <N>_pipettes: Number of pipettes to set up the pools using an n_channel_pipette.
    """
    true_positives = int(sum(sample))
    sample_size = len(sample)
    n = sample_size
    d = exp_d
    n_steps = 0
    n_tests = 0
    n_pipettes = {
        "{}_channel_pipette".format(channel): 0
        for channel in n_channel_pipette}
    while len(sample):
        if n <= (2 * d - 1):
            # move to individual testing at this point
            n_steps += 1
            n_tests += len(sample)
            for k in n_pipettes.keys():
                # individual pipettes for individual testing
                n_pipettes[k] += len(sample)
            break
        if d <= 0:
            # to ensure all defectives are found
            # perform binary search on remaining
            # samples
            pool = sample
            group_size = len(sample)
        else:
            l = n - d + 1
            a = math.floor(np.log2(l/d))
            group_size = 2**a
            pool = sample[:group_size]
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(
                channel)] += math.ceil(group_size / channel)
        n_tests += 1
        n_steps += 1
        sample = sample[group_size:]
        if np.any(pool):
            b_search_result = binary_search(pool, n_channel_pipette)
            for channel in n_channel_pipette:
                n_pipettes["{}_channel_pipette".format(
                    channel)] += b_search_result["{}_channel_pipette".format(
                        channel)]
            n_tests += b_search_result['n_tests']
            n_steps += b_search_result['n_steps']
            # add non tested groups back to the sample pile
            sample = np.append(sample, b_search_result['not_tested'])
            # remove one defective from the total because
            # one has been found
            d -= 1

        n = len(sample)

    return {
        'sample_size': sample_size,
        'n_steps': n_steps,
        'n_tests': n_tests,
        'true_positives': true_positives,
        'exp_positives': exp_d,
        **n_pipettes
    }


def run_generalized_binary_splitting_sims(N, K, R, P, E):
    data = {}
    for i, exp in enumerate(E):
        data[i] = generalized_binary_splitting(
                            exp,
                            get_random_sample_set(N, K),
                            P)
        data[i]['rep'] = R
    return pd.DataFrame.from_dict(data).T


def snakemake_main(outfile, N, K, R, P, E):
    df = run_generalized_binary_splitting_sims(N, K, R, P, E)
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    snakemake_main(
        snakemake.output[0],
        int(snakemake.params.N),
        int(snakemake.params.K),
        int(snakemake.params.R),
        snakemake.params.P,
        snakemake.params.E)
