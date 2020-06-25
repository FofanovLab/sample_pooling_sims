import math
import numpy as np
import pandas as pd
from itertools import groupby, chain

try:
    from workflow.scripts.utils import get_random_sample_set
except ImportError:
    from utils import get_random_sample_set

def get_K(n, s, d):
    """
    Calculate the number of samples per pool for each step
    """
    return [math.ceil((n/d)**((s-i)/s)) for i in range(1, s + 1)]


def get_S(n, d):
    """
    Calculate the number of steps
    """
    return math.ceil(np.log(n/d))


def s_stage(exp_d, sample, n_channel_pipette):
    """
    Runs Li's s-stage algorithm 
    simuliation on sample
    given an expected number of
    positive samples, and a sample set.
    Returns number of tests, steps and,
    individual pipettes required to 
    fully recover all defectives in the
    sample.
    Input
    -----
    exp_d: (int) Expected number of positive samples, used to calculate
                 the group sizes for each step. 
    sample: sample: (arr) A binary/boolean array.
    n_channel_pipette: (int) calculate number of pipettes
                        assuming an n_channel_pipette.
    Output
    ------
    Dictionary with values for
    sample_size: Number of samples provided
    n_groups: the number of groups for each subdivision
    true_positives: The number of defectives (True) in sample
    n_tests: Total number of tests. (The number of groups at each step)
    n_steps: Number of steps in the pooling scheme (calculated by the algorithm)
    n_pipettes: Number of pipettes to set up the pools using an n_channel_pipette.
    """
    N = len(sample)
    S = get_S(N, exp_d)
    k = get_K(N, S, exp_d)
    true_positives = sum(sample)
    n_tests = 0
    n_steps = S
    group_sizes = []
    n_pipettes = {
        "{}_channel_pipette".format(channel): 0
        for channel in n_channel_pipette}
    for r, group_size in enumerate(k):
        n_groups = math.floor(len(sample)/group_size)
        sample_split = [
            arr for arr in np.array_split(sample, n_groups)
            if len(arr)]
        group_sizes.append(max([len(l) for l in sample_split]))
        n_tests += len(sample_split)
        if r == 0:
            # initialize sample arrangement which defines
            # sequential samples to more accurately calculate
            # the total number of pipettes.
            sample_arrangement = np.array(
                [np.repeat(i, len(group)) for i, group in enumerate(sample_split)])
        sample_arrangement = np.array_split(
            np.concatenate(sample_arrangement), n_groups)
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(channel)] += (
                sum([math.ceil(j/channel) for j in
                     list(chain(*[[len(list(g)) for k, g in groupby(i)]
                                  for i in sample_arrangement]))])
            )
        positive = [group for group in sample_split if np.any(group)]
        sample = np.concatenate(positive)
        sample_arrangement = [sample_arrangement[i]
                              for i, group in enumerate(sample_split) if np.any(group)]
    return {
        'sample_size': N,
        'group_size': group_sizes,
        'n_steps': n_steps,
        'n_tests': n_tests,
        'true_positives': true_positives,
        'exp_positives': exp_d,
        **n_pipettes
    }


def run_s_stage_sims(N, K, R, P, E):
    data = {}
    for i, exp in enumerate(E):
        data[i] = s_stage(
            exp, get_random_sample_set(N, K),
            P
        )
        data[i]['rep'] = R
    return pd.DataFrame.from_dict(data).T


def snakemake_main(outfile, N, K, R, P, E):
    df = run_s_stage_sims(N, K, R, P, E)
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    snakemake_main(
        snakemake.output[0],
        int(snakemake.params.N),
        int(snakemake.params.K),
        int(snakemake.params.R),
        snakemake.params.P,
        snakemake.params.E)
