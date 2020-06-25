import math
import numpy as np
import pandas as pd
from itertools import groupby

try:
    from workflow.scripts.utils import get_random_sample_set
except ImportError:
    from utils import get_random_sample_set

def get_number_of_stages(n, d):
    """
    Calculate number of stages base on S-Stage algorithm but 
    constrain to be less than or equal to 3.
    """
    return min(3, math.ceil(np.log(n/d)))


def get_group_sizes(n, s, d):
    """
    Calculate the size of each group based on S-Stage algorithm using
    the number of samples (n), the number of stages (s), and the
    expected number of positive samples (s).
    """
    return [math.ceil((n/d)**((s-i)/s)) for i in range(1, s + 1)]


def get_number_of_groups(k, groups):
    """
    Return the number of groups based on the size of the previous 
    groups.
    """
    max_group_size = max([len(g) for g in groups])
    # avoid retesting the same groups if the math works out to be 1
    return max(math.floor(max_group_size/k), 2)


def get_sample_arrangements(n_groups, previous_arrangement):
    """
    Subdivide samples into groups for testing
    """
    new_arrangement = []
    for parent_group in previous_arrangement:
        new_arrangement += np.array_split(parent_group, n_groups)
    return new_arrangement


def get_positive_groups(groups):
    """
    Identify groups that will test positive because they
    contain at least one positive sample
    """
    return [group for group in groups if np.any(group)]


def modified_3_stage(exp_d, sample, n_channel_pipette):
    """
    Runs modified Li's s-stage algorithm 
    simuliation on sample. Contstrains
    the number of steps to be less than 3 and calculates
    groups sizes as equal sized subgroups
    of the groups in previous step.
    Returns number of tests, steps and,
    individual pipettes required to 
    fully recover all defectives in the
    sample.
    Input
    -----
    exp_d: (int) Expected number of postive samples
    sample: (arr) A binary/boolean array.
    n_channel_pipette: (arr) calculate number of pipettes
                        assuming n_channel_pipettes.
    Output
    ------
    Dictionary with values for
    sample_size: Number of samples provided
    n_groups: the number of groups for each subdivision,
    true_positives: The number of defectives (True) in sample
    n_tests: Total number of tests. 
    n_steps: Number of steps in the pooling scheme
    max_samples: Maximum number of samples included in a pool
    <N>_pipettes: Number of pipettes to set up the pools using an n_channel_pipette.
    """
    N = len(sample)
    # this algorithm constrains the number of tests to less than or equal to 3
    n_steps = get_number_of_stages(N, exp_d)
    group_sizes = get_group_sizes(N, n_steps, exp_d)
    groups = np.array(sample, ndmin=2)
    max_samples_per_group = 0
    true_group_sizes = []
    n_tests = 0
    n_pipettes = {
        "{}_channel_pipette".format(channel): 0
        for channel in n_channel_pipette}

    for k in group_sizes:
        n_groups = get_number_of_groups(k, groups)
        groups = get_sample_arrangements(n_groups, groups)
        true_group_sizes.append(max([len(l) for l in groups]))
        max_samples_per_group = max(
            max_samples_per_group, max([len(g) for g in groups]))
        n_tests += len(groups)
        for channel in n_channel_pipette:
            n_pipettes["{}_channel_pipette".format(
                channel)] += sum([math.ceil(len(g) / channel) for g in groups])
        groups = get_positive_groups(groups)

    return {
        'sample_size': N,
        'group_size': true_group_sizes,
        'n_steps': n_steps,
        'n_tests': n_tests,
        'true_positives': sum(sample),
        'exp_positives': exp_d,
        'max_samples': max_samples_per_group,
        **n_pipettes
    }


def run_modified_3_stage_sims(N, K, R, P, E):
    data = {}
    for i, exp in enumerate(E):
        data[i] = modified_3_stage(
            exp, get_random_sample_set(N, K),
            P
        )
        data[i]['rep'] = R
    return pd.DataFrame.from_dict(data).T


def snakemake_main(outfile, N, K, R, P, E):
    df = run_modified_3_stage_sims(N, K, R, P, E)
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    snakemake_main(
        snakemake.output[0],
        int(snakemake.params.N),
        int(snakemake.params.K),
        int(snakemake.params.R),
        snakemake.params.P,
        snakemake.params.E)
