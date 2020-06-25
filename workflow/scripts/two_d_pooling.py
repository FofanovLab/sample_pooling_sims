import math
import numpy as np
import pandas as pd

try:
    from workflow.scripts.utils import get_random_sample_set
except ImportError:
    from utils import get_random_sample_set

def get_number_of_arrays(n_samples, window):
    """
    Calculate the number of grids required for a given window size
    and number of samples
    """
    return math.ceil(n_samples/(window * window))


def two_dimensional_reshape(sample, window):
    """
    Place samples into grids
    """
    # add zero padding to dimensions
    n_arrays = get_number_of_arrays(len(sample), window)
    sample.resize((window * window) * n_arrays, refcheck=False)
    return sample.reshape(n_arrays, window, window)



def two_dimensional_pooling(window, sample, n_channel_pipette):
    """
    Runs 2D pooling simuliation on sample
    given a NxN array defined by window.
    Returns number of tests, steps and,
    individual pipettes required to 
    fully recover all defectives in the
    sample. 
    Input
    -----
    window: (int) The size of one of the dimensions in the
                    symmetric pooling array. 
    sample: (arr) A binary/boolean array. 
    n_channel_pipette: (arr) calculate number of pipettes
                        assuming n_channel_pipettes.
    Output
    ------
    Dictionary with values for
    sample_size: Number of samples provided
    windows: the window size provided,
    n_arrays: the number of arrays needed to accomodate the sample_size
    total_pools: Number of pools for pooling scheme. (=2 * window * n_arrays)
    obs_positives: The number of positive samples identified as the intersection of
        positive rows and columns (n_positives = n_positive_columns * n_positive_rows)
    true_positives: The number of defectives (True) in sample
    ambiguous: Bool, True if there were ambiguous results. (if there is more than
        one positive column and more than one positive row in an array)
    n_tests: Total number of tests including retesting ambiguous results (total_pools + n_ambiguous positives)
    n_steps: Number of steps in the pooling scheme (1 if non-ambiguous, 2 if ambiguous)
    <N>_pipettes: Number of pipettes to set up the pools using an n_channel_pipette.
    """
    n_samples = len(sample)
    n_arrays = get_number_of_arrays(n_samples, window)
    total_pools = window * 2 * n_arrays
    empty_wells = (window * window * n_arrays) - n_samples
    if empty_wells % window == 0:
        # If a full row/column is empty, remove it from the total
        # number of pools.
        total_pools -= empty_wells // window
    # arrange samples into arrays
    sample_2d = two_dimensional_reshape(sample, window)
    # count positive rows and columns
    n_positive_columns = np.sum(np.any(sample_2d, axis=1), axis=1)
    n_positive_rows = np.sum(np.any(sample_2d, axis=2), axis=1)
    true_positives = sum(sample)
    obs_positives_per_array = n_positive_columns * n_positive_rows
    obs_positives = np.sum(obs_positives_per_array)
    # 2d pooling is ambiguous if there is more than one postive column and row
    where_ambiguous = np.where(np.greater(
        np.minimum(n_positive_columns, n_positive_rows), 1))
    is_ambiguous = np.any(np.greater(np.minimum(
        n_positive_columns, n_positive_rows), 1))
    # add an additional step if there are ambiguous results
    n_steps = 1 + is_ambiguous
    # number of tests including any validation of ambiguous results
    n_tests = total_pools + np.sum(obs_positives_per_array[where_ambiguous])
    # calculate the number of pipettes per row/column given a n_channel pipette.
    # the number of pipettes is mulitipled by the window size and then again by
    # 2 to account for doing the columns and rows. Any ambiguous samples are
    # added on as well.
    sample_positions = two_dimensional_reshape(
        np.ones(n_samples), window)
    # get length of rows for layout (minus any empty wells)
    len_rows = sample_positions.sum(axis=2)
    # get length of columns for layout (minus any empty wells)
    len_cols = sample_positions.sum(axis=1)
    pool_sizes = np.append(len_rows, len_cols)
    n_pipettes = {}
    for channel in n_channel_pipette:
        n_pipettes["{}_channel_pipette".format(channel)] = int(
            sum([np.ceil(pool_sz / channel) for pool_sz in pool_sizes]) +
            (is_ambiguous * np.sum(obs_positives_per_array[where_ambiguous])))
    return {
        'sample_size': n_samples,
        'windows': window,
        'n_arrays': n_arrays,
        'obs_positives': obs_positives,
        'true_positives': true_positives,
        'ambiguous': is_ambiguous,
        'n_tests': n_tests,
        'n_steps': n_steps,
        **n_pipettes}



def get_two_d_pooling_windows(N):
    """
    Get all unique symmetrical grid sizes
    """
    return np.unique(
        np.array(
            [math.ceil(math.ceil(math.sqrt(N)) / i)
            for i in range(1, math.ceil(math.sqrt(N)))]))



def run_two_dimensional_pooling_sims(N, K, R, P):
    data = {}
    windows = get_two_d_pooling_windows(N)
    for i, window in enumerate(windows):
        data[i] = two_dimensional_pooling(
            window, get_random_sample_set(N, K), P)
        data[i]['rep'] = R
    return pd.DataFrame.from_dict(data).T


def snakemake_main(outfile, N, K, R, P):
    df = run_two_dimensional_pooling_sims(N, K, R, P)
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    snakemake_main(
        snakemake.output[0],
        int(snakemake.params.N),
        int(snakemake.params.K),
        int(snakemake.params.R),
        snakemake.params.P
    )
