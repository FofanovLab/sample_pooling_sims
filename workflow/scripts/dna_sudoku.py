import numpy as np
import pandas as pd


try:
    from workflow.scripts.utils import get_random_sample_set
except ImportError:
    from utils import get_random_sample_set
# Some hard coded sets of co-primes for different numbers of samples
SUDOKU_WINDOWS = {
    96: [10, 11, 13, 17, 19, 23],
    192: [15, 16, 17, 19, 23, 29, 31, 37],
    288: [17, 18, 19, 23, 25, 29, 31, 37, 41, 43],
    384: [20, 21, 23, 29, 31, 37, 41, 43, 47, 53],
    960: [31, 32, 33, 35, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
    1536: [40, 41, 43, 47, 49, 51, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
}


def make_modulus_matrix(interval, n_samples):
    """
    Provide the offsets for each pooling interval
    """
    indices = np.array([list(range(i, i + n_samples))
                        for i in range(interval)])
    x = np.logical_not(np.remainder(indices, interval))
    return x


def dna_sudoku(windows, sample):
    """
    Runs dna sudoku simulation on sample
    given window sizes where len(windows)
    is the weight (w) and d = w - 1.
    Retuns number of tests, steps and,
    individual pipettes required to 
    fully recover all defectives in the
    sample
    Input
    -----
    windows: (arr) The intervals for each group of pools.
    sample: (arr) A binary/boolean array.
    
    Output
    ------
    Dictionary with values for:
    sample_size: Number of samples provided
    windows: the provided intervals for each group of pools,
    obs_positives: The number of samples that appeared in the max number of positive pools.
    exp_positives: The number of defectives assumed in the scheme (len(windows) - 1)
    true_positives: The number of defectives (True) in sample
    ambiguous: Bool, True if there were ambiguous results. (n_obs > n_exp)
    n_tests: Total number of tests including retesting ambiguous results
    n_steps: Number of steps in the pooling scheme (1 if non-ambiguous, 2 if ambiguous)
    n_pipettes: Number of pipettes to set up the pools.
    
    """
    weight = len(windows)
    n_samples = len(sample)
    total_pools = sum(windows)
    test_design = np.concatenate(
        [make_modulus_matrix(
            interval, n_samples) for interval in windows])

    test_result = np.any(np.logical_and(test_design, sample), axis=1)
    decoder = np.sum(test_design[test_result, :], axis=0)
    positives = np.where(
        (decoder == np.max(decoder)))[0] if np.max(decoder) > 0 else np.array([])
    exp_positives = weight - 1
    obs_positives = len(positives)
    # Assume we have to run each positive sample
    # individually if correct decoding is not
    # guaranteed (d0 > d)
    ambiguous = obs_positives > exp_positives
    n_tests = total_pools + (obs_positives * ambiguous)
    # add an additional step if correct decoding is
    # not guaranteed
    n_steps = 1 + ambiguous
    n_pipettes = (n_samples * weight) + (obs_positives * ambiguous)
    return {
        'sample_size': n_samples,
        'weight': weight,
        'windows': windows,
        'obs_positives': obs_positives,
        'exp_positives': exp_positives,
        'true_positives': sum(sample),
        'ambiguous': ambiguous,
        'n_tests': n_tests,
        'n_steps': n_steps,
        'n_pipettes': n_pipettes}


def run_dna_sudoku_sims(N, K, R):
    data = {}
    try:
        windows = SUDOKU_WINDOWS[N]
    except KeyError:
        raise("""
        Simulation does not have co-prime values for this number of samples"""
        )
    for i in range(2, len(windows) + 1):    
        data[i] = dna_sudoku(
                    windows[:i], get_random_sample_set(N, K))
        data[i]['rep'] = R
    return pd.DataFrame.from_dict(data).T


def snakemake_main(outfile, N, K, R):
    df = run_dna_sudoku_sims(N, K, R)
    df.to_csv(outfile, index=False)

if __name__ == "__main__":
    
    snakemake_main(
        snakemake.output[0],
        int(snakemake.params.N),
        int(snakemake.params.K),
        int(snakemake.params.R))

