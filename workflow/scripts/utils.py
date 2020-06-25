import numpy as np
import pandas as pd



def get_random_sample_set(n, k):
    """
    Provide a binary array of size n
    with a random distribution of k
    defectives (1) and n-k non-defectives
    """
    s = np.append(np.zeros(n-k, dtype=bool), np.ones(k, dtype=bool))
    np.random.shuffle(s)
    return s


def cat_csvs(file_list, outfile):
    cat_df = pd.DataFrame([])
    for fn in file_list:
        df = pd.read_csv(fn)
        cat_df = cat_df.append(df)
    cat_df.to_csv(outfile, index=False)

