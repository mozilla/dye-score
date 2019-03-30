import numpy as np


def get_chebyshev_distances_xarray_ufunc(df_array, df_dye_array):
    def chebyshev(x):
        return np.abs(df_array[:, 0, :] - x).max(axis=1)
    result = np.apply_along_axis(chebyshev, 1, df_dye_array).T
    return result
