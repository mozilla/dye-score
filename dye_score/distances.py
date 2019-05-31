import numpy as np
from scipy.spatial import distance


def get_chebyshev_distances_xarray_ufunc(df_array, df_dye_array, **kwargs):
    def chebyshev(x):
        return np.abs(df_array[:, 0, :] - x).max(axis=1)
    result = np.apply_along_axis(chebyshev, 1, df_dye_array).T
    return result


def get_cityblock_distances_xarray_ufunc(df_array, df_dye_array, **kwargs):
    def cityblock(x):
        return np.abs(df_array[:, 0, :] - x).sum(axis=1)
    results = np.apply_along_axis(cityblock, 1, df_dye_array).T
    return results


def get_jaccard_distances_xarray_ufunc(df_array, df_dye_array, **kwargs):
    def jaccard(x):
        result = np.empty(df_array.shape[0])
        for i, row in enumerate(df_array[:, 0, :]):
            result[i] = distance.jaccard(row, x, **kwargs)
        return result
    results = np.apply_along_axis(jaccard, 1, df_dye_array).T
    return results


def get_cosine_distances_xarray_ufunc(df_array, df_dye_array, **kwargs):
    def cosine(x):
        result = np.empty((df_array.shape[0]))
        for i, row in enumerate(df_array[:, 0, :]):
            result[i] = distance.cosine(row, x, **kwargs)
        return result
    results = np.apply_along_axis(cosine, 1, df_dye_array).T
    return results


def get_mahalanobis_distances_xarray_ufunc(df_array, df_dye_array, VI):
    def mahalanobis(x):
        U_sub_V = df_array[:, 0, :] - x
        return np.sqrt(
            (np.tensordot(U_sub_V, VI, axes=1) * U_sub_V).sum(axis=1)
        )
    results = np.apply_along_axis(mahalanobis, 1, df_dye_array).T
    return results
