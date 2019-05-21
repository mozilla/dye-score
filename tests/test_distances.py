import numpy as np
from dye_score.distances import (
    get_chebyshev_distances_xarray_ufunc,
)
from scipy.spatial.distance import chebyshev


def test_chebyshev_func():
    """Note the injection of an extra dimension which happens
    when the xarray apply ufunc is put together.

    For (5,1,2) all data looks like:
    [[[0.34180806 0.92010143]],
     [[0.69717685 0.24012436]],
     [[0.3362796  0.08151153]],
     [[0.74861764 0.94125763]],
     [[0.25078923 0.3294995 ]]]

    dye data is just one row of this
     [[0.74861764 0.94125763]]

    returned data is:
     [[0.64183828],
      [0.34977852],
      [0.        ],
      [0.63256378],
      [0.06286615]]
    """
    random_array = np.random.rand(5, 1, 2)

    for dye_snippet in [0, 1, 4]:
        dye_snippet_result = get_chebyshev_distances_xarray_ufunc(
            random_array, random_array[dye_snippet]
        )
        assert dye_snippet_result.shape == (5, 1)
        for i, actual_result in enumerate(dye_snippet_result):
            expected_result = chebyshev(
                random_array[dye_snippet][0],
                random_array[i][0]
            )
            assert actual_result == expected_result
