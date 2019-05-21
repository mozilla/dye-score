#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dask.dataframe as dd
import numpy as np
import os
import pandas as pd
import pytest
import xarray as xr
import yaml

from scipy.spatial.distance import chebyshev
from pyarrow.lib import ArrowIOError
from yaml.scanner import ScannerError

from dye_score import DyeScore


def write_config_file(tmpdir, sample_config):
    config_file = os.path.join(tmpdir, 'config.yaml')
    with open(config_file, 'w') as f:
        f.write(yaml.dump(sample_config))
    return config_file


def daskify(df, npartitions=2):
    # Turn pandas df into dask
    return dd.from_pandas(df, npartitions=2)


##
# Test Data Validation
##

def test_data_validation_with_invalid_file(tmpdir, sample_config):
    # Set-up invalid data file and save config
    data_file = os.path.join(tmpdir, 'data.csv')
    df = pd.DataFrame({'a': [1, 2, 3]})
    df.to_csv(data_file)
    sample_config['INPUT_PARQUET_LOCATION'] = data_file
    config_file = write_config_file(tmpdir, sample_config)
    ds = DyeScore(config_file)
    # Test
    with pytest.raises(ArrowIOError):
        ds.validate_input_data()


def test_data_validation_with_valid_file(tmpdir, sample_config):
    # Set-up valid data file and save config
    data_file = os.path.join(tmpdir, 'data.parquet')
    daskify(pd.DataFrame({
        'top_level_url': ['a', 'b'],
        'document_url': ['a', 'b'],
        'script_url': ['c', 'd'],
        'symbol': ['e', 'f'],
        'func_name': ['g', 'h']
    })).to_parquet(data_file)
    sample_config['INPUT_PARQUET_LOCATION'] = data_file
    config_file = write_config_file(tmpdir, sample_config)
    ds = DyeScore(config_file)
    # Test
    assert ds.validate_input_data() is True


##
# Test Config
##

def test_config_requires_valid_file():
    with pytest.raises(ValueError):
        DyeScore('test.yaml')


def test_config_requires_valid_yaml(asset_dir):
    with pytest.raises(ScannerError):
        DyeScore(os.path.join(asset_dir, 'invalid_config.yaml'))


def test_config_sets_properties(asset_dir):
    ds = DyeScore(os.path.join(asset_dir, 'valid_config.yaml'))
    assert ds.config('INPUT_PARQUET_LOCATION') == 's3://inputdir'
    assert ds.config('DYESCORE_DATA_DIR') == 's3://datadir'
    assert ds.config('DYESCORE_RESULTS_DIR') == 's3://resultsdir'
    assert ds.config('AWS_ACCESS_KEY_ID') == 'jgdflkgjsld;gs'
    assert ds.config('AWS_SECRET_ACCESS_KEY') == 'dsil;guewort9q9vkdf/'


def test_config_with_use_aws_true(asset_dir):
    ds = DyeScore(os.path.join(asset_dir, 'valid_config.yaml'))
    assert ds.config('USE_AWS') is True


def test_config_with_use_aws_false(asset_dir):
    ds = DyeScore(os.path.join(asset_dir, 'valid_config_aws_false.yaml'))
    assert ds.config('USE_AWS') is False


###
# Test compute_distances_for_dye_snippets
###


def test_standard_chebyshev_call_works(tmpdir, sample_config):
    sample_config['DYESCORE_DATA_DIR'] = tmpdir.strpath
    ds = DyeScore(write_config_file(tmpdir, sample_config))

    random_array = np.random.rand(5, 2)
    snippet_ids = ['0', '1', '2', '3', '4']  # 0 index for sanity :D
    data = xr.DataArray(
        random_array,
        coords={
            'snippet': snippet_ids,
            'symbol': ['window.navigator', 'canvas.context'],
        },
        dims=('snippet', 'symbol')
    )
    f = ds.dye_score_data_file('snippets')
    data.to_dataset(name='data').to_zarr(store=ds.get_zarr_store(f))

    # Run Test
    dye_snippets = ['2']
    result_file = ds.compute_distances_for_dye_snippets(
        dye_snippets, override=True
    )

    # Check Results
    results = xr.open_zarr(store=ds.get_zarr_store(result_file))['data']
    assert results.shape == (5, 1)
    for s in snippet_ids:
        actual_result = results.sel(snippet=s, dye_snippet='2').values
        expected_result = chebyshev(random_array[2], random_array[int(s)])
        assert actual_result == expected_result


def test_passing_unavailable_string_fails(tmpdir, sample_config):
    # Setup
    sample_config['DYESCORE_DATA_DIR'] = tmpdir.strpath
    ds = DyeScore(write_config_file(tmpdir, sample_config))
    random_array = np.random.rand(5, 2)
    snippet_ids = ['0', '1', '2', '3', '4']
    data = xr.DataArray(
        random_array,
        coords={
            'snippet': snippet_ids,
            'symbol': ['window.navigator', 'canvas.context'],
        },
        dims=('snippet', 'symbol')
    )
    f = ds.dye_score_data_file('snippets')
    data.to_dataset(name='data').to_zarr(store=ds.get_zarr_store(f))

    # Run Test
    with pytest.raises(KeyError):
        ds.compute_distances_for_dye_snippets(
            ['2'], override=True, distance_function='euclidean',
        )
