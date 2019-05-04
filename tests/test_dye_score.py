#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dask.dataframe as dd
import os
import pandas as pd
import pytest
import yaml

from pyarrow.lib import ArrowIOError
from yaml.scanner import ScannerError

from dye_score import DyeScore


##
# Test Data Validation
##

def test_data_validation_with_invalid_file(tmpdir, sample_config):
    # Set-up invalid data file and save config
    data_dir = os.path.join(tmpdir, 'data.csv')
    config_file = os.path.join(tmpdir, 'config.yaml')
    df = pd.DataFrame({'a': [1, 2, 3]})
    df.to_csv(data_dir)
    sample_config['INPUT_PARQUET_LOCATION'] = data_dir
    with open(config_file, 'w') as f:
        f.write(yaml.dump(sample_config))
    ds = DyeScore(config_file)
    # Test
    with pytest.raises(ArrowIOError):
        ds.validate_input_data()


def test_data_validation_with_valid_file(tmpdir, sample_config):
    # Set-up valid data file and save config
    data_dir = os.path.join(tmpdir, 'data.parquet')
    config_file = os.path.join(tmpdir, 'config.yaml')
    df = pd.DataFrame({
        'top_level_url': ['a', 'b'],
        'document_url': ['a', 'b'],
        'script_url': ['c', 'd'],
        'symbol': ['e', 'f'],
        'func_name': ['g', 'h']
    })
    dfd = dd.from_pandas(df, npartitions=2)
    dfd.to_parquet(data_dir)
    sample_config['INPUT_PARQUET_LOCATION'] = data_dir
    with open(config_file, 'w') as f:
        f.write(yaml.dump(sample_config))
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
