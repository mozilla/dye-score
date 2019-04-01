# -*- coding: utf-8 -*

import os
import pytest


@pytest.fixture
def asset_dir():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, 'assets')


@pytest.fixture
def sample_config():
    return {
        "INPUT_PARQUET_LOCATION": "",
        "DYESCORE_DATA_DIR": "",
        "DYESCORE_RESULTS_DIR": "",
        "USE_AWS": False,
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
    }
