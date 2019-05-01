# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil
import yaml

from dask.dataframe import (
    from_pandas,
    read_csv,
    read_parquet,
)
from pandas import (
    DataFrame as pd_DataFrame,
    read_csv as pd_read_csv,
)
from pprint import pprint
from s3fs import S3FileSystem, S3Map
from xarray import (
    apply_ufunc,
    DataArray,
    open_zarr,
)
try:
    from pyspark.sql.functions import udf
except ModuleNotFoundError:
    print('PySpark not available for data processing.')

from .distances import (
    get_chebyshev_distances_xarray_ufunc,
)
from .utils import (
    get_netloc,
    get_path,
    get_end_of_path,
    get_clean_script,
)


def get_raw_snippet_from_row(row):
    script_url = row.script_url
    func_name = row.func_name
    if script_url == '':
        script_url = row.top_level_url
    netloc = get_netloc(script_url)
    path = get_path(script_url)
    path_end = get_end_of_path(path)
    return netloc + '||' + path_end + '||' + func_name


class DyeScore:
    """
    Args:
        config_file_path (str): The path of your config file that is used for dye score to interact with your
            environment.  Holds references to file paths and private data such as AWS API keys. Expects a YAML file with
            the following keys:

            * INPUT_PARQUET_LOCATION - the location of the raw or sampled OpenWPM input parquet folder
            * DYESCORE_DATA_DIR - location where you would like dye score to store data assets
            * DYESCORE_RESULTS_DIR - location where you would like dye score to store results assets
            * USE_AWS - default False - set true if data store is AWS
            * AWS_ACCESS_KEY_ID - optional - for storing and retrieving data on AWS
            * AWS_SECRET_ACCESS_KEY - optional - for storing and retrieving data on AWS

            Locations can be a local file path or a bucket.
        validate_config (bool, optional): Run ``DyeScore.validate_config`` method. Defaults to ``True``.
        print_config (bool, optional): Print out config once saved. Defaults to ``False``.
        sc (SparkContext, optional): If accessing s3 via s3a, pass spark context to set aws credentials
    """

    __conf = {
        "INPUT_PARQUET_LOCATION": "",
        "DYESCORE_DATA_DIR": "",
        "DYESCORE_RESULTS_DIR": "",
        "USE_AWS": False,
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
    }

    dye_score_columns = [
        'top_level_url',
        'script_url',
        'func_name',
        'symbol',
    ]

    dye_score_files = {
        'raw_snippet_call_df': 'raw_snippet_call_df.parquet',
        'raw_snippet_to_snippet_lookup': 'snippet_lookup.parquet',
        'snippets': 'snippets.zarr',
        'snippet_dyeing_map': 'snippet_dyeing_map.parquet',
    }

    def __init__(self, config_file_path, validate_config=True, print_config=False, sc=None):
        if not os.path.exists(config_file_path):
            raise ValueError(f'config_file_path `{config_file_path}` not found')

        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f.read())
            self.__conf['INPUT_PARQUET_LOCATION'] = config['INPUT_PARQUET_LOCATION']
            self.__conf['DYESCORE_DATA_DIR'] = config['DYESCORE_DATA_DIR']
            self.__conf['DYESCORE_RESULTS_DIR'] = config['DYESCORE_RESULTS_DIR']
            self.__conf['TMP_DIR'] = config.get('TMP_DIR', config['DYESCORE_DATA_DIR'])  # Default to data dir if not provided
            use_aws = config.get('USE_AWS', False)
            self.__conf['USE_AWS'] = bool(use_aws)
            self.__conf['SPARK_S3_PROTOCOL'] = config.get('SPARK_S3_PROTOCOL', 's3')
            self.__conf['AWS_ACCESS_KEY_ID'] = config.get('AWS_ACCESS_KEY_ID', '')
            self.__conf['AWS_SECRET_ACCESS_KEY'] = config.get('AWS_SECRET_ACCESS_KEY', '')
        if print_config:
            pprint(self.__conf)
        if validate_config:
            self.validate_config()
        if use_aws:
            self.s3 = S3FileSystem(**self.s3_storage_options)
        else:
            self.s3 = None
        if sc and use_aws and self.config('SPARK_S3_PROTOCOL') == 's3a':
            sc._jsc.hadoopConfiguration().set('fs.s3a.access.key', self.config('AWS_ACCESS_KEY_ID'))
            sc._jsc.hadoopConfiguration().set('fs.s3a.secret.key', self.config('AWS_SECRET_ACCESS_KEY'))

    @property
    def s3_storage_options(self):
        """s3 storage options built from config

        Returns:
            dict. if USE_AWS is True returns s3 options as dict, else None.
        """

        if self.config('USE_AWS') is True:
            return dict(
                anon=False,
                key=self.config('AWS_ACCESS_KEY_ID'),
                secret=self.config('AWS_SECRET_ACCESS_KEY')
            )
        else:
            return None

    @property
    def to_parquet_opts(self):
        """Options used when saving to parquet."""
        return dict(
            compression='snappy', engine='pyarrow', storage_options=self.s3_storage_options
        )

    @property
    def from_parquet_opts(self):
        """Options used when saving to parquet."""
        return dict(
            engine='pyarrow', storage_options=self.s3_storage_options
        )

    def get_zarr_store(self, file_path):
        if self.config('USE_AWS') is True:
            return S3Map(root=file_path.lstrip('s3://'), s3=self.s3)
        else:
            return file_path

    def config(self, option):
        """Method to retrieve config values

        Args:
            option (str): The desired config option key
        Returns:
            The config option value
        """
        return self.__conf[option]

    def validate_config(self):
        """Validate the config data.
        Currently just checks that values are correct for aws.

        Raises AssertionError if values are incorrect.
        """
        if self.config('USE_AWS') is True:
            assert self.config('INPUT_PARQUET_LOCATION').startswith(f's3://')
            assert self.config('DYESCORE_DATA_DIR').startswith(f's3://')

    def dye_score_data_file(self, filename):
        """Helper function to return standardized filename.

        DyeScore class holds a dictionary to standardize the file
        names that DyeScore saves. This method looks up filenames by their
        short name.

        Args:
            filename (str): data file name
        Returns:
            str. The path where the data file should reside
        """
        dyescoredir = self.config('DYESCORE_DATA_DIR')
        path = os.path.join(dyescoredir, self.dye_score_files[filename])
        return path

    def validate_input_data(self):
        """Checks for expected columns and types in input data."""
        in_file = self.config('INPUT_PARQUET_LOCATION')
        df = read_parquet(in_file, **self.from_parquet_opts)
        for column in self.dye_score_columns:
            assert column in df.columns, f'{column} missing from df.columns ({df.columns})'
            assert df[column].dtype == 'object', f'{column} does not have dtype `object`'
        return True

    def spark_convert(self, path):
        if self.config('USE_AWS') and self.config('SPARK_S3_PROTOCOL') == 's3a':
            return path.replace('s3://', 's3a://')
        else:
            return path

    def get_input_df(self, columns=None):
        """Helper function to return the input dataframe.

        Args:
            columns (list, optional): List of columns to retrieve. If None, all columns are returned.
        Returns:
            dask.DataFrame. Input dataframe with subset of columns requested.
        """
        if not columns:
            columns = self.dye_score_columns
        in_file = self.config('INPUT_PARQUET_LOCATION')
        df = read_parquet(in_file, columns=columns, **self.from_parquet_opts)
        return df

    ##
    # DATA PROCESSING
    #
    # In the next few methods we switch between dask and spark. We switch to spark
    # whenever we need to leverage its superior performance handling strings.
    ##

    def file_in_validation(self, inpath):
        """Check path exists.

        Raises ValueError if not. Used for input files, as these must exist to proceed.
        Args:
            inpath (str): Path of input file
        """
        if self.config('USE_AWS') is True:
            exists = self.s3.exists(inpath)
        else:
            exists = os.path.exists(inpath)
        if not exists:
            raise ValueError(f'File {inpath} does not exist. Cannot proceed.')

    def file_out_validation(self, outpath, override):
        """Check path exists.
        Raises ValueError if override is False. Otherwises removes the existing file.
        Args:
            outpath (str): Path of ourput file.
            override (bool): Whether to raise an error or remove existing data.
        """
        if self.config('USE_AWS') is True:
            exists = self.s3.exists(outpath)
        else:
            exists = os.path.exists(outpath)

        if exists and override is False:
            raise ValueError(f'File {outpath} already exists. Use `override=True` to remove and replace.')
        if exists and override is True:
            print(f'Removing existing file {outpath}')
            if self.config('USE_AWS') is True:
                self.s3.rm(outpath, recursive=True)
            else:
                shutil.rmtree(outpath)

    def build_raw_snippet_df(self, override=False, snippet_func=None):
        """Builds raw_snippets from input data

        Snippet function is ``script_url.netloc||script_url.path_end||func_name``
        If script_url is missing, location is used.

        Args:
            override (bool): True to replace any existing outputs

        Returns:
            str. The file path where output is saved
        """
        # File setup
        inpath = self.config('INPUT_PARQUET_LOCATION')
        outpath = self.dye_score_data_file('raw_snippet_call_df')
        self.file_in_validation(inpath)
        self.file_out_validation(outpath, override)
        # Process
        if not snippet_func:
            snippet_func = get_raw_snippet_from_row
        df = read_parquet(inpath, columns=self.dye_score_columns, **self.from_parquet_opts)
        df['raw_snippet'] = df.apply(snippet_func, axis=1, meta='O')
        df['called'] = 1
        print(df.head())
        df.to_parquet(outpath, **self.to_parquet_opts)
        return outpath

    def build_snippet_map(self, override=False):
        """Builds snippet ids and saves map of ids to raw snippets

        xarray cannot handle arbitrary length string indexes so we need to build a set of unique
        ids to reference snippets. This method creates the ids and saves the map of raw ids to snippets.

        Args:
            override (bool): True to replace any existing outputs

        Returns:
            str. The file path where output is saved
        """
        # File setup
        inpath = self.dye_score_data_file('raw_snippet_call_df')
        outpath = self.dye_score_data_file('raw_snippet_to_snippet_lookup')
        self.file_in_validation(inpath)
        self.file_out_validation(outpath, override)
        # Process
        df = read_parquet(inpath, columns=['raw_snippet'], **self.from_parquet_opts)
        snippet_lookup = df.raw_snippet.unique().to_frame()
        snippet_lookup['snippet'] = snippet_lookup.raw_snippet.apply(lambda x: hash(x), meta='int64')
        print(snippet_lookup.head())
        snippet_lookup.to_parquet(outpath, **self.to_parquet_opts)
        return outpath

    def build_snippets(self, spark, override=False):
        """Builds row-normalized snippet dataset

        * Dimensions are n snippets x s unique symbols in dataset.
        * Data is output in zarr format with processing by spark, dask, and xarray.
        * Creates an intermediate tmp file when converting from spark to dask.
        * Slow running operation - follow spark and dask status to see progress
        
        We use spark here because dask cannot memory efficiently compute a pivot
        table. This is the only function we need spark context for.

        Args:
            spark (pyspark.sql.session.SparkSession): spark instance
            override (bool): True to replace any existing outputs

        Returns:
            str. The file path where output is saved
        """
        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        print("""
        This function uses both spark and dask, make sure you have workers 
        available for both.""")

        # File setup
        outpath = self.dye_score_data_file('snippets')
        snippet_map = self.dye_score_data_file('raw_snippet_to_snippet_lookup')
        inpath = self.dye_score_data_file('raw_snippet_call_df')
        self.file_out_validation(outpath, override)
        self.file_in_validation(snippet_map)
        self.file_in_validation(inpath)

        # Get symbols
        inpath = self.dye_score_data_file('raw_snippet_call_df')
        self.file_in_validation(inpath)
        df = read_parquet(inpath, columns=['symbol'], **self.from_parquet_opts)
        symbols = df.symbol.unique().compute()
        symbols = sorted(list(symbols.values))
        print(f'Dataset has {len(symbols)} unique symbols')
        
        # Setup spark files
        snippet_map = self.spark_convert(snippet_map)
        inpath = self.spark_convert(inpath)

        # Process - pivot with spark and save to tmp file
        df_map = spark.read.parquet(snippet_map).select(['raw_snippet', 'snippet'])
        df = spark.read.parquet(inpath).select(['symbol', 'called', 'raw_snippet'])
        df_to_pivot = df.join(df_map, on='raw_snippet')
        df_to_pivot = df_to_pivot.drop('raw_snippet')
        pivot = df_to_pivot.groupBy('snippet').pivot('symbol', symbols).sum('called')
        pivot = pivot.na.fill(0)

        tmp_dir = self.config('TMP_DIR')
        tmp = os.path.join(tmp_dir, 'tmp.csv')
        if self.config('USE_AWS'):
            if self.s3.exists(tmp):
                self.s3.rm(tmp, recursive=True)
        else:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
        pivot.write.csv(self.spark_convert(tmp), header=True)

        # Process - set_index, normalize and save to zarr
        dtypes = {symbol: 'float64' for symbol in symbols}
        dtypes['snippet'] = 'object'
        pivot_table = read_csv(f'{tmp}/*.csv', dtype=dtypes, storage_options=self.s3_storage_options)
        pivot_table = pivot_table.set_index('snippet')

        row_normalize = pivot_table.div(pivot_table.sum(axis=1), axis=0)
        row_normalize_array = row_normalize.to_dask_array(lengths=True)

        # - xarray needs uniformly sized chunks
        row_normalize_array = row_normalize_array.rechunk({0: 'auto', 1: -1})
        data_array = DataArray(
                row_normalize_array,
                dims=['snippet', 'symbol'],
                coords={
                    'snippet': row_normalize.index.values,
                    'symbol': row_normalize.columns
                }
        )
        print(data_array)
        data_array.to_dataset(name='data').to_zarr(store=self.get_zarr_store(outpath))

        # Cleanup
        if self.config('USE_AWS'):
            self.s3.rm(tmp, recursive=True)
        else:
            shutil.rmtree(tmp)
        return outpath

    def build_snippet_snippet_dyeing_map(self, spark, override=False):
        """Build file used to join snippets to data for dyeing.

        Adds clean_script field to dataset. Saves parquet file with:
            * snippet - the int version, not raw_snippet
            * top_level_url
            * script_url
            * clean_script

        Args:
            spark (pyspark.sql.session.SparkSession): spark instance
            override (bool): True to replace any existing outputs

        Returns:
            str. The file path where output is saved

        """
        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        print("""
        This function uses both spark and dask, make sure you have workers 
        available for both.""")

        # File setup
        outpath = self.dye_score_data_file('snippet_dyeing_map')
        snippet_map = self.dye_score_data_file('raw_snippet_to_snippet_lookup')
        inpath = self.dye_score_data_file('raw_snippet_call_df')
        self.file_out_validation(outpath, override)
        self.file_in_validation(snippet_map)
        self.file_in_validation(inpath)

        tmp_dir = self.config('TMP_DIR')
        tmp = os.path.join(tmp_dir, 'tmp.parquet')
        if self.config('USE_AWS'):
            if self.s3.exists(tmp):
                self.s3.rm(tmp, recursive=True)
        else:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)

        # Process
        # - use spark to drop duplicates
        df = spark.read.parquet(self.spark_convert(inpath)).select(['top_level_url', 'script_url', 'func_name', 'raw_snippet'])
        df = df.drop_duplicates()
        df = df.coalesce(40)
        df = df.write.parquet(self.spark_convert(tmp))

        # - merge snippets in
        df = read_parquet(tmp + '/*.parquet', **self.from_parquet_opts)
        df['clean_script'] = df.script_url.apply(get_clean_script, meta='O')
        df_map = read_parquet(snippet_map, columns=['snippet', 'raw_snippet'], **self.from_parquet_opts)
        df = df.merge(df_map, on='raw_snippet')
        df = df.drop('raw_snippet', axis=1)
        df.to_parquet(outpath, **self.to_parquet_opts)

        # Cleanup
        if self.config('USE_AWS'):
            self.s3.rm(tmp, recursive=True)
        else:
            shutil.rmtree(tmp)

        return outpath

    ##
    # Dyeing and Scoring
    ##

    def compute_distances_for_dye_snippets(self, dye_snippets, filename_suffix='dye_snippets', override=False, snippet_chunksize=1000, dye_snippet_chunksize=1000):
        """Computes all pairwise distances from dye snippets to all other snippets.

        * Expects snippets file to exist.
        * Uses chebyshev distance.
        * Writes results to zarr with name ``snippets_dye_distances_from_{filename_suffix}``
        * This is a long-running function - see dask for progress

        Args:
            dye_snippets (np.array): Numpy array of snippets to be dyed. Must be a subset of snippets index.
            filename_suffix (str, optional): Change to differentiate between dye_snippet sets. Defaults to
                ``dye_snippets``
            override (bool, optional): Override output files. Defaults to ``False``.
        Returns:
            str. Path results were written to

        """
        # File setup
        snippet_file = self.dye_score_data_file('snippets')
        self.file_in_validation(snippet_file)
        resultsdir = self.config('DYESCORE_RESULTS_DIR')
        file_name = f'snippets_dye_distances_from_{filename_suffix}'
        outpath = os.path.join(resultsdir, file_name)
        self.file_out_validation(outpath, override)

        # Process distances
        df = open_zarr(store=self.get_zarr_store(snippet_file))['data']
        df = df.chunk({'symbol': -1})
        df_c = df.chunk({'snippet': snippet_chunksize})

        df_dye = df.loc[{'snippet': dye_snippets}]
        df_dye = df_dye.rename({'snippet': 'dye_snippet'})
        df_dye_c = df_dye.chunk({'dye_snippet': dye_snippet_chunksize})

        distance_array = apply_ufunc(
            get_chebyshev_distances_xarray_ufunc,
            df_c, df_dye_c,
            dask='parallelized',
            output_dtypes=[float],
            input_core_dims=[['symbol'], ['symbol']],
        )
        print(distance_array)
        distance_array.to_dataset(name='data').to_zarr(store=self.get_zarr_store(outpath))
        return outpath

    def compute_snippets_scores_for_thresholds(self, thresholds, filename_suffix='dye_snippets', override=False):
        """Get score for snippets for a range of distance thresholds.

        * Uses hard coded leaky threshold of 20%
        * Writes results to parquet files with name ``snippets_score_from_{filename_suffix}_{threshold}``

        Args:
            thresholds (list): List of distances to compute snippet scores for e.g. ``[0.23, 0.24, 0.25]``
            filename_suffix (str, optional): Change to differentiate between dye_snippet sets. Defaults to
                ``dye_snippets``
            override (bool, optional): Override output files. Defaults to ``False``.
        Returns:
            list. Paths results were written to
        """
        resultsdir = self.config('DYESCORE_RESULTS_DIR')
        file_name = f'snippets_dye_distances_from_{filename_suffix}'
        inpath = os.path.join(resultsdir, file_name)
        self.file_in_validation(inpath)
        distance_array = open_zarr(store=self.get_zarr_store(inpath))['data']

        LEAKY_THRESHOLD = 0.2
        n_sites = distance_array.shape[0]
        N_LEAKY_THRESHOLD = LEAKY_THRESHOLD * n_sites

        outpaths = []
        for threshold in thresholds:
            n_to_dye = np.sum(distance_array < threshold, axis=0).persist()
            non_leaky_sites = n_to_dye[n_to_dye < N_LEAKY_THRESHOLD].coords.to_index()
            distance_array_filtered = distance_array.loc[{'dye_snippet': non_leaky_sites}]

            site_counts = np.sum(distance_array_filtered < threshold, axis=1)
            site_counts_df = site_counts.to_dataframe()
            site_counts_df = site_counts_df.reset_index().rename(columns={'data': 'dye_count'})
            site_counts_df['snippet'] = site_counts_df.snippet.astype(int)
            outpath = os.path.join(resultsdir, f'snippets_score_from_{filename_suffix}_{threshold}')
            self.file_out_validation(outpath, override)
            from_pandas(site_counts_df, npartitions=1).to_parquet(outpath, **self.to_parquet_opts)
            outpaths.append(outpath)
        return outpaths

    def compute_dye_scores_for_thresholds(self, thresholds, filename_suffix='dye_snippets', override=False):
        """Get dye scores for a range of distance thresholds.

        * Uses results from ``compute_snippets_scores_for_thresholds``
        * Writes results to gzipped csv files with name ``dye_score_from_{filename_suffix}_{threshold}.csv.gz``

        Args:
            thresholds (list): List of distances to compute snippet scores for e.g. ``[0.23, 0.24, 0.25]``
            filename_suffix (str, optional): Change to differentiate between dye_snippet sets. Defaults to
                ``dye_snippets``
            override (bool, optional): Override output files. Defaults to ``False``.
        Returns:
            list. Paths results were written to
        """
        snippet_dyeing_map_file = self.dye_score_data_file('snippet_dyeing_map')
        snippet_data = read_parquet(snippet_dyeing_map_file, **self.from_parquet_opts)
        resultsdir = self.config('DYESCORE_RESULTS_DIR')

        outpaths = []
        for threshold in thresholds:
            inpath = os.path.join(resultsdir, f'snippets_score_from_{filename_suffix}_{threshold}')
            outpath = os.path.join(resultsdir, f'dye_score_from_{filename_suffix}_{threshold}.csv.gz')
            self.file_in_validation(inpath)
            self.file_out_validation(outpath, override)

            site_counts_df = read_parquet(inpath, **self.from_parquet_opts)
            script_to_dye = snippet_data.merge(site_counts_df, on='snippet')
            script_to_dye_max = script_to_dye[['clean_script', 'dye_count']].groupby('clean_script').max()
            script_to_dye_max = script_to_dye_max.rename(columns={'dye_count': 'dye_score'})
            script_to_dye_max.compute().to_csv(outpath, compression='gzip', storage_options=self.s3_storage_options)
            outpaths.append(outpath)
        return outpaths

    ##
    #  Dye score evaluation
    ##

    def _get_recall(self, dye_score_threshold, score_df, compare_list):
        retrieved = score_df[score_df.dye_score > dye_score_threshold]
        if len(retrieved) > 0:
            retrieved_and_relevant = retrieved[retrieved.clean_script.isin(compare_list)]
            return len(retrieved_and_relevant) / len(compare_list)
        else:
            return np.NaN

    def _build_plot_data_for_score_df(self, score_df, compare_list):
        pr = pd_DataFrame({'dye_score_threshold': np.linspace(0, score_df.dye_score.max(), 1000)})
        pr['recall'] = pr.dye_score_threshold.apply(
            self._get_recall, score_df=score_df, compare_list=compare_list
        )
        pr['n_over_threshold'] = pr.dye_score_threshold.apply(lambda x: (score_df.dye_score > x).sum())
        return pr

    def build_plot_data_for_thresholds(self, compare_list, thresholds, filename_suffix='dye_snippets', override=False):
        """Builds a dataframe for evaluation

        Contains the recall compared to the ``compare_list`` for  scripts under the threshold.

        Args:
            compare_list (list): List of dye scripts to compare for recall.
            thresholds (list): List of distances to compute snippet scores for e.g. ``[0.23, 0.24, 0.25]``
            filename_suffix (str, optional): Change to differentiate between dye_snippet sets. Defaults to
                ``dye_snippets``
            override (bool, optional): Override output files. Defaults to ``False``.
        Returns:
            list. Paths results were written to
        """
        resultsdir = self.config('DYESCORE_RESULTS_DIR')
        outpaths = []
        for threshold in thresholds:
            inpath = os.path.join(resultsdir, f'dye_score_from_{filename_suffix}_{threshold}.csv.gz')
            outpath = os.path.join(resultsdir, f'dye_score_plot_data_from_{filename_suffix}_{threshold}.csv.gz')
            self.file_in_validation(inpath)
            self.file_out_validation(outpath, override)
            dye_score_df = pd_read_csv(inpath, storage_options=self.s3_storage_options)
            plot_df = self._build_plot_data_for_score_df(dye_score_df, compare_list)
            plot_df.to_csv(outpath, compression='gzip', index=False, storage_options=self.s3_storage_options)
            outpaths.append(outpath)
        return outpaths
