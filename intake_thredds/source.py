import fnmatch

from intake_xarray.base import DataSourceMixin

from .cat import ThreddsCatalog

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class THREDDSMergedSource(DataSourceMixin):
    """Merges multiple datasets into a single datasets.

    This source takes a THREDDS URL and a path to descend down, and calls the
    combine function on all of the datasets found.

    Parameters
    ----------
    url : str
        Location of server
    path : list of str
        Subcats to follow; include glob characters (*, ?) in here for matching.
    driver : str
        Select driver to access data. Choose from 'netcdf' and 'opendap'.
    xarray_kwargs: dict
        kwargs to be passed to xr.open_dataset
    concat_kwargs: dict
        kwargs to be passed to xr.concat() filled by files opened by xr.open_dataset
        previously
    progressbar : bool
        If True, will print a progress bar. Requires `tqdm <https://github.com/tqdm/tqdm>`__
        to be installed.
    metadata : dict or None
        To associate with this source.

    Examples
    --------
    >>> import intake
    >>> cat_url = 'https://psl.noaa.gov/thredds/catalog.xml'
    >>> paths = ['Datasets', 'ncep.reanalysis.dailyavgs', 'surface', 'air.sig995.194*.nc']
    >>> src = intake.open_thredds_merged(cat_url, paths)
    >>> src
    sources:
    thredds_merged:
        args:
        path:
        - Datasets
        - ncep.reanalysis.dailyavgs
        - surface
        - air*sig995*194*.nc
        url: https://psl.noaa.gov/thredds/catalog.xml
        description: ''
        driver: intake_thredds.source.THREDDSMergedSource
        metadata: {}

    """

    version = '1.0'
    container = 'xarray'
    name = 'thredds_merged'
    partition_access = True

    def __init__(
        self,
        url,
        path,
        driver='opendap',
        progressbar=True,
        xarray_kwargs={},
        concat_kwargs=None,
        metadata=None,
    ):

        super(THREDDSMergedSource, self).__init__(metadata=metadata)
        self.urlpath = url
        if 'simplecache::' in url:
            self.metadata.update({'fsspec_pre_url': 'simplecache::'})
        self.path = path
        self.driver = driver
        self.xarray_kwargs = xarray_kwargs
        self.concat_kwargs = concat_kwargs
        self._ds = None
        self.progressbar = progressbar
        if self.progressbar and tqdm is None:
            raise ValueError(
                "Missing package 'tqdm' required for progress bars."
                'You can install tqdm via (1) python -m pip install tqdm or (2) conda install -c conda-forge tqdm.'
                "In case you don't want to install tqdm, please use `progressbar=False`."
            )

    def _open_dataset(self):
        import xarray as xr

        if self._ds is None:
            cat = ThreddsCatalog(self.urlpath, driver=self.driver)
            for i in range(len(self.path)):
                part = self.path[i]
                if '*' not in part and '?' not in part:
                    cat = cat[part](driver=self.driver)
                else:
                    break
            path = self.path[i:]
            if self.progressbar:
                data = [
                    ds(xarray_kwargs=self.xarray_kwargs).to_dask()
                    for ds in tqdm(_match(cat, path), desc='Dataset(s)', ncols=79)
                ]
            else:
                data = [ds(xarray_kwargs=self.xarray_kwargs).to_dask() for ds in _match(cat, path)]
            if self.concat_kwargs:
                self._ds = xr.concat(data, **self.concat_kwargs)
            else:
                self._ds = xr.combine_by_coords(data, combine_attrs='override')


def _match(cat, patterns):
    out = []
    for name in cat:
        if fnmatch.fnmatch(name, patterns[0]):
            if len(patterns) == 1:
                out.append(cat[name](chunks={}))
            else:
                out.extend(_match(cat[name](), patterns[1:]))
    return out
