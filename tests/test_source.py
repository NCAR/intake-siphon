import intake
import pytest
import xarray as xr


@pytest.fixture(scope='module')
def THREDDSMergedSource_cat():
    """THREDDSMergedSource looping through folders."""
    thredds_cat_url = 'https://psl.noaa.gov/thredds/catalog.xml'
    paths = [
        'Datasets',
        'ncep.reanalysis.dailyavgs',
        'surface',
        'air.sig995.194*.nc',
    ]
    cat = intake.open_thredds_merged(thredds_cat_url, paths)
    assert cat.urlpath == thredds_cat_url
    assert cat.path == paths
    return cat


@pytest.fixture(scope='module')
def THREDDSMergedSource_cat_short_url():
    return 'https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis.dailyavgs/surface/catalog.xml'


@pytest.fixture(scope='module')
def THREDDSMergedSource_cat_short_path():
    return ['air.sig995.194*.nc']


@pytest.fixture(scope='module')
def THREDDSMergedSource_cat_short(
    THREDDSMergedSource_cat_short_url, THREDDSMergedSource_cat_short_path
):
    """THREDDSMergedSource without the looping faster."""
    cat = intake.open_thredds_merged(
        THREDDSMergedSource_cat_short_url, THREDDSMergedSource_cat_short_path
    )
    assert cat.urlpath == THREDDSMergedSource_cat_short_url
    assert cat.path == THREDDSMergedSource_cat_short_path
    return cat


@pytest.fixture(scope='module')
def THREDDSMergedSource_cat_short_simplecache(
    THREDDSMergedSource_cat_short_url, THREDDSMergedSource_cat_short_path
):
    """Without the looping faster."""
    return intake.open_thredds_merged(
        f'simplecache::{THREDDSMergedSource_cat_short_url}',
        THREDDSMergedSource_cat_short_path,
        driver='netcdf',
    )


def test_THREDDSMergedSource(THREDDSMergedSource_cat):
    cat = THREDDSMergedSource_cat
    ds = cat.to_dask()
    assert dict(ds.dims) == {'lat': 73, 'lon': 144, 'time': 731}
    d = cat.discover()
    assert set(d['metadata']['coords']) == set(('lat', 'lon', 'time'))
    assert set(d['metadata']['data_vars'].keys()) == set(['air'])


def test_THREDDSMergedSource_long_short(THREDDSMergedSource_cat, THREDDSMergedSource_cat_short):
    ds = THREDDSMergedSource_cat.to_dask()
    ds_short = THREDDSMergedSource_cat_short.to_dask()
    for c in ds.coords:
        assert (ds[c] == ds_short[c]).all()
    assert ds.sizes == ds_short.sizes


def test_THREDDSMergedSource_simplecache_netcdf(THREDDSMergedSource_cat_short_simplecache):
    """Test that THREDDSMergedSource allows simplecache:: in url if netcdf as source."""
    import os

    import fsspec

    cache_storage = 'my_caching_folder'
    fsspec.config.conf['simplecache'] = {'cache_storage': cache_storage, 'same_names': True}
    cat = THREDDSMergedSource_cat_short_simplecache
    ds = cat.to_dask()
    assert isinstance(ds, xr.Dataset)
    # test files present
    cached_files = ['air.sig995.1948.nc', 'air.sig995.1949.nc']
    for f in cached_files:
        cached_file = os.path.join(cache_storage, f)
        assert os.path.exists(cached_file)
        os.remove(cached_file)
        assert not os.path.exists(cached_file)


def test_THREDDSMergedSource_simplecache_fails_opendap(THREDDSMergedSource_cat_short_url):
    """Test that THREDDSMergedSource simplecache:: in url with opendap."""
    with pytest.raises(ValueError, match=r'simplecache requires driver="netcdf"'):
        intake.open_thredds_cat(
            f'simplecache::{THREDDSMergedSource_cat_short_url}', driver='opendap'
        )


@pytest.mark.parametrize('driver', ['netcdf', 'opendap'])
@pytest.mark.parametrize('decode_times', [True, False])
def test_THREDDSMergedSource_xarray_kwargs(THREDDSMergedSource_cat_short_url, driver, decode_times):
    """Test THREDDSMergedSource with xarray_kwargs."""
    ds = intake.open_thredds_merged(
        'https://psl.noaa.gov/thredds/catalog.xml',
        [
            'Datasets',
            'ncep.reanalysis.dailyavgs',
            'surface',
            'air.sig995.194*.nc',
        ],
        driver=driver,
        xarray_kwargs={'decode_times': decode_times},
    ).to_dask()
    # check xarray_kwargs
    if decode_times:
        assert 'units' not in ds.time.attrs
    else:
        assert 'units' in ds.time.attrs


def test_concat_dim():
    """Test THREDDSMergedSource with concat_dim. Requires multiple files with same
    other coords to be concatinated along new dimension specified by concat_dim.
    Here get two ensemble members initialized 20200831 00:00 at 15.5 days = 372h"""
    url = 'simplecache::https://www.ncei.noaa.gov/thredds/catalog/model-gefs-003/202008/20200831/catalog.xml'
    ds = intake.open_thredds_merged(
        url,
        ['NCEP gens-a Grid 3 Member-Forecast 1[1-2]*-372 for 2020-08-31 00:00*'],
        driver='netcdf',
        xarray_kwargs=dict(
            engine='cfgrib',
            backend_kwargs=dict(
                filter_by_keys={'typeOfLevel': 'heightAboveGround', 'shortName': '2t'}
            ),
        ),
        concat_kwargs=dict(dim='number'),
    ).to_dask()
    assert 'number' in ds.dims
    assert 11 in ds.number
    assert 12 in ds.number
