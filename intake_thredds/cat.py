from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry


class ThreddsCatalog(Catalog):

    name = 'thredds_cat'

    def __init__(self, url: str, driver: str = 'opendap', **kwargs):
        """Intake catalog interface to a thredds catalog.

        Parameters
        ----------
        url : str
            Location of thredds catalog.
        driver : str
            Select driver to access data. Choose from 'netcdf' and 'opendap'.
        **kwargs :
            Additional keyword arguments are passed through to the
            :py:class:`~intake.catalog.Catalog` base class.
        """
        self.url = url
        self.driver = driver
        super().__init__(**kwargs)

    def _load(self):
        from siphon.catalog import TDSCatalog

        if 'simplecache::' in self.url:
            if self.driver == 'netcdf':
                self.cache = True
                self.url_no_simplecache = self.url.replace('simplecache::', '')
                self.metadata.update({'fsspec_pre_url': 'simplecache::'})
            else:
                raise ValueError(
                    f'simplecache requires driver="netcdf", found driver="{self.driver}".'
                )
        else:
            self.cache = False
            self.url_no_simplecache = self.url

        self.cat = TDSCatalog(self.url_no_simplecache)
        self.name = self.cat.catalog_name
        self.metadata.update(self.cat.metadata)

        # sub-cats
        self._entries = {
            r.title: LocalCatalogEntry(
                r.title,
                'THREDDS cat',
                'thredds_cat',
                True,
                {'url': r.href},
                [],
                [],
                self.metadata,
                None,
                catalog=self,
            )
            for r in self.cat.catalog_refs.values()
        }

        def access_urls(ds, self):
            if self.driver == 'opendap':
                driver_for_access_urls = 'OPENDAP'
            elif self.driver == 'netcdf':
                driver_for_access_urls = 'HTTPServer'
            url = ds.access_urls[driver_for_access_urls]
            if 'fsspec_pre_url' in self.metadata.keys():
                url = f'{self.metadata["fsspec_pre_url"]}{url}'
            return url

        self._entries.update(
            {
                ds.name: LocalCatalogEntry(
                    ds.name,
                    'THREDDS data',
                    self.driver,
                    True,
                    {'urlpath': access_urls(ds, self), 'chunks': None},
                    [],
                    [],
                    {},
                    None,
                    catalog=self,
                )
                for ds in self.cat.datasets.values()
            }
        )
