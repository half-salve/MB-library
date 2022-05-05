"""Basic DGL Dataset
"""

from __future__ import absolute_import

import os, sys, hashlib
import traceback
import abc
from .utils import download, extract_archive, get_download_dir, makedirs

class DGDataset(object):
    r"""The basic DGL dataset for creating graph datasets.
    This class defines a basic template class for DGL Dataset.
    The following steps will are executed automatically:

      1. Check whether there is a dataset cache on disk
         (already processed and stored on the disk) by
         invoking ``has_cache()``. If true, goto 5.
      2. Call ``download()`` to download the data.
      3. Call ``process()`` to process the data.
      4. Call ``save()`` to save the processed dataset on disk and goto 6.
      5. Call ``load()`` to load the processed dataset from disk.
      6. Done.

    Users can overwite these functions with their
    own data processing logic.

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: same as raw_dir
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
        Default: (), the corresponding hash value is ``'f9065fa7'``.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information

    Attributes
    ----------
    url : str
        The URL to download the dataset
    name : str
        The dataset name
    raw_dir : str
        Raw file directory contains the input data folder
    raw_path : str
        Directory contains the input data files.
        Default : ``os.path.join(self.raw_dir, self.name)``
    save_dir : str
        Directory to save the processed dataset
    save_path : str
        File path to save the processed dataset
    verbose : bool
        Whether to print information
    hash : str
        Hash value for the dataset and the setting.
    """
    def __init__(self, name, url=None, raw_dir=None, save_dir=None,
                 hash_key=(), force_reload=False, verbose=False):
        self._name = name
        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash = self._get_hash()

        # if no dir is provided, the default dgl download dir is used.
        if raw_dir is None:
            self._raw_dir = get_download_dir()
        else:
            self._raw_dir = raw_dir

        if save_dir is None:
            self._save_dir = self._raw_dir
        else:
            self._save_dir = save_dir

        self._load()

    def download(self):
        r"""Overwite to realize your own logic of downloading data.

        It is recommended to download the to the :obj:`self.raw_dir`
        folder. Can be ignored if the dataset is
        already in :obj:`self.raw_dir`.
        """
        pass

    def save(self):
        r"""Overwite to realize your own logic of
        saving the processed dataset into files.

        It is recommended to use ``dgl.utils.data.save_graphs``
        to save dgl graph into files and use
        ``dgl.utils.data.save_info`` to save extra
        information into files.
        """
        pass

    def load(self):
        r"""Overwite to realize your own logic of
        loading the saved dataset from files.

        It is recommended to use ``dgl.utils.data.load_graphs``
        to load dgl graph from files and use
        ``dgl.utils.data.load_info`` to load extra information
        into python dict object.
        """
        pass

    def process(self):
        r"""Overwrite to realize your own logic of processing the input data.
        """
        raise NotImplementedError

    def has_cache(self):
        r"""Overwrite to realize your own logic of
        deciding whether there exists a cached dataset.

        By default False.
        """
        return False

    def _download(self):
        r"""Download dataset by calling ``self.download()`` if the dataset does not exists under ``self.raw_path``.
            By default ``self.raw_path = os.path.join(self.raw_dir, self.name)``
            One can overwrite ``raw_path()`` function to change the path.
        """
        if os.path.exists(self.raw_path):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _load(self):
        r"""Entry point from __init__ to load the dataset.
            if the cache exists:
                Load the dataset from saved dgl graph and information files.
                If loadin process fails, re-download and process the dataset.
            else:
                1. Download the dataset if needed.
                2. Process the dataset and build the dgl graph.
                3. Save the processed dataset into files.
        """
        load_flag = not self._force_reload and self.has_cache()

        if load_flag:
            try:
                self.load()
                if self.verbose:
                    print('Done loading data from cached files.')
            except KeyboardInterrupt:
                raise
            except:
                load_flag = False
                if self.verbose:
                    print(traceback.format_exc())
                    print('Loading from cache failed, re-processing.')

        if not load_flag:
            self._download()
            self.process()
            self.save()
            if self.verbose:
                print('Done saving data into cached files.')

    def _get_hash(self):
        """Compute the hash of the input tuple

        Example
        -------
        Assume `self._hash_key = (10, False, True)`

        >>> hash_value = self._get_hash()
        >>> hash_value
        'a770b222'
        """
        hash_func = hashlib.sha1()
        hash_func.update(str(self._hash_key).encode('utf-8'))
        return hash_func.hexdigest()[:8]

    @property
    def url(self):
        r"""Get url to download the raw dataset.
        """
        return self._url

    @property
    def name(self):
        r"""Name of the dataset.
        """
        return self._name

    @property
    def raw_dir(self):
        r"""Raw file directory contains the input data folder.
        """
        return self._raw_dir

    @property
    def raw_path(self):
        r"""Directory contains the input data files.
            By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return os.path.join(self.raw_dir,self.name,"raw_data")

    @property
    def processed_path(self):
        r"""Directory contains the input data files.
            By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return os.path.join(self.raw_dir,self.name,"processed_data")

    @property
    def save_dir(self):
        r"""Directory to save the processed dataset.
        """
        return self._save_dir

    @property
    def save_path(self):
        r"""Path to save the processed dataset.
        """
        return os.path.join(self._save_dir, self.name)

    @property
    def verbose(self):
        r"""Whether to print information.
        """
        return self._verbose

    @property
    def hash(self):
        r"""Hash value for the dataset and the setting.
        """
        return self._hash


    @abc.abstractmethod
    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        r"""The number of examples in the dataset."""
        pass

class DGBuiltinDataset(DGDataset):
    r"""The Basic DGL Builtin Dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    url : str
        Url to download the raw dataset.
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: False
    """
    def __init__(self, name, url, raw_dir=None, force_reload=False, verbose=False):
        super(DGBuiltinDataset, self).__init__(name,
                                                url=url,
                                                raw_dir=raw_dir,
                                                save_dir=None,
                                                hash_key=hash_key,
                                                force_reload=force_reload,
                                                verbose=verbose)
        self._node_state=None
        self._edge_index = None
        self._node_features = None
        self._edge_features = None
        self._timestamp = None
        self._node_label = None


    def has_cache(self):
        assert os.path.exists(self.processed_path), 'The provided path does not exist'
        return os.path.exists(self.processed_path)

    def load(self):
        src_dst_timePath = os.path.join(self.processed_path, '{}_src_dst_time.npy'.format(self.name))
        edge_featuresPath = os.path.join(self.processed_path, '{}_edge_features.npy'.format(self.name))
        node_featuresPath = os.path.join(self.processed_path, '{}_node_features.npy'.format(self.name))
        node_labelPath = os.path.join(self.processed_path, '{}_node_label.npy'.format(self.name))
        node_statePath = os.path.join(self.processed_path, '{}_node_state.npy'.format(self.name))
        if os.path.exists(src_dst_timePath):
            src_dst_time = np.load(src_dst_timePath)
            self._edge_index = torch.LongTensor(src_dst_time[0:2, :])
            self._timestamp = torch.FloatTensor(src_dst_time[:, -1])

        if os.path.exists(edge_featuresPath):
            edge_features = np.load(edge_featuresPath)
            self._edge_features = torch.FloatTensor(edge_features)

        if os.path.exists(node_featuresPath):
            node_features = np.load(node_featuresPath)
            self._node_features = torch.FloatTensor(node_features)

        if os.path.exists(node_labelPath):
            id_label = np.load(node_labelPath)
            self._node_label = torch.FloatTensor(id_label)

        if os.path.exists(node_statePath):
            node_state = np.load(node_statePath)
            self._node_state = node_state

    @property
    def edge_index(self):
        r"""return edge index
        """
        assert self._edge_index, 'edge_index is not initialized please check the process function'

        return self._edge_index

    @property
    def node_features(self):
        r"""return node_features
        """
        assert self._node_features, 'node_features is not initialized please check the process function'
        return self._node_features

    @property
    def edge_features(self):
        r"""return edge_features
        """
        assert self._edge_features, 'edge_features is not initialized please check the process function'
        return self._edge_features

    @property
    def timestamp(self):
        r"""return timestamp.
        """
        assert self._timestamp, 'timestamp is not initialized please check the process function'
        return self._timestamp

    @property
    def node_label(self):
        r"""return node label.
        """
        assert self._node_label, 'node_label is not initialized please check the process function'
        return self._node_label

    @property
    def node_state(self):
        r"""return _node_state
        """
        assert self._node_state, 'node_state is not initialized please check the process function'

        return self._node_state


