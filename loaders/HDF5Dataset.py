import h5py
#import helpers
import numpy as np
from pathlib import Path
import glob
import torch
from torch.utils import data
#from torch.utils.data.DataLoader
import time
class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data=False, num_events=50000, data_cache_size=1, transform=None):
        super().__init__()
        self.transform = transform
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        # Search for all h5 files
        #files = sorted(glob.glob(file_path + '*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')
        self.events=[]
        init = time.time()
        for h5dataset_fp in files:
            print(h5dataset_fp)
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            self._get_events(str(h5dataset_fp.resolve()), load_data)
            self.num_events = sum(self.events)
            if self.num_events > num_events:
                self.num_events = num_events
                print(self.num_events)
                print('Time taken is {} sec'.format(time.time()-init))
                break
            
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for dname, ds in h5_file.items():
                # if data is not loaded its cache index is -1
                if dname in ['ECAL', 'target']:
                  idx = -1
                  if load_data:
                     # add data to the data cache
                     idx = self._add_to_cache(ds.value, file_path)
                  self.data_info.append({'file_path': file_path, 'type': dname, 'cache_idx': idx})
                                                                                                                                                                                                            
    def __getitem__(self, index):
        # get data
        x = self.get_data('ECAL', index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)
        y = self.get_data('target', index)[:, 1]
        print(y.shape)
        print(y[:5])

        y = torch.from_numpy(y)

        return (x, y)

    def __len__(self):
        return self.num_events
    
    def _get_events(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            events=0
            for dname, ds in h5_file.items():
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(ds.value, file_path)
                    
                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'data' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                if dname=='ECAL':
                   self.events.append(ds.value.shape[0])

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for dname, ds in h5_file.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    if dname in ['ECAL', 'target']:
                       idx = self._add_to_cache(ds.value, file_path)

                       # find the beginning index of the hdf5 file we are looking for
                       file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                       # the data info should have the same index since we loaded it in the same way
                       self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


def main():
    import HDF5Dataset as d
    file_path = '/bigdata/shared/LCD/NewV1/EleEscan/'

    dataset = d.HDF5Dataset(file_path=file_path, recursive=True, load_data=False, transform=None)

    dataloader = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    print(dataset.data_info)
    init = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        print(len(sample_batched))
        print("time {} sec".format(time.time()-init))
        if i_batch > 10:
            break

if __name__ == '__main__':
    main()
