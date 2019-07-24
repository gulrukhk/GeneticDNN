from __future__ import unicode_literals
import glob
import torch.utils.data as data
from timeit import default_timer as timer
import blunt_loader as loader
import h5py
import numpy as np
from memory_profiler import profile
#################################
# Load files and set up loaders #
#################################

#base_path = "/public/data/calo/RandomAngle/CLIC/"
base_path = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/"
particle =['Ele']
sample_path = []
for p in particle:
    sample_path.append(base_path + p + 'Escan/*.h5')
print(sample_path)

batch_size = 128
n_workers = 0
train_ratio = 0.9

n_classes = 1
# gather sample files for each type of particle
class_files = [[]] * n_classes
for i, class_path in enumerate(sample_path):
    class_files[i] = glob.glob(class_path)
files_per_class = min([len(files) for files in class_files])

# split the train, test, and validation files
# get lists of [[class1_file1, class2_file1], [class1_file2, class2_file2], ...]
train_files = []
test_files = []
n_train = int(files_per_class * train_ratio)
for i in range(files_per_class):
    new_files = []
    for j in range(n_classes):
        new_files.append(class_files[j][i])
    if i < n_train:
        train_files.append(new_files)
    else:
        test_files.append(new_files)

# prepare the generators
print('Preparing data loaders')
train_set = loader.HDF5Dataset(train_files)
test_set = loader.HDF5Dataset(test_files)
train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=1)#, sampler=loader.OrderedRandomSampler(train_set), num_workers=n_workers)
test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=1)#, sampler=loader.OrderedRandomSampler(test_set), num_workers=n_workers)

# lists of events
train_events = list(train_loader.sampler.__iter__())
test_events = list(test_loader.sampler.__iter__())


#########
# Tests #
#########
#get data for training
def GetDataAngle(datafile, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4):
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    ang = np.array(f.get(angtype))
    X =np.array(f.get('ECAL'))* xscale
    Y=np.array(f.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ang = ang.astype(np.float32)
    X = np.expand_dims(X, axis=-1)
    ecal = np.sum(X, axis=(1, 2, 3))
    if xpower !=1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal
                                                                
def time_my_loading():
    print('starting')
    file_index=0
    batch_size=128
    start = timer()
    x, y, ang, ecal = GetDataAngle(train_files[0][0])
    print('loading and processing file took {} sec'.format(timer()-start))
    start = timer()
    for index in np.arange(10):
        image_batch = x[(file_index * batch_size):(file_index  + 1) * batch_size]
        energy_batch = y[(file_index * batch_size):(file_index + 1) * batch_size]
        ecal_batch = ecal[(file_index *  batch_size):(file_index + 1) * batch_size]
        ang_batch = ang[(file_index * batch_size):(file_index + 1) * batch_size]
        print(index)
        print('{} seconds taken'.format(timer()-start))
    end = timer()
    print("Loading 10 batches took {} seconds.".format(end-start))
        
    
def time_loading():
    start = timer()
    print('starting')
    for i, _ in enumerate(train_loader):
        print(i)
        for key in _:
            if key=='energy':
                print(_[key][:5])
        print('{} seconds taken'.format(timer()-start))
        if i==3:
            break
    end = timer()
    print("Loading 10 batches took {} seconds.".format(end-start))


def test_event_overlap():
    train_files_list = [i for j in train_files for i in j]
    test_files_list = [i for j in test_files for i in j]
    all_files_list = train_files_list + test_files_list
    assert len(set(all_files_list)) == len(set(train_files_list)) + len(set(test_files_list))


def test_event_coverage():
    assert len(set(train_events)) + len(set(test_events)) == len(train_set) + len(test_set)


#################
# Perform tests #
#################

if __name__ == "__main__":
 time_my_loading()
 
