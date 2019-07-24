import numpy as np
import h5py
import glob
import time

# generator for hdf5 data files
class HDF5generator():
    def __init__(self, data_files, batch_size, num_events=None, shuffle=False):
       self.files = data_files  # list of data files 
       self.fiter = iter(self.files) # file iterator
       self.batch_size = batch_size # batch size
       self.shuffle = shuffle # if loaded events should be shuffled (useful for classification)
       self.keys = ['ECAL', 'energy', 'target', 'theta', 'class', 'sum'] # keys for hdf5 data set 
       if num_events: # number of events to use, if set to None then all events will be used
          self.num_events=num_events
       else:
          self.num_events=self.count_events()
       self.particles=['Ele', 'Pi0'] # particle types
       self.xpower = 1 # if Ecal energies to be raised to power
       self.xscale = 0.5 # if scaling is to be applied
       self.yscale = 100.0 # primary energy division factor
       self.dscale=1 # remove any scaling already present in dataset
       self.d=2 # number of dimensions for image
       self.data = self.read_data(self.fiter.next()) # read initial file
       self.num_batches_loaded = int(self.file_events/self.batch_size) # number of batches in cache
       self.total_batches = int(self.num_events/self.batch_size)# total batches
       self.index =0 # index for events
       self.batches_read =0 # number of batches read

    # iteration 
    def __iter__(self):
       while True:
          batch={}
          # end training if all batches are used
          if self.batches_read >= self.total_batches:
              raise StopIteration
          # if all batches currently loaded were used open the next file
          if self.index >= (self.num_batches_loaded * self.batch_size):
              for key in self.data:
                  batch[key]=self.data[key][self.index:self.batch_size + self.index] # get remaining events in batch
              self.data = self.read_data(self.fiter.next()) # read data from next file
              for i, key in enumerate(self.data): 
                  self.data[key] = np.concatenate((batch[key], self.data[key]), axis=0) # concatenate remaining events to new data
                  if i==0:
                      self.file_events = self.data[key].shape[0] # events loaded
                      self.num_batches_loaded = int(self.file_events/self.batch_size) # batches loaded
          for key in self.data:
            batch[key]=self.data[key][self.index:self.batch_size + self.index] # get a batch
          self.index += self.batch_size # increment index by batch size
          self.batches_read +=1 # increment batches read
          yield batch # yield the current batch

    # count total events 
    def count_events(self):
        l = 0
        for datafiles in self.files:
            for datafile in datafiles:
              f=h5py.File(datafile,'r')
              l+= f.get(self.keys[0]).shape[0]    
        return l

    # read data from file and preprocess
    def read_data(self, dfiles, nevents=10000):
        data = {}
        for n, dfile in enumerate(dfiles):
          print('reading {}'.format(dfile))
          f=h5py.File(dfile,'r')
          for key in self.keys:
              if key in f.keys():
                 if n==0: # first file tuple
                   if key=='target':
                       data[key] = np.array(f.get(key)[:nevents, 1], dtype=np.float32) # fixed angle primary energy 
                   else:
                       data[key] = np.array(f.get(key)[:nevents], dtype=np.float32) # other data sets
                 else: # all other file tuples
                   if key=='target':
                        data[key] = np.concatenate((data[key], np.array(f.get(key)[:nevents, 1], dtype=np.float32)), axis=0)
                   else:
                        data[key] = np.concatenate((data[key], np.array(f.get(key)[:nevents], dtype=np.float32)), axis=0)
              elif key == 'class': # class for first particle type will be zero and then 1 ....
                 if n==0: 
                   data[key] = n * np.ones(data['ECAL'].shape[0], dtype=np.float32)
                 else:
                   data[key] = np.concatenate((data[key], n * np.ones(data['ECAL'].shape[0], dtype=np.float32)), axis=0)

        self.file_events = data['ECAL'].shape[0] # get events loaded
        if 'sum' in self.keys: # if sum of energy depositions is to be included
            data['sum']=np.sum(data['ECAL'], axis=(1, 2, 3))

        # shuffling    
        if self.shuffle: 
          print('shuffling')
          for i, key in enumerate(data):
            if i==0:
               indexes = np.random.permutation(self.file_events)
            data[key] = data[key][indexes]
        self.preproc(data)
        self.index = 0
        return data 

    # any preprocessing for loaded data
    def preproc(self, data):
       for key in data:
           if key=='ECAL':
               if self.xscale!=1:
                   data[key] = data[key]* self.xscale
               if self.xpower!=1:
                   data[key] = np.power(data[key], self.xpower)
               if self.dscale !=1:
                   data[key] = data[key]/self.dscale
               if self.d==2:
                   data[key]= np.sum(data[key], axis=1)
           if key=='energy':
              if self.yscale!=1:
                  data[key] = data[key]/self.yscale
           if key=='target':
               if self.yscale!=1:
                  data[key] = data[key]/self.yscale
           if key=='sum':
               if self.xscale!=1:
                  data[key] = data[key]* self.xscale

                  
def test():
   #base_path = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/" # variable angle
   base_path = "/bigdata/shared/LCD/NewV1/" # fixed angle
   particle =['Ele', 'Pi0'] # particle types
   sample_path = [] 
   for p in particle:
     sample_path.append(base_path + p + 'Escan/*.h5')
   
   batch_size = 128
   train_ratio = 0.9
   shuffle=True

   n_classes = len(particle)
   # gather sample files for each type of particle
   class_files = [[]] * n_classes
   for i, class_path in enumerate(sample_path):
      class_files[i] = sorted(glob.glob(class_path))
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
   init = time.time()
   # data generator for train files
   data_gen = HDF5generator(train_files[:3], batch_size=batch_size, shuffle=shuffle) #, num_events=5000)
   data_gen.particles=particle
   print('Initialization in {} sec'.format(time.time()-init))
   init = time.time()
   for key in data_gen.data:
       print(key, data_gen.data[key].shape)
   for i, data in enumerate(data_gen):
      print(i)
      for key in data:
        if key =='ECAL':
            print(key, data[key].shape)
        else:
            print(key, data[key][:5])
      print('{} batches took {} sec'.format(i, time.time()-init))        
   print('{} batches loaded in {} sec'.format(i, time.time()-init))

if __name__ == "__main__":
   test()
