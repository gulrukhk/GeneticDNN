# make a directory                                                             
import glob
import numpy as np
                                                                  
def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)                                         
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
          raise exception

def fill_graph(graph, x, y):
   n = x.shape[0]
   for i in np.arange(n):
     graph.SetPoint(int(i), x[i], y[i])

def get_input_shape(dformat='channels_first', dim=2):
   if dim == 2:
     image_shape = (25, 25)
   else:
     image_shape = (25, 25, 25)

   if dformat == 'channels_first':
     input_shape = (1,) + image_shape
   else:
     input_shape = image_shape + (1,)
   return input_shape

def divide_train_test(base_path, particles, train_ratio):
   sample_path = []
   for p in particles:
     sample_path.append(base_path + p + 'Escan/*.h5')
   n_classes = len(particles)
   # gather sample files for each type of particle                                                                                                
   class_files = [[]] * n_classes
   for i, class_path in enumerate(sample_path):
      class_files[i] = sorted(glob.glob(class_path))
   files_per_class = min([len(files) for files in class_files])
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
   return train_files, test_files

# Fill root profile
def fill_profile(prof, x, y):
  n = x.shape[0]
  for i in range(n):
    prof.Fill(x[i], y[i])
