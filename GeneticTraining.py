from __future__ import print_function
from loaders.HDF5torch import HDF5generator # generator for data
seed = 0 # seed for reproducibility
import numpy as np
np.random.seed(seed)
import glob
import time
import torch

# added for reproducibility
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
from torchsummary import summary
from architectures.GeneticArch import arch 
from training.GA import GA

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import setGPU
except:
    pass

import argparse

def get_parser():
  parser = argparse.ArgumentParser(description='GA Params' )
  parser.add_argument('--epochs', action='store', type=int, default=500, help='Number of epochs to train for.')
  parser.add_argument('--batch_size', action='store', type=int, default=128, help='batch size per update')
  parser.add_argument('--train_ratio', action='store', default=0.9, help='test train ratio')
  parser.add_argument('--shuffle', action='store', default=False, help='shuffle data')
  parser.add_argument('--datapath', action='store', type=str, default="/bigdata/shared/LCD/NewV1/", help='base path for training data')
  parser.add_argument('--modelpath', action='store', type=str, default="results/models/", help='path to save model')
  parser.add_argument('--historypath', action='store', type=str, default="results/history/", help='path to save loss history')
  parser.add_argument('--name', action='store', type=str, default="GA", help='identifier for a training')
  parser.add_argument('--particles', nargs='+', default=['Ele'], help='particles used in training (Types should be written without quotes delimited by spaces)')
  parser.add_argument('--dim', action='store', type=int, default=2, help='Number of data dimensions to use')
  parser.add_argument('--dformat', action='store', type=str, default='channels_first')
  return parser

# percentage loss
def mape(pred, target):
   loss = torch.mean(torch.abs(target-pred)/target)
   return loss

def training():
   parser = get_parser()
   params = parser.parse_args()
   epochs = params.epochs
   batch_size = params.batch_size
   train_ratio = params.train_ratio
   shuffle = params.shuffle
   datapath = params.datapath
   MODEL_STORE_PATH = params.modelpath
   HISTORY_PATH = params.historypath
   modelfile = params.name + '_model.ckpt'
   historyfile = params.name + '_loss_history.pkl'
   particles = params.particles
   dim = params.dim
   dformat = params.dformat
   print(particles)
   input_shape = get_input_shape(dformat, dim)
   # define GA training object
   ga = GA(model=arch())
   summary(ga.model.cuda(), input_size=input_shape)

   # create random population
   parents = ga.random_population()
   criterion = mape
   
   train_files, test_files = divide_train_test(datapath, particles, train_ratio)
   init = time.time()
   loss_list=[]
   train_loss_list = []
   test_loss_list = []
   best_loss = 100
   ga.mutation_rate = 0.1
   for epoch in range(epochs):
     print('Epoch={}'.format(epoch))
     # Test the model
     test_loader = HDF5generator(test_files, batch_size=batch_size, shuffle=shuffle, num_events=20000)
     loss =[]
     for i, data in enumerate(test_loader):
        if (epoch==0) and (i==0):
          for model in parents:
            model.eval()
            with torch.no_grad():
              outputs = model(data['ECAL'])
              loss = criterion(outputs, data['target'])
              if loss.item() < best_loss:
                best_loss = loss.item()
                best_parent = model
                print('best parent', best_loss)     
        children = ga.return_children(best_parent)
        if epoch > 90:
           ga.mutation_rate =0.05
        if epoch > 190:
           ga.mutation_rate =0.01
        
        for model in children:
           model.eval()
           with torch.no_grad():
              outputs = model(data['ECAL'])
              loss = criterion(outputs, data['target'])
              if loss.item() <= best_loss:
                best_loss = loss.item()
                best_parent = model
                print('best child', best_loss)
     test_loss_list.append(best_loss)    
     print('Test loss of the model on the 20000 test images: {} %'.format(best_loss))
     pickle.dump({'train': train_loss_list, 'test': test_loss_list}, open(HISTORY_PATH + historyfile, 'wb'))
     # Save the model and plot
     torch.save(model.state_dict(), MODEL_STORE_PATH + modelfile)

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
 
if __name__ == "__main__":
    training()
