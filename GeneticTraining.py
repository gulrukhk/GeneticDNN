from __future__ import print_function
from __future__ import division

# setting seed
seed = 0 
import numpy as np
np.random.seed(seed)
import torch

# Also added for reproducibility
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# imports from torch
import torch.nn.functional as F
from torchsummary import summary

# user classes
from loaders.HDF5torch import HDF5generator # generator for data
from architectures.GeneticArch import arch  # architecture 
from training.GA import GA                  # genetic training
from analysis.utils import common as com    # utility functions

# imports from python
import copy
import glob
import time
import argparse
#from memory_profiler import profile 
try:
    import cPickle as pickle
except ImportError:
    import pickle

# setting gpu
try:
    import setGPU
except:
    pass

# command line arguments
def get_parser():
  parser = argparse.ArgumentParser(description='GA Params' )
  parser.add_argument('--epochs', action='store', type=int, default=50, help='Number of epochs to train for.')
  parser.add_argument('--batch_size', action='store', type=int, default=128, help='batch size per update')
  parser.add_argument('--train_ratio', action='store', default=0.9, help='test train ratio')
  parser.add_argument('--shuffle', action='store', default=False, help='shuffle data')
  parser.add_argument('--datapath', action='store', type=str, default="/bigdata/shared/LCD/NewV1/", help='base path for training data')
  parser.add_argument('--modelpath', action='store', type=str, default="results/models/", help='path to save model')
  parser.add_argument('--historypath', action='store', type=str, default="results/history/", help='path to save loss history')
  parser.add_argument('--mutation_power', action='store', default=1.0, help='this parameter controls the magnitude of weight updates/random weights')
  parser.add_argument('--random_mutation', action='store', default=True, help='mutate weights randomly instead of adding updates')
  parser.add_argument('--use_bias', action='store', default=False, help='using bias')
  parser.add_argument('--mutation_rate', action='store', default=0.1, help='this parameter determines the number of weights to be mutated')
  parser.add_argument('--num_parents', action='store', type=int, default=8, help='initial random population')
  parser.add_argument('--num_children', action='store', type=int, default=8, help='number of children')
  parser.add_argument('--num_events_train', action='store', type=int, default=5000, help='number of training events')
  parser.add_argument('--num_events_test', action='store', type=int, default=10000, help='number of training events')
  parser.add_argument('--particles', nargs='+', default=['Ele'], help='particles used in training (Types should be written without quotes delimited by spaces)')
  parser.add_argument('--dim', action='store', type=int, default=2, help='Number of data dimensions to use')
  parser.add_argument('--dformat', action='store', type=str, default='channels_first')
  parser.add_argument('--name', action='store', type=str, default="GA_training", help='identifier for a training')
  return parser

# percentage loss
def mape(pred, target):
   loss = torch.mean(torch.abs(target-pred)/target)
   return 100.0 * loss

# get mutation rate according to epochs
def get_mutation_rate(mu, epoch, total_epochs):
   if epoch < 10:
     return 0.1
   elif epoch < 20:
     return 0.01
   else:
     return 0.001

# get mutation rate varying linearly from 0.1 to 0.001
def get_mutation_linear(mp, epoch, total_epochs):
   return 0.1 - (((0.1-0.001) * epoch)/total_epochs)

# get mutation rate decaying exponentially from 0.1 to 0.0001
def get_mutation_exp(mp, epoch, total_epochs):
   #return 0.1 * np.exp(-6.0 * epoch/total_epochs)
   return 0.1 * np.exp(-6.9 * epoch/total_epochs)

# get mutation rate with a decay factor
def get_mutation_decay(mp, epoch, total_epochs):
   return 0.1 * 0.99**epoch

def GA_training():
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
   mutation_power = params.mutation_power
   random_mutation = params.random_mutation
   use_bias = params.use_bias
   mutation_rate = params.mutation_rate
   num_parents = params.num_parents
   num_children = params.num_children
   num_events_train = params.num_events_train
   num_events_test = params.num_events_test
   dim = params.dim
   dformat = params.dformat
   input_shape = com.get_input_shape(dformat, dim)
       
   # define GA training object
   ga = GA(arch=arch, mutation_rate = mutation_rate, mutation_power = mutation_power)
   
   # setting up training params
   ga.num_parents = num_parents
   ga.num_children = num_children
   ga.random_update = random_mutation
   ga.use_bias = use_bias

   # print info about architecture
   summary(ga.arch().cuda(), input_size=input_shape)
   print('The number of paramerets is {}.'.format(ga.size))

   # divide available data files into train and test   
   train_files, test_files = com.divide_train_test(datapath, particles, train_ratio)

   # initialization
   train_loss_list = []
   train_loss_list_batch = []
   test_loss_list = []
   time_list =[]
   time_list_batch =[]
   best_loss = 100
   batch_iter = 0

   # create random population
   parents = ga.random_population()
   criterion = mape

   # start training
   for epoch in range(epochs):
     init = time.time()
     
     # initialize data loader
     train_loader = HDF5generator(train_files, batch_size=batch_size, shuffle=shuffle, num_events=num_events_train)
     if epoch ==0:
        total_iter = train_loader.total_batches * epochs # total training iterations or batches
     
     # get the mutation rate for current epoch/batch 
     ga.mutation_rate = get_mutation_rate(ga.mutation_rate, epoch, epochs)
     
     # training for current epoch
     for i, data in enumerate(train_loader):
        init_batch = time.time()

        # For first batch of first epoch evaluate parent population
        if (epoch==0) and (i==0):
          print('Evaluating random population.........')
          for j, model in enumerate(parents):
            model.eval()
            with torch.no_grad():
              outputs = model(data['ECAL'])
              loss = criterion(outputs, data['target'])
              if loss.item() < best_loss:
                best_loss = loss.item()
                best_parent = copy.deepcopy(model)
                print('best parent', best_loss)
          
          # keep only best parent and delete the rest
          parents = []
       
        # get mp if not using random mutation
        if not random_mutation:
           ga.mutation_power = get_mutation_exp(ga.mutation_power, batch_iter, total_iter)
        # mutate best parent
        children = ga.return_children(best_parent)
        
        # evaluate children
        for j, model in enumerate(children):
           model.eval()
           with torch.no_grad():
              outputs = model(data['ECAL'])
              loss = criterion(outputs, data['target'])
              if loss.item() <= best_loss:
                best_loss = loss.item()
                best_parent = copy.deepcopy(model)
                print('best child loss = {}'.format(best_loss))
        
        # update train loss history for batch
        train_loss_list_batch.append(best_loss)
        time_list_batch.append(time.time()-init_batch) 
        batch_iter+=1
     
     # update epoch loss history
     train_loss_list.append(best_loss)
     time_list.append(time.time()-init)
     
     # Test the model
     test_init = time.time()
     test_loader = HDF5generator(test_files, batch_size=batch_size, shuffle=shuffle, num_events=num_events_test)
     best_parent.eval()
     with torch.no_grad():
       total = 0
       loss_list=[]
       for data in test_loader:
         outputs = best_parent(data['ECAL'])
         loss = criterion(outputs, data['target'])
         loss_list.append(loss.item())
       
     # update test loss history
     test_loss_list.append(np.mean(loss_list))

     # display info
     print('Epoch = {}'.format(epoch))
     print('Train loss = {:.4f}  %   Train time ={:.4f} sec'.format(train_loss_list[-1], time_list[-1]))
     print('Test loss  = {:.4f}  %   Test time = {:.4f} sec'.format(test_loss_list[-1], time.time()-test_init))
     print('================================================================')

     # save loss history
     pickle.dump({'train': train_loss_list, 'test': test_loss_list, 'timing':time_list, 'train_batch': train_loss_list_batch, 'timing_batch':time_list_batch}, open(HISTORY_PATH + historyfile, 'wb'))
     
     # save the model
     torch.save(model.state_dict(), MODEL_STORE_PATH + modelfile)

if __name__ == "__main__":
    GA_training()
