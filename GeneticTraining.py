from __future__ import print_function
from __future__ import division
from loaders.HDF5torch import HDF5generator # generator for data
seed = 0 # seed for reproducibility
import numpy as np
np.random.seed(seed)
import glob
import time
import torch
#from memory_profiler import profile
from analysis.utils import common as com
# added for reproducibility
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
from torchsummary import summary
from architectures.GeneticArch import arch 
from training.GA import GA
import copy

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
  parser.add_argument('--epochs', action='store', type=int, default=1000, help='Number of epochs to train for.')
  parser.add_argument('--batch_size', action='store', type=int, default=128, help='batch size per update')
  parser.add_argument('--train_ratio', action='store', default=0.9, help='test train ratio')
  parser.add_argument('--shuffle', action='store', default=False, help='shuffle data')
  parser.add_argument('--datapath', action='store', type=str, default="/bigdata/shared/LCD/NewV1/", help='base path for training data')
  parser.add_argument('--modelpath', action='store', type=str, default="results/models/", help='path to save model')
  parser.add_argument('--historypath', action='store', type=str, default="results/history/", help='path to save loss history')
  parser.add_argument('--mutation_power', action='store', default=0.02, help='this parameter is similar to lr and controls change in value of weight to be mutated')
  parser.add_argument('--mutation_rate', action='store', default=1, help='this parameter determines the number of weights to be mutated')
  parser.add_argument('--num_parents', action='store', type=int, default=4, help='initial random population')
  parser.add_argument('--num_children', action='store', type=int, default=4, help='number of children')
  parser.add_argument('--num_events', action='store', type=int, default=5000, help='number of training events')
  parser.add_argument('--particles', nargs='+', default=['Ele'], help='particles used in training (Types should be written without quotes delimited by spaces)')
  parser.add_argument('--dim', action='store', type=int, default=2, help='Number of data dimensions to use')
  parser.add_argument('--dformat', action='store', type=str, default='channels_first')
  parser.add_argument('--name', action='store', type=str, default="GA_5000events", help='identifier for a training')
  return parser

# percentage loss
def mape(pred, target):
   loss = torch.mean(torch.abs(target-pred)/target)
   return loss

def get_mutation_rate_plan5(mu, epoch, size):
   if epoch < 10:
     return 1.0
   elif epoch < 100:
     return 0.1
   else:
     return 0.01

def get_mutation_rate_plan4(mu, epoch, size):
   if epoch < 10:
     return 1.0
   elif epoch < 500:
     return 0.1
   else:
     return 0.01

def get_mutation_rate_plan3(mu, epoch, size): #plan 3
   if mu > 0.1:
     return mu-0.01
   else:
     return 0.1

#@profile
def get_mutation_rate_plan1(mu, epoch, size): #plan 1
   return 1.0

#@profile
def get_mutation_rate_plan2(mu, epoch, size): #plan 2
   return 0.1

def get_mutation_linear(mp, epoch, total_epochs):
   return 0.1 - (((0.1-0.001) * epoch)/total_epochs)

def get_mutation_exp(mp, epoch, total_epochs):
   #return 0.1 * np.exp(-6.0 * epoch/total_epochs)
   return 0.1 * np.exp(-6.9 * epoch/total_epochs)

def get_mutation_decay(mp, epoch, total_epochs):
   return 0.1 * 0.99**epoch

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
   mutation_power = params.mutation_power
   mutation_rate = params.mutation_rate
   num_parents = params.num_parents
   num_children = params.num_children
   num_evenst = params.num_events
   dim = params.dim
   dformat = params.dformat
   input_shape = com.get_input_shape(dformat, dim)
       
   # define GA training object
   ga = GA(arch=arch, mutation_rate = mutation_rate, mutation_power = mutation_power)
   ga.num_parents = num_parents
   ga.num_children = num_children
   ga.random_update = True
   summary(ga.arch().cuda(), input_size=input_shape)
   print('The number of paramerets is {}.'.format(ga.size))
   # create random population
   parents = ga.random_population()
   criterion = mape
   
   train_files, test_files = com.divide_train_test(datapath, particles, train_ratio)
   loss_list=[]
   train_loss_list = []
   test_loss_list = []
   time_list =[]
   best_loss = 100
   batch_iter = 0
   for epoch in range(epochs):
     init = time.time()
     # Test the model
     train_files = train_files[:1]     
     train_loader = HDF5generator(train_files, batch_size=batch_size, shuffle=shuffle, num_events=num_events)
     if epoch ==0:
        total_iter = train_loader.total_batches * epochs
        print(total_iter) 
     loss =[]
     print('Train generator initialized in {} sec.'.format(time.time()-init))
     ga.mutation_rate = get_mutation_rate_plan2(ga.mutation_rate, epoch, ga.size)
     for i, data in enumerate(train_loader):
        if (epoch==0) and (i==0):
          print('Evaluating random population.........')
          for i, model in enumerate(parents):
            model.eval()
            with torch.no_grad():
              outputs = model(data['ECAL'])
              loss = criterion(outputs, data['target'])
              if loss.item() < best_loss:
                best_loss = loss.item()
                best_parent = copy.deepcopy(model)
                print('best parent', best_loss)
          parents = []
        ga.mutation_power = get_mutation_exp(ga.mutation_power, batch_iter, total_iter)
        batch_iter+=1
        children = ga.return_children(best_parent)
        for j, model in enumerate(children):
           model.eval()
           with torch.no_grad():
              outputs = model(data['ECAL'])
              loss = criterion(outputs, data['target'])
              if loss.item() <= best_loss:
                best_loss = loss.item()
                best_parent = copy.deepcopy(model)
                print('best child loss = {}'.format(best_loss))
     
     train_loss_list.append(best_loss)
     time_list.append(time.time()-init)
     print('Epoch={} Mutation rate = {} Mutation power = {} Time taken ={} '.format(epoch, ga.mutation_rate, ga.mutation_power, time.time()-init))    
     # Test the model
     test_loader = HDF5generator(test_files, batch_size=batch_size, shuffle=shuffle, num_events=20000)
     best_parent.eval()
     test_init = time.time()
     with torch.no_grad():
       total = 0
       loss_list=[]
       for data in test_loader:
         outputs = best_parent(data['ECAL'])
         loss = criterion(outputs, data['target'])
         loss_list.append(loss.item())
       test_loss_list.append(np.mean(loss_list))
     print('================================================================')
     print('Train loss = {} Test loss={}'.format(train_loss_list[-1], test_loss_list[-1]))
     print('Test time = {} sec'.format(time.time()-test_init))
     pickle.dump({'train': train_loss_list, 'test': test_loss_list, 'timing':time_list}, open(HISTORY_PATH + historyfile, 'wb'))
     # Save the model and plot
     torch.save(model.state_dict(), MODEL_STORE_PATH + modelfile)

if __name__ == "__main__":
    training()
