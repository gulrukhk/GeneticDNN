import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np

class GA():
   def __init__(self, arch, mutation_rate=0.01, mutation_power=0.02):
     # initial params     
     self.mutation_rate= mutation_rate  
     self.mutation_power = mutation_power
     self.weight_init= torch.nn.init.xavier_uniform_
     self.arch = arch
     self.size = self.count_parameters(arch())
     self.num_parents = 1
     self.num_children = 4
     self.random_update = False
     self.use_bias = True
          
   # count net params
   def count_parameters(self, model):
     return sum(p.numel() for p in model.parameters())

   # initialize weights
   def init_weights(self, m):
     if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
      self.weight_init(m.weight)
      m.bias.data.fill_(0.00)
   
   # make a set of random individuals 
   def random_population(self):
     parents = []
     for _ in range(self.num_parents):
        parent = self.arch()
        parent.cuda()
        for param in parent.parameters():
            param.requires_grad = False
        self.init_weights(parent)
        parents.append(parent)
     return parents

   # mutate an individual
   def mutate(self, parent):
     # new individual is a copy of old
     child = copy.deepcopy(parent)
     
     # calculate number of mutations to apply and get random indexes
     param_size = self.count_parameters(parent)
     mutation_size = int(param_size * self.mutation_rate)
     genes_to_mutate = np.random.randint(param_size, size=mutation_size) # random indexes

     # sort indexes in ascending prder
     genes_to_mutate.sort() # sort in ascending order

     # get an array of updates
     update = self.mutation_power * np.random.randn(mutation_size) # random updates
     
     # update weights
     index = 0 # genes updated
     prev_size = 0 # size of all previous layers
     for num, p in enumerate(child.parameters()): # loop through layers
       for i, gene in enumerate(genes_to_mutate[index:]): # loop through indexes to be mutated
         if gene < (prev_size + np.prod(p.size())): # if gene in current layer
           loc = np.unravel_index(gene - prev_size, p.size()) # location of weight corressponding to gene
           if self.random_update:
              new_val = update[index] # updated weight will be equal to random mutation
           else:
              new_val = p[loc] + update[index] # updated weight will be equal to sum of weight and random mutation

           # update weights
           if self.use_bias:
             p[loc]= new_val
           elif len(p.shape)> 1: # to remove bias
             p[loc]= new_val
           index +=1 
       
       # update previous param size
       prev_size+= np.prod(p.size())
     return child
 
   # get mutated individuals
   def return_children(self, parent):
     children = []
     for i in range(self.num_children):
       children.append(self.mutate(parent))
     return children


  
